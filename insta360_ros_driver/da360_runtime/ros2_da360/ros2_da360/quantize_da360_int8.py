#!/usr/bin/env python3
"""Export DA360 and build a calibrated TensorRT INT8 engine.

The generated engine is static-shape and keeps the ROS/Python bridge
unchanged.  The runtime selects it automatically when the model path ends in
``.engine`` or ``.plan``.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import tensorrt as trt


MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
INV_STD = np.asarray(
    [1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225],
    dtype=np.float32,
).reshape(1, 3, 1, 1)


def load_frames(path):
    loaded = np.load(path, mmap_mode='r')
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            names = list(loaded.files)
            if not names:
                raise ValueError(f'calibration archive is empty: {path}')
            frames = loaded[names[0]]
        finally:
            loaded.close()
    else:
        frames = loaded
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(
            f'calibration data must have shape [N,H,W,3], got {frames.shape}'
        )
    if frames.dtype != np.uint8:
        frames = np.asarray(np.clip(frames, 0, 255), dtype=np.uint8)
    return frames


def normalize_frame(frame):
    if frame.dtype == np.uint8:
        value = frame.astype(np.float32) / 255.0
    else:
        value = np.asarray(frame, dtype=np.float32)
    value = value.transpose(2, 0, 1)[None, ...]
    value = (value - MEAN) * INV_STD
    return np.ascontiguousarray(value, dtype=np.float32)


class ExportWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tensor):
        return self.model(tensor)['pred_disp']


class TorchTensorCalibrator(trt.IInt8EntropyCalibrator2):
    """TensorRT calibrator backed by a persistent CUDA Torch tensor."""

    def __init__(self, frames, height, width, cache_path):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self._frames = frames
        self._height = height
        self._width = width
        self._cache_path = cache_path
        self._index = 0
        self._device_input = torch.empty(
            (1, 3, height, width),
            dtype=torch.float32,
            device='cuda',
        )

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        if self._index >= len(self._frames):
            return None
        value = normalize_frame(self._frames[self._index])
        if value.shape != (1, 3, self._height, self._width):
            raise ValueError(
                f'calibration frame shape mismatch: expected '
                f'{(self._height, self._width, 3)}, got {value.shape[2:] + (3,)}'
            )
        self._device_input.copy_(torch.from_numpy(value))
        torch.cuda.synchronize()
        self._index += 1
        return [int(self._device_input.data_ptr())]

    def read_calibration_cache(self):
        if self._cache_path.is_file():
            return self._cache_path.read_bytes()
        return None

    def write_calibration_cache(self, cache):
        self._cache_path.write_bytes(bytes(cache))


def load_reference_model(repo_root, checkpoint):
    os.environ['DA360_MIXED_PRECISION'] = '0'
    sys.path.insert(0, str(repo_root / 'ros2_da360'))
    from ros2_da360.da360_inference_worker import load_model

    model, device, height, width, _ = load_model(repo_root, checkpoint, '')
    if device.type != 'cuda':
        raise RuntimeError('TensorRT INT8 build requires a CUDA device')
    return model, device, height, width


def export_onnx(model, device, height, width, output_path):
    model.eval()
    wrapper = ExportWrapper(model).eval()
    sample = torch.zeros(
        (1, 3, height, width),
        dtype=torch.float32,
        device=device,
    )
    print(f'exporting static ONNX graph to {output_path}', flush=True)
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (sample,),
            str(output_path),
            input_names=['input'],
            output_names=['pred_disp'],
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
    import onnx

    graph = onnx.load(str(output_path))
    onnx.checker.check_model(graph)


def build_engine(onnx_path, engine_path, cache_path, frames, height, width, workspace_gb):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    with onnx_path.open('rb') as stream:
        if not parser.parse(stream.read()):
            errors = '\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f'TensorRT could not parse ONNX graph:\n{errors}')

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        int(workspace_gb * (1 << 30)),
    )
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.INT8)
    config.int8_calibrator = TorchTensorCalibrator(
        frames,
        height,
        width,
        cache_path,
    )
    print(
        f'building TensorRT INT8 engine with {len(frames)} calibration frames',
        flush=True,
    )
    start = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    elapsed = time.perf_counter() - start
    if serialized is None:
        raise RuntimeError('TensorRT returned no serialized engine')
    engine_path.write_bytes(bytes(serialized))
    print(
        f'built {engine_path} ({engine_path.stat().st_size / 1024.0 / 1024.0:.1f} MiB) '
        f'in {elapsed:.1f} s',
        flush=True,
    )


def depth_from_disparity(disparity):
    if isinstance(disparity, dict):
        disparity = disparity['pred_disp']
    depth = 1.0 / disparity.float()
    depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    depth_min = depth.amin(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    return depth / depth_min


def benchmark_and_compare(model, engine_path, frames, height, width, count):
    from ros2_da360.da360_inference_worker import TensorRTEnginePredictor

    predictor = TensorRTEnginePredictor(engine_path)
    samples = [
        torch.from_numpy(normalize_frame(frames[index])).cuda()
        for index in np.linspace(0, len(frames) - 1, min(count, len(frames)), dtype=int)
    ]
    model.mixed_precision = True

    with torch.inference_mode():
        for sample in samples[:2]:
            model(sample)
            predictor(sample)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for sample in samples:
            model(sample)
        torch.cuda.synchronize()
        torch_ms = (time.perf_counter() - start) * 1000.0 / len(samples)

        start = time.perf_counter()
        for sample in samples:
            predictor(sample)
        torch.cuda.synchronize()
        trt_ms = (time.perf_counter() - start) * 1000.0 / len(samples)

        abs_errors = []
        relative_errors = []
        for sample in samples:
            reference = depth_from_disparity(model(sample))
            quantized = depth_from_disparity(predictor(sample))
            difference = (quantized - reference).abs()
            abs_errors.append(float(difference.mean().cpu()))
            relative_errors.append(
                float((difference / reference.abs().clamp_min(1e-6)).mean().cpu())
            )

    result = {
        'torch_fp16_ms': torch_ms,
        'tensorrt_int8_ms': trt_ms,
        'speedup': torch_ms / trt_ms if trt_ms > 0 else 0.0,
        'mean_absolute_depth_error': float(np.mean(abs_errors)),
        'mean_relative_depth_error': float(np.mean(relative_errors)),
        'validation_frames': len(samples),
        'input_height': height,
        'input_width': width,
    }
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo-root', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--calibration', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--max-calibration', type=int, default=64)
    parser.add_argument('--validation-count', type=int, default=8)
    parser.add_argument('--workspace-gb', type=float, default=8.0)
    parser.add_argument(
        '--skip-build',
        action='store_true',
        help='reuse the existing engine in output-dir and only run validation',
    )
    args = parser.parse_args()

    args.repo_root = args.repo_root.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.calibration = args.calibration.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frames = load_frames(args.calibration)
    if args.max_calibration < 1:
        raise ValueError('--max-calibration must be positive')
    calibration_indices = np.linspace(
        0,
        len(frames) - 1,
        min(args.max_calibration, len(frames)),
        dtype=int,
    )
    calibration_frames = frames[calibration_indices]

    model, device, height, width = load_reference_model(
        args.repo_root,
        args.checkpoint,
    )
    if tuple(frames.shape[1:]) != (height, width, 3):
        raise ValueError(
            f'calibration frames are {tuple(frames.shape[1:])}, expected '
            f'{(height, width, 3)}'
        )

    onnx_path = args.output_dir / 'DA360_small_fp32.onnx'
    engine_path = args.output_dir / 'DA360_small_int8.engine'
    cache_path = args.output_dir / 'DA360_small_int8.cache'
    if args.skip_build:
        if not engine_path.is_file():
            raise FileNotFoundError(f'engine does not exist: {engine_path}')
    else:
        export_onnx(model, device, height, width, onnx_path)
        build_engine(
            onnx_path,
            engine_path,
            cache_path,
            calibration_frames,
            height,
            width,
            args.workspace_gb,
        )
    result = benchmark_and_compare(
        model,
        engine_path,
        frames,
        height,
        width,
        args.validation_count,
    )
    manifest = {
        'checkpoint': str(args.checkpoint),
        'calibration': str(args.calibration),
        'calibration_frames': len(calibration_frames),
        'engine': str(engine_path),
        'onnx': str(onnx_path),
        'result': result,
    }
    (args.output_dir / 'DA360_small_int8.json').write_text(
        json.dumps(manifest, indent=2) + '\n'
    )


if __name__ == '__main__':
    main()
