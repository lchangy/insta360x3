#!/usr/bin/env python3
"""DA360 inference worker used by the ROS 2 bridge.

The ROS-facing process and this worker are launched with a Python environment
containing CUDA Torch, NumPy, and OpenCV.  The worker
communicates raw BGR frames/depth arrays over stdin/stdout.  It is deliberately
not a ROS node; the ROS-facing process is the single node in
da360_realtime_node.py.
"""

import argparse
import contextlib
import os
import struct
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


REQUEST_HEADER = struct.Struct('<III')
RESPONSE_HEADER = struct.Struct('<III')


def read_exact(stream, size):
    data = bytearray()
    while len(data) < size:
        chunk = stream.read(size - len(data))
        if not chunk:
            return None
        data.extend(chunk)
    return bytes(data)


def read_exact_into(stream, buffer):
    view = memoryview(buffer)
    while view:
        count = stream.readinto(view)
        if not count:
            return False
        view = view[count:]
    return True


def write_all(stream, data):
    view = memoryview(data)
    while view:
        written = stream.write(view)
        if written is None:
            stream.flush()
            return
        view = view[written:]
    stream.flush()


def env_flag(name, default=True):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in ('0', 'false', 'no', 'off')


def load_model(repo_root, model_path, requested_net):
    sys.path.insert(0, str(repo_root))
    import networks

    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location='cpu')

    if not isinstance(checkpoint, dict):
        checkpoint = {'state_dict': checkpoint}

    net_name = requested_net or checkpoint.get('net', 'DA360')
    encoder = checkpoint.get('dinov2_encoder', 'vits')
    height = int(checkpoint.get('height', 518))
    width = int(checkpoint.get('width', 1036))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mixed_precision = device.type == 'cuda' and env_flag('DA360_MIXED_PRECISION', True)
    if device.type == 'cuda':
        try:
            torch.set_float32_matmul_precision('high')
        except AttributeError:
            pass

    net_class = getattr(networks, net_name)
    model_kwargs = {'dinov2_encoder': encoder}
    if net_name == 'DA360':
        model_kwargs['mixed_precision'] = mixed_precision
    model = net_class(height, width, **model_kwargs)
    state = model.state_dict()
    weights = {key: value for key, value in checkpoint.items() if key in state}
    model.load_state_dict(weights, strict=False)

    model.to(device)
    model.eval()
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    return model, device, height, width, mixed_precision


class TensorRTEnginePredictor:
    """TensorRT v10 predictor using Torch CUDA tensors for the buffers."""

    def __init__(self, engine_path):
        import tensorrt as trt

        self._trt = trt
        self._logger = trt.Logger(trt.Logger.ERROR)
        self._runtime = trt.Runtime(self._logger)
        with open(engine_path, 'rb') as stream:
            self._engine = self._runtime.deserialize_cuda_engine(stream.read())
        if self._engine is None:
            raise RuntimeError(f'failed to deserialize TensorRT engine: {engine_path}')
        self._context = self._engine.create_execution_context()
        if self._context is None:
            raise RuntimeError('failed to create TensorRT execution context')

        self._input_name = None
        self._output_name = None
        for index in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(index)
            mode = self._engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self._input_name = name
            else:
                self._output_name = name
        if self._input_name is None or self._output_name is None:
            raise RuntimeError('TensorRT engine must have one input and one output')

        input_shape = tuple(self._engine.get_tensor_shape(self._input_name))
        if len(input_shape) != 4 or any(int(value) <= 0 for value in input_shape):
            raise RuntimeError(f'TensorRT input shape must be static NCHW, got {input_shape}')
        if input_shape[0] != 1 or input_shape[1] != 3:
            raise RuntimeError(f'unexpected TensorRT input shape: {input_shape}')
        self.height = int(input_shape[2])
        self.width = int(input_shape[3])
        self.device = torch.device('cuda')
        self._output = None
        self._output_shape = None

    def __call__(self, tensor):
        if tensor.device.type != 'cuda' or tensor.dtype != torch.float32:
            raise TypeError('TensorRT predictor expects a CUDA float32 input tensor')
        if tuple(tensor.shape) != (1, 3, self.height, self.width):
            raise ValueError(
                f'TensorRT input shape mismatch: expected '
                f'(1, 3, {self.height}, {self.width}), got {tuple(tensor.shape)}'
            )

        if not self._context.set_input_shape(self._input_name, tuple(tensor.shape)):
            raise RuntimeError('TensorRT rejected the input shape')
        output_shape = tuple(self._context.get_tensor_shape(self._output_name))
        if any(int(value) <= 0 for value in output_shape):
            raise RuntimeError(f'TensorRT returned dynamic output shape: {output_shape}')
        if output_shape != self._output_shape:
            output_dtype = np.dtype(self._trt.nptype(
                self._engine.get_tensor_dtype(self._output_name)
            ))
            torch_dtype = torch.from_numpy(np.empty((), dtype=output_dtype)).dtype
            self._output = torch.empty(output_shape, dtype=torch_dtype, device=self.device)
            self._output_shape = output_shape

        self._context.set_tensor_address(self._input_name, int(tensor.data_ptr()))
        self._context.set_tensor_address(self._output_name, int(self._output.data_ptr()))
        stream_handle = torch.cuda.current_stream(self.device).cuda_stream
        if not self._context.execute_async_v3(stream_handle):
            raise RuntimeError('TensorRT execution failed')
        return {'pred_disp': self._output}


def load_tensorrt_engine(model_path):
    predictor = TensorRTEnginePredictor(model_path)
    return (
        predictor,
        predictor.device,
        predictor.height,
        predictor.width,
        False,
    )


class CudaGraphPredictor:
    """Static-shape CUDA graph wrapper with the same model output contract."""

    def __init__(self, model, device, height, width):
        self._model = model
        self._static_input = torch.empty(
            (1, 3, height, width),
            dtype=torch.float32,
            device=device,
        )
        warmup_input = torch.zeros_like(self._static_input)
        for _ in range(3):
            self._model(warmup_input)
        torch.cuda.synchronize(device)

        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._static_output = self._model(self._static_input)
        torch.cuda.synchronize(device)

    def __call__(self, tensor):
        self._static_input.copy_(tensor)
        self._graph.replay()
        return self._static_output


def _prediction_tensor(output):
    if isinstance(output, dict):
        return output['pred_disp']
    return output


def maybe_enable_cuda_graph(model, device, height, width):
    requested = os.environ.get('DA360_CUDA_GRAPH', 'auto').strip().lower()
    if device.type != 'cuda' or requested in ('0', 'false', 'no', 'off'):
        return model, False
    if requested not in ('auto', '1', 'true', 'yes', 'on'):
        print(f'unknown DA360_CUDA_GRAPH={requested!r}; using eager mode', file=sys.stderr)
        return model, False

    sample = torch.zeros((1, 3, height, width), dtype=torch.float32, device=device)
    try:
        with torch.inference_mode():
            eager_output = _prediction_tensor(model(sample)).detach().clone()
            torch.cuda.synchronize(device)
            graph_model = CudaGraphPredictor(model, device, height, width)
            graph_output = _prediction_tensor(graph_model(sample)).detach().clone()
            torch.cuda.synchronize(device)

            max_error = (eager_output - graph_output).abs().max().item()
            reference = eager_output.abs().clamp_min(1e-6)
            max_relative_error = (
                (eager_output - graph_output).abs() / reference
            ).max().item()
            if max_error > 1e-4 and max_relative_error > 1e-4:
                raise RuntimeError(
                    f'numerical mismatch max_abs={max_error:.3g}, '
                    f'max_rel={max_relative_error:.3g}'
                )

            def timed(callable_model):
                start = time.perf_counter()
                for _ in range(5):
                    callable_model(sample)
                torch.cuda.synchronize(device)
                return (time.perf_counter() - start) / 5.0

            eager_time = timed(model)
            graph_time = timed(graph_model)
            faster = graph_time < eager_time * 0.95
            if requested == 'auto' and not faster:
                print(
                    f'DA360 CUDA graph skipped: eager={eager_time * 1000.0:.1f} ms, '
                    f'graph={graph_time * 1000.0:.1f} ms',
                    file=sys.stderr,
                    flush=True,
                )
                return model, False
            print(
                f'DA360 CUDA graph enabled: eager={eager_time * 1000.0:.1f} ms, '
                f'graph={graph_time * 1000.0:.1f} ms, '
                f'max_abs={max_error:.3g}, max_rel={max_relative_error:.3g}',
                file=sys.stderr,
                flush=True,
            )
            return graph_model, True
    except Exception as exc:
        print(f'DA360 CUDA graph unavailable; using eager mode: {exc}', file=sys.stderr, flush=True)
        return model, False


def infer(
    model,
    device,
    model_height,
    model_width,
    rgb,
    source_height,
    source_width,
    output_stride,
    rgb_mean,
    rgb_inv_std,
):
    if (source_width, source_height) != (model_width, model_height):
        rgb = cv2.resize(rgb, (model_width, model_height), interpolation=cv2.INTER_CUBIC)
    tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).unsqueeze(0)
    tensor = tensor.to(device, dtype=torch.float32, non_blocking=True)
    tensor.mul_(1.0 / 255.0)
    tensor.sub_(rgb_mean).mul_(rgb_inv_std)

    with torch.inference_mode():
        outputs = model(tensor)
        disparity = outputs['pred_disp']
        if disparity.ndim == 4:
            disparity = disparity[:, 0]

        # Match the repository's test.py relative-depth convention.
        depth = 1.0 / disparity
        depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth_min = depth.amin(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        depth = depth / depth_min
        depth = depth[:, None]
        if tuple(depth.shape[-2:]) != (source_height, source_width):
            depth = F.interpolate(
                depth,
                size=(source_height, source_width),
                mode='bilinear',
                align_corners=True,
            )
        depth = depth[0, 0]
        if output_stride > 1:
            depth = depth[::output_stride, ::output_stride]

    return np.ascontiguousarray(depth.float().cpu().numpy(), dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo-root', required=True, type=Path)
    parser.add_argument('--model-path', required=True, type=Path)
    parser.add_argument('--net', default='')
    args = parser.parse_args()

    protocol_out = sys.stdout.buffer
    # Some model initialization code prints informational messages.  stdout
    # is reserved for the binary protocol, so redirect those messages.
    with contextlib.redirect_stdout(sys.stderr):
        backend_request = os.environ.get('DA360_BACKEND', 'auto').strip().lower()
        is_engine = args.model_path.suffix.lower() in ('.engine', '.plan')
        use_tensorrt = backend_request in ('tensorrt', 'trt', 'int8') or (
            backend_request == 'auto' and is_engine
        )
        if use_tensorrt:
            if not is_engine:
                raise ValueError(
                    'DA360_BACKEND=tensorrt requires a .engine or .plan model path'
                )
            model, device, model_height, model_width, mixed_precision = (
                load_tensorrt_engine(args.model_path)
            )
            cuda_graph_enabled = False
            backend_name = 'tensorrt'
        else:
            model, device, model_height, model_width, mixed_precision = load_model(
                args.repo_root,
                args.model_path,
                args.net,
            )
            model, cuda_graph_enabled = maybe_enable_cuda_graph(
                model,
                device,
                model_height,
                model_width,
            )
            backend_name = 'torch'

    print(
        f'DA360 worker ready: device={device}, model_input={model_width}x{model_height}, '
        f'backend={backend_name}, mixed_precision={mixed_precision}, '
        f'cuda_graph={cuda_graph_enabled}, '
        f'checkpoint={args.model_path}',
        file=sys.stderr,
        flush=True,
    )

    rgb_mean = torch.tensor(
        [0.485, 0.456, 0.406], dtype=torch.float32, device=device
    ).view(1, 3, 1, 1)
    rgb_inv_std = (
        torch.tensor(
            [1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225],
            dtype=torch.float32,
            device=device,
        )
        .view(1, 3, 1, 1)
    )

    input_stream = sys.stdin.buffer
    payload_buffer = None
    while True:
        header = read_exact(input_stream, REQUEST_HEADER.size)
        if header is None:
            return
        source_height, source_width, output_stride = REQUEST_HEADER.unpack(header)
        if output_stride < 1:
            raise ValueError(f'invalid output stride: {output_stride}')
        payload_size = source_height * source_width * 3
        if payload_buffer is None or len(payload_buffer) != payload_size:
            payload_buffer = bytearray(payload_size)
        if not read_exact_into(input_stream, payload_buffer):
            return

        rgb = np.frombuffer(
            payload_buffer,
            dtype=np.uint8,
        ).reshape(source_height, source_width, 3)
        start = time.perf_counter()
        with contextlib.redirect_stdout(sys.stderr):
            depth = infer(
                model,
                device,
                model_height,
                model_width,
                rgb,
                source_height,
                source_width,
                output_stride,
                rgb_mean,
                rgb_inv_std,
            )
        elapsed = time.perf_counter() - start
        out_height, out_width = depth.shape
        write_all(protocol_out, RESPONSE_HEADER.pack(out_height, out_width, output_stride))
        write_all(protocol_out, memoryview(depth).cast('B'))
        print(f'inference={elapsed * 1000.0:.1f} ms', file=sys.stderr, flush=True)


if __name__ == '__main__':
    main()
