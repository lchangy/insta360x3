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
import struct
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


REQUEST_HEADER = struct.Struct('<II')


def read_exact(stream, size):
    data = bytearray()
    while len(data) < size:
        chunk = stream.read(size - len(data))
        if not chunk:
            return None
        data.extend(chunk)
    return bytes(data)


def write_all(stream, data):
    view = memoryview(data)
    while view:
        written = stream.write(view)
        if written is None:
            stream.flush()
            return
        view = view[written:]
    stream.flush()


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

    net_class = getattr(networks, net_name)
    model = net_class(height, width, dinov2_encoder=encoder)
    state = model.state_dict()
    weights = {key: value for key, value in checkpoint.items() if key in state}
    model.load_state_dict(weights, strict=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    return model, device, height, width


def infer(model, device, model_height, model_width, bgr, source_height, source_width):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (model_width, model_height), interpolation=cv2.INTER_CUBIC)
    rgb = rgb.astype(np.float32) / 255.0
    rgb = (rgb - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).unsqueeze(0)
    tensor = tensor.to(device, non_blocking=True)

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
        depth = F.interpolate(
            depth[:, None],
            size=(source_height, source_width),
            mode='bilinear',
            align_corners=True,
        )[0, 0]

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
        model, device, model_height, model_width = load_model(
            args.repo_root,
            args.model_path,
            args.net,
        )

    print(
        f'DA360 worker ready: device={device}, model_input={model_width}x{model_height}, '
        f'checkpoint={args.model_path}',
        file=sys.stderr,
        flush=True,
    )

    input_stream = sys.stdin.buffer
    while True:
        header = read_exact(input_stream, REQUEST_HEADER.size)
        if header is None:
            return
        source_height, source_width = REQUEST_HEADER.unpack(header)
        payload_size = source_height * source_width * 3
        payload = read_exact(input_stream, payload_size)
        if payload is None:
            return

        bgr = np.frombuffer(payload, dtype=np.uint8).reshape(source_height, source_width, 3)
        start = time.perf_counter()
        with contextlib.redirect_stdout(sys.stderr):
            depth = infer(
                model,
                device,
                model_height,
                model_width,
                bgr,
                source_height,
                source_width,
            )
        elapsed = time.perf_counter() - start
        write_all(protocol_out, REQUEST_HEADER.pack(source_height, source_width))
        write_all(protocol_out, depth.tobytes(order='C'))
        print(f'inference={elapsed * 1000.0:.1f} ms', file=sys.stderr, flush=True)


if __name__ == '__main__':
    main()
