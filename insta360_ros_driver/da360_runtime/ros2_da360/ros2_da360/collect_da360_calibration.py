#!/usr/bin/env python3
"""Collect representative equirectangular frames for DA360 INT8 calibration."""

import argparse
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


def image_to_rgb(message):
    encoding = message.encoding.lower()
    channels = {
        'rgb8': 3,
        'bgr8': 3,
        'rgba8': 4,
        'bgra8': 4,
    }.get(encoding)
    if channels is None:
        raise ValueError(f'unsupported calibration image encoding: {message.encoding}')

    raw = np.frombuffer(memoryview(message.data), dtype=np.uint8)
    row_size = message.width * channels
    if message.step < row_size or raw.size < message.height * message.step:
        raise ValueError(
            f'invalid image step={message.step} for '
            f'{message.width}x{message.height} {message.encoding}'
        )
    image = raw.reshape(message.height, message.step)[:, :row_size]
    image = image.reshape(message.height, message.width, channels)
    if encoding == 'bgr8':
        image = image[:, :, ::-1]
    elif encoding == 'rgba8':
        image = image[:, :, :3]
    elif encoding == 'bgra8':
        image = image[:, :, 2::-1]
    return np.ascontiguousarray(image, dtype=np.uint8)


class CalibrationCollector(Node):
    def __init__(self, topic, target_count):
        super().__init__('da360_calibration_collector')
        self.target_count = target_count
        self.frames = []
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.subscription = self.create_subscription(
            Image,
            topic,
            self._callback,
            qos,
        )

    def _callback(self, message):
        if len(self.frames) >= self.target_count:
            return
        try:
            self.frames.append(image_to_rgb(message).copy())
        except ValueError as exc:
            self.get_logger().warning(str(exc))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--topic', default='/equirectangular/image')
    parser.add_argument('--count', type=int, default=64)
    parser.add_argument('--timeout', type=float, default=90.0)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.count < 1:
        raise ValueError('--count must be positive')

    rclpy.init()
    node = CalibrationCollector(args.topic, args.count)
    start = time.monotonic()
    try:
        while len(node.frames) < args.count:
            if time.monotonic() - start >= args.timeout:
                break
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        rclpy.shutdown()

    if not node.frames:
        raise RuntimeError(f'no calibration frames received from {args.topic}')
    shapes = {frame.shape for frame in node.frames}
    if len(shapes) != 1:
        raise RuntimeError(f'calibration frames have inconsistent shapes: {shapes}')

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames = np.stack(node.frames, axis=0)
    np.save(args.output, frames)
    print(
        f'collected {len(frames)} RGB frames with shape {tuple(frames.shape[1:])} '
        f'into {args.output}'
    )


if __name__ == '__main__':
    main()
