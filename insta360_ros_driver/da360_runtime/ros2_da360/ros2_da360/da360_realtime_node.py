#!/usr/bin/env python3
"""ROS 2 node: equirectangular image -> depth image + PointCloud2."""

import os
import queue
import shutil
import struct
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header


PROTOCOL_HEADER = struct.Struct('<II')


@dataclass
class PendingFrame:
    height: int
    width: int
    bgr: np.ndarray
    frame_id: str
    stamp_sec: int
    stamp_nanosec: int


def read_exact(stream, size):
    data = bytearray()
    while len(data) < size:
        chunk = stream.read(size - len(data))
        if not chunk:
            raise EOFError('DA360 worker closed its output stream')
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


class DA360RealtimeNode(Node):
    def __init__(self):
        super().__init__('da360_realtime')

        bundled_root = Path(__file__).resolve().parents[2]
        default_root = os.environ.get('DA360_ROOT', str(bundled_root))
        default_python = os.environ.get('DA360_PYTHON', sys.executable)
        default_model = str(Path(default_root) / 'checkpoints' / 'DA360_small.pth')

        self.declare_parameter('repo_root', default_root)
        self.declare_parameter('worker_python', default_python)
        self.declare_parameter('model_path', default_model)
        self.declare_parameter('net', '')
        self.declare_parameter('input_topic', '/equirectangular/image')
        self.declare_parameter('depth_topic', '/da360/depth')
        self.declare_parameter('pointcloud_topic', '/da360/points')
        self.declare_parameter('frame_id', '')
        self.declare_parameter('point_stride', 4)

        repo_root = Path(str(self.get_parameter('repo_root').value)).expanduser().resolve()
        worker_python = str(self.get_parameter('worker_python').value)
        model_path = Path(str(self.get_parameter('model_path').value)).expanduser()
        if not model_path.is_absolute():
            model_path = repo_root / model_path
        model_path = model_path.resolve()

        self._frame_override = str(self.get_parameter('frame_id').value)
        self._point_stride = max(1, int(self.get_parameter('point_stride').value))
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._pending_lock = threading.Lock()
        self._pending = None
        self._warned_encoding = False
        self._processed = 0
        self._last_rate_log = time.monotonic()
        self._last_rate_count = 0

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        input_topic = str(self.get_parameter('input_topic').value)
        depth_topic = str(self.get_parameter('depth_topic').value)
        pointcloud_topic = str(self.get_parameter('pointcloud_topic').value)
        self._depth_pub = self.create_publisher(Image, depth_topic, qos)
        self._points_pub = self.create_publisher(PointCloud2, pointcloud_topic, qos)
        self._subscription = self.create_subscription(Image, input_topic, self._image_callback, qos)

        if not repo_root.is_dir():
            raise FileNotFoundError(f'DA360 repo does not exist: {repo_root}')
        if not model_path.is_file():
            raise FileNotFoundError(f'DA360 checkpoint does not exist: {model_path}')
        if not Path(worker_python).is_file():
            resolved_python = shutil.which(worker_python)
            if resolved_python is None:
                raise FileNotFoundError(f'Inference Python does not exist: {worker_python}')
            worker_python = resolved_python

        worker_script = repo_root / 'ros2_da360' / 'ros2_da360' / 'da360_inference_worker.py'
        command = [
            worker_python,
            '-u',
            str(worker_script),
            '--repo-root',
            str(repo_root),
            '--model-path',
            str(model_path),
        ]
        net = str(self.get_parameter('net').value)
        if net:
            command.extend(['--net', net])

        self.get_logger().info(f'Starting DA360 worker: {" ".join(command)}')
        worker_env = os.environ.copy()
        worker_env['PYTHONUNBUFFERED'] = '1'
        self._worker = subprocess.Popen(
            command,
            cwd=str(repo_root),
            env=worker_env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            bufsize=0,
        )
        self._worker_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._worker_thread.start()
        self.get_logger().info(
            f'Subscribed to {input_topic}; publishing {depth_topic} and {pointcloud_topic}; '
            f'point_stride={self._point_stride}'
        )

    @staticmethod
    def _to_bgr(msg):
        encoding = msg.encoding.lower()
        if encoding in ('rgb8', 'bgr8'):
            channels = 3
        elif encoding in ('rgba8', 'bgra8'):
            channels = 4
        elif encoding == 'mono8':
            channels = 1
        else:
            raise ValueError(f'unsupported image encoding: {msg.encoding}')

        raw = np.frombuffer(memoryview(msg.data), dtype=np.uint8)
        expected_row = msg.width * channels
        if msg.step < expected_row or raw.size < msg.height * msg.step:
            raise ValueError(f'invalid Image step={msg.step} for {msg.width}x{msg.height} {msg.encoding}')
        image = raw.reshape(msg.height, msg.step)[:, :expected_row]
        image = image.reshape(msg.height, msg.width, channels)

        if encoding == 'rgb8':
            return np.ascontiguousarray(image[:, :, ::-1])
        if encoding == 'rgba8':
            return np.ascontiguousarray(image[:, :, 2::-1])
        if encoding == 'bgra8':
            return np.ascontiguousarray(image[:, :, :3])
        if encoding == 'mono8':
            return np.ascontiguousarray(np.repeat(image, 3, axis=2))
        return np.ascontiguousarray(image)

    def _image_callback(self, msg):
        try:
            bgr = self._to_bgr(msg)
        except ValueError as exc:
            if not self._warned_encoding:
                self.get_logger().error(str(exc))
                self._warned_encoding = True
            return

        frame = PendingFrame(
            height=int(msg.height),
            width=int(msg.width),
            bgr=bgr,
            frame_id=self._frame_override or msg.header.frame_id or 'camera_frame',
            stamp_sec=int(msg.header.stamp.sec),
            stamp_nanosec=int(msg.header.stamp.nanosec),
        )
        # Latest-frame-only buffering keeps the output live when the source is
        # faster than the selected DA360 model.
        with self._pending_lock:
            self._pending = frame
        self._wake.set()

    def _take_pending(self):
        with self._pending_lock:
            frame = self._pending
            self._pending = None
        return frame

    @staticmethod
    def _header(frame):
        header = Header()
        header.stamp.sec = frame.stamp_sec
        header.stamp.nanosec = frame.stamp_nanosec
        header.frame_id = frame.frame_id
        return header

    def _make_depth_message(self, frame, depth):
        msg = Image()
        msg.header = self._header(frame)
        msg.height = frame.height
        msg.width = frame.width
        msg.encoding = '32FC1'
        msg.is_bigendian = 0
        msg.step = frame.width * 4
        msg.data = np.ascontiguousarray(depth, dtype=np.float32).tobytes()
        return msg

    def _make_pointcloud_message(self, frame, depth):
        stride = self._point_stride
        depth = np.ascontiguousarray(depth[::stride, ::stride], dtype=np.float32)
        rgb = frame.bgr[::stride, ::stride, ::-1]
        height, width = depth.shape

        theta = np.pi - (np.arange(height, dtype=np.float32) + 0.5) * np.pi / height
        phi = (np.arange(width, dtype=np.float32) + 0.5) * 2.0 * np.pi / width - np.pi
        sin_theta = np.sin(theta)[:, None]
        cos_theta = np.cos(theta)[:, None]
        sin_phi = np.sin(phi)[None, :]
        cos_phi = np.cos(phi)[None, :]

        x = depth * sin_theta * sin_phi
        y = depth * cos_theta
        z = depth * sin_theta * cos_phi
        valid = np.isfinite(depth) & (depth > 0.0)
        if not np.any(valid):
            return None

        cloud_dtype = np.dtype([
            ('x', '<f4'),
            ('y', '<f4'),
            ('z', '<f4'),
            ('rgb', '<f4'),
        ])
        cloud = np.empty(int(valid.sum()), dtype=cloud_dtype)
        cloud['x'] = x[valid]
        cloud['y'] = y[valid]
        cloud['z'] = z[valid]
        colors = np.ascontiguousarray(rgb[valid], dtype=np.uint8)
        packed_rgb = (
            (colors[:, 0].astype(np.uint32) << 16)
            | (colors[:, 1].astype(np.uint32) << 8)
            | colors[:, 2].astype(np.uint32)
        )
        cloud['rgb'] = packed_rgb.view('<f4')

        msg = PointCloud2()
        msg.header = self._header(frame)
        msg.height = 1
        msg.width = cloud.shape[0]
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = cloud_dtype.itemsize
        msg.row_step = msg.point_step * msg.width
        msg.data = cloud.tobytes()
        msg.is_dense = True
        return msg

    def _inference_loop(self):
        try:
            while not self._stop.is_set():
                self._wake.wait(0.1)
                self._wake.clear()
                while not self._stop.is_set():
                    frame = self._take_pending()
                    if frame is None:
                        break
                    if self._worker.poll() is not None:
                        raise RuntimeError(f'DA360 worker exited with code {self._worker.returncode}')

                    request_header = PROTOCOL_HEADER.pack(frame.height, frame.width)
                    write_all(self._worker.stdin, request_header)
                    write_all(self._worker.stdin, frame.bgr.tobytes(order='C'))

                    response = read_exact(self._worker.stdout, PROTOCOL_HEADER.size)
                    out_height, out_width = PROTOCOL_HEADER.unpack(response)
                    if (out_height, out_width) != (frame.height, frame.width):
                        raise RuntimeError(
                            f'DA360 worker returned {out_width}x{out_height}, '
                            f'expected {frame.width}x{frame.height}'
                        )
                    depth_bytes = read_exact(self._worker.stdout, out_height * out_width * 4)
                    depth = np.frombuffer(depth_bytes, dtype='<f4').reshape(out_height, out_width).copy()

                    self._depth_pub.publish(self._make_depth_message(frame, depth))
                    pointcloud = self._make_pointcloud_message(frame, depth)
                    if pointcloud is not None:
                        self._points_pub.publish(pointcloud)

                    self._processed += 1
                    now = time.monotonic()
                    if now - self._last_rate_log >= 5.0:
                        count = self._processed - self._last_rate_count
                        rate = count / (now - self._last_rate_log)
                        self.get_logger().info(f'published {rate:.2f} depth/point-cloud frames per second')
                        self._last_rate_log = now
                        self._last_rate_count = self._processed
        except (BrokenPipeError, EOFError, OSError, RuntimeError) as exc:
            if not self._stop.is_set():
                self.get_logger().error(f'DA360 inference loop stopped: {exc}')

    def close(self):
        self._stop.set()
        self._wake.set()
        if hasattr(self, '_worker_thread'):
            self._worker_thread.join(timeout=2.0)
        worker = getattr(self, '_worker', None)
        if worker is None:
            return
        if worker.poll() is None:
            worker.terminate()
            try:
                worker.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait()
        for pipe in (worker.stdin, worker.stdout):
            if pipe is not None:
                pipe.close()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = DA360RealtimeNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
