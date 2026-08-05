#!/usr/bin/env python3
"""ROS 2 node: equirectangular image -> depth image + PointCloud2."""

import os
import shutil
import struct
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header


PROTOCOL_HEADER = struct.Struct('<III')


def env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in ('0', 'false', 'no', 'off')


@dataclass
class PendingFrame:
    height: int
    width: int
    rgb: np.ndarray
    frame_id: str
    stamp_sec: int
    stamp_nanosec: int
    data_owner: object


@dataclass
class PendingPointCloud:
    frame: PendingFrame
    depth: np.ndarray
    sample_stride: int


@dataclass
class PendingDepth:
    height: int
    width: int
    frame_id: str
    stamp_sec: int
    stamp_nanosec: int
    data: bytes


def read_exact(stream, size):
    data = bytearray()
    while len(data) < size:
        chunk = stream.read(size - len(data))
        if not chunk:
            raise EOFError('DA360 worker closed its output stream')
        data.extend(chunk)
    return bytes(data)


def read_exact_into(stream, buffer):
    view = memoryview(buffer)
    while view:
        count = stream.readinto(view)
        if not count:
            raise EOFError('DA360 worker closed its output stream')
        view = view[count:]


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
        self._profile = env_flag('DA360_PROFILE', False)
        self._metrics_lock = threading.Lock()
        self._inference_count = 0
        self._pointcloud_count = 0
        self._pointcloud_drops = 0
        self._inference_total = 0.0
        self._pointcloud_total = 0.0
        self._publish_total = 0.0
        self._last_stats_log = time.monotonic()
        self._last_stats_inference = 0
        self._last_stats_pointcloud = 0
        self._ray_cache = {}
        self._cloud_dtype = np.dtype([
            ('x', '<f4'),
            ('y', '<f4'),
            ('z', '<f4'),
            ('rgb', '<f4'),
        ])
        self._point_fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        self._pointcloud_pending_lock = threading.Lock()
        self._pointcloud_wake = threading.Event()
        self._pending_pointcloud = None
        self._depth_pending_lock = threading.Lock()
        self._depth_wake = threading.Event()
        self._pending_depth = None
        try:
            self._depth_max_fps = max(0, int(os.environ.get('DA360_DEPTH_MAX_FPS', '1')))
        except ValueError:
            self._depth_max_fps = 1
        try:
            self._depth_stride = max(1, int(os.environ.get('DA360_DEPTH_STRIDE', '2')))
        except ValueError:
            self._depth_stride = 2
        self._last_depth_enqueue_time = 0.0

        qos_mode = os.environ.get('DA360_QOS', 'best_effort').strip().lower()
        reliability = (
            ReliabilityPolicy.RELIABLE
            if qos_mode == 'reliable'
            else ReliabilityPolicy.BEST_EFFORT
        )
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=reliability,
            durability=DurabilityPolicy.VOLATILE,
        )
        input_topic = str(self.get_parameter('input_topic').value)
        depth_topic = str(self.get_parameter('depth_topic').value)
        pointcloud_topic = str(self.get_parameter('pointcloud_topic').value)
        self._depth_pub = self.create_publisher(Image, depth_topic, qos)
        self._points_pub = self.create_publisher(PointCloud2, pointcloud_topic, qos)
        self._subscription = self.create_subscription(Image, input_topic, self._image_callback, qos)
        self.get_logger().info(
            f'DA360 sensor QoS: {qos_mode if qos_mode == "reliable" else "best_effort"}; '
            f'depth_max_fps={self._depth_max_fps or "unlimited"}; '
            f'depth_stride={self._depth_stride}'
        )

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
        self._depth_thread = threading.Thread(target=self._depth_publish_loop, daemon=True)
        self._pointcloud_thread = threading.Thread(
            target=self._pointcloud_publish_loop,
            daemon=True,
        )
        self._depth_thread.start()
        self._pointcloud_thread.start()
        self._worker_thread.start()
        self.get_logger().info(
            f'Subscribed to {input_topic}; publishing {depth_topic} and {pointcloud_topic}; '
            f'point_stride={self._point_stride}'
        )

    @staticmethod
    def _to_rgb(msg):
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
            return np.ascontiguousarray(image)
        if encoding == 'bgr8':
            return np.ascontiguousarray(image[:, :, ::-1])
        if encoding == 'rgba8':
            return np.ascontiguousarray(image[:, :, :3])
        if encoding == 'bgra8':
            return np.ascontiguousarray(image[:, :, 2::-1])
        if encoding == 'mono8':
            return np.ascontiguousarray(np.repeat(image, 3, axis=2))
        return np.ascontiguousarray(image)

    def _image_callback(self, msg):
        try:
            rgb = self._to_rgb(msg)
        except ValueError as exc:
            if not self._warned_encoding:
                self.get_logger().error(str(exc))
                self._warned_encoding = True
            return

        frame = PendingFrame(
            height=int(msg.height),
            width=int(msg.width),
            rgb=rgb,
            frame_id=self._frame_override or msg.header.frame_id or 'camera_frame',
            stamp_sec=int(msg.header.stamp.sec),
            stamp_nanosec=int(msg.header.stamp.nanosec),
            # Keep the ROS message buffer alive while this latest frame is
            # waiting for the worker.  rgb8 can therefore remain zero-copy.
            data_owner=msg.data,
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

    def _make_depth_message(self, pending):
        msg = Image()
        msg.header = self._header(pending)
        msg.height = pending.height
        msg.width = pending.width
        msg.encoding = '32FC1'
        msg.is_bigendian = 0
        msg.step = pending.width * 4
        msg.data = pending.data
        return msg

    def _depth_publish_loop(self):
        while not self._stop.is_set():
            self._depth_wake.wait(0.1)
            self._depth_wake.clear()
            with self._depth_pending_lock:
                pending = self._pending_depth
                self._pending_depth = None
            if pending is None or self._depth_pub.get_subscription_count() <= 0:
                continue
            self._depth_pub.publish(self._make_depth_message(pending))

    def _make_pointcloud_message(self, frame, depth, sample_stride):
        if self._point_stride % sample_stride:
            raise ValueError(
                f'point stride {self._point_stride} is not divisible by '
                f'worker stride {sample_stride}'
            )
        point_step = self._point_stride // sample_stride
        depth = np.ascontiguousarray(depth[::point_step, ::point_step], dtype=np.float32)
        rgb = frame.rgb[::self._point_stride, ::self._point_stride]
        height, width = depth.shape

        # The ERP projection uses the camera-local convention x=right,
        # y=up, z=forward. Publish the cloud in camera_frame using the ROS
        # body-frame convention x=forward, y=left, z=up:
        # [x_ros, y_ros, z_ros] = [z_erp, -x_erp, y_erp].
        rays = self._ray_cache.get((height, width))
        if rays is None:
            theta = (np.arange(height, dtype=np.float32) + 0.5) * np.pi / height
            phi = (np.arange(width, dtype=np.float32) + 0.5) * 2.0 * np.pi / width - np.pi
            sin_theta = np.sin(theta)[:, None]
            cos_theta = np.cos(theta)[:, None]
            sin_phi = np.sin(phi)[None, :]
            cos_phi = np.cos(phi)[None, :]
            rays = np.empty((height * width, 3), dtype=np.float32)
            # ERP ray in camera-local coordinates:
            # x_erp = sin(theta) * sin(phi)
            # y_erp = cos(theta)
            # z_erp = sin(theta) * cos(phi)
            rays[:, 0] = (sin_theta * cos_phi).reshape(-1)
            rays[:, 1] = -(sin_theta * sin_phi).reshape(-1)
            rays[:, 2] = np.broadcast_to(cos_theta, (height, width)).reshape(-1)
            self._ray_cache[(height, width)] = rays

        depth_flat = depth.reshape(-1)
        valid = np.isfinite(depth_flat) & (depth_flat > 0.0)
        if not np.any(valid):
            return None

        colors = np.asarray(rgb, dtype=np.uint8).reshape(-1, 3)
        if np.all(valid):
            cloud = np.empty((depth_flat.size, 4), dtype=np.float32)
            valid_rays = rays
            valid_depth = depth_flat
        else:
            valid_indices = np.flatnonzero(valid)
            cloud = np.empty((valid_indices.size, 4), dtype=np.float32)
            valid_rays = rays[valid_indices]
            valid_depth = depth_flat[valid_indices]
            colors = colors[valid_indices]

        np.multiply(valid_rays, valid_depth[:, None], out=cloud[:, :3])
        if np.all(valid):
            colors = colors[:depth_flat.size]
        packed_rgb = cloud[:, 3].view(np.uint32)
        packed_rgb[:] = colors[:, 0].astype(np.uint32)
        packed_rgb <<= 16
        packed_rgb |= colors[:, 1].astype(np.uint32) << 8
        packed_rgb |= colors[:, 2].astype(np.uint32)

        msg = PointCloud2()
        msg.header = self._header(frame)
        msg.height = 1
        msg.width = cloud.shape[0]
        msg.fields = self._point_fields
        msg.is_bigendian = False
        msg.point_step = self._cloud_dtype.itemsize
        msg.row_step = msg.point_step * msg.width
        msg.data = cloud.tobytes(order='C')
        msg.is_dense = True
        return msg

    def _take_pending_pointcloud(self):
        with self._pointcloud_pending_lock:
            pending = self._pending_pointcloud
            self._pending_pointcloud = None
        return pending

    def _enqueue_pointcloud(self, pending):
        with self._pointcloud_pending_lock:
            if self._pending_pointcloud is not None:
                with self._metrics_lock:
                    self._pointcloud_drops += 1
            self._pending_pointcloud = pending
        self._pointcloud_wake.set()

    def _log_metrics(self):
        now = time.monotonic()
        with self._metrics_lock:
            elapsed = now - self._last_stats_log
            if elapsed < 5.0:
                return
            inference_count = self._inference_count
            pointcloud_count = self._pointcloud_count
            inference_delta = inference_count - self._last_stats_inference
            pointcloud_delta = pointcloud_count - self._last_stats_pointcloud
            self._last_stats_inference = inference_count
            self._last_stats_pointcloud = pointcloud_count
            inference_rate = inference_delta / elapsed
            pointcloud_rate = pointcloud_delta / elapsed
            inference_average = (
                self._inference_total / inference_count * 1000.0
                if inference_count
                else 0.0
            )
            pointcloud_average = (
                self._pointcloud_total / pointcloud_count * 1000.0
                if pointcloud_count
                else 0.0
            )
            publish_average = (
                self._publish_total / pointcloud_count * 1000.0
                if pointcloud_count
                else 0.0
            )
            drops = self._pointcloud_drops
            self._last_stats_log = now

        message = (
            f'published {pointcloud_rate:.2f} point-cloud frames per second; '
            f'inference responses={inference_rate:.2f} Hz'
        )
        if self._profile:
            message += (
                f'; avg roundtrip={inference_average:.1f} ms'
                f'; avg pointcloud={pointcloud_average:.1f} ms'
                f'; avg publish={publish_average:.1f} ms'
                f'; pointcloud_drops={drops}'
            )
        self.get_logger().info(message)

    def _pointcloud_publish_loop(self):
        try:
            while not self._stop.is_set():
                self._pointcloud_wake.wait(0.1)
                self._pointcloud_wake.clear()
                while not self._stop.is_set():
                    pending = self._take_pending_pointcloud()
                    if pending is None:
                        break

                    pointcloud_start = time.perf_counter()
                    pointcloud = self._make_pointcloud_message(
                        pending.frame,
                        pending.depth,
                        pending.sample_stride,
                    )
                    pointcloud_elapsed = time.perf_counter() - pointcloud_start
                    if pointcloud is None:
                        self._log_metrics()
                        continue

                    publish_start = time.perf_counter()
                    self._points_pub.publish(pointcloud)
                    publish_elapsed = time.perf_counter() - publish_start
                    with self._metrics_lock:
                        self._pointcloud_count += 1
                        self._pointcloud_total += pointcloud_elapsed
                        self._publish_total += publish_elapsed
                    self._log_metrics()
        except Exception as exc:
            if not self._stop.is_set():
                self.get_logger().error(f'Point-cloud publish loop stopped: {exc}')

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

                    depth_has_subscriber = self._depth_pub.get_subscription_count() > 0
                    now = time.monotonic()
                    depth_due = (
                        self._depth_max_fps == 0
                        or now - self._last_depth_enqueue_time >= 1.0 / self._depth_max_fps
                    )
                    request_stride = (
                        gcd(self._point_stride, self._depth_stride)
                        if depth_has_subscriber and depth_due
                        else self._point_stride
                    )
                    request_header = PROTOCOL_HEADER.pack(
                        frame.height,
                        frame.width,
                        request_stride,
                    )
                    inference_start = time.perf_counter()
                    write_all(self._worker.stdin, request_header)
                    write_all(self._worker.stdin, memoryview(frame.rgb).cast('B'))

                    response = read_exact(self._worker.stdout, PROTOCOL_HEADER.size)
                    out_height, out_width, response_stride = PROTOCOL_HEADER.unpack(response)
                    if response_stride < 1:
                        raise RuntimeError(f'DA360 worker returned invalid stride {response_stride}')
                    expected_height = (frame.height + response_stride - 1) // response_stride
                    expected_width = (frame.width + response_stride - 1) // response_stride
                    if response_stride != request_stride:
                        raise RuntimeError(
                            f'DA360 worker returned stride {response_stride}, '
                            f'expected {request_stride}'
                        )
                    if (out_height, out_width) != (expected_height, expected_width):
                        raise RuntimeError(
                            f'DA360 worker returned {out_width}x{out_height}, '
                            f'expected {expected_width}x{expected_height}'
                        )
                    depth_buffer = bytearray(out_height * out_width * 4)
                    read_exact_into(self._worker.stdout, depth_buffer)
                    depth = np.frombuffer(
                        depth_buffer,
                        dtype='<f4',
                    ).reshape(out_height, out_width)
                    inference_elapsed = time.perf_counter() - inference_start
                    with self._metrics_lock:
                        self._inference_count += 1
                        self._inference_total += inference_elapsed

                    self._enqueue_pointcloud(PendingPointCloud(
                        frame=frame,
                        depth=depth,
                        sample_stride=response_stride,
                    ))

                    # Depth publication is asynchronous and latest-frame-only;
                    # publishing an Image must not block the inference pipeline.
                    if depth_has_subscriber and depth_due:
                        self._last_depth_enqueue_time = now
                        depth_step = self._depth_stride // response_stride
                        depth_for_publish = depth[::depth_step, ::depth_step]
                        depth_for_publish = np.ascontiguousarray(
                            depth_for_publish,
                            dtype=np.float32,
                        )
                        pending_depth = PendingDepth(
                            height=depth_for_publish.shape[0],
                            width=depth_for_publish.shape[1],
                            frame_id=frame.frame_id,
                            stamp_sec=frame.stamp_sec,
                            stamp_nanosec=frame.stamp_nanosec,
                            data=depth_for_publish.tobytes(),
                        )
                        with self._depth_pending_lock:
                            self._pending_depth = pending_depth
                        self._depth_wake.set()
                    self._log_metrics()
        except (BrokenPipeError, EOFError, OSError, RuntimeError) as exc:
            if not self._stop.is_set():
                self.get_logger().error(f'DA360 inference loop stopped: {exc}')

    def close(self):
        self._stop.set()
        self._wake.set()
        self._depth_wake.set()
        self._pointcloud_wake.set()
        if hasattr(self, '_worker_thread'):
            self._worker_thread.join(timeout=2.0)
        if hasattr(self, '_depth_thread'):
            self._depth_thread.join(timeout=2.0)
        if hasattr(self, '_pointcloud_thread'):
            self._pointcloud_thread.join(timeout=2.0)
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
