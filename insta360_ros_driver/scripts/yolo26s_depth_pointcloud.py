#!/usr/bin/env python3
"""Publish YOLO26s-depth point clouds for the left and right cubemap faces.

The cubemap node produces 90-degree rectilinear images.  YOLO26s-depth
returns a dense metric depth map for each image; this node projects that map
through the corresponding cube-face rays and publishes both clouds in the
shared camera_frame coordinate system (x=forward, y=left, z=up).
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header


FACE_NAMES = ("left", "right")
FACE_AXES = {
    # Optical-axis directions after converting from the cubemap convention
    # (x=right, y=up, z=forward) to camera_frame (x=forward, y=left, z=up).
    "left": np.array((0.0, 1.0, 0.0), dtype=np.float32),
    "right": np.array((0.0, -1.0, 0.0), dtype=np.float32),
}


@dataclass(frozen=True)
class PendingFrame:
    side: str
    bgr: np.ndarray
    height: int
    width: int
    frame_id: str
    stamp_sec: int
    stamp_nanosec: int


class Yolo26sDepthPointCloud(Node):
    """Run YOLO26s-depth on two cubemap streams and publish PointCloud2."""

    def __init__(self) -> None:
        super().__init__("yolo26s_depth_pointcloud")

        self.declare_parameter(
            "model_path",
            os.environ.get("YOLO26S_DEPTH_MODEL", "yolo26s-depth.pt"),
        )
        self.declare_parameter("left_image_topic", "/cubemap/left/image")
        self.declare_parameter("right_image_topic", "/cubemap/right/image")
        self.declare_parameter(
            "left_depth_topic", "/yolo26s_depth/left/depth"
        )
        self.declare_parameter(
            "right_depth_topic", "/yolo26s_depth/right/depth"
        )
        self.declare_parameter(
            "left_pointcloud_topic", "/yolo26s_depth/left/points"
        )
        self.declare_parameter(
            "right_pointcloud_topic", "/yolo26s_depth/right/points"
        )
        self.declare_parameter("frame_id", "camera_frame")
        self.declare_parameter("point_stride", 2)
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("device", "")
        self.declare_parameter("half", False)
        self.declare_parameter("face_fov_deg", 90.0)
        self.declare_parameter("depth_mode", "range")
        self.declare_parameter("depth_min", 0.05)
        self.declare_parameter("depth_max", 100.0)
        self.declare_parameter("publish_depth", True)

        self._model_path = str(self.get_parameter("model_path").value)
        self._frame_id = str(self.get_parameter("frame_id").value)
        self._point_stride = max(1, int(self.get_parameter("point_stride").value))
        self._imgsz = max(32, int(self.get_parameter("imgsz").value))
        self._device = str(self.get_parameter("device").value)
        self._half = bool(self.get_parameter("half").value)
        self._depth_mode = str(self.get_parameter("depth_mode").value).lower()
        if self._depth_mode not in ("range", "optical_z"):
            raise ValueError("depth_mode must be 'range' or 'optical_z'")

        self._depth_min = max(0.0, float(self.get_parameter("depth_min").value))
        self._depth_max = float(self.get_parameter("depth_max").value)
        self._face_fov_rad = np.deg2rad(
            float(self.get_parameter("face_fov_deg").value)
        )
        if not 0.0 < self._face_fov_rad < np.pi:
            raise ValueError("face_fov_deg must be between 0 and 180")
        self._publish_depth = bool(self.get_parameter("publish_depth").value)

        self._pending_lock = threading.Lock()
        self._pending: dict[str, PendingFrame | None] = {
            side: None for side in FACE_NAMES
        }
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._warned_inference: set[str] = set()
        self._processed = 0
        self._last_rate_log = time.monotonic()
        self._last_rate_count = 0

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        cloud_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        image_topics = {
            "left": str(self.get_parameter("left_image_topic").value),
            "right": str(self.get_parameter("right_image_topic").value),
        }
        depth_topics = {
            "left": str(self.get_parameter("left_depth_topic").value),
            "right": str(self.get_parameter("right_depth_topic").value),
        }
        pointcloud_topics = {
            "left": str(self.get_parameter("left_pointcloud_topic").value),
            "right": str(self.get_parameter("right_pointcloud_topic").value),
        }

        self._depth_publishers = {}
        if self._publish_depth:
            self._depth_publishers = {
                side: self.create_publisher(Image, depth_topics[side], cloud_qos)
                for side in FACE_NAMES
            }
        self._pointcloud_publishers = {
            side: self.create_publisher(
                PointCloud2, pointcloud_topics[side], cloud_qos
            )
            for side in FACE_NAMES
        }
        self._subscriptions = [
            self.create_subscription(
                Image,
                image_topics[side],
                lambda message, current_side=side: self._image_callback(
                    current_side, message
                ),
                qos,
            )
            for side in FACE_NAMES
        ]

        self.get_logger().info(
            f"Loading YOLO26s-depth model {self._model_path!r}; "
            f"device={self._device or 'auto'}, imgsz={self._imgsz}"
        )
        try:
            from ultralytics import YOLO
            from ultralytics.nn.tasks import DepthModel as _DepthModel  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "the selected Python environment lacks YOLO26 depth support; "
                "run 'uv sync --frozen' to install the pinned upstream Ultralytics source"
            ) from exc

        self._model = YOLO(self._model_path)
        self._worker = threading.Thread(
            target=self._inference_loop,
            name="yolo26s-depth-inference",
            daemon=True,
        )
        self._worker.start()
        self.get_logger().info(
            f"Subscribed to {image_topics['left']} and {image_topics['right']}; "
            f"publishing {pointcloud_topics['left']} and "
            f"{pointcloud_topics['right']}; point_stride={self._point_stride}"
        )

    @staticmethod
    def _to_bgr(message: Image) -> np.ndarray:
        encoding = message.encoding.lower()
        if encoding in ("rgb8", "bgr8"):
            channels = 3
        elif encoding in ("rgba8", "bgra8"):
            channels = 4
        elif encoding == "mono8":
            channels = 1
        else:
            raise ValueError(f"unsupported image encoding: {message.encoding}")

        raw = np.frombuffer(memoryview(message.data), dtype=np.uint8)
        expected_row = int(message.width) * channels
        if message.step < expected_row or raw.size < int(message.height) * message.step:
            raise ValueError(
                f"invalid Image step={message.step} for "
                f"{message.width}x{message.height} {message.encoding}"
            )
        image = raw.reshape(int(message.height), int(message.step))[:, :expected_row]
        image = image.reshape(int(message.height), int(message.width), channels)

        if encoding == "rgb8":
            return np.ascontiguousarray(image[:, :, ::-1])
        if encoding == "rgba8":
            return np.ascontiguousarray(image[:, :, 2::-1])
        if encoding == "bgra8":
            return np.ascontiguousarray(image[:, :, :3])
        if encoding == "mono8":
            return np.ascontiguousarray(np.repeat(image, 3, axis=2))
        return np.ascontiguousarray(image)

    def _image_callback(self, side: str, message: Image) -> None:
        try:
            bgr = self._to_bgr(message)
        except ValueError as exc:
            if side not in self._warned_inference:
                self.get_logger().error(f"{side} cubemap image rejected: {exc}")
                self._warned_inference.add(side)
            return

        frame = PendingFrame(
            side=side,
            bgr=bgr,
            height=int(message.height),
            width=int(message.width),
            frame_id=self._frame_id or message.header.frame_id or "camera_frame",
            stamp_sec=int(message.header.stamp.sec),
            stamp_nanosec=int(message.header.stamp.nanosec),
        )
        # Keep only the newest frame for each face so inference cannot build a
        # long queue when the camera is faster than the selected model.
        with self._pending_lock:
            self._pending[side] = frame
        self._wake.set()

    def _take_pending(self) -> list[PendingFrame]:
        with self._pending_lock:
            frames = [frame for frame in self._pending.values() if frame is not None]
            for side in FACE_NAMES:
                self._pending[side] = None
        return frames

    @staticmethod
    def _header(frame: PendingFrame) -> Header:
        header = Header()
        header.stamp.sec = frame.stamp_sec
        header.stamp.nanosec = frame.stamp_nanosec
        header.frame_id = frame.frame_id
        return header

    def _predict_depth(self, frame: PendingFrame) -> np.ndarray:
        kwargs = {
            "imgsz": self._imgsz,
            "verbose": False,
        }
        if self._device:
            kwargs["device"] = self._device
        if self._half:
            kwargs["half"] = True

        results = self._model.predict(source=frame.bgr, **kwargs)
        if not results or getattr(results[0], "depth", None) is None:
            raise RuntimeError("YOLO26s-depth returned no depth map")

        depth_data = results[0].depth.data
        if hasattr(depth_data, "detach"):
            depth_data = depth_data.detach().cpu().numpy()
        depth = np.asarray(depth_data, dtype=np.float32).squeeze()
        if depth.ndim != 2:
            raise RuntimeError(f"unexpected depth map shape: {depth.shape}")
        if depth.shape != (frame.height, frame.width):
            depth = cv2.resize(
                depth,
                (frame.width, frame.height),
                interpolation=cv2.INTER_LINEAR,
            )
        return np.ascontiguousarray(depth, dtype=np.float32)

    def _face_rays(self, side: str, height: int, width: int) -> np.ndarray:
        stride = self._point_stride
        rows = np.arange(0, height, stride, dtype=np.float32)
        cols = np.arange(0, width, stride, dtype=np.float32)
        tan_half_fov = np.tan(self._face_fov_rad / 2.0)
        u = (2.0 * (cols + 0.5) / width - 1.0) * tan_half_fov
        v = (2.0 * (rows + 0.5) / height - 1.0) * tan_half_fov
        u, v = np.meshgrid(u, v)

        if side == "left":
            cubemap_rays = np.stack((-np.ones_like(u), -v, u), axis=-1)
        elif side == "right":
            cubemap_rays = np.stack((np.ones_like(u), -v, -u), axis=-1)
        else:
            raise ValueError(f"unsupported cubemap side: {side}")
        # Cubemap rays are x=right, y=up, z=forward. Convert to the ROS
        # body-frame convention x=forward, y=left, z=up.
        rays = np.stack(
            (cubemap_rays[..., 2], -cubemap_rays[..., 0], cubemap_rays[..., 1]),
            axis=-1,
        )
        rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
        return rays.astype(np.float32, copy=False)

    def _make_depth_message(self, frame: PendingFrame, depth: np.ndarray) -> Image:
        message = Image()
        message.header = self._header(frame)
        message.height = frame.height
        message.width = frame.width
        message.encoding = "32FC1"
        message.is_bigendian = 0
        message.step = frame.width * 4
        message.data = np.ascontiguousarray(depth, dtype=np.float32).tobytes()
        return message

    def _make_pointcloud_message(
        self, frame: PendingFrame, depth: np.ndarray
    ) -> PointCloud2 | None:
        stride = self._point_stride
        sampled_depth = np.ascontiguousarray(depth[::stride, ::stride], dtype=np.float32)
        rays = self._face_rays(frame.side, frame.height, frame.width)
        sampled_bgr = frame.bgr[::stride, ::stride]

        if self._depth_mode == "optical_z":
            face_axis = FACE_AXES[frame.side]
            axis_projection = np.sum(rays * face_axis, axis=-1)
            sampled_depth = sampled_depth / np.maximum(axis_projection, 1e-6)

        valid = np.isfinite(sampled_depth) & (sampled_depth > self._depth_min)
        if self._depth_max > 0.0:
            valid &= sampled_depth <= self._depth_max
        if not np.any(valid):
            return None

        xyz = rays * sampled_depth[..., None]
        cloud_dtype = np.dtype(
            [
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("rgb", "<f4"),
            ]
        )
        cloud = np.empty(int(valid.sum()), dtype=cloud_dtype)
        cloud["x"] = xyz[..., 0][valid]
        cloud["y"] = xyz[..., 1][valid]
        cloud["z"] = xyz[..., 2][valid]

        colors = np.ascontiguousarray(sampled_bgr[valid, ::-1], dtype=np.uint8)
        packed_rgb = (
            (colors[:, 0].astype(np.uint32) << 16)
            | (colors[:, 1].astype(np.uint32) << 8)
            | colors[:, 2].astype(np.uint32)
        )
        cloud["rgb"] = packed_rgb.view("<f4")

        message = PointCloud2()
        message.header = self._header(frame)
        message.height = 1
        message.width = cloud.shape[0]
        message.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        message.is_bigendian = False
        message.point_step = cloud_dtype.itemsize
        message.row_step = message.point_step * message.width
        message.data = cloud.tobytes()
        message.is_dense = True
        return message

    def _inference_loop(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(0.1)
            self._wake.clear()
            for frame in self._take_pending():
                if self._stop.is_set():
                    break
                try:
                    depth = self._predict_depth(frame)
                    if self._stop.is_set():
                        break
                    if self._publish_depth:
                        self._depth_publishers[frame.side].publish(
                            self._make_depth_message(frame, depth)
                        )
                    pointcloud = self._make_pointcloud_message(frame, depth)
                    if pointcloud is not None:
                        self._pointcloud_publishers[frame.side].publish(pointcloud)
                    self._processed += 1
                except Exception as exc:  # keep the other face alive on one bad frame
                    if frame.side not in self._warned_inference:
                        self.get_logger().error(
                            f"YOLO26s-depth inference failed for {frame.side}: {exc}"
                        )
                        self._warned_inference.add(frame.side)

            now = time.monotonic()
            if now - self._last_rate_log >= 5.0:
                count = self._processed - self._last_rate_count
                rate = count / (now - self._last_rate_log)
                self.get_logger().info(
                    f"published {rate:.2f} YOLO26s-depth point-cloud frames/s"
                )
                self._last_rate_log = now
                self._last_rate_count = self._processed

    def close(self) -> None:
        self._stop.set()
        self._wake.set()
        if hasattr(self, "_worker"):
            self._worker.join(timeout=2.0)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = Yolo26sDepthPointCloud()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
