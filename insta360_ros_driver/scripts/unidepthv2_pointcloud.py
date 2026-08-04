#!/usr/bin/env python3
"""Publish UniDepthV2-Small point clouds for all four cubemap faces.

The cubemap publisher is intentionally not started or modified here.  This
node only subscribes to its existing front/right/back/left image topics, so it
can be launched alongside the current camera/depth pipeline without changing
that pipeline's parameters or processes.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rclpy
import torch
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


FACE_NAMES = ("front", "right", "back", "left")


@dataclass(frozen=True)
class PendingFrame:
    side: str
    rgb: np.ndarray
    height: int
    width: int
    frame_id: str
    stamp_sec: int
    stamp_nanosec: int


class UniDepthV2PointCloud(Node):
    """Run UniDepthV2-Small on four cubemap streams and publish PointCloud2."""

    def __init__(self) -> None:
        super().__init__("unidepthv2_pointcloud")

        self.declare_parameter(
            "model_name",
            os.environ.get(
                "UNIDEPTHV2_MODEL", "lpiccinelli/unidepth-v2-vits14"
            ),
        )
        self.declare_parameter(
            "unidepth_repo", os.environ.get("UNIDEPTH_REPO", "")
        )
        for face in FACE_NAMES:
            self.declare_parameter(
                f"{face}_image_topic", f"/cubemap/{face}/image"
            )
            self.declare_parameter(
                f"{face}_depth_topic", f"/unidepthv2/{face}/depth"
            )
            self.declare_parameter(
                f"{face}_pointcloud_topic", f"/unidepthv2/{face}/points"
            )
        self.declare_parameter("frame_id", "camera_frame")
        self.declare_parameter("point_stride", 2)
        self.declare_parameter("face_fov_deg", 90.0)
        self.declare_parameter("range_min", 0.05)
        self.declare_parameter("range_max", 100.0)
        self.declare_parameter("confidence_threshold", 0.0)
        self.declare_parameter("publish_depth", True)
        self.declare_parameter("device", "")
        self.declare_parameter("resolution_level", 5)
        self.declare_parameter("interpolation_mode", "bilinear")

        self._model_name = str(self.get_parameter("model_name").value)
        self._frame_id = str(self.get_parameter("frame_id").value)
        self._point_stride = max(1, int(self.get_parameter("point_stride").value))
        self._face_fov_rad = np.deg2rad(
            float(self.get_parameter("face_fov_deg").value)
        )
        if not 0.0 < self._face_fov_rad < np.pi:
            raise ValueError("face_fov_deg must be between 0 and 180")

        self._range_min = max(0.0, float(self.get_parameter("range_min").value))
        self._range_max = float(self.get_parameter("range_max").value)
        self._confidence_threshold = float(
            self.get_parameter("confidence_threshold").value
        )
        self._publish_depth = bool(self.get_parameter("publish_depth").value)

        device_name = str(self.get_parameter("device").value).strip()
        if not device_name:
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device_name)
        if self._device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"UniDepth device {self._device} was requested, but CUDA is unavailable"
            )

        resolution_level = int(self.get_parameter("resolution_level").value)
        if not -1 <= resolution_level < 10:
            raise ValueError("resolution_level must be -1 or an integer in [0, 10)")
        self._resolution_level = resolution_level
        self._interpolation_mode = str(
            self.get_parameter("interpolation_mode").value
        )

        self._pending_lock = threading.Lock()
        self._pending: dict[str, PendingFrame | None] = {
            side: None for side in FACE_NAMES
        }
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._warned_encoding: set[str] = set()
        self._warned_inference: set[str] = set()
        self._processed = 0
        self._last_rate_log = time.monotonic()
        self._last_rate_count = 0

        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        output_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        image_topics = {
            face: str(self.get_parameter(f"{face}_image_topic").value)
            for face in FACE_NAMES
        }
        depth_topics = {
            face: str(self.get_parameter(f"{face}_depth_topic").value)
            for face in FACE_NAMES
        }
        pointcloud_topics = {
            face: str(self.get_parameter(f"{face}_pointcloud_topic").value)
            for face in FACE_NAMES
        }

        self._depth_publishers = {}
        if self._publish_depth:
            self._depth_publishers = {
                side: self.create_publisher(Image, depth_topics[side], output_qos)
                for side in FACE_NAMES
            }
        self._pointcloud_publishers = {
            side: self.create_publisher(
                PointCloud2, pointcloud_topics[side], output_qos
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
                image_qos,
            )
            for side in FACE_NAMES
        ]

        self._model, self._pinhole_type = self._load_model(
            str(self.get_parameter("unidepth_repo").value)
        )
        self._worker = threading.Thread(
            target=self._inference_loop,
            name="unidepthv2-inference",
            daemon=True,
        )
        self._worker.start()

        self.get_logger().info(
            "Subscribed to "
            + ", ".join(image_topics[face] for face in FACE_NAMES)
            + "; publishing "
            + ", ".join(pointcloud_topics[face] for face in FACE_NAMES)
            + f"; point_stride={self._point_stride}; device={self._device}"
        )

    def _load_model(self, unidepth_repo: str):
        repo_path = Path(unidepth_repo).expanduser() if unidepth_repo else None
        if repo_path is not None:
            repo_path = repo_path.resolve()
            if not repo_path.is_dir():
                raise FileNotFoundError(
                    f"UniDepth repository does not exist: {repo_path}"
                )
            sys.path.insert(0, str(repo_path))

        try:
            from unidepth.models import UniDepthV2
            from unidepth.utils.camera import Pinhole
        except ImportError as exc:
            raise RuntimeError(
                "UniDepth is not importable. Install the lpiccinelli-eth/unidepth "
                "package in the selected Python environment, or set unidepth_repo "
                "to its checkout."
            ) from exc

        self.get_logger().info(
            f"Loading UniDepthV2 model {self._model_name!r} on {self._device}; "
            "the first run may download the Hugging Face checkpoint"
        )
        model = UniDepthV2.from_pretrained(self._model_name)
        if self._resolution_level >= 0:
            model.resolution_level = self._resolution_level
        if self._interpolation_mode:
            model.interpolation_mode = self._interpolation_mode
        model = model.to(self._device).eval()
        self.get_logger().info(
            f"UniDepthV2 ready: model={self._model_name!r}, "
            f"resolution_level={self._resolution_level}, "
            f"interpolation={self._interpolation_mode or 'model default'}"
        )
        return model, Pinhole

    @staticmethod
    def _to_rgb(message: Image) -> np.ndarray:
        encoding = message.encoding.lower()
        if encoding in ("rgb8", "bgr8"):
            channels = 3
        elif encoding in ("rgba8", "bgra8"):
            channels = 4
        elif encoding == "mono8":
            channels = 1
        else:
            raise ValueError(f"unsupported image encoding: {message.encoding}")

        height = int(message.height)
        width = int(message.width)
        step = int(message.step)
        expected_row = width * channels
        raw = np.frombuffer(memoryview(message.data), dtype=np.uint8)
        if step < expected_row or raw.size < height * step:
            raise ValueError(
                f"invalid Image step={step} for "
                f"{width}x{height} {message.encoding}"
            )
        image = raw.reshape(height, step)[:, :expected_row]
        image = image.reshape(height, width, channels)

        if encoding == "rgb8":
            return np.ascontiguousarray(image)
        if encoding == "bgr8":
            return np.ascontiguousarray(image[:, :, ::-1])
        if encoding == "rgba8":
            return np.ascontiguousarray(image[:, :, :3])
        if encoding == "bgra8":
            return np.ascontiguousarray(image[:, :, 2::-1])
        return np.ascontiguousarray(np.repeat(image, 3, axis=2))

    def _image_callback(self, side: str, message: Image) -> None:
        try:
            rgb = self._to_rgb(message)
        except ValueError as exc:
            if side not in self._warned_encoding:
                self.get_logger().error(f"{side} cubemap image rejected: {exc}")
                self._warned_encoding.add(side)
            return

        frame = PendingFrame(
            side=side,
            rgb=rgb,
            height=int(message.height),
            width=int(message.width),
            frame_id=self._frame_id or message.header.frame_id or "camera_frame",
            stamp_sec=int(message.header.stamp.sec),
            stamp_nanosec=int(message.header.stamp.nanosec),
        )
        # Keep only the newest frame for each face.  UniDepth inference is
        # intentionally decoupled from ROS callbacks so it cannot block the
        # image publisher or build an unbounded queue.
        with self._pending_lock:
            self._pending[side] = frame
        self._wake.set()

    def _take_pending(self) -> list[PendingFrame]:
        with self._pending_lock:
            frames = [frame for frame in self._pending.values() if frame is not None]
            for side in FACE_NAMES:
                self._pending[side] = None
        return frames

    def _make_camera(self, height: int, width: int):
        tan_half_fov = np.tan(self._face_fov_rad / 2.0)
        # ros_cubemap_view uses pixel-center normalized coordinates, so this
        # K exactly matches the 90-degree cubemap ray construction.
        fx = width / (2.0 * tan_half_fov)
        fy = height / (2.0 * tan_half_fov)
        K = torch.tensor(
            [
                [fx, 0.0, (width - 1.0) / 2.0],
                [0.0, fy, (height - 1.0) / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
            device=self._device,
        ).unsqueeze(0)
        return self._pinhole_type(K=K)

    def _predict(self, frame: PendingFrame):
        rgb = torch.from_numpy(frame.rgb).permute(2, 0, 1).contiguous()
        camera = self._make_camera(frame.height, frame.width)
        predictions = self._model.infer(rgb, camera)

        points = predictions.get("points")
        depth = predictions.get("depth")
        if points is None or depth is None:
            raise RuntimeError("UniDepthV2 returned no points/depth prediction")
        if points.ndim != 4 or points.shape[0] != 1 or points.shape[1] != 3:
            raise RuntimeError(f"unexpected UniDepth points shape: {tuple(points.shape)}")
        if depth.ndim != 4 or depth.shape[0] != 1 or depth.shape[1] != 1:
            raise RuntimeError(f"unexpected UniDepth depth shape: {tuple(depth.shape)}")
        if tuple(points.shape[-2:]) != (frame.height, frame.width):
            raise RuntimeError(
                "UniDepth output size "
                f"{tuple(points.shape[-2:])} does not match input "
                f"{(frame.height, frame.width)}"
            )
        if tuple(depth.shape[-2:]) != (frame.height, frame.width):
            raise RuntimeError(
                "UniDepth depth output size "
                f"{tuple(depth.shape[-2:])} does not match input "
                f"{(frame.height, frame.width)}"
            )

        points_np = points[0].detach().float().cpu().numpy()
        depth_np = depth[0, 0].detach().float().cpu().numpy()
        confidence_np = None
        confidence = predictions.get("confidence")
        if confidence is not None:
            if confidence.ndim == 4 and confidence.shape[0] == 1:
                confidence_np = confidence[0, 0].detach().float().cpu().numpy()
                if confidence_np.shape != (frame.height, frame.width):
                    confidence_np = None

        points_local = np.transpose(points_np, (1, 2, 0))
        # UniDepth/Pinhole uses x-right, y-down, z-forward image coordinates.
        # The cubemap node's shared camera frame is x-right, y-up, z-forward.
        # These transforms exactly match cubemap_projection.build_face_map():
        # front=(+x,-y,+z), right=(+z,-y,-x), back=(-x,-y,-z),
        # left=(-z,-y,+x).
        if frame.side == "front":
            points_camera = np.stack(
                (
                    points_local[..., 0],
                    -points_local[..., 1],
                    points_local[..., 2],
                ),
                axis=-1,
            )
        elif frame.side == "right":
            points_camera = np.stack(
                (
                    points_local[..., 2],
                    -points_local[..., 1],
                    -points_local[..., 0],
                ),
                axis=-1,
            )
        elif frame.side == "back":
            points_camera = np.stack(
                (
                    -points_local[..., 0],
                    -points_local[..., 1],
                    -points_local[..., 2],
                ),
                axis=-1,
            )
        elif frame.side == "left":
            points_camera = np.stack(
                (
                    -points_local[..., 2],
                    -points_local[..., 1],
                    points_local[..., 0],
                ),
                axis=-1,
            )
        else:
            raise ValueError(f"unsupported cubemap side: {frame.side}")

        return (
            np.ascontiguousarray(points_camera, dtype=np.float32),
            np.ascontiguousarray(depth_np, dtype=np.float32),
            None
            if confidence_np is None
            else np.ascontiguousarray(confidence_np, dtype=np.float32),
        )

    @staticmethod
    def _header(frame: PendingFrame) -> Header:
        header = Header()
        header.stamp.sec = frame.stamp_sec
        header.stamp.nanosec = frame.stamp_nanosec
        header.frame_id = frame.frame_id
        return header

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
        self,
        frame: PendingFrame,
        points: np.ndarray,
        depth: np.ndarray,
        confidence: np.ndarray | None,
    ) -> PointCloud2 | None:
        stride = self._point_stride
        sampled_points = points[::stride, ::stride]
        sampled_depth = depth[::stride, ::stride]
        sampled_rgb = frame.rgb[::stride, ::stride]
        sampled_range = np.linalg.norm(sampled_points, axis=-1)

        valid = np.isfinite(sampled_points).all(axis=-1)
        valid &= np.isfinite(sampled_range)
        valid &= np.isfinite(sampled_depth) & (sampled_depth > 0.0)
        valid &= sampled_range >= self._range_min
        if self._range_max > 0.0:
            valid &= sampled_range <= self._range_max
        if self._confidence_threshold > 0.0 and confidence is not None:
            valid &= confidence[::stride, ::stride] >= self._confidence_threshold
        if not np.any(valid):
            return None

        cloud_dtype = np.dtype(
            [
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("rgb", "<f4"),
            ]
        )
        cloud = np.empty(int(valid.sum()), dtype=cloud_dtype)
        cloud["x"] = sampled_points[..., 0][valid]
        cloud["y"] = sampled_points[..., 1][valid]
        cloud["z"] = sampled_points[..., 2][valid]

        colors = np.ascontiguousarray(sampled_rgb[valid], dtype=np.uint8)
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
                    points, depth, confidence = self._predict(frame)
                    if self._publish_depth:
                        self._depth_publishers[frame.side].publish(
                            self._make_depth_message(frame, depth)
                        )
                    pointcloud = self._make_pointcloud_message(
                        frame, points, depth, confidence
                    )
                    if pointcloud is not None:
                        self._pointcloud_publishers[frame.side].publish(pointcloud)
                    self._processed += 1
                except Exception as exc:  # keep the other face alive on one bad frame
                    if frame.side not in self._warned_inference:
                        self.get_logger().error(
                            f"UniDepthV2 inference failed for {frame.side}: {exc}"
                        )
                        self._warned_inference.add(frame.side)

            now = time.monotonic()
            if now - self._last_rate_log >= 5.0:
                count = self._processed - self._last_rate_count
                rate = count / (now - self._last_rate_log)
                self.get_logger().info(
                    f"published {rate:.2f} UniDepthV2 point-cloud frames/s"
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
        node = UniDepthV2PointCloud()
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
