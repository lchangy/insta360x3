#!/usr/bin/env python3
"""Convert six Gazebo cube faces into the equirectangular image DA360 consumes."""

from __future__ import annotations

import math
import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Header


FACES = ("front", "right", "back", "left", "top", "bottom")


class GazeboEquirectangular(Node):
    """Publish a 2:1 ERP image from the six square camera topics."""

    def __init__(self) -> None:
        super().__init__("gazebo_equirectangular")
        self.declare_parameter("face_size", 360)
        self.declare_parameter("output_width", 0)
        self.declare_parameter("output_height", 0)
        self.declare_parameter("publish_rate", 5.0)
        self.declare_parameter("output_topic", "/equirectangular/image")
        self.declare_parameter("frame_id", "camera_frame")
        self.declare_parameter("topic_prefix", "/da360_sim/cubemap")

        self.face_size = max(32, int(self.get_parameter("face_size").value))
        output_width = int(self.get_parameter("output_width").value)
        output_height = int(self.get_parameter("output_height").value)
        self.output_width = output_width or self.face_size * 4
        self.output_height = output_height or self.face_size * 2
        self.publish_rate = max(0.1, float(self.get_parameter("publish_rate").value))
        self.output_topic = str(self.get_parameter("output_topic").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        topic_prefix = str(self.get_parameter("topic_prefix").value).rstrip("/")

        sensor_qos = QoSProfile(
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

        self._faces: Dict[str, Tuple[Header, np.ndarray]] = {}
        self._warned_encoding = set()
        self._publisher = self.create_publisher(Image, self.output_topic, output_qos)
        self._subscriptions = [
            self.create_subscription(
                Image,
                f"{topic_prefix}/{face}",
                lambda msg, name=face: self._image_callback(name, msg),
                sensor_qos,
            )
            for face in FACES
        ]
        self._timer = self.create_timer(1.0 / self.publish_rate, self._publish)
        self._published = 0
        self._last_log = time.monotonic()

        self.get_logger().info(
            f"six-face ERP: {self.output_width}x{self.output_height} from "
            f"{self.face_size}x{self.face_size} faces; output={self.output_topic}"
        )

    @staticmethod
    def _decode(msg: Image) -> np.ndarray:
        encoding = msg.encoding.lower()
        channels = {
            "rgb8": 3,
            "bgr8": 3,
            "rgba8": 4,
            "bgra8": 4,
            "mono8": 1,
        }.get(encoding)
        if channels is None:
            raise ValueError(f"unsupported image encoding: {msg.encoding}")

        row_bytes = msg.width * channels
        step = msg.step or row_bytes
        raw = np.frombuffer(memoryview(msg.data), dtype=np.uint8)
        if step < row_bytes or raw.size < msg.height * step:
            raise ValueError(
                f"invalid image step={msg.step} for {msg.width}x{msg.height} {msg.encoding}"
            )
        image = raw.reshape(msg.height, step)[:, :row_bytes]
        image = image.reshape(msg.height, msg.width, channels)
        if encoding == "rgb8":
            image = image[:, :, ::-1]
        elif encoding == "rgba8":
            image = image[:, :, 2::-1]
        elif encoding == "bgra8":
            image = image[:, :, :3]
        elif encoding == "mono8":
            image = np.repeat(image, 3, axis=2)
        return np.ascontiguousarray(image)

    def _image_callback(self, face: str, msg: Image) -> None:
        try:
            image = self._decode(msg)
        except ValueError as exc:
            if face not in self._warned_encoding:
                self.get_logger().error(f"{face}: {exc}")
                self._warned_encoding.add(face)
            return

        if image.shape[:2] != (self.face_size, self.face_size):
            image = cv2.resize(image, (self.face_size, self.face_size), interpolation=cv2.INTER_AREA)
        self._faces[face] = (msg.header, image)

    @staticmethod
    def _face_map(direction: Tuple[np.ndarray, np.ndarray, np.ndarray], face: str):
        """Return cube-face coordinates using cubemap_projection.py conventions."""
        dx, dy, dz = direction
        if face == "front":
            denominator = np.where(np.abs(dz) > 1e-6, dz, 1e-6)
            a, b = dx / denominator, -dy / denominator
        elif face == "back":
            denominator = np.where(np.abs(dz) > 1e-6, dz, -1e-6)
            a, b = dx / denominator, dy / denominator
        elif face == "right":
            denominator = np.where(np.abs(dx) > 1e-6, dx, 1e-6)
            a, b = -dz / denominator, -dy / denominator
        elif face == "left":
            denominator = np.where(np.abs(dx) > 1e-6, dx, -1e-6)
            a, b = -dz / denominator, dy / denominator
        elif face == "top":
            denominator = np.where(np.abs(dy) > 1e-6, dy, 1e-6)
            a, b = dx / denominator, dz / denominator
        elif face == "bottom":
            denominator = np.where(np.abs(dy) > 1e-6, dy, -1e-6)
            a, b = -dx / denominator, dz / denominator
        else:
            raise ValueError(f"unknown face: {face}")

        map_x = (a + 1.0) * 0.5 * np.float32(direction[0].shape[1]) - 0.5
        map_y = (b + 1.0) * 0.5 * np.float32(direction[0].shape[0]) - 0.5
        return map_x.astype(np.float32), map_y.astype(np.float32)

    def _compose(self) -> Optional[np.ndarray]:
        if any(face not in self._faces for face in FACES):
            return None

        rows = self.output_height
        cols = self.output_width
        longitude = (np.arange(cols, dtype=np.float32) + 0.5) * (2.0 * math.pi / cols) - math.pi
        latitude = math.pi / 2.0 - (np.arange(rows, dtype=np.float32) + 0.5) * (math.pi / rows)
        cos_lat = np.cos(latitude)[:, None]
        # ERP convention used by the repository cubemap projector:
        # longitude 0 is +X in the projector frame and +Y is up.
        dx = cos_lat * np.cos(longitude)[None, :]
        dy = np.sin(latitude)[:, None] * np.ones((1, cols), dtype=np.float32)
        dz = cos_lat * np.sin(longitude)[None, :]
        direction = (dx, dy, dz)
        abs_dx, abs_dy, abs_dz = np.abs(dx), np.abs(dy), np.abs(dz)

        masks = {
            "front": (dz >= 0) & (abs_dz >= abs_dx) & (abs_dz >= abs_dy),
            "back": (dz < 0) & (abs_dz >= abs_dx) & (abs_dz >= abs_dy),
            "right": (dx >= 0) & (abs_dx > abs_dz) & (abs_dx >= abs_dy),
            "left": (dx < 0) & (abs_dx > abs_dz) & (abs_dx >= abs_dy),
            "top": (dy >= 0) & (abs_dy > abs_dx) & (abs_dy > abs_dz),
            "bottom": (dy < 0) & (abs_dy > abs_dx) & (abs_dy > abs_dz),
        }

        erp = np.zeros((rows, cols, 3), dtype=np.uint8)
        for face in FACES:
            image = self._faces[face][1]
            map_x, map_y = self._face_map(direction, face)
            sampled = cv2.remap(
                image,
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            erp[masks[face]] = sampled[masks[face]]
        return erp

    def _publish(self) -> None:
        erp = self._compose()
        if erp is None:
            return

        latest_header = max(
            (header for header, _ in self._faces.values()),
            key=lambda header: (header.stamp.sec, header.stamp.nanosec),
        )
        msg = Image()
        msg.header = Header()
        msg.header.stamp = latest_header.stamp
        msg.header.frame_id = self.frame_id
        msg.height, msg.width = erp.shape[:2]
        msg.encoding = "bgr8"
        msg.is_bigendian = False
        msg.step = msg.width * 3
        msg.data = erp.tobytes()
        self._publisher.publish(msg)

        self._published += 1
        now = time.monotonic()
        if now - self._last_log >= 5.0:
            rate = self._published / (now - self._last_log) if self._last_log else 0.0
            self.get_logger().info(f"published {self.output_width}x{self.output_height} ERP ({rate:.2f} Hz)")
            self._published = 0
            self._last_log = now


def main() -> None:
    rclpy.init()
    node = GazeboEquirectangular()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
