#!/usr/bin/env python3
"""Display the four rectilinear Insta360 surround-view ROS topics."""

import threading

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


TOPICS = {
    "FRONT": "/surround/front/image",
    "RIGHT": "/surround/right/image",
    "BACK": "/surround/back/image",
    "LEFT": "/surround/left/image",
}


class SurroundViewer(Node):
    def __init__(self):
        super().__init__("insta360_surround_viewer")
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.frames = {name: None for name in TOPICS}
        self.counts = {name: 0 for name in TOPICS}
        self.subscriptions_ = [
            self.create_subscription(
                Image,
                topic,
                lambda msg, name=name: self.on_image(name, msg),
                qos,
            )
            for name, topic in TOPICS.items()
        ]

    def on_image(self, name, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            with self.lock:
                self.frames[name] = frame.copy()
                self.counts[name] += 1
                count = self.counts[name]
            if count == 1:
                print(f"[surround-view] {name}: {msg.width}x{msg.height} {msg.encoding}", flush=True)
        except Exception as exc:  # cv_bridge can reject a malformed frame
            print(f"[surround-view] {name} conversion error: {exc}", flush=True)

    def render(self):
        with self.lock:
            frames = dict(self.frames)
            counts = dict(self.counts)

        panels = []
        for name in TOPICS:
            frame = frames[name]
            if frame is None or frame.size == 0:
                panel = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(
                    panel,
                    f"waiting for image... ({counts[name]})",
                    (150, 245),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (220, 220, 220),
                    2,
                    cv2.LINE_AA,
                )
            else:
                panel = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)

            cv2.rectangle(panel, (0, 0), (180, 42), (0, 0, 0), -1)
            cv2.putText(
                panel,
                name,
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                panel,
                f"frames: {counts[name]}",
                (15, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            panels.append(panel)

        top = np.hstack((panels[0], panels[1]))
        bottom = np.hstack((panels[2], panels[3]))
        cv2.imshow("Insta360 X3 - Surround Views", np.vstack((top, bottom)))


def main():
    rclpy.init()
    node = SurroundViewer()
    cv2.namedWindow("Insta360 X3 - Surround Views", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Insta360 X3 - Surround Views", 1280, 960)
    cv2.moveWindow("Insta360 X3 - Surround Views", 80, 50)
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.03)
            node.render()
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
