#!/usr/bin/env python3
import os
import threading

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


TOPICS = {
    "dual": "/dual_fisheye/image",
    "equirectangular": "/equirectangular/image",
    "front": "/surround/front/image",
    "right": "/surround/right/image",
    "back": "/surround/back/image",
    "left": "/surround/left/image",
}


class Capture(Node):
    def __init__(self):
        super().__init__("capture_surround_once")
        self.bridge = CvBridge()
        self.saved = set()
        self.subscriptions_ = [
            self.create_subscription(
                Image, topic,
                lambda msg, name=name: self.save(name, msg), 10)
            for name, topic in TOPICS.items()
        ]

    def save(self, name, msg):
        if name in self.saved:
            return
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        path = f"/tmp/insta360_{name}.png"
        cv2.imwrite(path, frame)
        self.saved.add(name)
        print(f"saved {name}: {frame.shape} -> {path}", flush=True)
        if len(self.saved) == len(TOPICS):
            os._exit(0)


def main():
    rclpy.init()
    node = Capture()
    threading.Timer(8.0, lambda: os._exit(2)).start()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
