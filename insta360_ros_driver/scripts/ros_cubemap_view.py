#!/usr/bin/env python3
"""Show and publish four horizontal cubemap faces from a ROS ERP image."""

from __future__ import annotations

import argparse

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import Image

from insta360_ros_driver.cubemap_projection import build_face_map, remap_face


FACE_NAMES = ('front', 'right', 'back', 'left')


class RosCubemapView(Node):
    def __init__(self, topic: str, face_size: int, show: bool) -> None:
        super().__init__('insta360_ros_cubemap_view')
        self.bridge = CvBridge()
        self.face_size = face_size
        self.show = show
        self.maps = {
            name: build_face_map(face_size, name) for name in FACE_NAMES
        }

        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.face_publishers = {
            name: self.create_publisher(Image, f'/cubemap/{name}/image', qos)
            for name in FACE_NAMES
        }
        self.mosaic_publisher = self.create_publisher(
            Image,
            '/cubemap/horizontal/image',
            qos,
        )
        self.subscription = self.create_subscription(
            Image,
            topic,
            self.image_callback,
            qos,
        )
        self.get_logger().info(
            f'Listening on {topic}; publishing four {face_size}x{face_size} '
            f'cubemap faces; gui={show}'
        )

    def image_callback(self, message: Image) -> None:
        try:
            erp_bgr = self.bridge.imgmsg_to_cv2(message, desired_encoding='bgr8')
        except CvBridgeError as exc:
            self.get_logger().error(f'ROS image conversion failed: {exc}')
            return

        faces: dict[str, np.ndarray] = {}
        for name in FACE_NAMES:
            map_x, map_y = self.maps[name]
            face = remap_face(
                erp_bgr,
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_WRAP,
            )
            faces[name] = face

            output = self.bridge.cv2_to_imgmsg(face, encoding='bgr8')
            output.header = message.header
            output.header.frame_id = f'cubemap_{name}'
            self.face_publishers[name].publish(output)

        mosaic = np.vstack((
            np.hstack((faces['front'], faces['right'])),
            np.hstack((faces['back'], faces['left'])),
        ))
        for label, x, y in (
            ('FRONT', 8, 24),
            ('RIGHT', self.face_size + 8, 24),
            ('BACK', 8, self.face_size + 24),
            ('LEFT', self.face_size + 8, self.face_size + 24),
        ):
            cv2.putText(
                mosaic,
                label,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        mosaic_message = self.bridge.cv2_to_imgmsg(mosaic, encoding='bgr8')
        mosaic_message.header = message.header
        mosaic_message.header.frame_id = 'cubemap_horizontal'
        self.mosaic_publisher.publish(mosaic_message)

        if self.show:
            cv2.imshow('Cubemap: front | right / back | left', mosaic)
            cv2.waitKey(1)

    def destroy_node(self) -> bool:
        if self.show:
            cv2.destroyAllWindows()
        return super().destroy_node()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--topic', default='/equirectangular/image')
    parser.add_argument('--face-size', type=int, default=360)
    parser.add_argument('--gui', choices=('true', 'false'), default='true')
    parser.add_argument('--no-gui', action='store_true')
    args = parser.parse_args(remove_ros_args()[1:])
    if args.face_size <= 0:
        parser.error('--face-size must be a positive integer')

    show = args.gui == 'true' and not args.no_gui
    rclpy.init()
    node = RosCubemapView(args.topic, args.face_size, show)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
