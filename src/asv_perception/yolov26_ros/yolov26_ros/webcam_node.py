#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2


class USBCameraNode(Node):
    def __init__(self):
        super().__init__("usb_camera_node")

        # Publisher
        self.pub = self.create_publisher(Image, "/usb_cam/image_raw", 10)

        self.bridge = CvBridge()

        # Open default USB camera (0)
        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

        if not self.cap.isOpened():
            self.get_logger().error("Failed to open USB camera (/dev/video0)")
            return

        # Timer at ~30 Hz
        self.timer = self.create_timer(1.0 / 30.0, self.timer_cb)

        self.get_logger().info("USB camera node started")
        self.get_logger().info("Publishing: /usb_cam/image_raw")

    def timer_cb(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("Failed to read frame from camera")
            return

        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "usb_camera"

        self.pub.publish(msg)

    def destroy_node(self):
        if self.cap.isOpened():
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = USBCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
