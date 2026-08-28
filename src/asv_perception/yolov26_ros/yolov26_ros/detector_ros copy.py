#!/usr/bin/env python3
# YOLO26 ROS2 Node (Ultralytics)
# Created by: Xavier Vicent
# Publishes:
#   - /yolov26/detections                (ALL classes)
#   - /yolov26/annotated_image           (ALL classes)
#   - /yolov26/detections_black_buoy     (BLACK BUOY only)
#   - /yolov26/annotated_image_black_buoy (BLACK BUOY only)

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import numpy as np

from ultralytics import YOLO

from yolov26_msgs.msg import YoloDetection
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy


class YOLO26Node(Node):
    def __init__(self):
        super().__init__("yolov26_detector")

        # --- Parameters ---
        self.weights = self.declare_parameter("weights", "yolo26n.pt").value
        self.input_topic = self.declare_parameter("input_topic", "/image_raw").value

        # Existing outputs (KEEP)
        self.output_topic_image = self.declare_parameter(
            "output_topic_image", "/yolov26/annotated_image"
        ).value

        self.output_topic_detections = self.declare_parameter(
            "output_topic_detections", "/yolov26/detections"
        ).value

        # NEW: black buoy outputs
        self.output_topic_black_det = self.declare_parameter(
            "output_topic_black_detections", "/yolov26/detections_black_buoy"
        ).value

        self.output_topic_black_img = self.declare_parameter(
            "output_topic_black_image", "/yolov26/annotated_image_black_buoy"
        ).value

        self.black_buoy_class_name = self.declare_parameter(
            "black_buoy_class_name", "black_buoy"
        ).value

        # Model params
        self.conf = float(self.declare_parameter("conf", 0.25).value)
        self.iou = float(self.declare_parameter("iou", 0.45).value)
        self.imgsz = int(self.declare_parameter("imgsz", 640).value)
        self.device = str(self.declare_parameter("device", "cpu").value)
        self.max_det = int(self.declare_parameter("max_det", 300).value)

        self.get_logger().info(f"Loading YOLO model: {self.weights}")
        self.model = YOLO(self.weights)

        # Warmup
        try:
            dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            _ = self.model.predict(dummy, imgsz=self.imgsz,
                                   conf=self.conf, iou=self.iou,
                                   device=self.device,
                                   max_det=self.max_det, verbose=False)
        except Exception as e:
            self.get_logger().warn(f"Warmup failed: {e}")

        self.bridge = CvBridge()

        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
        )

        self.sub = self.create_subscription(Image, self.input_topic, self.image_cb, qos)

        # Publishers
        self.pub_img_all = self.create_publisher(Image, self.output_topic_image, 10)
        self.pub_det_all = self.create_publisher(YoloDetection, self.output_topic_detections, 10)

        self.pub_img_black = self.create_publisher(Image, self.output_topic_black_img, 10)
        self.pub_det_black = self.create_publisher(YoloDetection, self.output_topic_black_det, 10)

        self.get_logger().info("YOLO26 Node Ready")

    @staticmethod
    def empty_det_msg():
        m = YoloDetection()
        m.class_name = []
        m.confidence = []
        m.x_min = []
        m.y_min = []
        m.x_max = []
        m.y_max = []
        m.x_center = []
        m.y_center = []
        return m

    def image_cb(self, msg: Image):

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"cv_bridge failed: {e}")
            return

        try:
            results = self.model.predict(
                frame,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                device=self.device,
                max_det=self.max_det,
                verbose=False
            )
        except Exception as e:
            self.get_logger().error(f"YOLO failed: {e}")
            return

        if not results:
            return

        r0 = results[0]

        det_all = self.empty_det_msg()
        det_black = self.empty_det_msg()

        annotated_all = r0.plot()

        # Create black-only image copy
        annotated_black = frame.copy()

        boxes = getattr(r0, "boxes", None)
        if boxes is not None and boxes.xyxy is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            names = r0.names if hasattr(r0, "names") else {}

            target = str(self.black_buoy_class_name)

            for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):

                cls_name = str(names.get(int(k), int(k)))
                xc = float((x1 + x2) / 2.0)
                yc = float((y1 + y2) / 2.0)

                # ----- ALL detections -----
                det_all.class_name.append(cls_name)
                det_all.confidence.append(float(c))
                det_all.x_min.append(float(x1))
                det_all.y_min.append(float(y1))
                det_all.x_max.append(float(x2))
                det_all.y_max.append(float(y2))
                det_all.x_center.append(xc)
                det_all.y_center.append(yc)

                # ----- BLACK BUOY ONLY -----
                if cls_name == target:
                    det_black.class_name.append(cls_name)
                    det_black.confidence.append(float(c))
                    det_black.x_min.append(float(x1))
                    det_black.y_min.append(float(y1))
                    det_black.x_max.append(float(x2))
                    det_black.y_max.append(float(y2))
                    det_black.x_center.append(xc)
                    det_black.y_center.append(yc)

                    # Draw ONLY black buoy on black image
                    cv2.rectangle(
                        annotated_black,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (0, 255, 255), 2
                    )
                    cv2.putText(
                        annotated_black,
                        f"{cls_name} {c:.2f}",
                        (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        2
                    )

        # Publish detections
        self.pub_det_all.publish(det_all)
        self.pub_det_black.publish(det_black)

        # Publish annotated images
        out_all = self.bridge.cv2_to_imgmsg(annotated_all, encoding="bgr8")
        out_all.header = msg.header
        self.pub_img_all.publish(out_all)

        out_black = self.bridge.cv2_to_imgmsg(annotated_black, encoding="bgr8")
        out_black.header = msg.header
        self.pub_img_black.publish(out_black)


def main(args=None):
    rclpy.init(args=args)
    node = YOLO26Node()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
