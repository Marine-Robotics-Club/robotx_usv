#!/usr/bin/env python3
# YOLOv8 ROS2 Node using Ultralytics
# Created/modified by: Xavier Vicent
#
# Publishes:
#   - /yolov8/detections
#   - /yolov8/annotated_image
#   - /yolov8/annotated_image/compressed
#   - /yolov8/detections_black_buoy
#   - /yolov8/annotated_image_black_buoy
#   - /yolov8/annotated_image_black_buoy/compressed

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge

import cv2
import numpy as np

from ultralytics import YOLO

from yolov26_msgs.msg import YoloDetection

from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy


class YOLOv8Node(Node):
    def __init__(self):
        super().__init__("yolov8_detector")

        # ============================================================
        # Parameters
        # ============================================================

        self.weights = self.declare_parameter("weights", "yolov8n.pt").value

        self.input_topic = self.declare_parameter(
            "input_topic",
            "/zed/zed_node/rgb/color/rect/image",
        ).value

        # Output topics: all detections
        self.output_topic_image = self.declare_parameter(
            "output_topic_image",
            "/yolov8/annotated_image",
        ).value

        self.output_topic_image_compressed = self.declare_parameter(
            "output_topic_image_compressed",
            "/yolov8/annotated_image/compressed",
        ).value

        self.output_topic_detections = self.declare_parameter(
            "output_topic_detections",
            "/yolov8/detections",
        ).value

        # Output topics: black buoy only
        self.output_topic_black_det = self.declare_parameter(
            "output_topic_black_detections",
            "/yolov8/detections_black_buoy",
        ).value

        self.output_topic_black_img = self.declare_parameter(
            "output_topic_black_image",
            "/yolov8/annotated_image_black_buoy",
        ).value

        self.output_topic_black_img_compressed = self.declare_parameter(
            "output_topic_black_image_compressed",
            "/yolov8/annotated_image_black_buoy/compressed",
        ).value

        self.black_buoy_class_name = self.declare_parameter(
            "black_buoy_class_name",
            "black_buoy",
        ).value

        # Model inference parameters
        self.conf = float(self.declare_parameter("conf", 0.25).value)
        self.iou = float(self.declare_parameter("iou", 0.45).value)
        self.imgsz = int(self.declare_parameter("imgsz", 640).value)
        self.device = str(self.declare_parameter("device", "cpu").value)
        self.max_det = int(self.declare_parameter("max_det", 300).value)

        # JPEG compression quality
        self.jpeg_quality = int(self.declare_parameter("jpeg_quality", 70).value)
        self.jpeg_quality = max(1, min(100, self.jpeg_quality))

        # ============================================================
        # Load model
        # ============================================================

        self.get_logger().info("======================================")
        self.get_logger().info("Starting YOLOv8 detector")
        self.get_logger().info("======================================")
        self.get_logger().info(f"Weights:          {self.weights}")
        self.get_logger().info(f"Input topic:      {self.input_topic}")
        self.get_logger().info(f"All det:          {self.output_topic_detections}")
        self.get_logger().info(f"All image:        {self.output_topic_image}")
        self.get_logger().info(f"All compressed:   {self.output_topic_image_compressed}")
        self.get_logger().info(f"Black det:        {self.output_topic_black_det}")
        self.get_logger().info(f"Black image:      {self.output_topic_black_img}")
        self.get_logger().info(f"Black compressed: {self.output_topic_black_img_compressed}")
        self.get_logger().info(f"conf:             {self.conf}")
        self.get_logger().info(f"iou:              {self.iou}")
        self.get_logger().info(f"imgsz:            {self.imgsz}")
        self.get_logger().info(f"device:           {self.device}")
        self.get_logger().info(f"jpeg_quality:     {self.jpeg_quality}")
        self.get_logger().info("======================================")

        try:
            self.model = YOLO(self.weights)
        except Exception as e:
            self.get_logger().error(f"Could not load YOLOv8 model: {e}")
            raise

        try:
            self.class_names = self.model.names
            self.get_logger().info(f"Model classes: {self.class_names}")
        except Exception:
            self.class_names = {}
            self.get_logger().warn("Could not read model class names.")

        # Warmup
        try:
            dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            _ = self.model.predict(
                dummy,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                device=self.device,
                max_det=self.max_det,
                verbose=False,
            )
            self.get_logger().info("YOLOv8 warmup complete.")
        except Exception as e:
            self.get_logger().warn(f"Warmup failed: {e}")

        # ============================================================
        # ROS interfaces
        # ============================================================

        self.bridge = CvBridge()

        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
        )

        self.sub = self.create_subscription(
            Image,
            self.input_topic,
            self.image_cb,
            qos,
        )

        # Normal image publishers
        self.pub_img_all = self.create_publisher(
            Image,
            self.output_topic_image,
            10,
        )

        self.pub_img_black = self.create_publisher(
            Image,
            self.output_topic_black_img,
            10,
        )

        # Compressed image publishers
        self.pub_img_all_compressed = self.create_publisher(
            CompressedImage,
            self.output_topic_image_compressed,
            10,
        )

        self.pub_img_black_compressed = self.create_publisher(
            CompressedImage,
            self.output_topic_black_img_compressed,
            10,
        )

        # Detection publishers
        self.pub_det_all = self.create_publisher(
            YoloDetection,
            self.output_topic_detections,
            10,
        )

        self.pub_det_black = self.create_publisher(
            YoloDetection,
            self.output_topic_black_det,
            10,
        )

        self.get_logger().info("YOLOv8 detector ready.")

    # ============================================================
    # Message helpers
    # ============================================================

    @staticmethod
    def empty_det_msg():
        msg = YoloDetection()
        msg.class_name = []
        msg.confidence = []
        msg.x_min = []
        msg.y_min = []
        msg.x_max = []
        msg.y_max = []
        msg.x_center = []
        msg.y_center = []
        return msg

    @staticmethod
    def get_class_name(names, class_id: int) -> str:
        if isinstance(names, dict):
            return str(names.get(class_id, class_id))

        if isinstance(names, list):
            if 0 <= class_id < len(names):
                return str(names[class_id])
            return str(class_id)

        return str(class_id)

    def cv2_to_compressed_msg(self, image_bgr, header):
        msg = CompressedImage()
        msg.header = header
        msg.format = "jpeg"

        encode_params = [
            int(cv2.IMWRITE_JPEG_QUALITY),
            int(self.jpeg_quality),
        ]

        ok, encoded = cv2.imencode(".jpg", image_bgr, encode_params)

        if not ok:
            return None

        msg.data = encoded.tobytes()
        return msg

    # ============================================================
    # Main callback
    # ============================================================

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
                verbose=False,
            )
        except Exception as e:
            self.get_logger().error(f"YOLOv8 inference failed: {e}")
            return

        if not results:
            return

        result = results[0]

        det_all = self.empty_det_msg()
        det_black = self.empty_det_msg()

        try:
            annotated_all = result.plot()
        except Exception:
            annotated_all = frame.copy()

        annotated_black = frame.copy()

        boxes = getattr(result, "boxes", None)

        if boxes is not None and boxes.xyxy is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)

            names = result.names if hasattr(result, "names") else self.class_names
            target_black_name = str(self.black_buoy_class_name)

            for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, clss):
                cls_id = int(cls_id)
                cls_name = self.get_class_name(names, cls_id)

                x1 = float(x1)
                y1 = float(y1)
                x2 = float(x2)
                y2 = float(y2)

                xc = float((x1 + x2) / 2.0)
                yc = float((y1 + y2) / 2.0)

                # -----------------------------
                # All detections
                # -----------------------------
                det_all.class_name.append(cls_name)
                det_all.confidence.append(float(conf))
                det_all.x_min.append(x1)
                det_all.y_min.append(y1)
                det_all.x_max.append(x2)
                det_all.y_max.append(y2)
                det_all.x_center.append(xc)
                det_all.y_center.append(yc)

                # -----------------------------
                # Black buoy only
                # -----------------------------
                if cls_name == target_black_name:
                    det_black.class_name.append(cls_name)
                    det_black.confidence.append(float(conf))
                    det_black.x_min.append(x1)
                    det_black.y_min.append(y1)
                    det_black.x_max.append(x2)
                    det_black.y_max.append(y2)
                    det_black.x_center.append(xc)
                    det_black.y_center.append(yc)

                    cv2.rectangle(
                        annotated_black,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (0, 255, 255),
                        2,
                    )

                    cv2.putText(
                        annotated_black,
                        f"{cls_name} {conf:.2f}",
                        (int(x1), max(20, int(y1) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

        # ============================================================
        # Publish detections
        # ============================================================

        self.pub_det_all.publish(det_all)
        self.pub_det_black.publish(det_black)

        # ============================================================
        # Publish normal annotated images
        # ============================================================

        try:
            out_all = self.bridge.cv2_to_imgmsg(annotated_all, encoding="bgr8")
            out_all.header = msg.header
            self.pub_img_all.publish(out_all)
        except Exception as e:
            self.get_logger().warn(f"Could not publish all annotated image: {e}")

        try:
            out_black = self.bridge.cv2_to_imgmsg(annotated_black, encoding="bgr8")
            out_black.header = msg.header
            self.pub_img_black.publish(out_black)
        except Exception as e:
            self.get_logger().warn(f"Could not publish black buoy annotated image: {e}")

        # ============================================================
        # Publish compressed annotated images
        # ============================================================

        try:
            compressed_all = self.cv2_to_compressed_msg(annotated_all, msg.header)
            if compressed_all is not None:
                self.pub_img_all_compressed.publish(compressed_all)
        except Exception as e:
            self.get_logger().warn(f"Could not publish compressed all image: {e}")

        try:
            compressed_black = self.cv2_to_compressed_msg(annotated_black, msg.header)
            if compressed_black is not None:
                self.pub_img_black_compressed.publish(compressed_black)
        except Exception as e:
            self.get_logger().warn(f"Could not publish compressed black image: {e}")


def main(args=None):
    rclpy.init(args=args)

    node = YOLOv8Node()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()