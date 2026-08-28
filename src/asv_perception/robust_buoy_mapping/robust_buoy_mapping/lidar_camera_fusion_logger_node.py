#!/usr/bin/env python3

"""Log camera + LiDAR buoy-detection fusion candidates.

This version expects the LiDAR buoy detector from your other package to publish
buoy detections already.  It does NOT cluster the raw point cloud again.

Supported LiDAR detection messages:
    fau_msgs/msg/ObjectPosition on /<wamv>/vision/output/buoy_objects
        fields: object_names[], x_object[], y_object[], z_object[], radii_object[]

    lidar_msgs/msg/BuoyDetected on /<wamv>/vision/output/buoy_detected
        fields: name[], x[], y[], z[]
"""

import csv
import os
from typing import Any, Dict, List, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import Pose2D
from yolov26_msgs.msg import ZedDetection

try:
    from fau_msgs.msg import ObjectPosition
except Exception:  # pragma: no cover - custom package may not be installed in sandbox
    ObjectPosition = None

try:
    from lidar_msgs.msg import BuoyDetected
except Exception:  # pragma: no cover - custom package may not be installed in sandbox
    BuoyDetected = None

from robust_buoy_mapping.dynamic_kf_buoy_mapper_node import finite
from robust_buoy_mapping.lidar_camera_fusion_common import (
    CAMERA_X_OFFSET_M,
    CSV_COLUMNS,
    LIDAR_X_OFFSET_M,
    as_bool_param,
    associate_lidar_detections_one_to_one,
    build_fusion_candidate,
    extract_camera_detections,
    extract_lidar_detections_from_buoy_detected,
    extract_lidar_detections_from_fau_objects,
)


class LidarCameraFusionLoggerNode(Node):
    def __init__(self):
        super().__init__("lidar_camera_fusion_logger_node")

        self.declare_parameter("wamv", "wamv1")
        self.declare_parameter("camera_topic", "/zed_custom_detections")
        self.declare_parameter("lidar_detection_source", "fau_objects")  # fau_objects, buoy_detected, both
        self.declare_parameter("lidar_objects_topic", "/wamv1/vision/output/buoy_objects")
        self.declare_parameter("lidar_buoy_detected_topic", "/wamv1/vision/output/buoy_detected")
        self.declare_parameter("pose_topic", "/asv/vehicle_pose")
        self.declare_parameter("log_csv", "/home/highlevel/roboboat_vehicle_data/logs/lidar_camera_fusion_candidates.csv")

        # Sensor extrinsics wrt GPS/body origin. Inputs are NWU/FLU (+y left); fusion converts to mapper body (+y right).
        self.declare_parameter("camera_x_offset_m", CAMERA_X_OFFSET_M)
        self.declare_parameter("camera_y_offset_m", 0.0)
        self.declare_parameter("camera_yaw_offset_rad", 0.061087)
        self.declare_parameter("camera_y_is_left", True)  # True for NWU/FLU detections

        self.declare_parameter("lidar_x_offset_m", LIDAR_X_OFFSET_M)
        self.declare_parameter("lidar_y_offset_m", 0.0)
        self.declare_parameter("lidar_z_offset_m", 0.0)
        self.declare_parameter("lidar_yaw_offset_rad", 0.0)
        self.declare_parameter("lidar_y_is_left", True)   # True for NWU/FLU detections

        self.declare_parameter("min_confidence", 45.0)
        self.declare_parameter("min_range_xy", 0.05)
        self.declare_parameter("max_range_xy", 25.0)

        self.declare_parameter("lidar_min_range_m", 0.3)
        self.declare_parameter("lidar_max_range_m", 35.0)
        self.declare_parameter("lidar_min_z_m", -5.0)
        self.declare_parameter("lidar_max_z_m", 3.0)

        self.declare_parameter("association_radius_m", 1.8)
        self.declare_parameter("lidar_blend_weight", 0.75)
        self.declare_parameter("max_lidar_age_s", 0.5)

        self.wamv = str(self.get_parameter("wamv").value)
        self.camera_topic = str(self.get_parameter("camera_topic").value)
        self.lidar_detection_source = str(self.get_parameter("lidar_detection_source").value).strip().lower()
        self.lidar_objects_topic = str(self.get_parameter("lidar_objects_topic").value)
        self.lidar_buoy_detected_topic = str(self.get_parameter("lidar_buoy_detected_topic").value)
        self.pose_topic = str(self.get_parameter("pose_topic").value)
        self.log_csv = str(self.get_parameter("log_csv").value)

        self.pose_x = 0.0
        self.pose_y = 0.0
        self.pose_yaw = 0.0
        self.have_pose = False

        self.latest_lidar_dets: List[Dict[str, Any]] = []
        self.latest_lidar_t: Optional[float] = None
        self.latest_lidar_source = "none"

        os.makedirs(os.path.dirname(self.log_csv), exist_ok=True)
        self.csv_file = open(self.log_csv, "w", newline="")
        self.writer = csv.DictWriter(self.csv_file, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        self.writer.writeheader()

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE)

        self.create_subscription(Pose2D, self.pose_topic, self.pose_cb, qos)
        self.create_subscription(ZedDetection, self.camera_topic, self.camera_cb, qos)
        self.create_lidar_detection_subscriptions(qos)

        self.get_logger().info("LIDAR-CAMERA FUSION LOGGER STARTED")
        self.get_logger().info(f"camera_topic:              {self.camera_topic}")
        self.get_logger().info(f"lidar_detection_source:    {self.lidar_detection_source}")
        self.get_logger().info(f"lidar_objects_topic:       {self.lidar_objects_topic}")
        self.get_logger().info(f"lidar_buoy_detected_topic: {self.lidar_buoy_detected_topic}")
        self.get_logger().info(f"pose_topic:                {self.pose_topic}")
        self.get_logger().info(f"log_csv:                   {self.log_csv}")
        self.get_logger().info("Using GPS/body origin: camera_x=26 in, lidar_x=20 in unless overridden.")

    def create_lidar_detection_subscriptions(self, qos: QoSProfile):
        source = self.lidar_detection_source
        if source not in ("fau_objects", "buoy_detected", "both"):
            self.get_logger().warn(f"Unknown lidar_detection_source='{source}'. Using fau_objects.")
            source = "fau_objects"

        if source in ("fau_objects", "both"):
            if ObjectPosition is None:
                self.get_logger().error("Cannot subscribe to fau_msgs/msg/ObjectPosition; fau_msgs is not available.")
            else:
                self.create_subscription(ObjectPosition, self.lidar_objects_topic, self.lidar_objects_cb, qos)
                self.get_logger().info(f"Subscribed to LiDAR fau_msgs objects: {self.lidar_objects_topic}")

        if source in ("buoy_detected", "both"):
            if BuoyDetected is None:
                self.get_logger().error("Cannot subscribe to lidar_msgs/msg/BuoyDetected; lidar_msgs is not available.")
            else:
                self.create_subscription(BuoyDetected, self.lidar_buoy_detected_topic, self.lidar_buoy_detected_cb, qos)
                self.get_logger().info(f"Subscribed to LiDAR buoy_detected: {self.lidar_buoy_detected_topic}")

    def pose_cb(self, msg: Pose2D):
        if not (finite(msg.x) and finite(msg.y) and finite(msg.theta)):
            return
        self.pose_x = float(msg.x)
        self.pose_y = float(msg.y)
        self.pose_yaw = float(msg.theta)
        self.have_pose = True

    def _lidar_extract_kwargs(self):
        return dict(
            lidar_x_offset_m=float(self.get_parameter("lidar_x_offset_m").value),
            lidar_y_offset_m=float(self.get_parameter("lidar_y_offset_m").value),
            lidar_z_offset_m=float(self.get_parameter("lidar_z_offset_m").value),
            lidar_yaw_offset_rad=float(self.get_parameter("lidar_yaw_offset_rad").value),
            lidar_y_is_left=as_bool_param(self.get_parameter("lidar_y_is_left").value),
            min_range_m=float(self.get_parameter("lidar_min_range_m").value),
            max_range_m=float(self.get_parameter("lidar_max_range_m").value),
            min_z_m=float(self.get_parameter("lidar_min_z_m").value),
            max_z_m=float(self.get_parameter("lidar_max_z_m").value),
        )

    def lidar_objects_cb(self, msg):
        self.latest_lidar_dets = extract_lidar_detections_from_fau_objects(msg, **self._lidar_extract_kwargs())
        self.latest_lidar_t = self.get_clock().now().nanoseconds * 1e-9
        self.latest_lidar_source = "fau_objects"
        self.get_logger().debug(f"LiDAR objects received: {len(self.latest_lidar_dets)}")

    def lidar_buoy_detected_cb(self, msg):
        self.latest_lidar_dets = extract_lidar_detections_from_buoy_detected(msg, **self._lidar_extract_kwargs())
        self.latest_lidar_t = self.get_clock().now().nanoseconds * 1e-9
        self.latest_lidar_source = "buoy_detected"
        self.get_logger().debug(f"LiDAR buoy_detected received: {len(self.latest_lidar_dets)}")

    def camera_cb(self, msg: ZedDetection):
        if not self.have_pose or self.latest_lidar_t is None:
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        lidar_age = now - float(self.latest_lidar_t)
        if lidar_age > float(self.get_parameter("max_lidar_age_s").value):
            self.get_logger().warn(f"Skipping camera frame because LiDAR detections are stale: {lidar_age:.2f}s")
            return

        camera_dets = extract_camera_detections(
            msg,
            min_confidence=float(self.get_parameter("min_confidence").value),
            min_range_xy=float(self.get_parameter("min_range_xy").value),
            max_range_xy=float(self.get_parameter("max_range_xy").value),
            camera_x_offset_m=float(self.get_parameter("camera_x_offset_m").value),
            camera_y_offset_m=float(self.get_parameter("camera_y_offset_m").value),
            camera_yaw_offset_rad=float(self.get_parameter("camera_yaw_offset_rad").value),
            camera_y_is_left=as_bool_param(self.get_parameter("camera_y_is_left").value),
        )
        if not camera_dets:
            return

        # Close-buoy safe association: each LiDAR detection can be assigned
        # to only one camera detection.  This avoids reusing or swapping the
        # same LiDAR buoy when two buoys are close together.
        matched_lidar_clusters = associate_lidar_detections_one_to_one(
            camera_dets,
            self.latest_lidar_dets,
            association_radius_m=float(self.get_parameter("association_radius_m").value),
        )

        rows = []
        for det, cluster in zip(camera_dets, matched_lidar_clusters):
            row = build_fusion_candidate(
                det,
                cluster,
                vehicle_x_m=self.pose_x,
                vehicle_y_m=self.pose_y,
                vehicle_yaw_rad=self.pose_yaw,
                now_s=now,
                camera_msg_age_s=0.0,
                lidar_msg_age_s=lidar_age,
                lidar_blend_weight=float(self.get_parameter("lidar_blend_weight").value),
            )
            row["lidar_source"] = self.latest_lidar_source
            rows.append(row)
            self.writer.writerow(row)

        self.csv_file.flush()
        n_lidar = sum(1 for r in rows if int(r.get("lidar_has_cluster", 0)) == 1)
        self.get_logger().info(
            f"fusion_logger camera_dets={len(camera_dets)} lidar_dets={len(self.latest_lidar_dets)} "
            f"rows={len(rows)} matched_lidar={n_lidar} source={self.latest_lidar_source}"
        )

    def destroy_node(self):
        try:
            self.csv_file.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LidarCameraFusionLoggerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()