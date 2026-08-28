#!/usr/bin/env python3

"""Run XGBoost LiDAR-camera buoy fusion and publish fused detections.

This node subscribes to your existing LiDAR buoy detector outputs instead of
clustering the raw point cloud again.
"""

import json
import math
import os
from typing import Any, Dict, List, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None

from geometry_msgs.msg import Pose2D
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from yolov26_msgs.msg import ZedDetection

try:
    from fau_msgs.msg import ObjectPosition
except Exception:  # pragma: no cover
    ObjectPosition = None

try:
    from lidar_msgs.msg import BuoyDetected
except Exception:  # pragma: no cover
    BuoyDetected = None

from robust_buoy_mapping.dynamic_kf_buoy_mapper_node import finite
from robust_buoy_mapping.lidar_camera_fusion_common import (
    CAMERA_X_OFFSET_M,
    LIDAR_X_OFFSET_M,
    as_bool_param,
    associate_lidar_detections_one_to_one,
    build_fusion_candidate,
    extract_camera_detections,
    extract_lidar_detections_from_buoy_detected,
    extract_lidar_detections_from_fau_objects,
    feature_vector_from_candidate,
    heuristic_fusion_sigma,
)


class LidarCameraFusionInferenceNode(Node):
    def __init__(self):
        super().__init__("lidar_camera_fusion_inference_node")

        self.declare_parameter("wamv", "wamv1")
        self.declare_parameter("camera_topic", "/zed_custom_detections")
        self.declare_parameter("lidar_detection_source", "fau_objects")  # fau_objects, buoy_detected, both
        self.declare_parameter("lidar_objects_topic", "/wamv1/vision/output/buoy_objects")
        self.declare_parameter("lidar_buoy_detected_topic", "/wamv1/vision/output/buoy_detected")
        self.declare_parameter("pose_topic", "/asv/vehicle_pose")
        self.declare_parameter("fused_topic", "/asv/perception/fused_buoy_detections")
        self.declare_parameter("fused_marker_topic", "/asv/perception/fused_buoy_markers")

        self.declare_parameter("model_dir", "/home/highlevel/roboboat_usv/src/asv_perception/robust_buoy_mapping/models/lidar_camera_fusion")
        self.declare_parameter("reliable_model_path", "")
        self.declare_parameter("dx_model_path", "")
        self.declare_parameter("dy_model_path", "")
        self.declare_parameter("sigma_model_path", "")
        self.declare_parameter("p_reliable_threshold", 0.50)
        self.declare_parameter("publish_without_model", True)
        self.declare_parameter("allow_camera_only", False)

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
        self.declare_parameter("max_correction_m", 2.0)
        self.declare_parameter("sigma_min_m", 0.25)
        self.declare_parameter("sigma_max_m", 4.0)

        self.wamv = str(self.get_parameter("wamv").value)
        self.camera_topic = str(self.get_parameter("camera_topic").value)
        self.lidar_detection_source = str(self.get_parameter("lidar_detection_source").value).strip().lower()
        self.lidar_objects_topic = str(self.get_parameter("lidar_objects_topic").value)
        self.lidar_buoy_detected_topic = str(self.get_parameter("lidar_buoy_detected_topic").value)
        self.pose_topic = str(self.get_parameter("pose_topic").value)
        self.fused_topic = str(self.get_parameter("fused_topic").value)
        self.fused_marker_topic = str(self.get_parameter("fused_marker_topic").value)

        self.pose_x = 0.0
        self.pose_y = 0.0
        self.pose_yaw = 0.0
        self.have_pose = False
        self.latest_lidar_dets: List[Dict[str, Any]] = []
        self.latest_lidar_t: Optional[float] = None
        self.latest_lidar_source = "none"
        self.latest_detections = []

        self.reliable_model = None
        self.dx_model = None
        self.dy_model = None
        self.sigma_model = None
        self.load_models()

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE)
        transient_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)

        self.create_subscription(Pose2D, self.pose_topic, self.pose_cb, qos)
        self.create_subscription(ZedDetection, self.camera_topic, self.camera_cb, qos)
        self.create_lidar_detection_subscriptions(qos)

        self.fused_pub = self.create_publisher(String, self.fused_topic, transient_qos)
        self.marker_pub = self.create_publisher(MarkerArray, self.fused_marker_topic, 10)

        self.get_logger().info("LIDAR-CAMERA FUSION INFERENCE STARTED")
        self.get_logger().info(f"camera_topic:              {self.camera_topic}")
        self.get_logger().info(f"lidar_detection_source:    {self.lidar_detection_source}")
        self.get_logger().info(f"lidar_objects_topic:       {self.lidar_objects_topic}")
        self.get_logger().info(f"lidar_buoy_detected_topic: {self.lidar_buoy_detected_topic}")
        self.get_logger().info(f"fused_topic:               {self.fused_topic}")
        self.get_logger().info("FUSION FRAME: camera/LiDAR input detections are treated as NWU/FLU (+y left).")
        self.get_logger().info("FUSION FRAME: features/output are converted to mapper body frame (+x forward, +y right).")
        self.get_logger().info("Output goes to fused_dynamic_kf_ai_mapper_node, then into the mapping AI/KF stage.")

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

    def model_path(self, explicit_param: str, filename: str) -> str:
        explicit = str(self.get_parameter(explicit_param).value).strip()
        if explicit:
            return explicit
        return os.path.join(str(self.get_parameter("model_dir").value), filename)

    def load_one(self, path: str, name: str):
        if joblib is None:
            self.get_logger().warn("joblib is not available. Fusion AI models disabled.")
            return None
        if not os.path.exists(path):
            self.get_logger().warn(f"{name} model not found: {path}")
            return None
        self.get_logger().info(f"loaded {name}: {path}")
        return joblib.load(path)

    def load_models(self):
        self.reliable_model = self.load_one(self.model_path("reliable_model_path", "fusion_reliable_xgb.joblib"), "fusion reliability")
        self.dx_model = self.load_one(self.model_path("dx_model_path", "fusion_dx_xgb.joblib"), "fusion dx correction")
        self.dy_model = self.load_one(self.model_path("dy_model_path", "fusion_dy_xgb.joblib"), "fusion dy correction")
        self.sigma_model = self.load_one(self.model_path("sigma_model_path", "fusion_sigma_xgb.joblib"), "fusion sigma")

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

    def predict_probability(self, X, has_lidar: bool) -> float:
        if self.reliable_model is None:
            if as_bool_param(self.get_parameter("publish_without_model").value):
                return 0.65 if has_lidar else 0.40
            return 0.0
        try:
            if hasattr(self.reliable_model, "predict_proba"):
                return float(self.reliable_model.predict_proba(X)[0, 1])
            pred = float(self.reliable_model.predict(X)[0])
            return pred if 0.0 <= pred <= 1.0 else 1.0 / (1.0 + math.exp(-pred))
        except Exception as exc:
            self.get_logger().warn(f"fusion reliability inference failed: {exc}")
            return 0.0

    def predict_regression(self, model, X, fallback: float, limit_abs: Optional[float] = None) -> float:
        if model is None:
            return float(fallback)
        try:
            y = float(model.predict(X)[0])
            if limit_abs is not None:
                y = float(np.clip(y, -abs(limit_abs), abs(limit_abs)))
            return y
        except Exception as exc:
            self.get_logger().warn(f"fusion regression inference failed: {exc}")
            return float(fallback)

    def camera_cb(self, msg: ZedDetection):
        if not self.have_pose or self.latest_lidar_t is None:
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        lidar_age = now - float(self.latest_lidar_t)
        if lidar_age > float(self.get_parameter("max_lidar_age_s").value):
            self.get_logger().warn(f"Skipping fusion because LiDAR detections are stale: {lidar_age:.2f}s")
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

        output_dets = []
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

            has_lidar = int(row.get("lidar_has_cluster", 0)) == 1
            if (not has_lidar) and (not as_bool_param(self.get_parameter("allow_camera_only").value)):
                continue

            X = feature_vector_from_candidate(row)
            p = self.predict_probability(X, has_lidar=has_lidar)
            if p < float(self.get_parameter("p_reliable_threshold").value):
                continue

            max_corr = float(self.get_parameter("max_correction_m").value)
            dx_corr = self.predict_regression(self.dx_model, X, 0.0, limit_abs=max_corr)
            dy_corr = self.predict_regression(self.dy_model, X, 0.0, limit_abs=max_corr)
            sigma = self.predict_regression(self.sigma_model, X, heuristic_fusion_sigma(row, p), limit_abs=None)
            sigma = float(np.clip(sigma, float(self.get_parameter("sigma_min_m").value), float(self.get_parameter("sigma_max_m").value)))

            fused_x = float(row["raw_fused_map_x_m"]) + dx_corr
            fused_y = float(row["raw_fused_map_y_m"]) + dy_corr
            rng = float(row["raw_fused_range_m"])

            conf = float(np.clip(float(row["camera_confidence"]) * (0.55 + 0.45 * p), 1.0, 100.0))
            output_dets.append({
                "color": str(row["color"]),
                "class": f"{row['color']}_buoy",
                "class_name": f"{row['color']}_buoy",
                "x": fused_x,
                "y": fused_y,
                "north_m": fused_x,
                "east_m": fused_y,
                "confidence": conf,
                "range_xy": rng,
                "sigma_m": sigma,
                "p_reliable": float(p),
                "has_lidar": bool(has_lidar),
                "lidar_source": self.latest_lidar_source,
                "camera_confidence": float(row["camera_confidence"]),
                "camera_x_body_m": float(row["camera_x_body_m"]),
                "camera_y_body_m": float(row["camera_y_body_m"]),
                "lidar_x_body_m": float(row["lidar_x_body_m"]),
                "lidar_y_body_m": float(row["lidar_y_body_m"]),
                "lidar_radius_m": float(row.get("lidar_radius_m", 0.0)),
                "lidar_point_count": int(row["lidar_point_count"]),
                "cam_lidar_dist_m": float(row["cam_lidar_dist_m"]),
                "raw_fused_map_x_m": float(row["raw_fused_map_x_m"]),
                "raw_fused_map_y_m": float(row["raw_fused_map_y_m"]),
                "dx_correction_m": float(dx_corr),
                "dy_correction_m": float(dy_corr),
                "source": "xgboost_lidar_camera_detection_fusion",
            })

        stamp = self.get_clock().now().to_msg()
        payload = {
            "stamp": {"sec": int(stamp.sec), "nanosec": int(stamp.nanosec)},
            "frame_id": "map",
            "source": "lidar_camera_fusion_inference",
            "camera_x_offset_m": float(self.get_parameter("camera_x_offset_m").value),
            "lidar_x_offset_m": float(self.get_parameter("lidar_x_offset_m").value),
            "lidar_detection_source": self.latest_lidar_source,
            "camera_detection_count": len(camera_dets),
            "lidar_detection_count": len(self.latest_lidar_dets),
            "detection_count": len(output_dets),
            "detections": output_dets,
        }
        out = String()
        out.data = json.dumps(payload)
        self.fused_pub.publish(out)
        self.latest_detections = output_dets
        self.marker_pub.publish(self.make_markers(stamp, output_dets))
        self.get_logger().info(
            f"fusion_inference camera={len(camera_dets)} lidar={len(self.latest_lidar_dets)} "
            f"fused={len(output_dets)} source={self.latest_lidar_source}"
        )

    def make_markers(self, stamp, dets):
        arr = MarkerArray()
        delete = Marker()
        delete.header.stamp = stamp
        delete.header.frame_id = "map"
        delete.action = Marker.DELETEALL
        arr.markers.append(delete)

        for i, d in enumerate(dets):
            m = Marker()
            m.header.stamp = stamp
            m.header.frame_id = "map"
            m.ns = "lidar_camera_fused_buoys"
            m.id = i + 1
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(d["x"])
            m.pose.position.y = float(d["y"])
            m.pose.position.z = 0.65
            m.pose.orientation.w = 1.0
            m.scale.x = 0.35
            m.scale.y = 0.35
            m.scale.z = 0.35
            if d["color"] == "red":
                m.color.r = 1.0
                m.color.g = 0.0
                m.color.b = 0.0
            else:
                m.color.r = 0.0
                m.color.g = 1.0
                m.color.b = 0.0
            m.color.a = 0.70
            arr.markers.append(m)

            txt = Marker()
            txt.header.stamp = stamp
            txt.header.frame_id = "map"
            txt.ns = "lidar_camera_fused_labels"
            txt.id = 1000 + i
            txt.type = Marker.TEXT_VIEW_FACING
            txt.action = Marker.ADD
            txt.pose.position.x = float(d["x"])
            txt.pose.position.y = float(d["y"])
            txt.pose.position.z = 1.25
            txt.pose.orientation.w = 1.0
            txt.scale.z = 0.45
            txt.color.r = 1.0
            txt.color.g = 1.0
            txt.color.b = 1.0
            txt.color.a = 1.0
            txt.text = f"fusion {d['color']}\np={d['p_reliable']:.2f} σ={d['sigma_m']:.2f}\nLiDAR={d['lidar_source']}"
            arr.markers.append(txt)
        return arr


def main(args=None):
    rclpy.init(args=args)
    node = LidarCameraFusionInferenceNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()