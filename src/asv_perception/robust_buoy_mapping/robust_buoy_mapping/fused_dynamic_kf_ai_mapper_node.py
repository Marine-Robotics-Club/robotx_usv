#!/usr/bin/env python3

import json
import math
from typing import Any, Dict, List

import numpy as np
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from std_msgs.msg import String

from robust_buoy_mapping.dynamic_kf_ai_assisted_mapper_node import DynamicKFAIAssistedMapperNode


class FusedDynamicKFAIMapperNode(DynamicKFAIAssistedMapperNode):
    """Second-stage mapper that consumes LiDAR-camera fusion output.

    This node keeps the existing dynamic-KF + AI association/birth/sigma logic, but
    its measurements are fused map-frame detections instead of raw ZED detections.
    """

    def __init__(self):
        super().__init__()
        self.declare_parameter("fusion_topic", "/asv/perception/fused_buoy_detections")
        self.fusion_topic = str(self.get_parameter("fusion_topic").value)
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE)
        self.create_subscription(String, self.fusion_topic, self.fusion_cb, qos)
        self.get_logger().info("FUSED DYNAMIC KF + AI MAPPER STARTED")
        self.get_logger().info(f"fusion_topic: {self.fusion_topic}")

    def zed_cb(self, msg):
        # The parent subscription can still exist if launched with the original camera_topic.
        # Ignore raw ZED here so only fused LiDAR-camera detections feed the mapper.
        return

    def extract_fused_detections(self, payload: Dict[str, Any], now: float) -> List[Dict[str, Any]]:
        dets = []
        for d in payload.get("detections", []):
            color = str(d.get("color", "")).lower().strip()
            if color not in ("red", "green"):
                continue

            try:
                x = float(d.get("x", d.get("north_m")))
                y = float(d.get("y", d.get("east_m")))
                conf = float(d.get("confidence", 50.0))
                rng = float(d.get("range_xy", 0.0))
                sigma = float(d.get("sigma_m", d.get("sigma", 1.0)))
            except Exception:
                continue

            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(sigma)):
                continue
            if sigma <= 0.0:
                continue

            sigma = float(np.clip(sigma, self.measurement_sigma_min, self.measurement_sigma_max))
            R = np.diag([sigma ** 2, sigma ** 2]).astype(float)
            dets.append({
                "t": now,
                "color": color,
                "x": x,
                "y": y,
                "confidence": conf,
                "range_xy": rng,
                "sigma": sigma,
                "R": R,
                "source": str(d.get("source", "lidar_camera_fusion")),
                "p_reliable": float(d.get("p_reliable", 1.0)),
                "has_lidar": bool(d.get("has_lidar", True)),
            })

        dets.sort(key=lambda q: (-q["confidence"], q["range_xy"]))
        return dets

    def fusion_cb(self, msg: String):
        now = self.get_clock().now().nanoseconds * 1e-9
        try:
            payload = json.loads(msg.data)
        except Exception as exc:
            self.get_logger().warn(f"Bad fused detection JSON: {exc}")
            return

        self.predict_all(now)
        dets = self.extract_fused_detections(payload, now)
        self.raw_latest = dets[-80:]
        if not dets:
            return

        self.associate_and_update(dets, now)
        self.merge_duplicates(now)
        self.delete_bad_tracks(now)

        confirmed = sum(1 for t in self.tracks if t.confirmed)
        tentative = len(self.tracks) - confirmed
        self.get_logger().info(
            f"fused_ai_mapper obs={len(dets)} tracks={len(self.tracks)} confirmed={confirmed} tentative={tentative} "
            f"red={sum(1 for t in self.tracks if t.color == 'red')} green={sum(1 for t in self.tracks if t.color == 'green')}"
        )

    def make_semantic_msg(self, stamp, tracks):
        msg = super().make_semantic_msg(stamp, tracks)
        try:
            payload = json.loads(msg.data)
            payload["source"] = "lidar_camera_fusion_to_dynamic_kf_ai_mapper"
            for b in payload.get("buoys", []):
                b["source"] = "lidar_camera_fusion_dynamic_kf_ai_mapper"
            msg.data = json.dumps(payload)
        except Exception:
            pass
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = FusedDynamicKFAIMapperNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()