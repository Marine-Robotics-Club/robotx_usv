#!/usr/bin/env python3

import json
import math
from typing import Any, Dict, List

import numpy as np
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from std_msgs.msg import String

from robust_buoy_mapping.dynamic_kf_ai_quality_logger_node import DynamicKFAIQualityLoggerNode


class FusedDynamicKFAIQualityLoggerNode(DynamicKFAIQualityLoggerNode):
    """Self-supervised mapping logger for LiDAR-camera fused buoy detections.

    The original DynamicKFAIQualityLoggerNode logs pair_candidates.csv,
    birth_candidates.csv, and track_snapshots.csv from raw camera detections.
    This subclass keeps the same logging and dynamic-KF logic, but feeds it
    detections from /asv/perception/fused_buoy_detections instead.
    """

    def __init__(self):
        super().__init__()

        self.declare_parameter("fusion_topic", "/asv/perception/fused_buoy_detections")
        self.fusion_topic = str(self.get_parameter("fusion_topic").value)

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(String, self.fusion_topic, self.fusion_cb, qos)

        self.get_logger().info("FUSED DYNAMIC KF + AI QUALITY LOGGER STARTED")
        self.get_logger().info(f"fusion_topic: {self.fusion_topic}")
        self.get_logger().info(f"ai_log_dir: {self.ai_log_dir}")

    def zed_cb(self, msg):
        """Ignore the parent raw-camera subscription.

        The parent class creates a camera subscription for compatibility with the
        original mapper. For fused mapping training we only want the fused output.
        """
        return

    def extract_fused_detections(self, payload: Dict[str, Any], now: float) -> List[Dict[str, Any]]:
        dets: List[Dict[str, Any]] = []

        for d in payload.get("detections", []):
            color = str(d.get("color", "")).lower().strip()
            if color not in ("red", "green"):
                continue

            try:
                x = float(d.get("x", d.get("north_m")))
                y = float(d.get("y", d.get("east_m")))
                conf = float(d.get("confidence", 50.0))
                rng = float(d.get("range_xy", d.get("range_m", 0.0)))
                sigma = float(d.get("sigma_m", d.get("sigma", 1.0)))
                p_reliable = float(d.get("p_reliable", 1.0))
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
                "p_reliable": p_reliable,
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

        if dets:
            self.associate_and_update(dets, now)
            self.merge_duplicates(now)
            self.delete_bad_tracks(now)

        # The parent timer also snapshots, but snapshotting here guarantees logs
        # are written even if detections arrive slower than the timer period.
        self.snapshot_tracks(now)

        confirmed = sum(1 for t in self.tracks if t.confirmed)
        tentative = len(self.tracks) - confirmed
        self.get_logger().info(
            f"fused_ai_quality_logger obs={len(dets)} tracks={len(self.tracks)} "
            f"confirmed={confirmed} tentative={tentative} "
            f"red={sum(1 for t in self.tracks if t.color == 'red')} "
            f"green={sum(1 for t in self.tracks if t.color == 'green')}"
        )

    def make_semantic_msg(self, stamp, tracks):
        msg = super().make_semantic_msg(stamp, tracks)
        try:
            payload = json.loads(msg.data)
            payload["source"] = "lidar_camera_fusion_to_dynamic_kf_ai_quality_logger"
            for b in payload.get("buoys", []):
                b["source"] = "lidar_camera_fusion_dynamic_kf_ai_quality_logger"
            msg.data = json.dumps(payload)
        except Exception:
            pass
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = FusedDynamicKFAIQualityLoggerNode()
    try:
        rclpy.spin(node)
    finally:
        for f in [node.pair_file, node.birth_file, node.snapshot_file]:
            if f is not None:
                f.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()