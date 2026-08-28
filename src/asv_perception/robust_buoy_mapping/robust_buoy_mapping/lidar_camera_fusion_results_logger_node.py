#!/usr/bin/env python3
"""CSV logger for final LiDAR-camera fusion + dynamic-KF mapping results.

This node is meant to run alongside lidar_camera_fusion_ai_mapping.launch.py.
It does not change the mapper. It only subscribes to the fused detection topic,
the final semantic map topic, and optionally the vehicle pose topic, then writes
CSV files that are useful for paper/dissertation plots.
"""
from __future__ import annotations

import csv
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import Pose2D
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String


def finite(v: Any) -> bool:
    try:
        return math.isfinite(float(v))
    except Exception:
        return False


def stamp_to_float(payload: Dict[str, Any]) -> float:
    st = payload.get("stamp", {}) if isinstance(payload, dict) else {}
    try:
        return float(st.get("sec", 0)) + 1e-9 * float(st.get("nanosec", 0))
    except Exception:
        return 0.0


def get_float(d: Dict[str, Any], *keys: str, default: float = float("nan")) -> float:
    for k in keys:
        if k in d and finite(d[k]):
            return float(d[k])
    return float(default)


def get_int(d: Dict[str, Any], *keys: str, default: int = -1) -> int:
    for k in keys:
        if k in d:
            try:
                return int(d[k])
            except Exception:
                pass
    return int(default)


def get_bool_int(d: Dict[str, Any], *keys: str) -> int:
    for k in keys:
        if k in d:
            return 1 if bool(d[k]) else 0
    return 0


def color_from_dict(d: Dict[str, Any]) -> str:
    color = str(d.get("color", "")).strip().lower()
    if color in ("red", "green"):
        return color
    cls = str(d.get("class", d.get("class_name", d.get("label", "")))).lower()
    if "green" in cls:
        return "green"
    if "red" in cls:
        return "red"
    return "unknown"


class LidarCameraFusionResultsLoggerNode(Node):
    def __init__(self):
        super().__init__("lidar_camera_fusion_results_logger_node")

        self.declare_parameter("fused_topic", "/asv/perception/fused_buoy_detections")
        self.declare_parameter("semantic_buoys_topic", "/asv/map/semantic_buoys")
        self.declare_parameter("pose_topic", "/asv/vehicle_pose")
        self.declare_parameter("gt_csv", "/home/highlevel/roboboat_vehicle_data/ground_truth/gt_buoys_test_day.csv")
        self.declare_parameter("log_dir", "/home/highlevel/roboboat_vehicle_data/results/lidar_camera_fusion_mapping")
        self.declare_parameter("match_gate_m", 5.0)
        self.declare_parameter("log_fused_detections", True)
        self.declare_parameter("log_pose", True)
        self.declare_parameter("print_period_s", 2.0)

        self.fused_topic = str(self.get_parameter("fused_topic").value)
        self.semantic_topic = str(self.get_parameter("semantic_buoys_topic").value)
        self.pose_topic = str(self.get_parameter("pose_topic").value)
        self.gt_csv = str(self.get_parameter("gt_csv").value)
        self.log_dir = str(self.get_parameter("log_dir").value)
        self.match_gate_m = float(self.get_parameter("match_gate_m").value)
        self.log_fused_detections = bool(self.get_parameter("log_fused_detections").value)
        self.log_pose = bool(self.get_parameter("log_pose").value)
        self.print_period_s = float(self.get_parameter("print_period_s").value)

        os.makedirs(self.log_dir, exist_ok=True)

        self.gt_buoys = self._load_gt(self.gt_csv)
        self.latest_pose: Optional[Pose2D] = None
        self.last_print_t = 0.0

        self.fused_file = open(os.path.join(self.log_dir, "fused_detections.csv"), "w", newline="")
        self.map_file = open(os.path.join(self.log_dir, "map_tracks.csv"), "w", newline="")
        self.metrics_file = open(os.path.join(self.log_dir, "map_metrics.csv"), "w", newline="")
        self.pose_file = open(os.path.join(self.log_dir, "vehicle_pose.csv"), "w", newline="")
        self.gt_file = open(os.path.join(self.log_dir, "gt_buoys_used.csv"), "w", newline="")

        self.fused_w = csv.writer(self.fused_file)
        self.map_w = csv.writer(self.map_file)
        self.metrics_w = csv.writer(self.metrics_file)
        self.pose_w = csv.writer(self.pose_file)
        self.gt_w = csv.writer(self.gt_file)

        self._write_headers()

        qos = QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE)
        transient_qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)

        self.create_subscription(String, self.fused_topic, self.fused_cb, qos)
        self.create_subscription(String, self.semantic_topic, self.semantic_cb, transient_qos)
        if self.log_pose:
            self.create_subscription(Pose2D, self.pose_topic, self.pose_cb, qos)

        self.get_logger().info("LiDAR-camera fusion results CSV logger started")
        self.get_logger().info(f"fused_topic: {self.fused_topic}")
        self.get_logger().info(f"semantic_buoys_topic: {self.semantic_topic}")
        self.get_logger().info(f"pose_topic: {self.pose_topic}")
        self.get_logger().info(f"gt_csv: {self.gt_csv} | GT count={len(self.gt_buoys)}")
        self.get_logger().info(f"log_dir: {self.log_dir}")

    def _write_headers(self) -> None:
        self.fused_w.writerow([
            "ros_time_s", "msg_stamp_s", "det_index", "color",
            "x_map_m", "y_map_m", "confidence", "range_xy_m", "sigma_m", "p_reliable",
            "has_lidar", "lidar_source", "camera_confidence",
            "camera_x_body_m", "camera_y_body_m", "lidar_x_body_m", "lidar_y_body_m",
            "lidar_radius_m", "lidar_point_count", "cam_lidar_dist_m",
            "raw_fused_map_x_m", "raw_fused_map_y_m", "dx_correction_m", "dy_correction_m",
            "nearest_gt_id", "nearest_gt_color", "nearest_gt_error_m",
        ])
        self.map_w.writerow([
            "ros_time_s", "msg_stamp_s", "track_id", "color",
            "x_map_m", "y_map_m", "vx_mps", "vy_mps", "speed_mps",
            "confirmed", "hits", "misses", "position_sigma_m", "last_confidence", "last_range_m",
            "matched_gt_id", "matched_gt_color", "matched_error_m",
            "nearest_gt_id", "nearest_gt_color", "nearest_gt_error_m",
        ])
        self.metrics_w.writerow([
            "ros_time_s", "msg_stamp_s", "source", "gt_count", "live_count",
            "matched_count", "false_positive_count", "missed_gt_count",
            "mean_error_m", "median_error_m", "rmse_error_m", "max_error_m",
            "red_live_count", "green_live_count", "red_gt_count", "green_gt_count",
            "match_details_json",
        ])
        self.pose_w.writerow(["ros_time_s", "x_m", "y_m", "yaw_rad"])
        self.gt_w.writerow(["gt_id", "color", "x_map_m", "y_map_m"])
        for g in self.gt_buoys:
            self.gt_w.writerow([g["id"], g["color"], f'{g["x"]:.6f}', f'{g["y"]:.6f}'])
        self._flush_all()

    def _flush_all(self) -> None:
        for f in (self.fused_file, self.map_file, self.metrics_file, self.pose_file, self.gt_file):
            f.flush()

    def _load_gt(self, path: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if not path or not os.path.exists(path):
            self.get_logger().warn(f"GT CSV not found; error metrics will be empty: {path}")
            return out
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                color = str(row.get("color", "")).strip().lower()
                if color not in ("red", "green"):
                    continue
                x = row.get("north_m", row.get("x", None))
                y = row.get("east_m", row.get("y", None))
                if not (finite(x) and finite(y)):
                    continue
                try:
                    gid = int(row.get("id", len(out) + 1))
                except Exception:
                    gid = len(out) + 1
                out.append({"id": gid, "color": color, "x": float(x), "y": float(y)})
        return out

    def pose_cb(self, msg: Pose2D) -> None:
        self.latest_pose = msg
        if not self.log_pose:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        self.pose_w.writerow([f"{now:.6f}", f"{float(msg.x):.6f}", f"{float(msg.y):.6f}", f"{float(msg.theta):.9f}"])
        self.pose_file.flush()

    def _nearest_gt(self, x: float, y: float, color: Optional[str] = None) -> Tuple[int, str, float]:
        best_id = -1
        best_color = ""
        best_d = float("nan")
        best = float("inf")
        for g in self.gt_buoys:
            if color in ("red", "green") and g["color"] != color:
                continue
            d = math.hypot(x - float(g["x"]), y - float(g["y"]))
            if d < best:
                best = d
                best_id = int(g["id"])
                best_color = str(g["color"])
                best_d = float(d)
        return best_id, best_color, best_d

    def _extract_live(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        live: List[Dict[str, Any]] = []
        for b in payload.get("buoys", []):
            color = color_from_dict(b)
            if color not in ("red", "green"):
                continue
            x = get_float(b, "x", "north_m")
            y = get_float(b, "y", "east_m")
            if not (finite(x) and finite(y)):
                continue
            live.append({
                "id": get_int(b, "track_id", "id", default=-1),
                "track_id": get_int(b, "track_id", "id", default=-1),
                "color": color,
                "x": float(x),
                "y": float(y),
                "vx": get_float(b, "vx", default=0.0),
                "vy": get_float(b, "vy", default=0.0),
                "speed": get_float(b, "speed_mps", default=0.0),
                "confirmed": get_bool_int(b, "confirmed"),
                "hits": get_int(b, "hits", default=0),
                "misses": get_int(b, "misses", default=0),
                "position_sigma_m": get_float(b, "position_sigma_m", default=float("nan")),
                "last_confidence": get_float(b, "last_confidence", default=float("nan")),
                "last_range_m": get_float(b, "last_range_m", default=float("nan")),
            })
        return live

    def _match_live_to_gt(self, live: List[Dict[str, Any]]) -> Tuple[Dict[int, Dict[str, Any]], List[float], List[Dict[str, Any]]]:
        # Greedy global matching by same color and distance.
        pairs: List[Tuple[float, int, int]] = []
        for gi, g in enumerate(self.gt_buoys):
            for li, l in enumerate(live):
                if g["color"] != l["color"]:
                    continue
                d = math.hypot(float(g["x"]) - float(l["x"]), float(g["y"]) - float(l["y"]))
                if d <= self.match_gate_m:
                    pairs.append((float(d), gi, li))
        pairs.sort(key=lambda q: q[0])
        used_g = set()
        used_l = set()
        by_live_idx: Dict[int, Dict[str, Any]] = {}
        errors: List[float] = []
        details: List[Dict[str, Any]] = []
        for d, gi, li in pairs:
            if gi in used_g or li in used_l:
                continue
            used_g.add(gi)
            used_l.add(li)
            g = self.gt_buoys[gi]
            l = live[li]
            match = {"gt_id": int(g["id"]), "gt_color": g["color"], "error_m": float(d)}
            by_live_idx[li] = match
            errors.append(float(d))
            details.append({
                "gt_id": int(g["id"]),
                "gt_color": g["color"],
                "track_id": int(l["track_id"]),
                "track_color": l["color"],
                "error_m": float(d),
            })
        return by_live_idx, errors, details

    def fused_cb(self, msg: String) -> None:
        if not self.log_fused_detections:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        try:
            payload = json.loads(msg.data)
        except Exception as exc:
            self.get_logger().warn(f"Bad fused JSON: {exc}")
            return
        msg_stamp = stamp_to_float(payload)
        dets = payload.get("detections", [])
        for i, d in enumerate(dets):
            if not isinstance(d, dict):
                continue
            color = color_from_dict(d)
            x = get_float(d, "x", "north_m")
            y = get_float(d, "y", "east_m")
            ngt_id, ngt_color, ngt_err = self._nearest_gt(x, y, color if color in ("red", "green") else None)
            self.fused_w.writerow([
                f"{now:.6f}", f"{msg_stamp:.6f}", i, color,
                f"{x:.6f}" if finite(x) else "", f"{y:.6f}" if finite(y) else "",
                f'{get_float(d, "confidence"):.6f}' if finite(get_float(d, "confidence")) else "",
                f'{get_float(d, "range_xy", "range_m"):.6f}' if finite(get_float(d, "range_xy", "range_m")) else "",
                f'{get_float(d, "sigma_m", "sigma"):.6f}' if finite(get_float(d, "sigma_m", "sigma")) else "",
                f'{get_float(d, "p_reliable"):.6f}' if finite(get_float(d, "p_reliable")) else "",
                get_bool_int(d, "has_lidar"), str(d.get("lidar_source", "")),
                f'{get_float(d, "camera_confidence"):.6f}' if finite(get_float(d, "camera_confidence")) else "",
                f'{get_float(d, "camera_x_body_m"):.6f}' if finite(get_float(d, "camera_x_body_m")) else "",
                f'{get_float(d, "camera_y_body_m"):.6f}' if finite(get_float(d, "camera_y_body_m")) else "",
                f'{get_float(d, "lidar_x_body_m"):.6f}' if finite(get_float(d, "lidar_x_body_m")) else "",
                f'{get_float(d, "lidar_y_body_m"):.6f}' if finite(get_float(d, "lidar_y_body_m")) else "",
                f'{get_float(d, "lidar_radius_m"):.6f}' if finite(get_float(d, "lidar_radius_m")) else "",
                get_int(d, "lidar_point_count", default=0),
                f'{get_float(d, "cam_lidar_dist_m"):.6f}' if finite(get_float(d, "cam_lidar_dist_m")) else "",
                f'{get_float(d, "raw_fused_map_x_m"):.6f}' if finite(get_float(d, "raw_fused_map_x_m")) else "",
                f'{get_float(d, "raw_fused_map_y_m"):.6f}' if finite(get_float(d, "raw_fused_map_y_m")) else "",
                f'{get_float(d, "dx_correction_m"):.6f}' if finite(get_float(d, "dx_correction_m")) else "",
                f'{get_float(d, "dy_correction_m"):.6f}' if finite(get_float(d, "dy_correction_m")) else "",
                ngt_id, ngt_color, f"{ngt_err:.6f}" if finite(ngt_err) else "",
            ])
        self.fused_file.flush()

    def semantic_cb(self, msg: String) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        try:
            payload = json.loads(msg.data)
        except Exception as exc:
            self.get_logger().warn(f"Bad semantic JSON: {exc}")
            return

        msg_stamp = stamp_to_float(payload)
        source = str(payload.get("source", "unknown"))
        live = self._extract_live(payload)
        match_by_live_idx, errors, details = self._match_live_to_gt(live)

        for li, l in enumerate(live):
            match = match_by_live_idx.get(li, {})
            ngt_id, ngt_color, ngt_err = self._nearest_gt(l["x"], l["y"], l["color"])
            self.map_w.writerow([
                f"{now:.6f}", f"{msg_stamp:.6f}", int(l["track_id"]), l["color"],
                f'{l["x"]:.6f}', f'{l["y"]:.6f}',
                f'{l["vx"]:.6f}', f'{l["vy"]:.6f}', f'{l["speed"]:.6f}',
                int(l["confirmed"]), int(l["hits"]), int(l["misses"]),
                f'{l["position_sigma_m"]:.6f}' if finite(l["position_sigma_m"]) else "",
                f'{l["last_confidence"]:.6f}' if finite(l["last_confidence"]) else "",
                f'{l["last_range_m"]:.6f}' if finite(l["last_range_m"]) else "",
                match.get("gt_id", -1), match.get("gt_color", ""),
                f'{float(match["error_m"]):.6f}' if "error_m" in match else "",
                ngt_id, ngt_color, f"{ngt_err:.6f}" if finite(ngt_err) else "",
            ])

        if errors:
            arr = np.asarray(errors, dtype=float)
            mean = float(np.mean(arr))
            median = float(np.median(arr))
            rmse = float(np.sqrt(np.mean(arr * arr)))
            maxe = float(np.max(arr))
        else:
            mean = median = rmse = maxe = float("nan")

        red_live = sum(1 for l in live if l["color"] == "red")
        green_live = sum(1 for l in live if l["color"] == "green")
        red_gt = sum(1 for g in self.gt_buoys if g["color"] == "red")
        green_gt = sum(1 for g in self.gt_buoys if g["color"] == "green")
        matched = len(errors)
        fp = max(0, len(live) - matched)
        missed = max(0, len(self.gt_buoys) - matched)

        self.metrics_w.writerow([
            f"{now:.6f}", f"{msg_stamp:.6f}", source, len(self.gt_buoys), len(live),
            matched, fp, missed,
            f"{mean:.6f}" if finite(mean) else "",
            f"{median:.6f}" if finite(median) else "",
            f"{rmse:.6f}" if finite(rmse) else "",
            f"{maxe:.6f}" if finite(maxe) else "",
            red_live, green_live, red_gt, green_gt,
            json.dumps(details, separators=(",", ":")),
        ])
        self.map_file.flush()
        self.metrics_file.flush()

        if now - self.last_print_t >= self.print_period_s:
            self.last_print_t = now
            if errors:
                self.get_logger().info(
                    f"results live={len(live)} matched={matched}/{len(self.gt_buoys)} "
                    f"fp={fp} missed={missed} mean={mean:.3f} rmse={rmse:.3f}"
                )
            else:
                self.get_logger().warn(f"results live={len(live)} matched=0/{len(self.gt_buoys)} fp={fp} missed={missed}")

    def destroy_node(self):
        self._flush_all()
        for f in (self.fused_file, self.map_file, self.metrics_file, self.pose_file, self.gt_file):
            try:
                f.close()
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LidarCameraFusionResultsLoggerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()