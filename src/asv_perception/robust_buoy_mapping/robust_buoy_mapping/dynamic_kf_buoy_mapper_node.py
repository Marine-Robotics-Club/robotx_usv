#!/usr/bin/env python3

import json
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from geometry_msgs.msg import Pose2D
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from yolov26_msgs.msg import ZedDetection


def finite(v: Any) -> bool:
    try:
        return math.isfinite(float(v))
    except Exception:
        return False


def arr_len(msg, name: str) -> int:
    if not hasattr(msg, name):
        return 0
    try:
        return len(getattr(msg, name))
    except Exception:
        return 0


def arr_val(msg, name: str, i: int, default=None):
    if not hasattr(msg, name):
        return default
    a = getattr(msg, name)
    try:
        if i < len(a):
            return a[i]
    except Exception:
        pass
    return default


def color_from_class(name: str) -> str:
    s = str(name).lower().strip().replace("greeb", "green")
    if "green" in s:
        return "green"
    if "red" in s:
        return "red"
    return "unknown"


def confidence_percent(c: float) -> float:
    c = float(c)
    if c <= 1.5:
        return 100.0 * c
    return c


def clamp_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


class BuoyKFTrack:
    def __init__(
        self,
        track_id: int,
        color: str,
        x: float,
        y: float,
        t: float,
        initial_position_sigma: float,
        initial_velocity_sigma: float,
    ):
        self.id = int(track_id)
        self.color = str(color)

        self.x = np.zeros((4, 1), dtype=float)
        self.x[0, 0] = float(x)
        self.x[1, 0] = float(y)
        self.x[2, 0] = 0.0
        self.x[3, 0] = 0.0

        self.P = np.diag([
            initial_position_sigma ** 2,
            initial_position_sigma ** 2,
            initial_velocity_sigma ** 2,
            initial_velocity_sigma ** 2,
        ]).astype(float)

        self.last_t = float(t)
        self.last_update_t = float(t)

        self.hits = 1
        self.misses = 0
        self.confirmed = False

        self.last_confidence = 0.0
        self.last_range = 0.0

    def predict(
        self,
        t: float,
        process_accel_sigma: float,
        velocity_decay_time_s: float,
        max_speed_mps: float,
    ):
        t = float(t)
        dt = max(1e-3, min(2.0, t - self.last_t))
        self.last_t = t

        if velocity_decay_time_s > 1e-3:
            decay = math.exp(-dt / velocity_decay_time_s)
        else:
            decay = 1.0

        F = np.array([
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, decay, 0.0],
            [0.0, 0.0, 0.0, decay],
        ], dtype=float)

        q = float(process_accel_sigma) ** 2
        Q = q * np.array([
            [dt ** 4 / 4.0, 0.0, dt ** 3 / 2.0, 0.0],
            [0.0, dt ** 4 / 4.0, 0.0, dt ** 3 / 2.0],
            [dt ** 3 / 2.0, 0.0, dt ** 2, 0.0],
            [0.0, dt ** 3 / 2.0, 0.0, dt ** 2],
        ], dtype=float)

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

        self.clamp_speed(max_speed_mps)

    def clamp_speed(self, max_speed_mps: float):
        vx = float(self.x[2, 0])
        vy = float(self.x[3, 0])
        spd = math.hypot(vx, vy)

        if spd > max_speed_mps > 1e-6:
            scale = max_speed_mps / spd
            self.x[2, 0] *= scale
            self.x[3, 0] *= scale

    def innovation_stats(self, z: np.ndarray, R: np.ndarray):
        H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ], dtype=float)

        y = z.reshape((2, 1)) - H @ self.x
        S = H @ self.P @ H.T + R

        try:
            Sinv = np.linalg.inv(S)
            d2 = float(y.T @ Sinv @ y)
        except Exception:
            d2 = float("inf")

        return y, S, H, d2

    def update(
        self,
        z_x: float,
        z_y: float,
        R: np.ndarray,
        t: float,
        confidence: float,
        range_xy: float,
        max_speed_mps: float,
    ) -> float:
        z = np.array([float(z_x), float(z_y)], dtype=float)
        y, S, H, d2 = self.innovation_stats(z, R)

        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y

        I = np.eye(4)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T

        self.last_update_t = float(t)
        self.hits += 1
        self.misses = 0
        self.last_confidence = float(confidence)
        self.last_range = float(range_xy)

        self.clamp_speed(max_speed_mps)

        return d2

    def pos(self) -> Tuple[float, float]:
        return float(self.x[0, 0]), float(self.x[1, 0])

    def vel(self) -> Tuple[float, float]:
        return float(self.x[2, 0]), float(self.x[3, 0])

    def pos_sigma(self) -> float:
        return float(math.sqrt(max(float(self.P[0, 0]), float(self.P[1, 1]), 1e-9)))

    def age_since_update(self, t: float) -> float:
        return float(t - self.last_update_t)

    def score(self, t: float) -> float:
        recent_bonus = max(0.0, 10.0 - self.age_since_update(t))
        conf_bonus = 0.02 * self.last_confidence
        cov_penalty = self.pos_sigma()
        confirmed_bonus = 10.0 if self.confirmed else 0.0
        return confirmed_bonus + self.hits + recent_bonus + conf_bonus - cov_penalty


class DynamicKFBuoyMapper(Node):
    def __init__(self):
        super().__init__("dynamic_kf_buoy_mapper_node")

        self.declare_parameter("asv", "asv")

        self.declare_parameter("camera_topic", "/zed_custom_detections")
        self.declare_parameter("pose_topic", "/asv/vehicle_pose")

        self.declare_parameter("semantic_buoys_topic", "map/semantic_buoys")
        self.declare_parameter("map_topic", "map/local_occupancy_2")
        self.declare_parameter("semantic_marker_topic", "/asv/camera_mapping/semantic_map_markers")
        self.declare_parameter("raw_marker_topic", "/asv/camera_mapping/raw_detection_markers")

        self.declare_parameter("frame_id", "map")
        self.declare_parameter("occupancy_frame_id", "map")

        # Calibrated camera transform
        self.declare_parameter("camera_x_offset_m", 0.6604)
        self.declare_parameter("camera_y_offset_m", 0.0)
        self.declare_parameter("camera_yaw_offset_rad", 0.061087)

        # ZED detection filtering
        self.declare_parameter("min_confidence", 45.0)
        self.declare_parameter("min_range_xy", 0.05)
        self.declare_parameter("max_range_xy", 22.0)

        # KF parameters
        self.declare_parameter("initial_position_sigma", 1.5)
        self.declare_parameter("initial_velocity_sigma", 0.25)
        self.declare_parameter("process_accel_sigma", 0.08)
        self.declare_parameter("velocity_decay_time_s", 8.0)
        self.declare_parameter("max_speed_mps", 0.35)

        # Measurement noise
        self.declare_parameter("measurement_sigma_base", 0.35)
        self.declare_parameter("measurement_sigma_per_meter", 0.08)
        self.declare_parameter("measurement_sigma_min", 0.35)
        self.declare_parameter("measurement_sigma_max", 2.5)

        # Association / track management
        self.declare_parameter("mahalanobis_gate_confirmed", 9.21)
        self.declare_parameter("mahalanobis_gate_tentative", 16.0)
        self.declare_parameter("confirm_hits", 3)
        self.declare_parameter("delete_tentative_after_missing_s", 3.0)
        self.declare_parameter("delete_confirmed_after_missing_s", 60.0)
        self.declare_parameter("birth_min_separation_m", 1.2)
        self.declare_parameter("merge_distance_m", 0.9)
        self.declare_parameter("merge_mahalanobis_gate", 4.0)

        # Publishing
        self.declare_parameter("publish_tentative_tracks", False)
        self.declare_parameter("publish_period_s", 0.5)
        self.declare_parameter("marker_diameter_m", 0.70)
        self.declare_parameter("raw_marker_diameter_m", 0.35)

        # Occupancy grid
        self.declare_parameter("occupancy_resolution_m", 0.25)
        self.declare_parameter("occupancy_width_m", 80.0)
        self.declare_parameter("occupancy_height_m", 80.0)
        self.declare_parameter("occupancy_obstacle_radius_m", 0.60)
        self.declare_parameter("occupancy_inflation_radius_m", 0.30)
        self.declare_parameter("occupancy_covariance_scale", 2.0)
        self.declare_parameter("occupancy_use_fixed_radius", True)
        self.declare_parameter("occupancy_fixed_radius_m", 0.40)

        self.asv = str(self.get_parameter("asv").value).strip("/")

        self.camera_topic = str(self.get_parameter("camera_topic").value)
        self.pose_topic = str(self.get_parameter("pose_topic").value)

        self.semantic_topic = self.resolve_topic(str(self.get_parameter("semantic_buoys_topic").value))
        self.map_topic = self.resolve_topic(str(self.get_parameter("map_topic").value))
        self.semantic_marker_topic = str(self.get_parameter("semantic_marker_topic").value)
        self.raw_marker_topic = str(self.get_parameter("raw_marker_topic").value)

        self.frame_id = str(self.get_parameter("frame_id").value)
        self.occupancy_frame_id = str(self.get_parameter("occupancy_frame_id").value)

        self.camera_x_offset_m = float(self.get_parameter("camera_x_offset_m").value)
        self.camera_y_offset_m = float(self.get_parameter("camera_y_offset_m").value)
        self.camera_yaw_offset_rad = float(self.get_parameter("camera_yaw_offset_rad").value)

        self.min_confidence = float(self.get_parameter("min_confidence").value)
        self.min_range_xy = float(self.get_parameter("min_range_xy").value)
        self.max_range_xy = float(self.get_parameter("max_range_xy").value)

        self.initial_position_sigma = float(self.get_parameter("initial_position_sigma").value)
        self.initial_velocity_sigma = float(self.get_parameter("initial_velocity_sigma").value)
        self.process_accel_sigma = float(self.get_parameter("process_accel_sigma").value)
        self.velocity_decay_time_s = float(self.get_parameter("velocity_decay_time_s").value)
        self.max_speed_mps = float(self.get_parameter("max_speed_mps").value)

        self.measurement_sigma_base = float(self.get_parameter("measurement_sigma_base").value)
        self.measurement_sigma_per_meter = float(self.get_parameter("measurement_sigma_per_meter").value)
        self.measurement_sigma_min = float(self.get_parameter("measurement_sigma_min").value)
        self.measurement_sigma_max = float(self.get_parameter("measurement_sigma_max").value)

        self.mahalanobis_gate_confirmed = float(self.get_parameter("mahalanobis_gate_confirmed").value)
        self.mahalanobis_gate_tentative = float(self.get_parameter("mahalanobis_gate_tentative").value)
        self.confirm_hits = int(self.get_parameter("confirm_hits").value)
        self.delete_tentative_after_missing_s = float(self.get_parameter("delete_tentative_after_missing_s").value)
        self.delete_confirmed_after_missing_s = float(self.get_parameter("delete_confirmed_after_missing_s").value)
        self.birth_min_separation_m = float(self.get_parameter("birth_min_separation_m").value)
        self.merge_distance_m = float(self.get_parameter("merge_distance_m").value)
        self.merge_mahalanobis_gate = float(self.get_parameter("merge_mahalanobis_gate").value)

        self.publish_tentative_tracks = bool(self.get_parameter("publish_tentative_tracks").value)
        self.publish_period_s = float(self.get_parameter("publish_period_s").value)
        self.marker_diameter_m = float(self.get_parameter("marker_diameter_m").value)
        self.raw_marker_diameter_m = float(self.get_parameter("raw_marker_diameter_m").value)

        self.occ_res = float(self.get_parameter("occupancy_resolution_m").value)
        self.occ_width_m = float(self.get_parameter("occupancy_width_m").value)
        self.occ_height_m = float(self.get_parameter("occupancy_height_m").value)
        self.occ_radius_m = float(self.get_parameter("occupancy_obstacle_radius_m").value)
        self.occ_inflation_m = float(self.get_parameter("occupancy_inflation_radius_m").value)
        self.occ_cov_scale = float(self.get_parameter("occupancy_covariance_scale").value)
        self.occ_use_fixed_radius = bool(self.get_parameter("occupancy_use_fixed_radius").value)
        self.occ_fixed_radius_m = float(self.get_parameter("occupancy_fixed_radius_m").value)

        self.pose_x = 0.0
        self.pose_y = 0.0
        self.pose_yaw = 0.0
        self.have_pose = False

        self.tracks: List[BuoyKFTrack] = []
        self.next_id = 1
        self.raw_latest: List[Dict[str, Any]] = []

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE)
        transient_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)

        self.create_subscription(Pose2D, self.pose_topic, self.pose_cb, qos)
        self.create_subscription(ZedDetection, self.camera_topic, self.zed_cb, qos)

        self.semantic_pub = self.create_publisher(String, self.semantic_topic, transient_qos)
        self.grid_pub = self.create_publisher(OccupancyGrid, self.map_topic, transient_qos)
        self.semantic_marker_pub = self.create_publisher(MarkerArray, self.semantic_marker_topic, 10)
        self.raw_marker_pub = self.create_publisher(MarkerArray, self.raw_marker_topic, 10)

        self.timer = self.create_timer(self.publish_period_s, self.timer_cb)

        self.get_logger().info("DYNAMIC KF BUOY MAPPER STARTED")
        self.get_logger().info("No GT prior. Unknown number of buoys. Tracks are born/confirmed/deleted automatically.")
        self.get_logger().info(f"camera_topic: {self.camera_topic}")
        self.get_logger().info(f"pose_topic: {self.pose_topic}")
        self.get_logger().info(f"semantic_topic: {self.semantic_topic}")
        self.get_logger().info(f"map_topic: {self.map_topic}")

    def resolve_topic(self, topic: str) -> str:
        topic = topic.strip()
        if topic.startswith("/"):
            return topic
        return f"/{self.asv}/{topic}".replace("//", "/")

    def pose_cb(self, msg: Pose2D):
        if not (finite(msg.x) and finite(msg.y) and finite(msg.theta)):
            return
        self.pose_x = float(msg.x)
        self.pose_y = float(msg.y)
        self.pose_yaw = float(msg.theta)
        self.have_pose = True

    def measurement_sigma(self, range_xy: float, confidence: float) -> float:
        sigma = self.measurement_sigma_base + self.measurement_sigma_per_meter * float(range_xy)

        conf = max(1.0, min(100.0, float(confidence)))
        # Lower confidence means larger R.
        sigma *= math.sqrt(75.0 / conf)

        sigma = max(self.measurement_sigma_min, min(self.measurement_sigma_max, sigma))
        return sigma

    def zed_to_body(self, x_loc: float, y_loc: float):
        # ZED assumed: x forward, y left. Vehicle body: x forward, y right.
        x_cam = float(x_loc)
        y_right = -float(y_loc)

        c = math.cos(self.camera_yaw_offset_rad)
        s = math.sin(self.camera_yaw_offset_rad)

        x_rot = c * x_cam - s * y_right
        y_rot = s * x_cam + c * y_right

        x_body = x_rot + self.camera_x_offset_m
        y_body = y_rot + self.camera_y_offset_m

        return x_body, y_body

    def body_to_map(self, x_body: float, y_body: float):
        c = math.cos(self.pose_yaw)
        s = math.sin(self.pose_yaw)

        north = self.pose_x + c * x_body - s * y_body
        east = self.pose_y + s * x_body + c * y_body

        return north, east

    def predict_all(self, now: float):
        for tr in self.tracks:
            tr.predict(
                t=now,
                process_accel_sigma=self.process_accel_sigma,
                velocity_decay_time_s=self.velocity_decay_time_s,
                max_speed_mps=self.max_speed_mps,
            )

    def extract_detections(self, msg: ZedDetection, now: float) -> List[Dict[str, Any]]:
        dets = []

        n = min(
            arr_len(msg, "class_name"),
            arr_len(msg, "confidence"),
            arr_len(msg, "x_loc"),
            arr_len(msg, "y_loc"),
            arr_len(msg, "z_loc"),
        )

        for i in range(n):
            cls = str(arr_val(msg, "class_name", i, "unknown"))
            color = color_from_class(cls)

            if color not in ("red", "green"):
                continue

            conf = confidence_percent(float(arr_val(msg, "confidence", i, 0.0)))
            if conf < self.min_confidence:
                continue

            x_loc = arr_val(msg, "x_loc", i, float("nan"))
            y_loc = arr_val(msg, "y_loc", i, float("nan"))
            z_loc = arr_val(msg, "z_loc", i, float("nan"))

            if not (finite(x_loc) and finite(y_loc) and finite(z_loc)):
                continue

            xb, yb = self.zed_to_body(float(x_loc), float(y_loc))
            rng = math.hypot(xb, yb)

            if rng < self.min_range_xy or rng > self.max_range_xy:
                continue

            mx, my = self.body_to_map(xb, yb)

            if not (finite(mx) and finite(my)):
                continue

            sigma = self.measurement_sigma(rng, conf)
            R = np.diag([sigma ** 2, sigma ** 2]).astype(float)

            dets.append({
                "t": now,
                "color": color,
                "x": float(mx),
                "y": float(my),
                "confidence": float(conf),
                "range_xy": float(rng),
                "sigma": float(sigma),
                "R": R,
            })

        # More confident and closer detections first.
        dets.sort(key=lambda d: (-d["confidence"], d["range_xy"]))
        return dets

    def associate_and_update(self, dets: List[Dict[str, Any]], now: float):
        used_tracks = set()
        used_dets = set()

        candidates = []

        for di, d in enumerate(dets):
            z = np.array([d["x"], d["y"]], dtype=float)

            for ti, tr in enumerate(self.tracks):
                if ti in used_tracks:
                    continue
                if tr.color != d["color"]:
                    continue

                gate = self.mahalanobis_gate_confirmed if tr.confirmed else self.mahalanobis_gate_tentative

                _, _, _, d2 = tr.innovation_stats(z, d["R"])

                if d2 <= gate:
                    candidates.append((float(d2), ti, di))

        candidates.sort(key=lambda x: x[0])

        assignments = []
        for d2, ti, di in candidates:
            if ti in used_tracks or di in used_dets:
                continue
            used_tracks.add(ti)
            used_dets.add(di)
            assignments.append((ti, di, d2))

        for ti, di, _d2 in assignments:
            tr = self.tracks[ti]
            d = dets[di]

            tr.update(
                z_x=d["x"],
                z_y=d["y"],
                R=d["R"],
                t=now,
                confidence=d["confidence"],
                range_xy=d["range_xy"],
                max_speed_mps=self.max_speed_mps,
            )

            if tr.hits >= self.confirm_hits:
                tr.confirmed = True

        # Unassigned existing tracks get a miss.
        for ti, tr in enumerate(self.tracks):
            if ti not in used_tracks:
                tr.misses += 1

        # Birth new tentative tracks from unassigned detections.
        births = 0
        for di, d in enumerate(dets):
            if di in used_dets:
                continue

            if self.too_close_to_existing_same_color(d):
                continue

            tr = BuoyKFTrack(
                track_id=self.next_id,
                color=d["color"],
                x=d["x"],
                y=d["y"],
                t=now,
                initial_position_sigma=self.initial_position_sigma,
                initial_velocity_sigma=self.initial_velocity_sigma,
            )
            tr.last_confidence = d["confidence"]
            tr.last_range = d["range_xy"]

            self.next_id += 1
            self.tracks.append(tr)
            births += 1

        if births > 0:
            self.get_logger().info(f"new tentative KF buoy tracks: {births}")

    def too_close_to_existing_same_color(self, det: Dict[str, Any]) -> bool:
        for tr in self.tracks:
            if tr.color != det["color"]:
                continue
            tx, ty = tr.pos()
            d = math.hypot(det["x"] - tx, det["y"] - ty)

            # If a detection is close to an existing track but failed Mahalanobis,
            # it is probably noisy or duplicate. Do not create a new track.
            if d < self.birth_min_separation_m:
                return True

        return False

    def merge_duplicates(self, now: float):
        changed = True

        while changed:
            changed = False
            n = len(self.tracks)

            for i in range(n):
                if changed:
                    break

                for j in range(i + 1, n):
                    a = self.tracks[i]
                    b = self.tracks[j]

                    if a.color != b.color:
                        continue

                    ax, ay = a.pos()
                    bx, by = b.pos()
                    dist = math.hypot(ax - bx, ay - by)

                    if dist > self.merge_distance_m:
                        continue

                    # Merge only if they are close spatially.
                    # Winner = higher score, loser removed.
                    if a.score(now) >= b.score(now):
                        winner = a
                        loser = b
                        loser_index = j
                    else:
                        winner = b
                        loser = a
                        loser_index = i

                    self.merge_into(winner, loser, now)
                    del self.tracks[loser_index]

                    changed = True
                    break

    def merge_into(self, winner: BuoyKFTrack, loser: BuoyKFTrack, now: float):
        # Conservative merge. Keep winner identity, average position using information weights.
        sw = max(1.0, 1.0 / max(winner.pos_sigma() ** 2, 1e-6))
        sl = max(1.0, 1.0 / max(loser.pos_sigma() ** 2, 1e-6))

        winner.x = (sw * winner.x + sl * loser.x) / (sw + sl)
        winner.P = winner.P * 0.75

        winner.hits += loser.hits
        winner.confirmed = winner.confirmed or loser.confirmed
        winner.last_update_t = max(winner.last_update_t, loser.last_update_t)
        winner.last_t = now

    def delete_bad_tracks(self, now: float):
        keep = []

        for tr in self.tracks:
            missing_s = tr.age_since_update(now)

            if not tr.confirmed and missing_s > self.delete_tentative_after_missing_s:
                continue

            if tr.confirmed and missing_s > self.delete_confirmed_after_missing_s:
                continue

            # If covariance explodes, remove tentative tracks but keep confirmed longer.
            if not tr.confirmed and tr.pos_sigma() > 6.0:
                continue

            if tr.confirmed and tr.pos_sigma() > 15.0:
                continue

            keep.append(tr)

        removed = len(self.tracks) - len(keep)
        self.tracks = keep

        if removed > 0:
            self.get_logger().info(f"removed stale/noisy KF tracks: {removed}")

    def zed_cb(self, msg: ZedDetection):
        if not self.have_pose:
            return

        now = self.get_clock().now().nanoseconds * 1e-9

        self.predict_all(now)

        dets = self.extract_detections(msg, now)
        self.raw_latest = dets[-80:]

        if not dets:
            return

        self.associate_and_update(dets, now)
        self.merge_duplicates(now)
        self.delete_bad_tracks(now)

        confirmed = sum(1 for t in self.tracks if t.confirmed)
        tentative = len(self.tracks) - confirmed

        self.get_logger().info(
            f"kf_mapper obs={len(dets)} tracks={len(self.tracks)} confirmed={confirmed} tentative={tentative} "
            f"red={sum(1 for t in self.tracks if t.color == 'red')} green={sum(1 for t in self.tracks if t.color == 'green')}"
        )

    def timer_cb(self):
        now = self.get_clock().now().nanoseconds * 1e-9

        self.predict_all(now)
        self.merge_duplicates(now)
        self.delete_bad_tracks(now)

        tracks = self.publishable_tracks()

        stamp = self.get_clock().now().to_msg()

        self.semantic_pub.publish(self.make_semantic_msg(stamp, tracks))
        self.grid_pub.publish(self.make_grid(stamp, tracks))
        self.semantic_marker_pub.publish(self.make_track_markers(stamp, tracks))
        self.raw_marker_pub.publish(self.make_raw_markers(stamp))

    def publishable_tracks(self) -> List[BuoyKFTrack]:
        if self.publish_tentative_tracks:
            tracks = list(self.tracks)
        else:
            tracks = [tr for tr in self.tracks if tr.confirmed]

        tracks.sort(key=lambda tr: (tr.color, tr.id))
        return tracks

    def make_semantic_msg(self, stamp, tracks: List[BuoyKFTrack]) -> String:
        buoys = []

        for tr in tracks:
            px, py = tr.pos()
            vx, vy = tr.vel()
            cls = f"{tr.color}_buoy"

            buoys.append({
                "id": int(tr.id),
                "track_id": int(tr.id),
                "class": cls,
                "class_name": cls,
                "label": cls,
                "color": tr.color,

                "x": float(px),
                "y": float(py),
                "north_m": float(px),
                "east_m": float(py),

                "vx": float(vx),
                "vy": float(vy),
                "speed_mps": float(math.hypot(vx, vy)),

                "confirmed": bool(tr.confirmed),
                "hits": int(tr.hits),
                "misses": int(tr.misses),
                "position_sigma_m": float(tr.pos_sigma()),
                "last_confidence": float(tr.last_confidence),
                "last_range_m": float(tr.last_range),
                "source": "dynamic_kf_buoy_mapper",
            })

        payload = {
            "stamp": {"sec": int(stamp.sec), "nanosec": int(stamp.nanosec)},
            "frame_id": self.frame_id,
            "source": "dynamic_kf_buoy_mapper_26in_camera_offset",
            "buoy_count": len(buoys),
            "buoys": buoys,
        }

        msg = String()
        msg.data = json.dumps(payload)
        return msg

    def make_grid(self, stamp, tracks: List[BuoyKFTrack]) -> OccupancyGrid:
        width = int(round(self.occ_width_m / self.occ_res))
        height = int(round(self.occ_height_m / self.occ_res))

        if self.have_pose:
            cx = self.pose_x
            cy = self.pose_y
        elif tracks:
            cx = sum(tr.pos()[0] for tr in tracks) / len(tracks)
            cy = sum(tr.pos()[1] for tr in tracks) / len(tracks)
        else:
            cx = 0.0
            cy = 0.0

        ox = cx - 0.5 * self.occ_width_m
        oy = cy - 0.5 * self.occ_height_m

        data = [0] * (width * height)

        for tr in tracks:
            bx, by = tr.pos()

            if self.occ_use_fixed_radius:
                total_radius = self.occ_fixed_radius_m
            else:
                cov_radius = self.occ_cov_scale * tr.pos_sigma()
                total_radius = self.occ_radius_m + self.occ_inflation_m + cov_radius
                total_radius = min(total_radius, 5.0)

            rad_cells = max(1, int(math.ceil(total_radius / self.occ_res)))
            r2 = total_radius * total_radius

            gx0 = int((bx - ox) / self.occ_res)
            gy0 = int((by - oy) / self.occ_res)

            if gx0 < 0 or gx0 >= width or gy0 < 0 or gy0 >= height:
                continue

            for gy in range(max(0, gy0 - rad_cells), min(height - 1, gy0 + rad_cells) + 1):
                wy = oy + (gy + 0.5) * self.occ_res
                for gx in range(max(0, gx0 - rad_cells), min(width - 1, gx0 + rad_cells) + 1):
                    wx = ox + (gx + 0.5) * self.occ_res
                    if (wx - bx) ** 2 + (wy - by) ** 2 <= r2:
                        data[gy * width + gx] = 100

        msg = OccupancyGrid()
        msg.header.stamp = stamp
        msg.header.frame_id = self.occupancy_frame_id
        msg.info.resolution = float(self.occ_res)
        msg.info.width = int(width)
        msg.info.height = int(height)
        msg.info.origin.position.x = float(ox)
        msg.info.origin.position.y = float(oy)
        msg.info.origin.orientation.w = 1.0
        msg.data = data
        return msg

    def make_track_markers(self, stamp, tracks: List[BuoyKFTrack]) -> MarkerArray:
        arr = MarkerArray()

        delete = Marker()
        delete.header.stamp = stamp
        delete.header.frame_id = self.frame_id
        delete.action = Marker.DELETEALL
        arr.markers.append(delete)

        for tr in tracks:
            px, py = tr.pos()
            vx, vy = tr.vel()

            m = Marker()
            m.header.stamp = stamp
            m.header.frame_id = self.frame_id
            m.ns = "dynamic_kf_buoys"
            m.id = int(tr.id)
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(px)
            m.pose.position.y = float(py)
            m.pose.position.z = 0.35
            m.pose.orientation.w = 1.0
            m.scale.x = self.marker_diameter_m
            m.scale.y = self.marker_diameter_m
            m.scale.z = self.marker_diameter_m

            if tr.color == "red":
                m.color.r = 1.0
                m.color.g = 0.0
                m.color.b = 0.0
            else:
                m.color.r = 0.0
                m.color.g = 1.0
                m.color.b = 0.0

            m.color.a = 0.95 if tr.confirmed else 0.45
            arr.markers.append(m)

            txt = Marker()
            txt.header.stamp = stamp
            txt.header.frame_id = self.frame_id
            txt.ns = "dynamic_kf_labels"
            txt.id = int(10000 + tr.id)
            txt.type = Marker.TEXT_VIEW_FACING
            txt.action = Marker.ADD
            txt.pose.position.x = float(px)
            txt.pose.position.y = float(py)
            txt.pose.position.z = 1.15
            txt.pose.orientation.w = 1.0
            txt.scale.z = 0.55
            txt.color.r = 1.0
            txt.color.g = 1.0
            txt.color.b = 1.0
            txt.color.a = 1.0
            txt.text = f"{tr.id}:{tr.color}\\nh={tr.hits} σ={tr.pos_sigma():.2f}\\nv={math.hypot(vx,vy):.2f}"
            arr.markers.append(txt)

        return arr

    def make_raw_markers(self, stamp) -> MarkerArray:
        arr = MarkerArray()

        delete = Marker()
        delete.header.stamp = stamp
        delete.header.frame_id = self.frame_id
        delete.action = Marker.DELETEALL
        arr.markers.append(delete)

        for i, d in enumerate(self.raw_latest):
            m = Marker()
            m.header.stamp = stamp
            m.header.frame_id = self.frame_id
            m.ns = "raw_zed_dynamic_kf"
            m.id = int(i + 1)
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(d["x"])
            m.pose.position.y = float(d["y"])
            m.pose.position.z = 0.8
            m.pose.orientation.w = 1.0
            m.scale.x = self.raw_marker_diameter_m
            m.scale.y = self.raw_marker_diameter_m
            m.scale.z = self.raw_marker_diameter_m

            if d["color"] == "red":
                m.color.r = 1.0
                m.color.g = 0.0
                m.color.b = 0.0
            else:
                m.color.r = 0.0
                m.color.g = 1.0
                m.color.b = 0.0

            m.color.a = 0.35
            arr.markers.append(m)

        return arr


def main(args=None):
    rclpy.init(args=args)
    node = DynamicKFBuoyMapper()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
