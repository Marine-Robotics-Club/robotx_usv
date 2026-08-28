#!/usr/bin/env python3
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose2D
from fau_msgs.msg import ObjectPositionArray
from visualization_msgs.msg import Marker, MarkerArray


TRACK_CLASSES = {"green_buoy", "red_buoy", "yellow_buoy", "black_buoy"}


class KalmanTracker2D:
    """Static position-only KF (random walk): state [x,y]."""
    def __init__(self, dt: float = 0.1, r_meas: float = 1.2, q_proc: float = 0.002, p0: float = 25.0):
        self.dt = float(dt)
        self.x = np.zeros((2, 1), dtype=float)

        self.F = np.eye(2, dtype=float)
        self.H = np.eye(2, dtype=float)

        self.R = np.eye(2, dtype=float) * float(r_meas)
        self.P = np.eye(2, dtype=float) * float(p0)
        self.Q = np.eye(2, dtype=float) * float(q_proc)

        self.initialized = False

    def init_from_measurement(self, x: float, y: float):
        self.x[0, 0] = float(x)
        self.x[1, 0] = float(y)
        self.initialized = True

    def predict(self):
        if not self.initialized:
            return
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: np.ndarray):
        if not self.initialized:
            self.init_from_measurement(float(z[0, 0]), float(z[1, 0]))
            return

        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(2, dtype=float)
        self.P = (I - K @ self.H) @ self.P

    def get_position(self) -> Tuple[float, float]:
        return float(self.x[0, 0]), float(self.x[1, 0])


@dataclass
class Track:
    cls: str
    kf: KalmanTracker2D
    z: float

    hits: int = 1
    consecutive_hits: int = 1
    confirmed: bool = False

    updated_this_cycle: bool = False


class BuoyMapperKF(Node):
    """
    Buoy tracking + mapping with multi-camera cone gating (center/left/right).
    Publishes tracked obstacles and RViz markers for green/red/yellow/black buoys.

    ✅ Integrated fusion "other boat" marker (NO TIMEOUT):
      - Subscribes /vision/output/fusion (ObjectPositionArray)
      - Looks for yellow_triangle and/or black_triangle (configurable)
      - Maintains ONE "other boat" target
      - Same-boat gating: if new detection is within fusion_same_boat_radius_m of current target,
        it's treated as the SAME boat (updates position). If outside:
          * fusion_allow_switch_target=False -> ignore (stay locked)
          * fusion_allow_switch_target=True  -> switch to new target
    """

    def __init__(self):
        super().__init__("buoy_mapper_kf")

        # -------------------- Params --------------------
        self.declare_parameter("asv_name", "asv")

        # Detection topics (buoys)
        self.declare_parameter("detections_topic", "/vision/output/position_estimates")
        self.declare_parameter("detections_topic_left", "/vision/output/position_estimates/left")
        self.declare_parameter("detections_topic_right", "/vision/output/position_estimates/right")

        self.declare_parameter("pose_topic", "")  # default: /{asv_name}/vehicle_pose

        self.declare_parameter("tracked_topic", "/vision/tracked/obstacles")
        self.declare_parameter("marker_topic", "/vision/tracked/markers")
        self.declare_parameter("frame_id", "map")

        # Association / spawning (general)
        self.declare_parameter("assoc_radius_m", 2.0)
        self.declare_parameter("assoc_radius_confirmed_m", 3.5)
        self.declare_parameter("add_radius_m", 5.0)

        # Local cone gate (new buoys only)
        self.declare_parameter("cone_half_angle_deg", 35.0)
        self.declare_parameter("require_forward_x", False)

        # Camera centers (deg) w.r.t boat forward
        self.declare_parameter("cam_center_deg", 0.0)
        self.declare_parameter("cam_left_center_deg", 108.0)
        self.declare_parameter("cam_right_center_deg", -108.0)

        # Confirmation (general)
        self.declare_parameter("min_consecutive_hits", 4)

        # Outlier rejection (general)
        self.declare_parameter("max_update_jump_confirmed_m", 1.0)
        self.declare_parameter("max_update_jump_unconfirmed_m", 1.5)

        # KF params
        self.declare_parameter("dt", 0.1)
        self.declare_parameter("kf_r_meas", 1.2)
        self.declare_parameter("kf_q_proc", 0.002)
        self.declare_parameter("kf_p0", 25.0)

        # RViz markers (buoys + ego boat)
        self.declare_parameter("buoy_marker_diameter_m", 0.5)
        self.declare_parameter("boat_marker_length_m", 1.2)
        self.declare_parameter("boat_marker_width_m", 0.4)

        # Same-color spawn suppression (general)
        self.declare_parameter("same_color_keep_radius_m", 2.0)

        # -------------------- Yellow-specific params --------------------
        self.declare_parameter("yellow_enable_assoc", True)
        self.declare_parameter("yellow_assoc_radius_m", 6.0)
        self.declare_parameter("yellow_max_update_jump_m", 4.0)
        self.declare_parameter("yellow_add_radius_m", 80.0)
        self.declare_parameter("yellow_min_consecutive_hits", 2)
        self.declare_parameter("yellow_same_color_keep_radius_m", 4.0)

        # -------------------- Fusion boat marker params (NO TIMEOUT) --------------------
        self.declare_parameter("fusion_topic", "/vision/output/fusion")

        self.declare_parameter("fusion_target_marker_length_m", 1.6)
        self.declare_parameter("fusion_target_marker_width_m", 0.7)
        self.declare_parameter("fusion_target_marker_height_m", 0.7)

        self.declare_parameter("fusion_use_yellow_triangle", True)
        self.declare_parameter("fusion_use_black_triangle", True)

        # ✅ "same boat" radius gating for fusion target
        self.declare_parameter("fusion_same_boat_radius_m", 5.0)
        self.declare_parameter("fusion_allow_switch_target", False)

        # -------------------- Read params --------------------
        self.asv_name = str(self.get_parameter("asv_name").value)

        self.detections_topic = str(self.get_parameter("detections_topic").value).strip()
        self.detections_topic_left = str(self.get_parameter("detections_topic_left").value).strip()
        self.detections_topic_right = str(self.get_parameter("detections_topic_right").value).strip()

        self.pose_topic = str(self.get_parameter("pose_topic").value).strip()
        if not self.pose_topic:
            self.pose_topic = f"/{self.asv_name}/vehicle_pose"

        self.tracked_topic = str(self.get_parameter("tracked_topic").value).strip()
        self.marker_topic = str(self.get_parameter("marker_topic").value).strip()
        self.frame_id = str(self.get_parameter("frame_id").value).strip()

        self.assoc_r = float(self.get_parameter("assoc_radius_m").value)
        self.assoc_r2 = self.assoc_r * self.assoc_r

        self.assoc_r_conf = float(self.get_parameter("assoc_radius_confirmed_m").value)
        self.assoc_r_conf2 = self.assoc_r_conf * self.assoc_r_conf

        self.add_r = float(self.get_parameter("add_radius_m").value)
        self.add_r2 = self.add_r * self.add_r

        self.cone_half_angle_deg = float(self.get_parameter("cone_half_angle_deg").value)
        self.require_forward_x = bool(self.get_parameter("require_forward_x").value)

        self.cam_center_deg = float(self.get_parameter("cam_center_deg").value)
        self.cam_left_center_deg = float(self.get_parameter("cam_left_center_deg").value)
        self.cam_right_center_deg = float(self.get_parameter("cam_right_center_deg").value)

        self.min_consecutive_hits = int(self.get_parameter("min_consecutive_hits").value)

        self.max_jump_conf = float(self.get_parameter("max_update_jump_confirmed_m").value)
        self.max_jump_conf2 = self.max_jump_conf * self.max_jump_conf
        self.max_jump_unconf = float(self.get_parameter("max_update_jump_unconfirmed_m").value)
        self.max_jump_unconf2 = self.max_jump_unconf * self.max_jump_unconf

        self.dt = float(self.get_parameter("dt").value)
        self.kf_r_meas = float(self.get_parameter("kf_r_meas").value)
        self.kf_q_proc = float(self.get_parameter("kf_q_proc").value)
        self.kf_p0 = float(self.get_parameter("kf_p0").value)

        self.buoy_d = float(self.get_parameter("buoy_marker_diameter_m").value)
        self.boat_len = float(self.get_parameter("boat_marker_length_m").value)
        self.boat_w = float(self.get_parameter("boat_marker_width_m").value)

        self.same_color_keep_radius_m = float(self.get_parameter("same_color_keep_radius_m").value)
        self.same_color_keep_radius2 = self.same_color_keep_radius_m * self.same_color_keep_radius_m

        # Yellow-specific reads
        self.yellow_enable_assoc = bool(self.get_parameter("yellow_enable_assoc").value)
        self.yellow_assoc_r = float(self.get_parameter("yellow_assoc_radius_m").value)
        self.yellow_assoc_r2 = self.yellow_assoc_r * self.yellow_assoc_r
        self.yellow_max_jump = float(self.get_parameter("yellow_max_update_jump_m").value)
        self.yellow_max_jump2 = self.yellow_max_jump * self.yellow_max_jump
        self.yellow_add_r = float(self.get_parameter("yellow_add_radius_m").value)
        self.yellow_add_r2 = self.yellow_add_r * self.yellow_add_r
        self.yellow_min_hits = int(self.get_parameter("yellow_min_consecutive_hits").value)
        self.yellow_keep_r = float(self.get_parameter("yellow_same_color_keep_radius_m").value)
        self.yellow_keep_r2 = self.yellow_keep_r * self.yellow_keep_r

        # Fusion reads
        self.fusion_topic = str(self.get_parameter("fusion_topic").value).strip()
        self.fusion_tgt_len = float(self.get_parameter("fusion_target_marker_length_m").value)
        self.fusion_tgt_w = float(self.get_parameter("fusion_target_marker_width_m").value)
        self.fusion_tgt_h = float(self.get_parameter("fusion_target_marker_height_m").value)
        self.fusion_use_yellow = bool(self.get_parameter("fusion_use_yellow_triangle").value)
        self.fusion_use_black = bool(self.get_parameter("fusion_use_black_triangle").value)

        self.fusion_same_boat_r = float(self.get_parameter("fusion_same_boat_radius_m").value)
        self.fusion_same_boat_r2 = self.fusion_same_boat_r * self.fusion_same_boat_r
        self.fusion_allow_switch = bool(self.get_parameter("fusion_allow_switch_target").value)

        # -------------------- Boat pose (GLOBAL NED) --------------------
        self.x_usv_NED = 0.0
        self.y_usv_NED = 0.0
        self.psi_usv_NED = 0.0
        self.have_pose = False

        # -------------------- Tracks --------------------
        self.tracks: Dict[str, List[Track]] = {c: [] for c in TRACK_CLASSES}

        # -------------------- Fusion target cache (GLOBAL NED) --------------------
        self._fusion_have_target = False
        self._fusion_target_xg = 0.0
        self._fusion_target_yg = 0.0
        self._fusion_target_label = ""  # "yellow_triangle" / "black_triangle"

        # -------------------- ROS I/O --------------------
        self.sub_pose = self.create_subscription(Pose2D, self.pose_topic, self.on_pose, 10)

        # Buoy detections: lambdas so we know which camera produced it
        self.sub_det_center = None
        self.sub_det_left = None
        self.sub_det_right = None

        if self.detections_topic:
            self.sub_det_center = self.create_subscription(
                ObjectPositionArray,
                self.detections_topic,
                lambda msg: self.on_detections(msg, "center"),
                10,
            )
        if self.detections_topic_left:
            self.sub_det_left = self.create_subscription(
                ObjectPositionArray,
                self.detections_topic_left,
                lambda msg: self.on_detections(msg, "left"),
                10,
            )
        if self.detections_topic_right:
            self.sub_det_right = self.create_subscription(
                ObjectPositionArray,
                self.detections_topic_right,
                lambda msg: self.on_detections(msg, "right"),
                10,
            )

        # Fusion detections (boat-like)
        self.sub_fusion = self.create_subscription(
            ObjectPositionArray,
            self.fusion_topic,
            self.on_fusion,
            10
        )

        self.pub_tracks = self.create_publisher(ObjectPositionArray, self.tracked_topic, 10)
        self.pub_markers = self.create_publisher(MarkerArray, self.marker_topic, 10)

        self.timer = self.create_timer(self.dt, self.on_timer)

    # -------------------- Pose --------------------
    def on_pose(self, msg: Pose2D):
        self.x_usv_NED = float(msg.x)
        self.y_usv_NED = float(msg.y)
        self.psi_usv_NED = float(msg.theta)
        self.have_pose = True

    # -------------------- Fusion callback (other boat marker) --------------------
    def on_fusion(self, msg: ObjectPositionArray):
        if not self.have_pose:
            return

        n = min(len(msg.object_names), len(msg.x_object), len(msg.y_object))
        if n <= 0:
            return

        for i in range(n):
            name = (msg.object_names[i] or "").strip()

            if name == "yellow_triangle" and not self.fusion_use_yellow:
                continue
            if name == "black_triangle" and not self.fusion_use_black:
                continue
            if name not in ("yellow_triangle", "black_triangle"):
                continue

            x_local = float(msg.x_object[i])
            y_local = float(msg.y_object[i])

            xg, yg = self.local_nwu_to_global_ned(x_local, y_local)

            # Same-boat gating in GLOBAL NED
            if self._fusion_have_target:
                dx = float(xg) - float(self._fusion_target_xg)
                dy = float(yg) - float(self._fusion_target_yg)
                d2 = dx * dx + dy * dy

                if d2 <= self.fusion_same_boat_r2:
                    # same boat -> update
                    self._fusion_target_xg = float(xg)
                    self._fusion_target_yg = float(yg)
                    self._fusion_target_label = name
                    return

                # different boat candidate
                if not self.fusion_allow_switch:
                    # stay locked on the original boat
                    return

                # allowed to switch
                self._fusion_target_xg = float(xg)
                self._fusion_target_yg = float(yg)
                self._fusion_target_label = name
                return

            # no current target -> accept first detection
            self._fusion_have_target = True
            self._fusion_target_xg = float(xg)
            self._fusion_target_yg = float(yg)
            self._fusion_target_label = name
            return

    # -------------------- Local cone gate (NEW buoys only) --------------------
    @staticmethod
    def _wrap_deg(d: float) -> float:
        return (d + 180.0) % 360.0 - 180.0

    def _camera_center_deg(self, camera: str) -> float:
        if camera == "left":
            return self.cam_left_center_deg
        if camera == "right":
            return self.cam_right_center_deg
        return self.cam_center_deg

    def _local_cone_ok(self, x_local: float, y_local: float, camera: str) -> bool:
        x_local = float(x_local)
        y_local = float(y_local)

        if self.require_forward_x and x_local <= 0.0:
            return False

        ang = math.degrees(math.atan2(y_local, x_local))
        cam_center = float(self._camera_center_deg(camera))
        diff = self._wrap_deg(ang - cam_center)
        return abs(diff) <= self.cone_half_angle_deg

    # -------------------- Transform (LOCAL NWU -> GLOBAL NED) --------------------
    def local_nwu_to_global_ned(self, x_local: float, y_local: float) -> Tuple[float, float]:
        d_L_G = np.array([self.x_usv_NED, self.y_usv_NED, 0.0], dtype=float)

        c = math.cos(self.psi_usv_NED)
        s = math.sin(self.psi_usv_NED)
        R_L_G = np.array(
            [[c, -s, 0.0],
             [s,  c, 0.0],
             [0.0, 0.0, 1.0]],
            dtype=float
        )

        d_P_L = np.array([float(x_local), -float(y_local), 0.0], dtype=float)  # NWU -> NED

        H_L_G = np.eye(4, dtype=float)
        H_L_G[:3, :3] = R_L_G
        H_L_G[:3, 3] = d_L_G

        H_P_L = np.eye(4, dtype=float)
        H_P_L[:3, 3] = d_P_L

        H_P_G = H_L_G @ H_P_L
        return float(H_P_G[0, 3]), float(H_P_G[1, 3])

    # -------------------- Nearest track in class --------------------
    def _nearest_track_index(self, cls: str, x: float, y: float) -> Tuple[int, float]:
        best_i = -1
        best_d2 = float("inf")
        for i, tr in enumerate(self.tracks[cls]):
            tx, ty = tr.kf.get_position()
            dx = x - tx
            dy = y - ty
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best_i = i
        return best_i, best_d2

    # Only consider confirmed buoys for spawn suppression
    def _near_any_identified_same_color(self, cls: str, x: float, y: float, r2: float) -> bool:
        for tr in self.tracks[cls]:
            if not tr.confirmed:
                continue
            tx, ty = tr.kf.get_position()
            dx = x - tx
            dy = y - ty
            if (dx * dx + dy * dy) <= r2:
                return True
        return False

    # -------------------- Update --------------------
    def _update_track(self, tr: Track, xg: float, yg: float, z_local: float):
        z = np.array([[xg], [yg]], dtype=float)
        tr.kf.update(z)
        tr.z = float(z_local)
        tr.hits += 1
        tr.consecutive_hits += 1
        tr.updated_this_cycle = True

        min_hits = self.yellow_min_hits if tr.cls == "yellow_buoy" else self.min_consecutive_hits
        if (not tr.confirmed) and (tr.consecutive_hits >= min_hits):
            tr.confirmed = True

    # -------------------- Detections (buoys) --------------------
    def on_detections(self, msg: ObjectPositionArray, camera: str = "center"):
        if not self.have_pose:
            return

        n = min(len(msg.object_names), len(msg.x_object), len(msg.y_object), len(msg.z_object))
        if n == 0:
            return

        for i in range(n):
            cls = (msg.object_names[i] or "").strip()
            if cls not in TRACK_CLASSES:
                continue

            x_local = float(msg.x_object[i])
            y_local = float(msg.y_object[i])
            z_local = float(msg.z_object[i])

            xg, yg = self.local_nwu_to_global_ned(x_local, y_local)

            dx_b = xg - self.x_usv_NED
            dy_b = yg - self.y_usv_NED
            d2_boat = dx_b * dx_b + dy_b * dy_b

            # --- Try associate/update existing buoy of same class ---
            best_i, _ = self._nearest_track_index(cls, xg, yg)
            if best_i >= 0:
                tr = self.tracks[cls][best_i]
                tx, ty = tr.kf.get_position()
                dx = xg - tx
                dy = yg - ty
                d2 = dx * dx + dy * dy

                if cls == "yellow_buoy" and self.yellow_enable_assoc:
                    gate2 = self.yellow_assoc_r2
                    maxjump2 = self.yellow_max_jump2
                else:
                    gate2 = self.assoc_r_conf2 if tr.confirmed else self.assoc_r2
                    maxjump2 = self.max_jump_conf2 if tr.confirmed else self.max_jump_unconf2

                if d2 <= gate2:
                    if d2 <= maxjump2:
                        self._update_track(tr, xg, yg, z_local)
                    continue

            # --- Spawn NEW buoy with gates ---
            if cls == "yellow_buoy":
                if d2_boat > self.yellow_add_r2:
                    continue
            else:
                if d2_boat > self.add_r2:
                    continue

            if not self._local_cone_ok(x_local, y_local, camera):
                continue

            keep_r2 = self.yellow_keep_r2 if cls == "yellow_buoy" else self.same_color_keep_radius2
            if self._near_any_identified_same_color(cls, xg, yg, keep_r2):
                continue

            kf = KalmanTracker2D(dt=self.dt, r_meas=self.kf_r_meas, q_proc=self.kf_q_proc, p0=self.kf_p0)
            kf.init_from_measurement(xg, yg)
            self.tracks[cls].append(
                Track(
                    cls=cls,
                    kf=kf,
                    z=z_local,
                    hits=1,
                    consecutive_hits=1,
                    confirmed=False,
                    updated_this_cycle=True,
                )
            )

    # -------------------- Timer: predict + publish --------------------
    def on_timer(self):
        if not self.have_pose:
            return

        for cls in TRACK_CLASSES:
            for tr in self.tracks[cls]:
                tr.kf.predict()
                if not tr.updated_this_cycle and not tr.confirmed:
                    tr.consecutive_hits = 0
                tr.updated_this_cycle = False

        self.publish_tracks()
        self.publish_markers()

    # -------------------- Publishing --------------------
    def _is_visible(self, tr: Track) -> bool:
        if tr.confirmed:
            return True
        min_hits = self.yellow_min_hits if tr.cls == "yellow_buoy" else self.min_consecutive_hits
        return tr.consecutive_hits >= min_hits

    def publish_tracks(self):
        out = ObjectPositionArray()
        for cls in sorted(TRACK_CLASSES):
            for k, tr in enumerate(self.tracks[cls]):
                if not self._is_visible(tr):
                    continue
                x, y = tr.kf.get_position()
                out.object_names.append(f"{cls}_{k}")
                out.x_object.append(float(x))
                out.y_object.append(float(y))
                out.z_object.append(float(tr.z))
        self.pub_tracks.publish(out)

    def _buoy_rgba(self, cls: str) -> Tuple[float, float, float, float]:
        if cls == "green_buoy":
            return (0.1, 0.9, 0.1, 0.9)
        if cls == "red_buoy":
            return (0.9, 0.1, 0.1, 0.9)
        if cls == "yellow_buoy":
            return (0.9, 0.9, 0.1, 0.9)
        if cls == "black_buoy":
            return (0.05, 0.05, 0.05, 0.95)
        return (0.8, 0.8, 0.8, 0.9)

    @staticmethod
    def _yaw_to_quat_z_w(yaw: float) -> Tuple[float, float]:
        return (math.sin(yaw * 0.5), math.cos(yaw * 0.5))

    def publish_markers(self):
        ma = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        mid = 0

        # 1) Ego boat arrow (keep!)
        boat = Marker()
        boat.header.frame_id = self.frame_id
        boat.header.stamp = stamp
        boat.ns = "ego_boat"
        boat.id = mid
        mid += 1
        boat.type = Marker.ARROW
        boat.action = Marker.ADD
        boat.pose.position.x = float(self.x_usv_NED)
        boat.pose.position.y = float(self.y_usv_NED)
        boat.pose.position.z = 0.0
        yaw = float(self.psi_usv_NED)
        boat.pose.orientation.z = math.sin(yaw * 0.5)
        boat.pose.orientation.w = math.cos(yaw * 0.5)
        boat.scale.x = float(self.boat_len)
        boat.scale.y = float(self.boat_w)
        boat.scale.z = float(self.boat_w * 1.5)
        boat.color.r, boat.color.g, boat.color.b, boat.color.a = 0.2, 0.6, 1.0, 1.0
        boat.lifetime.sec = 0
        boat.lifetime.nanosec = 0
        ma.markers.append(boat)

        # 2) Buoy spheres
        for cls in sorted(TRACK_CLASSES):
            r, g, b, a = self._buoy_rgba(cls)
            for tr in self.tracks[cls]:
                if not self._is_visible(tr):
                    continue
                x, y = tr.kf.get_position()

                m = Marker()
                m.header.frame_id = self.frame_id
                m.header.stamp = stamp
                m.ns = cls
                m.id = mid
                mid += 1
                m.type = Marker.SPHERE
                m.action = Marker.ADD
                m.pose.position.x = float(x)
                m.pose.position.y = float(y)
                m.pose.position.z = 0.0
                m.pose.orientation.w = 1.0
                m.scale.x = float(self.buoy_d)
                m.scale.y = float(self.buoy_d)
                m.scale.z = float(self.buoy_d)
                m.color.r, m.color.g, m.color.b, m.color.a = float(r), float(g), float(b), float(a)
                m.lifetime.sec = 0
                m.lifetime.nanosec = 0
                ma.markers.append(m)

        # 3) Fusion "other boat" marker (NO TIMEOUT, no delete)
        if self._fusion_have_target:
            yaw_t = math.atan2(self._fusion_target_yg - self.y_usv_NED,
                               self._fusion_target_xg - self.x_usv_NED)
            qz, qw = self._yaw_to_quat_z_w(yaw_t)

            tgt = Marker()
            tgt.header.frame_id = self.frame_id
            tgt.header.stamp = stamp
            tgt.ns = "fusion_boat"
            tgt.id = 0
            tgt.type = Marker.ARROW
            tgt.action = Marker.ADD
            tgt.pose.position.x = float(self._fusion_target_xg)
            tgt.pose.position.y = float(self._fusion_target_yg)
            tgt.pose.position.z = 0.0
            tgt.pose.orientation.z = float(qz)
            tgt.pose.orientation.w = float(qw)
            tgt.scale.x = float(self.fusion_tgt_len)
            tgt.scale.y = float(self.fusion_tgt_w)
            tgt.scale.z = float(self.fusion_tgt_h)

            if self._fusion_target_label == "yellow_triangle":
                tgt.color.r, tgt.color.g, tgt.color.b, tgt.color.a = 0.95, 0.95, 0.10, 0.95
            else:
                tgt.color.r, tgt.color.g, tgt.color.b, tgt.color.a = 0.95, 0.45, 0.10, 0.95

            tgt.lifetime.sec = 0
            tgt.lifetime.nanosec = 0
            ma.markers.append(tgt)

        self.pub_markers.publish(ma)


def main():
    rclpy.init()
    node = BuoyMapperKF()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()