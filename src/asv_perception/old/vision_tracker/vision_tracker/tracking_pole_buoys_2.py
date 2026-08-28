#!/usr/bin/env python3
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose2D
from visualization_msgs.msg import Marker, MarkerArray
from fau_msgs.msg import ObjectPositionArray

from lidar_msgs.msg import BuoyDetected  # name[], x[], y[], z[]


TRACK_CLASSES = ("buoy",)  # single class


class KalmanTracker2D:
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


class BuoyMapperFromBuoyDetected(Node):
    """
    Subscribes to /vision/output/buoy_detected (LOCAL NWU points).
    Tracks ALL buoys as a single class (no left/right pairing).
    Keeps up to max_tracks total (default 6).
    """

    def __init__(self):
        super().__init__("buoy_mapper_from_buoy_detected")

        # ---- Params ----
        self.declare_parameter("asv_name", "asv")
        self.declare_parameter("pose_topic", "")  # default /{asv}/vehicle_pose
        self.declare_parameter("buoy_detected_topic", "/vision/output/buoy_detected")

        self.declare_parameter("tracked_topic", "/vision/tracked/obstacles")
        self.declare_parameter("marker_topic", "/vision/tracked/markers")
        self.declare_parameter("frame_id", "map")

        # Keep only N total buoys (no per-side logic)
        self.declare_parameter("max_tracks", 6)

        # Filtering
        self.declare_parameter("require_forward_x", True)
        self.declare_parameter("min_range_m", 0.0)
        self.declare_parameter("max_range_m", 0.0)

        # KF
        self.declare_parameter("dt", 0.1)
        self.declare_parameter("kf_r_meas", 1.2)
        self.declare_parameter("kf_q_proc", 0.002)
        self.declare_parameter("kf_p0", 25.0)

        # Association / stability
        self.declare_parameter("assoc_radius_m", 2.0)
        self.declare_parameter("assoc_radius_confirmed_m", 2.5)
        self.declare_parameter("min_consecutive_hits", 1)

        # Reject big jumps
        self.declare_parameter("max_update_jump_confirmed_m", 0.6)
        self.declare_parameter("max_update_jump_unconfirmed_m", 1.0)

        # Optional freeze
        self.declare_parameter("freeze_when_confirmed", False)

        # Markers
        self.declare_parameter("buoy_marker_diameter_m", 0.8)
        self.declare_parameter("boat_marker_length_m", 1.2)
        self.declare_parameter("boat_marker_width_m", 0.4)

        # ---- Read params ----
        self.asv_name = str(self.get_parameter("asv_name").value)

        self.pose_topic = str(self.get_parameter("pose_topic").value).strip()
        if not self.pose_topic:
            self.pose_topic = f"/{self.asv_name}/vehicle_pose"

        self.buoy_detected_topic = str(self.get_parameter("buoy_detected_topic").value)
        self.tracked_topic = str(self.get_parameter("tracked_topic").value)
        self.marker_topic = str(self.get_parameter("marker_topic").value)
        self.frame_id = str(self.get_parameter("frame_id").value)

        self.max_tracks = int(self.get_parameter("max_tracks").value)

        self.require_forward_x = bool(self.get_parameter("require_forward_x").value)
        self.min_range_m = float(self.get_parameter("min_range_m").value)
        self.max_range_m = float(self.get_parameter("max_range_m").value)

        self.dt = float(self.get_parameter("dt").value)
        self.kf_r_meas = float(self.get_parameter("kf_r_meas").value)
        self.kf_q_proc = float(self.get_parameter("kf_q_proc").value)
        self.kf_p0 = float(self.get_parameter("kf_p0").value)

        self.assoc_r = float(self.get_parameter("assoc_radius_m").value)
        self.assoc_r2 = self.assoc_r * self.assoc_r
        self.assoc_r_conf = float(self.get_parameter("assoc_radius_confirmed_m").value)
        self.assoc_r_conf2 = self.assoc_r_conf * self.assoc_r_conf
        self.min_consecutive_hits = int(self.get_parameter("min_consecutive_hits").value)

        self.max_jump_conf = float(self.get_parameter("max_update_jump_confirmed_m").value)
        self.max_jump_conf2 = self.max_jump_conf * self.max_jump_conf
        self.max_jump_unconf = float(self.get_parameter("max_update_jump_unconfirmed_m").value)
        self.max_jump_unconf2 = self.max_jump_unconf * self.max_jump_unconf

        self.freeze_when_confirmed = bool(self.get_parameter("freeze_when_confirmed").value)

        self.buoy_d = float(self.get_parameter("buoy_marker_diameter_m").value)
        self.boat_len = float(self.get_parameter("boat_marker_length_m").value)
        self.boat_w = float(self.get_parameter("boat_marker_width_m").value)

        # ---- Pose (GLOBAL NED) ----
        self.x_usv_NED = 0.0
        self.y_usv_NED = 0.0
        self.psi_usv_NED = 0.0
        self.have_pose = False

        # ---- Tracks (single class) ----
        self.tracks: List[Track] = []

        # ---- ROS I/O ----
        self.sub_pose = self.create_subscription(Pose2D, self.pose_topic, self.on_pose, 10)
        self.sub_buoy = self.create_subscription(BuoyDetected, self.buoy_detected_topic, self.on_buoy_detected, 10)

        self.pub_tracks = self.create_publisher(ObjectPositionArray, self.tracked_topic, 10)
        self.pub_markers = self.create_publisher(MarkerArray, self.marker_topic, 10)

        self.timer = self.create_timer(self.dt, self.on_timer)

        self.get_logger().info(
            f"BuoyMapperFromBuoyDetected: pose={self.pose_topic} buoy={self.buoy_detected_topic} "
            f"max_tracks={self.max_tracks}"
        )

    # ---------------- Pose ----------------
    def on_pose(self, msg: Pose2D):
        self.x_usv_NED = float(msg.x)
        self.y_usv_NED = float(msg.y)
        self.psi_usv_NED = float(msg.theta)
        self.have_pose = True

    # ---------------- Transform: LOCAL NWU -> GLOBAL NED ----------------
    def local_nwu_to_global_ned(self, x_local: float, y_local: float) -> Tuple[float, float]:
        dx = float(x_local)
        dy = -float(y_local)

        c = math.cos(self.psi_usv_NED)
        s = math.sin(self.psi_usv_NED)

        xg = self.x_usv_NED + (c * dx - s * dy)
        yg = self.y_usv_NED + (s * dx + c * dy)
        return float(xg), float(yg)

    # ---------------- Helpers ----------------
    def _nearest_track_index(self, x: float, y: float) -> Tuple[int, float]:
        best_i = -1
        best_d2 = float("inf")
        for i, tr in enumerate(self.tracks):
            tx, ty = tr.kf.get_position()
            dx = x - tx
            dy = y - ty
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best_i = i
        return best_i, best_d2

    def _update_track(self, tr: Track, xg: float, yg: float, z_local: float):
        if self.freeze_when_confirmed and tr.confirmed:
            tr.z = float(z_local)
            tr.hits += 1
            tr.consecutive_hits += 1
            tr.updated_this_cycle = True
            return

        z = np.array([[xg], [yg]], dtype=float)
        tr.kf.update(z)
        tr.z = float(z_local)
        tr.hits += 1
        tr.consecutive_hits += 1
        tr.updated_this_cycle = True
        if (not tr.confirmed) and (tr.consecutive_hits >= self.min_consecutive_hits):
            tr.confirmed = True

    def _is_visible(self, tr: Track) -> bool:
        return tr.confirmed or (tr.consecutive_hits >= self.min_consecutive_hits)

    def _spawn_track(self, xg: float, yg: float, z_local: float):
        kf = KalmanTracker2D(dt=self.dt, r_meas=self.kf_r_meas, q_proc=self.kf_q_proc, p0=self.kf_p0)
        kf.init_from_measurement(xg, yg)
        self.tracks.append(
            Track(cls="buoy", kf=kf, z=z_local, hits=1, consecutive_hits=1, confirmed=False, updated_this_cycle=True)
        )

    def _trim_to_max_tracks_keep_closest(self):
        if len(self.tracks) <= self.max_tracks:
            return
        # Keep closest to boat (by range in global NED)
        scored = []
        for tr in self.tracks:
            x, y = tr.kf.get_position()
            dx = x - self.x_usv_NED
            dy = y - self.y_usv_NED
            scored.append((dx * dx + dy * dy, tr))
        scored.sort(key=lambda t: t[0])
        self.tracks = [tr for (_d2, tr) in scored[: self.max_tracks]]

    # ---------------- Main callback ----------------
    def on_buoy_detected(self, msg: BuoyDetected):
        if not self.have_pose:
            return

        n = min(len(msg.x), len(msg.y), len(msg.z))
        if n == 0:
            return

        # Build detection list (no class split)
        dets: List[Tuple[float, float, float, float]] = []  # (rng, x_local, y_local, z_local)

        for i in range(n):
            x_local = float(msg.x[i])
            y_local = float(msg.y[i])
            z_local = float(msg.z[i])

            if self.require_forward_x and x_local <= 0.0:
                continue

            rng = math.hypot(x_local, y_local)

            if self.min_range_m > 0.0 and rng < self.min_range_m:
                continue
            if self.max_range_m > 0.0 and rng > self.max_range_m:
                continue

            dets.append((rng, x_local, y_local, z_local))

        if not dets:
            return

        # Sort by range: closest first
        dets.sort(key=lambda t: t[0])

        # Update/associate each detection
        for (_rng, x_local, y_local, z_local) in dets:
            xg, yg = self.local_nwu_to_global_ned(x_local, y_local)

            best_i, best_d2 = self._nearest_track_index(xg, yg)

            if best_i >= 0:
                tr = self.tracks[best_i]
                tx, ty = tr.kf.get_position()
                dx = xg - tx
                dy = yg - ty
                d2 = dx * dx + dy * dy

                gate2 = self.assoc_r_conf2 if tr.confirmed else self.assoc_r2
                maxjump2 = self.max_jump_conf2 if tr.confirmed else self.max_jump_unconf2

                if d2 <= gate2:
                    if d2 <= maxjump2:
                        self._update_track(tr, xg, yg, z_local)
                    continue

            # Spawn if we have capacity
            if len(self.tracks) < self.max_tracks:
                self._spawn_track(xg, yg, z_local)
            else:
                # If full: only allow update of nearest if it's very close (prevents stealing)
                if best_i >= 0 and best_d2 <= (self.assoc_r_conf2 * 1.5):
                    tr = self.tracks[best_i]
                    tx, ty = tr.kf.get_position()
                    dx = xg - tx
                    dy = yg - ty
                    d2 = dx * dx + dy * dy
                    maxjump2 = self.max_jump_conf2 if tr.confirmed else self.max_jump_unconf2
                    if d2 <= maxjump2:
                        self._update_track(tr, xg, yg, z_local)

        # Optional: trim to max tracks (keeps closest ones)
        self._trim_to_max_tracks_keep_closest()

    # ---------------- Timer ----------------
    def on_timer(self):
        if not self.have_pose:
            return

        for tr in self.tracks:
            tr.kf.predict()
            if not tr.updated_this_cycle and not tr.confirmed:
                tr.consecutive_hits = 0
            tr.updated_this_cycle = False

        self.publish_tracks()
        self.publish_markers()

    # ---------------- Publishing ----------------
    def publish_tracks(self):
        out = ObjectPositionArray()
        for k, tr in enumerate(self.tracks):
            if not self._is_visible(tr):
                continue
            x, y = tr.kf.get_position()
            out.object_names.append(f"buoy_{k}")
            out.x_object.append(float(x))
            out.y_object.append(float(y))
            out.z_object.append(float(tr.z))
        self.pub_tracks.publish(out)

    def publish_markers(self):
        ma = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        mid = 0

        # Boat marker
        boat = Marker()
        boat.header.frame_id = self.frame_id
        boat.header.stamp = stamp
        boat.ns = "boat"
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
        ma.markers.append(boat)

        # Buoy markers (single color, no pairing)
        for tr in self.tracks:
            if not self._is_visible(tr):
                continue

            x, y = tr.kf.get_position()

            m = Marker()
            m.header.frame_id = self.frame_id
            m.header.stamp = stamp
            m.ns = "buoy"
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

            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.8, 0.1, 0.9  # yellow-ish

            ma.markers.append(m)

        self.pub_markers.publish(ma)


def main(args=None):
    rclpy.init(args=args)
    node = BuoyMapperFromBuoyDetected()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()