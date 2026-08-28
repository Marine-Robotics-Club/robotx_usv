#!/usr/bin/env python3
from __future__ import annotations

import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from fau_msgs.msg import ObjectPositionArray
from geometry_msgs.msg import Pose2D
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration


class KalmanTracker2D:
    """Static position-only KF (random walk): state [x,y]."""
    def __init__(self, dt: float = 0.1):
        self.dt = float(dt)
        self.x = np.zeros((2, 1), dtype=float)

        self.F = np.eye(2, dtype=float)
        self.H = np.eye(2, dtype=float)

        self.R = np.eye(2, dtype=float) * 0.6
        self.P = np.eye(2, dtype=float) * 50.0
        self.Q = np.eye(2, dtype=float) * 0.02

        self.initialized = False

    def init_from_measurement(self, x: float, y: float):
        self.x[0, 0] = float(x)
        self.x[1, 0] = float(y)
        self.initialized = True

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: np.ndarray):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(2, dtype=float)
        self.P = (I - K @ self.H) @ self.P

    def get_position(self) -> tuple[float, float]:
        return float(self.x[0, 0]), float(self.x[1, 0])


class PoleTrack:
    """One KF per buoy class (green pole, red pole). Tracks the first one seen of that class."""
    def __init__(self, name: str, dt: float):
        self.name = name
        self.kf = KalmanTracker2D(dt=dt)
        self.tracking = False
        self.last_raw_global: tuple[float, float] | None = None
        self.hits = 0


class PoleBuoyKFGlobal(Node):
    def __init__(self):
        super().__init__("pole_buoy_kf_global")

        # ---- params ----
        self.declare_parameter("asv", "asv")
        self.declare_parameter("in_topic", "vision/output/fusion")  # ObjectPositionArray
        self.declare_parameter("odom_topic", "vehicle_pose")        # Pose2D (GLOBAL NED)

        self.declare_parameter("dt", 0.1)
        self.declare_parameter("gate_dist", 2.0)  # meters in GLOBAL frame (keep “same buoy”)
        self.declare_parameter("global_frame", "map")  # IMPORTANT: this is what RViz Fixed Frame must match

        # transform (your function)
        self.declare_parameter("lidar_pitch_deg", 3.5)
        self.declare_parameter("gps_behind_lidar_m", 1.03)

        # marker sizes (buoy radius ~0.5m)
        self.declare_parameter("buoy_marker_diameter_m", 1.0)  # 2*0.5
        self.declare_parameter("buoy_marker_z", 0.25)
        self.declare_parameter("text_height_m", 0.35)
        self.declare_parameter("marker_lifetime", 0.3)  # keep re-publishing

        # ---- read params ----
        self.wamv = str(self.get_parameter("asv").value)
        self.in_topic = str(self.get_parameter("in_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)

        self.dt = float(self.get_parameter("dt").value)
        self.gate_dist = float(self.get_parameter("gate_dist").value)
        self.global_frame = str(self.get_parameter("global_frame").value)

        self.lidar_pitch_deg = float(self.get_parameter("lidar_pitch_deg").value)
        self.gps_behind_lidar_m = float(self.get_parameter("gps_behind_lidar_m").value)

        self.buoy_marker_diam = float(self.get_parameter("buoy_marker_diameter_m").value)
        self.buoy_marker_z = float(self.get_parameter("buoy_marker_z").value)
        self.text_height_m = float(self.get_parameter("text_height_m").value)
        self.marker_lifetime = float(self.get_parameter("marker_lifetime").value)

        # ---- boat pose (GLOBAL NED) ----
        self.have_odom = False
        self.boat_x = 0.0
        self.boat_y = 0.0
        self.boat_psi = 0.0

        # ---- trackers (different buoys) ----
        self.green = PoleTrack("green_pole_buoy", dt=self.dt)
        self.red = PoleTrack("red_pole_buoy", dt=self.dt)

        # ---- subs ----
        self.create_subscription(
            Pose2D,
            f"/{self.wamv}/{self.odom_topic}".replace("//", "/"),
            self._odom_cb,
            10,
        )
        self.create_subscription(
            ObjectPositionArray,
            f"/{self.in_topic}".replace("//", "/"),
            self._fusion_cb,
            10,
        )

        # ---- marker pub (global) ----
        # RViz subscribes fine to volatile markers
        self.marker_pub = self.create_publisher(
            MarkerArray,
            f"/{self.wamv}/viz/pole_buoys_global".replace("//", "/"),
            10,
        )

        self.timer = self.create_timer(self.dt, self._timer_publish_markers)

        self.get_logger().info(
            f"pole_buoy_kf_global | sub=/{self.in_topic} odom=/{self.wamv}/{self.odom_topic} "
            f"markers=/{self.wamv}/viz/pole_buoys_global global_frame={self.global_frame}\n"
            f"RViz: set Fixed Frame = '{self.global_frame}' (NOT base_link)."
        )

    # ---------- your transform ----------
    def body_to_global_ned(self, xb, yb, zb, x_usv_ned, y_usv_ned, psi_usv_ned):
        theta = math.radians(self.lidar_pitch_deg)
        ct = math.cos(theta)
        st = math.sin(theta)

        x1 = ct * xb + st * zb
        y1 = yb
        z1 = -st * xb + ct * zb

        x1 = x1 + self.gps_behind_lidar_m  # GPS behind lidar

        x_body = x1
        y_body = -y1  # NWU -> NED
        z_body = z1

        c = math.cos(psi_usv_ned)
        s = math.sin(psi_usv_ned)

        dN = c * x_body - s * y_body
        dE = s * x_body + c * y_body

        xg = x_usv_ned + dN
        yg = y_usv_ned + dE
        zg = z_body
        return xg, yg, zg

    # ---------- label helpers ----------
    @staticmethod
    def _canon(name: str) -> str:
        s = (name or "").strip()
        # tolerate your typo
        if s.lower() == "greeb_pole_buoy":
            return "green_pole_buoy"
        return s

    def _odom_cb(self, msg: Pose2D):
        self.boat_x = float(msg.x)
        self.boat_y = float(msg.y)
        self.boat_psi = float(msg.theta)
        self.have_odom = True

    def _fusion_cb(self, msg: ObjectPositionArray):
        n = min(len(msg.object_names), len(msg.x_object), len(msg.y_object))
        if n == 0:
            return

        # print raw (preview)
        preview = []
        for i in range(min(n, 8)):
            nm = self._canon(str(msg.object_names[i]))
            x = float(msg.x_object[i])
            y = float(msg.y_object[i])
            z = float(msg.z_object[i]) if i < len(msg.z_object) else 0.0
            preview.append(f"{nm}@({x:.2f},{y:.2f},{z:.2f})")
        self.get_logger().info(f"RAW /{self.in_topic}: n={n} " + " | ".join(preview))

        if not self.have_odom:
            self.get_logger().warn("No /vehicle_pose yet; cannot convert to global NED.")
            return

        # Find FIRST detection per class in this message (green and red are DIFFERENT buoys)
        idx_green = None
        idx_red = None
        for i in range(n):
            nm = self._canon(str(msg.object_names[i]))
            if idx_green is None and nm == "green_pole_buoy":
                idx_green = i
            elif idx_red is None and nm == "red_pole_buoy":
                idx_red = i
            if idx_green is not None and idx_red is not None:
                break

        if idx_green is not None:
            self._handle_one(self.green, msg, idx_green)

        if idx_red is not None:
            self._handle_one(self.red, msg, idx_red)

    def _handle_one(self, tr: PoleTrack, msg: ObjectPositionArray, idx: int):
        xb = float(msg.x_object[idx])
        yb = float(msg.y_object[idx])
        zb = float(msg.z_object[idx]) if idx < len(msg.z_object) else 0.0

        xg, yg, zg = self.body_to_global_ned(
            xb=xb, yb=yb, zb=zb,
            x_usv_ned=self.boat_x,
            y_usv_ned=self.boat_y,
            psi_usv_ned=self.boat_psi,
        )
        tr.last_raw_global = (xg, yg)

        if not tr.tracking:
            tr.kf.init_from_measurement(xg, yg)
            tr.tracking = True
            tr.hits = 1
            xk, yk = tr.kf.get_position()
            self.get_logger().info(
                f"INIT {tr.name}: raw_body=({xb:.2f},{yb:.2f},{zb:.2f}) -> "
                f"raw_global=({xg:.2f},{yg:.2f},{zg:.2f}) | kf=({xk:.2f},{yk:.2f})"
            )
            return

        # gate in GLOBAL frame so we keep updating the same buoy of that class
        xk, yk = tr.kf.get_position()
        d = math.hypot(xg - xk, yg - yk)
        if d > self.gate_dist:
            self.get_logger().warn(
                f"{tr.name} ignored (gate): raw_global=({xg:.2f},{yg:.2f}) "
                f"kf=({xk:.2f},{yk:.2f}) d={d:.2f}m > gate={self.gate_dist:.2f}"
            )
            return

        tr.kf.predict()
        tr.kf.update(np.array([[xg], [yg]], dtype=float))
        tr.hits += 1
        xk2, yk2 = tr.kf.get_position()

        self.get_logger().info(
            f"{tr.name} update: raw_global=({xg:.2f},{yg:.2f}) -> kf=({xk2:.2f},{yk2:.2f}) d={d:.2f} hits={tr.hits}"
        )

    # ---------- markers ----------
    def _timer_publish_markers(self):
        stamp = self.get_clock().now().to_msg()
        ma = MarkerArray()

        lifetime = Duration(sec=int(self.marker_lifetime),
                            nanosec=int((self.marker_lifetime % 1.0) * 1e9))

        def add_buoy(tr: PoleTrack, base_id: int, rgba: tuple[float, float, float, float]):
            if not tr.tracking:
                return
            x, y = tr.kf.get_position()

            m = Marker()
            m.header.frame_id = self.global_frame  # ✅ GLOBAL FRAME
            m.header.stamp = stamp
            m.ns = "pole_buoys"
            m.id = base_id
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(x)
            m.pose.position.y = float(y)
            m.pose.position.z = float(self.buoy_marker_z)
            m.pose.orientation.w = 1.0

            # smaller (buoy radius ~0.5m => diameter ~1.0m)
            d = max(0.2, float(self.buoy_marker_diam))
            m.scale.x = d
            m.scale.y = d
            m.scale.z = min(d, 0.8)  # keep it squat

            m.color.r, m.color.g, m.color.b, m.color.a = rgba
            m.lifetime = lifetime
            ma.markers.append(m)

            t = Marker()
            t.header.frame_id = self.global_frame  # ✅ GLOBAL FRAME
            t.header.stamp = stamp
            t.ns = "pole_buoy_labels"
            t.id = base_id + 1
            t.type = Marker.TEXT_VIEW_FACING
            t.action = Marker.ADD
            t.pose.position.x = float(x)
            t.pose.position.y = float(y)
            t.pose.position.z = float(self.buoy_marker_z + 0.9)
            t.pose.orientation.w = 1.0
            t.scale.z = float(self.text_height_m)
            t.color.r, t.color.g, t.color.b, t.color.a = 1.0, 1.0, 1.0, 1.0
            t.text = f"{tr.name} hits={tr.hits} ({x:.1f},{y:.1f})"
            t.lifetime = lifetime
            ma.markers.append(t)

        add_buoy(self.green, base_id=10, rgba=(0.0, 1.0, 0.0, 0.9))
        add_buoy(self.red, base_id=20, rgba=(1.0, 0.0, 0.0, 0.9))

        if ma.markers:
            self.marker_pub.publish(ma)


def main(args=None):
    rclpy.init(args=args)
    node = PoleBuoyKFGlobal()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()