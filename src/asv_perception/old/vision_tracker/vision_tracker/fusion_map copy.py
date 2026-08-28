#!/usr/bin/env python3
from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from geometry_msgs.msg import Pose2D
from nav_msgs.msg import OccupancyGrid
from builtin_interfaces.msg import Duration
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import CameraInfo

from fau_msgs.msg import ObjectPosition
from yolov26_msgs.msg import YoloDetection


# -------------------- helpers --------------------
def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def pixel_to_bearing_intrinsics(x_center_px: float, fx: float, cx: float) -> float:
    # radians, +right in image
    return math.atan2(float(x_center_px) - float(cx), float(fx))


# -------------------- small KF (same spirit as yours) --------------------
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
        # z is (2,1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(2, dtype=float)
        self.P = (I - K @ self.H) @ self.P

    def get_position(self) -> Tuple[float, float]:
        return float(self.x[0, 0]), float(self.x[1, 0])


@dataclass
class CamDet:
    cls: str
    conf: float
    xc: float
    yc: float


class Track:
    def __init__(self, track_id: int, cls: str, x: float, y: float, now: float, dt: float):
        self.id = int(track_id)
        self.cls = str(cls)
        self.kf = KalmanTracker2D(dt=dt)
        self.kf.init_from_measurement(x, y)
        self.last_update = float(now)
        self.hits = 1

        # persistent marker behavior
        self.locked_color = False
        self.color = (0.2, 0.8, 1.0, 0.95)  # default
        self.published_persistent = False
        self.marker_id: Optional[int] = None


# -------------------- node --------------------
class CameraMapAngleCompareKF(Node):
    def __init__(self):
        super().__init__("camera_map_angle_compare_kf")

        # ---- params ----
        self.declare_parameter("wamv", "wamv")

        self.declare_parameter("yolo_topic", "/yolov26/detections")
        self.declare_parameter("pose_topic", "vehicle_pose")
        self.declare_parameter("map_topic", "map/local_occupancy_2")
        self.declare_parameter("camera_info_topic", "/cam132/image_raw")
        self.declare_parameter("camera_info_timeout_s", 2.0)

        # outputs
        self.declare_parameter("out_topic", "vision/output/map_hits_tracked")          # ObjectPosition
        self.declare_parameter("marker_topic", "viz/map_hits_tracked_persistent")     # MarkerArray
        self.declare_parameter("vehicle_marker_topic", "viz/vehicle_pose")

        # raycast
        self.declare_parameter("occ_threshold", 80)
        self.declare_parameter("ray_step_m", 0.25)
        self.declare_parameter("ray_max_m", 80.0)
        self.declare_parameter("ray_start_m", 0.5)

        # KF tracking (like your buoy tracker)
        self.declare_parameter("dt", 0.1)
        self.declare_parameter("gate_dist", 3.5)       # association gate
        self.declare_parameter("timeout", 0.0)         # <=0 keep forever
        self.declare_parameter("min_hits", 5)          # confirm after 5 hits
        self.declare_parameter("spawn_suppression_dist", 3.0)  # don't spawn new if near existing

        # requested radius behavior
        self.declare_parameter("hit_radius_m", 3.0)    # ✅ YOU ASKED: radius 3 meters
        self.declare_parameter("marker_z", 0.25)

        # printing
        self.declare_parameter("min_conf", 0.0)
        self.declare_parameter("print_angles", False)
        self.declare_parameter("print_period_s", 0.5)

        # vehicle marker
        self.declare_parameter("vehicle_marker_lifetime", 0.6)

        # ---- read ----
        self.wamv = str(self.get_parameter("wamv").value)

        self.yolo_topic = str(self.get_parameter("yolo_topic").value)
        self.pose_topic = str(self.get_parameter("pose_topic").value)
        self.map_topic = str(self.get_parameter("map_topic").value)
        self.camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        self.camera_info_timeout_s = float(self.get_parameter("camera_info_timeout_s").value)

        self.out_topic = str(self.get_parameter("out_topic").value)
        self.marker_topic = str(self.get_parameter("marker_topic").value)
        self.vehicle_marker_topic = str(self.get_parameter("vehicle_marker_topic").value)

        self.occ_threshold = int(self.get_parameter("occ_threshold").value)
        self.ray_step_m = float(self.get_parameter("ray_step_m").value)
        self.ray_max_m = float(self.get_parameter("ray_max_m").value)
        self.ray_start_m = float(self.get_parameter("ray_start_m").value)

        self.dt = float(self.get_parameter("dt").value)
        self.gate_dist = float(self.get_parameter("gate_dist").value)
        self.timeout = float(self.get_parameter("timeout").value)
        self.min_hits = int(self.get_parameter("min_hits").value)
        self.spawn_suppression_dist = float(self.get_parameter("spawn_suppression_dist").value)

        self.hit_radius_m = float(self.get_parameter("hit_radius_m").value)
        self.marker_z = float(self.get_parameter("marker_z").value)

        self.min_conf = float(self.get_parameter("min_conf").value)
        self.print_angles = bool(self.get_parameter("print_angles").value)
        self.print_period_s = float(self.get_parameter("print_period_s").value)

        self.vehicle_marker_lifetime = float(self.get_parameter("vehicle_marker_lifetime").value)

        # ---- state ----
        self.have_pose = False
        self.x_usv = 0.0
        self.y_usv = 0.0
        self.psi_usv = 0.0

        self.map_msg: Optional[OccupancyGrid] = None

        self.have_caminfo = False
        self.fx = 0.0
        self.cx = 0.0
        self._last_caminfo_t = 0.0

        self._cam: List[CamDet] = []
        self._last_warn_t: Dict[str, float] = {}
        self._last_print_t = 0.0

        # KF tracks
        self.tracks: List[Track] = []
        self.next_id: int = 0

        # persistent marker IDs (never reused)
        self._next_marker_id: int = 0

        # ---- QoS ----
        pose_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE)
        yolo_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
        map_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        caminfo_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE)

        # ---- subs ----
        self.create_subscription(Pose2D, f"/{self.wamv}/{self.pose_topic}".replace("//", "/"), self._pose_cb, pose_qos)
        self.create_subscription(OccupancyGrid, f"/{self.wamv}/{self.map_topic}".replace("//", "/"), self._map_cb, map_qos)
        self.create_subscription(YoloDetection, self.yolo_topic, self._yolo_cb, yolo_qos)
        self.create_subscription(CameraInfo, self.camera_info_topic, self._caminfo_cb, caminfo_qos)

        # ---- pubs ----
        self.pub_tracked = self.create_publisher(ObjectPosition, f"/{self.wamv}/{self.out_topic}".replace("//", "/"), 10)
        self.marker_pub = self.create_publisher(MarkerArray, f"/{self.wamv}/{self.marker_topic}".replace("//", "/"), 1)
        self.vehicle_marker_pub = self.create_publisher(Marker, f"/{self.wamv}/{self.vehicle_marker_topic}".replace("//", "/"), 1)

        self.timer = self.create_timer(self.dt, self._tick)

        self.get_logger().info(
            "Camera↔Map angle compare + KF tracking (persistent markers)\n"
            f"sub yolo={self.yolo_topic}\n"
            f"sub caminfo={self.camera_info_topic}\n"
            f"sub pose=/{self.wamv}/{self.pose_topic}\n"
            f"sub map=/{self.wamv}/{self.map_topic}\n"
            f"pub tracked=/{self.wamv}/{self.out_topic}\n"
            f"pub markers=/{self.wamv}/{self.marker_topic}\n"
            f"hit_radius_m={self.hit_radius_m:.2f} (diameter={2.0*self.hit_radius_m:.2f})\n"
        )

    # ---- time / throttle ----
    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _warn_throttle(self, period_s: float, msg: str):
        now = self._now()
        last = self._last_warn_t.get(msg, 0.0)
        if (now - last) >= float(period_s):
            self.get_logger().warn(msg)
            self._last_warn_t[msg] = now

    def _grid_frame(self) -> str:
        if self.map_msg is not None and self.map_msg.header.frame_id:
            return str(self.map_msg.header.frame_id)
        return "map"

    # ---- callbacks ----
    def _pose_cb(self, msg: Pose2D):
        self.x_usv = float(msg.x)
        self.y_usv = float(msg.y)
        self.psi_usv = float(msg.theta)
        self.have_pose = True

    def _map_cb(self, msg: OccupancyGrid):
        self.map_msg = msg

    def _caminfo_cb(self, msg: CameraInfo):
        if len(msg.k) < 9:
            return
        fx = float(msg.k[0])
        cx = float(msg.k[2])
        if not (math.isfinite(fx) and math.isfinite(cx)) or fx <= 1e-6:
            return
        self.fx = fx
        self.cx = cx
        self.have_caminfo = True
        self._last_caminfo_t = self._now()

    def _yolo_cb(self, msg: YoloDetection):
        n = min(len(msg.class_name), len(msg.confidence), len(msg.x_center), len(msg.y_center))
        dets: List[CamDet] = []
        for i in range(n):
            cls = str(msg.class_name[i]).strip()
            conf = float(msg.confidence[i])
            xc = float(msg.x_center[i])
            yc = float(msg.y_center[i])
            if not cls:
                continue
            if not (math.isfinite(conf) and math.isfinite(xc) and math.isfinite(yc)):
                continue
            if self.min_conf > 0.0 and conf < self.min_conf:
                continue
            dets.append(CamDet(cls=cls, conf=conf, xc=xc, yc=yc))
        self._cam = dets

    # ---- angle helpers ----
    def _bearing_from_px(self, xc: float) -> float:
        if not self.have_caminfo:
            return float("nan")
        if (self._now() - self._last_caminfo_t) > self.camera_info_timeout_s:
            self._warn_throttle(2.0, "CameraInfo seems stale; check camera_info publisher.")
        return pixel_to_bearing_intrinsics(xc, self.fx, self.cx)

    def _bearing_from_map_hit(self, hx: float, hy: float) -> float:
        # map-derived REL bearing to (hx,hy)
        dN = float(hx) - float(self.x_usv)
        dE = float(hy) - float(self.y_usv)
        psi_hit = math.atan2(dE, dN)
        return wrap_angle(psi_hit - float(self.psi_usv))

    # ---- grid ops ----
    def _grid_world_to_cell(self, grid: OccupancyGrid, x: float, y: float) -> Optional[Tuple[int, int]]:
        res = float(grid.info.resolution)
        ox = float(grid.info.origin.position.x)
        oy = float(grid.info.origin.position.y)
        gx = int((x - ox) / res)
        gy = int((y - oy) / res)
        if gx < 0 or gy < 0 or gx >= int(grid.info.width) or gy >= int(grid.info.height):
            return None
        return gx, gy

    def _grid_get(self, grid: OccupancyGrid, gx: int, gy: int) -> int:
        idx = gy * int(grid.info.width) + gx
        if idx < 0 or idx >= len(grid.data):
            return -1
        return int(grid.data[idx])

    def _raycast_to_map_hit(self, bearing_cam: float) -> Tuple[bool, Tuple[float, float], float]:
        grid = self.map_msg
        if grid is None:
            return (False, (0.0, 0.0), 0.0)

        yaw_global = float(self.psi_usv) + float(bearing_cam)
        c = math.cos(yaw_global)
        s = math.sin(yaw_global)

        start = float(self.ray_start_m)
        step = max(0.05, float(self.ray_step_m), float(grid.info.resolution))
        maxd = max(step, float(self.ray_max_m))

        d = start
        while d <= maxd:
            x = float(self.x_usv) + d * c
            y = float(self.y_usv) + d * s

            cell = self._grid_world_to_cell(grid, x, y)
            if cell is not None:
                gx, gy = cell
                val = self._grid_get(grid, gx, gy)
                if val >= int(self.occ_threshold):
                    return (True, (x, y), float(d))
            d += step

        return (False, (0.0, 0.0), float(maxd))

    # ---- tracking helpers ----
    def _spawn_track(self, cls: str, x: float, y: float, now: float):
        tr = Track(self.next_id, cls, x, y, now, dt=self.dt)
        tr.color = self._color_for_class(cls)  # choose once at creation
        tr.locked_color = True
        self.tracks.append(tr)
        self.next_id += 1

    def _associate_and_update(self, detections_xy: List[Tuple[str, float, float]], now: float):
        if not detections_xy:
            return

        if not self.tracks:
            for (cls, x, y) in detections_xy:
                self._spawn_track(cls, x, y, now)
            return

        # cost list: (track_idx, det_idx, dist)
        costs = []
        for ti, tr in enumerate(self.tracks):
            tx, ty = tr.kf.get_position()
            for di, (cls, x, y) in enumerate(detections_xy):
                d = math.hypot(x - tx, y - ty)
                costs.append((ti, di, d))
        costs.sort(key=lambda t: t[2])

        assigned_tracks = set()
        assigned_dets = set()

        # greedy assignment like your code
        for ti, di, d in costs:
            if d > self.gate_dist:
                break
            if ti in assigned_tracks or di in assigned_dets:
                continue

            tr = self.tracks[ti]
            cls, x, y = detections_xy[di]

            tr.kf.update(np.array([[x], [y]], dtype=float))
            tr.last_update = float(now)
            tr.hits += 1

            # keep the original locked color forever (your request),
            # but we can still store latest class name for debug/publishing
            tr.cls = str(cls)

            assigned_tracks.add(ti)
            assigned_dets.add(di)

        # spawn new detections if not assigned and not near any existing track
        for di, (cls, x, y) in enumerate(detections_xy):
            if di in assigned_dets:
                continue
            too_close = False
            for tr in self.tracks:
                tx, ty = tr.kf.get_position()
                if math.hypot(x - tx, y - ty) < self.spawn_suppression_dist:
                    too_close = True
                    break
            if not too_close:
                self._spawn_track(cls, x, y, now)

    def _color_for_class(self, cls: str) -> Tuple[float, float, float, float]:
        lname = cls.lower()
        if "red" in lname:
            return (1.0, 0.0, 0.0, 0.95)
        if "green" in lname:
            return (0.0, 1.0, 0.0, 0.95)
        if "yellow" in lname:
            return (1.0, 1.0, 0.0, 0.95)
        if "black" in lname:
            return (0.05, 0.05, 0.05, 0.98)
        return (0.2, 0.8, 1.0, 0.95)

    # ---- persistent marker creation (once per confirmed track) ----
    def _make_persistent_marker_for_track(self, stamp_msg, tr: Track) -> Marker:
        x, y = tr.kf.get_position()

        m = Marker()
        m.header.frame_id = self._grid_frame()
        m.header.stamp = stamp_msg
        m.ns = "map_hits_tracked_persistent"
        m.id = int(self._next_marker_id)
        self._next_marker_id += 1

        m.type = Marker.SPHERE
        m.action = Marker.ADD

        m.pose.position.x = float(x)
        m.pose.position.y = float(y)
        m.pose.position.z = float(self.marker_z)
        m.pose.orientation.w = 1.0

        # ✅ radius=3m => diameter=6m
        diam = 1.0
        m.scale.x = diam
        m.scale.y = diam
        m.scale.z = 0.5

        r, g, b, a = tr.color
        m.color.r, m.color.g, m.color.b, m.color.a = float(r), float(g), float(b), float(a)

        # forever
        m.lifetime = Duration(sec=0, nanosec=0)

        tr.marker_id = m.id
        tr.published_persistent = True
        return m

    # ---- vehicle marker ----
    def _publish_vehicle_marker(self, stamp_msg):
        if not self.have_pose:
            return

        lifetime = Duration(
            sec=int(self.vehicle_marker_lifetime),
            nanosec=int((self.vehicle_marker_lifetime % 1.0) * 1e9),
        )

        m = Marker()
        m.header.frame_id = self._grid_frame()
        m.header.stamp = stamp_msg
        m.ns = "vehicle_pose"
        m.id = 0
        m.type = Marker.ARROW
        m.action = Marker.ADD

        m.pose.position.x = float(self.x_usv)
        m.pose.position.y = float(self.y_usv)
        m.pose.position.z = 0.2

        yaw = float(self.psi_usv)
        m.pose.orientation.z = math.sin(0.5 * yaw)
        m.pose.orientation.w = math.cos(0.5 * yaw)

        m.scale.x = 0.25
        m.scale.y = 0.45
        m.scale.z = 0.8

        m.color.r = 1.0
        m.color.g = 0.5
        m.color.b = 0.0
        m.color.a = 0.95

        m.lifetime = lifetime
        self.vehicle_marker_pub.publish(m)

    # ---- main loop ----
    def _tick(self):
        stamp_msg = self.get_clock().now().to_msg()
        now = self._now()

        # predict tracks
        for tr in self.tracks:
            tr.kf.predict()

        if not self.have_pose:
            self._warn_throttle(2.0, "No /vehicle_pose yet; skipping.")
            self._publish_vehicle_marker(stamp_msg)
            return

        if not self.have_caminfo:
            self._warn_throttle(2.0, "No CameraInfo yet; bearings wrong without intrinsics.")
            self._publish_vehicle_marker(stamp_msg)
            return

        if self.map_msg is None:
            self._warn_throttle(2.0, "No occupancy grid yet; cannot raycast.")
            self._publish_vehicle_marker(stamp_msg)
            return

        if not self._cam:
            self._publish_vehicle_marker(stamp_msg)
            return

        do_print = self.print_angles and ((now - self._last_print_t) >= self.print_period_s)

        # collect detections as map hits for KF update
        dets_xy: List[Tuple[str, float, float]] = []

        for d in self._cam:
            a_cam = self._bearing_from_px(d.xc)
            if not math.isfinite(a_cam):
                continue

            hit, (hx, hy), hit_rng = self._raycast_to_map_hit(a_cam)
            if not hit:
                if do_print:
                    self.get_logger().info(
                        f"[CMP] cls={d.cls} conf={d.conf:.2f} "
                        f"cam_rel={math.degrees(a_cam):+.2f}deg "
                        f"NO_HIT up_to={hit_rng:.1f}m"
                    )
                continue

            a_map = self._bearing_from_map_hit(hx, hy)
            diff = wrap_angle(a_map - a_cam)

            if do_print:
                self.get_logger().info(
                    f"[CMP] cls={d.cls} conf={d.conf:.2f} "
                    f"cam_rel={math.degrees(a_cam):+.2f}deg "
                    f"map_rel={math.degrees(a_map):+.2f}deg "
                    f"diff(map-cam)={math.degrees(diff):+.2f}deg "
                    f"hit=({hx:+.2f},{hy:+.2f}) rng={hit_rng:.2f}m"
                )

            dets_xy.append((d.cls, float(hx), float(hy)))

        if do_print:
            self._last_print_t = now

        # KF update
        self._associate_and_update(dets_xy, now)

        # timeout pruning (optional)
        if self.timeout > 0.0:
            self.tracks = [tr for tr in self.tracks if (now - tr.last_update) <= self.timeout]

        # publish confirmed tracks + publish new persistent markers
        out = ObjectPosition()
        new_markers = MarkerArray()

        for tr in self.tracks:
            if tr.hits < self.min_hits:
                continue

            x, y = tr.kf.get_position()

            out.object_names.append(f"{tr.cls}_T{tr.id}")
            out.x_object.append(float(x))
            out.y_object.append(float(y))
            out.z_object.append(0.0)
            out.radii_object.append(float(self.hit_radius_m))  # ✅ radius 3m

            # persistent marker published once
            if not tr.published_persistent:
                new_markers.markers.append(self._make_persistent_marker_for_track(stamp_msg, tr))

        # publish outputs
        self.pub_tracked.publish(out)

        if new_markers.markers:
            self.marker_pub.publish(new_markers)

        self._publish_vehicle_marker(stamp_msg)


def main(args=None):
    rclpy.init(args=args)
    node = CameraMapAngleCompareKF()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()