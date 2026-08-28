#!/usr/bin/env python3
from __future__ import annotations

import math
import numpy as np
from collections import defaultdict, deque
from typing import List, Tuple, Dict, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose, Pose2D
from builtin_interfaces.msg import Duration
from visualization_msgs.msg import Marker, MarkerArray

from fau_msgs.msg import ObjectPosition


# ---------------- KF ----------------
class KalmanTracker2D:
    """Static position-only KF (random walk): state [x,y]."""
    def __init__(self, dt=0.1):
        self.dt = float(dt)
        self.x = np.zeros((2, 1), dtype=float)

        self.F = np.eye(2, dtype=float)
        self.H = np.eye(2, dtype=float)

        self.R = np.eye(2, dtype=float) * 6.0
        self.P = np.eye(2, dtype=float) * 10.0
        self.Q = np.eye(2, dtype=float) * 0.01

        self.initialized = False

    def init_from_measurement(self, x, y):
        self.x[0, 0] = float(x)
        self.x[1, 0] = float(y)
        self.initialized = True

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(2, dtype=float)
        self.P = (I - K @ self.H) @ self.P

    def get_position(self):
        return float(self.x[0, 0]), float(self.x[1, 0])


# ---------------- Tracks ----------------
class Track:
    def __init__(self, track_id: int, class_name: str, x: float, y: float, z: float,
                 radius: float, t: float, dt: float):
        self.id = track_id
        self.class_name = class_name

        self.kf = KalmanTracker2D(dt=dt)
        self.kf.init_from_measurement(x, y)

        self.z = float(z)
        self.radius = float(radius)

        self.last_update = float(t)
        self.hits = 1

        self.class_votes = defaultdict(int)
        self.class_votes[class_name] += 1


def body_to_global_ned(
    xb: float, yb: float, zb: float,
    x_usv: float, y_usv: float, psi_usv: float
) -> Tuple[float, float, float]:
    """
    EXACT match to your C++:
      x_body = xb
      y_body = -yb
      dN = c*x_body - s*y_body
      dE = s*x_body + c*y_body
      xg = x_usv + dN
      yg = y_usv + dE
      zg = zb
    """
    x_body = float(xb)
    y_body = -float(yb)

    c = math.cos(float(psi_usv))
    s = math.sin(float(psi_usv))

    dN = c * x_body - s * y_body
    dE = s * x_body + c * y_body

    return (float(x_usv) + dN, float(y_usv) + dE, float(zb))


class BuoyObjectsKFTracker(Node):
    def __init__(self):
        super().__init__("vision_tracker")

        # ------------ params ------------
        self.declare_parameter("wamv", "wamv")
        self.declare_parameter("objects_topic", "vision/output/buoy_objects")
        self.declare_parameter("pose_topic", "vehicle_pose")

        self.declare_parameter("dt", 0.1)
        self.declare_parameter("gate_dist", 0.5)
        self.declare_parameter("timeout", 0.0)
        self.declare_parameter("min_hits", 2)

        self.declare_parameter("radius_alpha", 0.1)
        self.declare_parameter("inflate_extra_m", 0.0)

        self.declare_parameter("spawn_suppression_dist", 2.0)
        self.declare_parameter("track_merge_dist", 2.5)

        self.declare_parameter("grid_frame", "map")
        self.declare_parameter("grid_resolution", 0.2)
        self.declare_parameter("grid_width_m", 500.0)
        self.declare_parameter("grid_height_m", 500.0)
        self.declare_parameter("grid_unknown_by_default", False)

        self.declare_parameter("out_topic", "vision/output/buoy_tracked_global")
        self.declare_parameter("map_topic", "map/local_occupancy_2")
        self.declare_parameter("marker_topic", "viz/buoy_tracks")
        self.declare_parameter("vehicle_marker_topic", "viz/vehicle_pose")
        self.declare_parameter("marker_lifetime", 0.6)

        self.declare_parameter("vehicle_marker_type", "arrow")
        self.declare_parameter("vehicle_scale_xy", 1.2)
        self.declare_parameter("vehicle_scale_z", 0.3)

        # Approximate sync using callback receipt time
        self.declare_parameter("pose_buffer_size", 500)
        self.declare_parameter("max_pose_age_for_match", 2.0)

        # ------------ read ------------
        self.wamv = str(self.get_parameter("wamv").value)
        self.objects_topic = str(self.get_parameter("objects_topic").value)
        self.pose_topic = str(self.get_parameter("pose_topic").value)

        self.dt = float(self.get_parameter("dt").value)
        self.gate_dist = float(self.get_parameter("gate_dist").value)
        self.timeout = float(self.get_parameter("timeout").value)
        self.min_hits = int(self.get_parameter("min_hits").value)

        self.radius_alpha = float(self.get_parameter("radius_alpha").value)
        self.inflate_extra_m = float(self.get_parameter("inflate_extra_m").value)

        self.spawn_suppression_dist = float(self.get_parameter("spawn_suppression_dist").value)
        self.track_merge_dist = float(self.get_parameter("track_merge_dist").value)

        self.grid_frame = str(self.get_parameter("grid_frame").value)
        self.grid_resolution = float(self.get_parameter("grid_resolution").value)
        self.grid_width_m = float(self.get_parameter("grid_width_m").value)
        self.grid_height_m = float(self.get_parameter("grid_height_m").value)
        self.grid_unknown_by_default = bool(self.get_parameter("grid_unknown_by_default").value)

        self.out_topic = str(self.get_parameter("out_topic").value)
        self.map_topic = str(self.get_parameter("map_topic").value)
        self.marker_topic = str(self.get_parameter("marker_topic").value)
        self.vehicle_marker_topic = str(self.get_parameter("vehicle_marker_topic").value)
        self.marker_lifetime = float(self.get_parameter("marker_lifetime").value)

        self.vehicle_marker_type = str(self.get_parameter("vehicle_marker_type").value).lower().strip()
        self.vehicle_scale_xy = float(self.get_parameter("vehicle_scale_xy").value)
        self.vehicle_scale_z = float(self.get_parameter("vehicle_scale_z").value)

        self.pose_buffer_size = int(self.get_parameter("pose_buffer_size").value)
        self.max_pose_age_for_match = float(self.get_parameter("max_pose_age_for_match").value)

        # ------------ state ------------
        self.tracks: List[Track] = []
        self.next_id = 0

        self.have_pose = False
        self.x_usv = 0.0
        self.y_usv = 0.0
        self.psi_usv = 0.0

        # store pose as (receipt_time, x, y, yaw)
        self.pose_buffer = deque(maxlen=self.pose_buffer_size)

        # ------------ QoS ------------
        objects_qos = QoSProfile(depth=10)
        objects_qos.reliability = ReliabilityPolicy.RELIABLE
        objects_qos.durability = DurabilityPolicy.VOLATILE

        pose_qos = QoSProfile(depth=10)
        pose_qos.reliability = ReliabilityPolicy.RELIABLE
        pose_qos.durability = DurabilityPolicy.VOLATILE

        # ------------ subs ------------
        self.objects_sub = self.create_subscription(
            ObjectPosition,
            f"/{self.wamv}/{self.objects_topic}".replace("//", "/"),
            self._objects_cb,
            objects_qos
        )

        self.pose_sub = self.create_subscription(
            Pose2D,
            f"/{self.wamv}/{self.pose_topic}".replace("//", "/"),
            self._pose_cb,
            pose_qos
        )

        # ------------ pubs ------------
        self.pub_tracked = self.create_publisher(
            ObjectPosition,
            f"/{self.wamv}/{self.out_topic}".replace("//", "/"),
            10
        )

        grid_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        self.grid_pub = self.create_publisher(
            OccupancyGrid,
            f"/{self.wamv}/{self.map_topic}".replace("//", "/"),
            grid_qos
        )

        self.marker_pub = self.create_publisher(
            MarkerArray,
            f"/{self.wamv}/{self.marker_topic}".replace("//", "/"),
            1
        )

        self.vehicle_marker_pub = self.create_publisher(
            Marker,
            f"/{self.wamv}/{self.vehicle_marker_topic}".replace("//", "/"),
            1
        )

        self.timer = self.create_timer(self.dt, self._timer_tick)

        self.get_logger().info(
            "KF tracker using callback-time approximate sync\n"
            f"sub objects=/{self.wamv}/{self.objects_topic} | sub pose=/{self.wamv}/{self.pose_topic}\n"
            f"pub tracked=/{self.wamv}/{self.out_topic}\n"
            f"pub grid=/{self.wamv}/{self.map_topic}\n"
            f"gate_dist={self.gate_dist:.2f} min_hits={self.min_hits} "
            f"spawn_suppression_dist={self.spawn_suppression_dist:.2f} "
            f"track_merge_dist={self.track_merge_dist:.2f} "
            f"max_pose_age_for_match={self.max_pose_age_for_match:.2f}"
        )

    # ------------ time helpers ------------
    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _warn_throttle(self, period_s: float, msg: str):
        now = self._now()
        if not hasattr(self, "_last_warn_t"):
            self._last_warn_t: Dict[str, float] = {}
        last = self._last_warn_t.get(msg, 0.0)
        if (now - last) >= float(period_s):
            self.get_logger().warn(msg)
            self._last_warn_t[msg] = now

    def _get_pose_nearest(self, t_query: float) -> Optional[Tuple[float, float, float, float]]:
        if not self.pose_buffer:
            return None

        best = min(self.pose_buffer, key=lambda p: abs(p[0] - t_query))
        if abs(best[0] - t_query) > self.max_pose_age_for_match:
            return None
        return best

    # ------------ pose ------------
    def _pose_cb(self, msg: Pose2D):
        t_recv = self._now()

        self.x_usv = float(msg.x)
        self.y_usv = float(msg.y)
        self.psi_usv = float(msg.theta)
        self.have_pose = True

        self.pose_buffer.append((t_recv, self.x_usv, self.y_usv, self.psi_usv))

    # ------------ objects callback ------------
    def _objects_cb(self, msg: ObjectPosition):
        if not self.have_pose:
            self._warn_throttle(2.0, "No /vehicle_pose yet; skipping detections.")
            return

        n = min(
            len(msg.object_names),
            len(msg.x_object),
            len(msg.y_object),
            len(msg.z_object),
            len(msg.radii_object)
        )
        if n <= 0:
            return

        # Use callback receipt time as detection time proxy
        t_det = self._now()
        pose = self._get_pose_nearest(t_det)

        if pose is None:
            self._warn_throttle(2.0, "No sufficiently close pose by callback time; skipping detections.")
            return

        _, x_usv, y_usv, psi_usv = pose

        dets_global = []
        for i in range(n):
            name = str(msg.object_names[i])
            xb = float(msg.x_object[i])
            yb = float(msg.y_object[i])
            zb = float(msg.z_object[i])
            r = float(msg.radii_object[i])

            if not (math.isfinite(xb) and math.isfinite(yb) and math.isfinite(zb) and math.isfinite(r)):
                continue

            xg, yg, zg = body_to_global_ned(xb, yb, zb, x_usv, y_usv, psi_usv)
            dets_global.append((name, xg, yg, zg, max(r, 0.1)))

        if dets_global:
            self._associate_and_update(dets_global, self._now())

    # ------------ tracker maintenance (merge) ------------
    def _merge_close_tracks(self):
        if self.track_merge_dist <= 0.0:
            return

        merged = True
        while merged:
            merged = False
            for i in range(len(self.tracks)):
                if i >= len(self.tracks):
                    break
                ti = self.tracks[i]
                xi, yi = ti.kf.get_position()

                for j in range(i + 1, len(self.tracks)):
                    tj = self.tracks[j]
                    xj, yj = tj.kf.get_position()

                    if math.hypot(xi - xj, yi - yj) > self.track_merge_dist:
                        continue

                    if ti.hits != tj.hits:
                        keep_i = (ti.hits > tj.hits)
                    else:
                        keep_i = (ti.last_update >= tj.last_update)

                    keeper = ti if keep_i else tj
                    removed = tj if keep_i else ti

                    keeper.z = 0.5 * (keeper.z + removed.z)
                    keeper.radius = 0.5 * (keeper.radius + removed.radius)
                    keeper.hits = max(keeper.hits, removed.hits)
                    keeper.last_update = max(keeper.last_update, removed.last_update)

                    self.tracks.remove(removed)
                    merged = True
                    break

                if merged:
                    break

    # ------------ tracker association ------------
    def _spawn_track(self, name: str, x: float, y: float, z: float, radius: float, now: float):
        tr = Track(self.next_id, name, x, y, z, radius, now, dt=self.dt)
        self.tracks.append(tr)
        self.next_id += 1

    def _associate_and_update(self, detections_global: List[Tuple[str, float, float, float, float]], now: float):
        if not detections_global:
            return

        if len(self.tracks) == 0:
            for (name, x, y, z, r) in detections_global:
                self._spawn_track(name, x, y, z, r, now)
            return

        costs = []
        for ti, tr in enumerate(self.tracks):
            tx, ty = tr.kf.get_position()
            for di, (name, x, y, z, r) in enumerate(detections_global):
                d = math.hypot(x - tx, y - ty)
                costs.append((ti, di, d))
        costs.sort(key=lambda c: c[2])

        assigned_tracks = set()
        assigned_dets = set()

        for ti, di, d in costs:
            if d > self.gate_dist:
                break
            if ti in assigned_tracks or di in assigned_dets:
                continue

            tr = self.tracks[ti]
            name, x, y, z, r = detections_global[di]

            tr.kf.update(np.array([[x], [y]], dtype=float))
            tr.z = float(z)
            tr.last_update = now
            tr.hits += 1

            tr.class_votes[name] += 1
            tr.class_name = name

            a = max(0.0, min(1.0, self.radius_alpha))
            tr.radius = (1.0 - a) * tr.radius + a * float(r)

            assigned_tracks.add(ti)
            assigned_dets.add(di)

        for di, (name, x, y, z, r) in enumerate(detections_global):
            if di in assigned_dets:
                continue

            too_close = False
            for tr in self.tracks:
                tx, ty = tr.kf.get_position()
                if math.hypot(x - tx, y - ty) < self.spawn_suppression_dist:
                    too_close = True
                    break

            if not too_close:
                self._spawn_track(name, x, y, z, r, now)

    # ------------ occupancy grid ------------
    def _make_empty_grid(self) -> OccupancyGrid:
        res = self.grid_resolution
        w = max(1, int(math.ceil(self.grid_width_m / res)))
        h = max(1, int(math.ceil(self.grid_height_m / res)))

        grid = OccupancyGrid()
        grid.header.frame_id = self.grid_frame
        grid.info.resolution = float(res)
        grid.info.width = int(w)
        grid.info.height = int(h)

        origin = Pose()
        origin.position.x = -0.5 * self.grid_width_m
        origin.position.y = -0.5 * self.grid_height_m
        origin.position.z = 0.0
        origin.orientation.w = 1.0
        grid.info.origin = origin

        grid.data = ([-1] if self.grid_unknown_by_default else [0]) * (w * h)
        return grid

    def _world_to_grid(self, x: float, y: float, grid: OccupancyGrid):
        res = grid.info.resolution
        ox = grid.info.origin.position.x
        oy = grid.info.origin.position.y
        gx = int((x - ox) / res)
        gy = int((y - oy) / res)
        if gx < 0 or gy < 0 or gx >= grid.info.width or gy >= grid.info.height:
            return None
        return gx, gy

    def _set_cell(self, grid: OccupancyGrid, gx: int, gy: int, val: int):
        grid.data[gy * grid.info.width + gx] = int(val)

    def _paint_disk(self, grid: OccupancyGrid, x: float, y: float, radius_m: float):
        center = self._world_to_grid(x, y, grid)
        if center is None:
            return
        res = grid.info.resolution
        r_cells = max(1, int(math.ceil(radius_m / res)))
        cx, cy = center
        for dy in range(-r_cells, r_cells + 1):
            for dx in range(-r_cells, r_cells + 1):
                if dx * dx + dy * dy > r_cells * r_cells:
                    continue
                gx = cx + dx
                gy = cy + dy
                if 0 <= gx < grid.info.width and 0 <= gy < grid.info.height:
                    self._set_cell(grid, gx, gy, 100)

    # ------------ markers (buoys) ------------
    def _publish_markers(self, stamp_msg):
        ma = MarkerArray()
        lifetime = Duration(
            sec=int(self.marker_lifetime),
            nanosec=int((self.marker_lifetime % 1.0) * 1e9)
        )

        for tr in self.tracks:
            if tr.hits < self.min_hits:
                continue
            x, y = tr.kf.get_position()

            m = Marker()
            m.header.frame_id = self.grid_frame
            m.header.stamp = stamp_msg
            m.ns = "buoy_tracks"
            m.id = tr.id
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(x)
            m.pose.position.y = float(y)
            m.pose.position.z = 0.25
            m.pose.orientation.w = 1.0

            diam = max(0.2, 2.0 * (tr.radius + self.inflate_extra_m))
            m.scale.x = diam
            m.scale.y = diam
            m.scale.z = 0.5

            m.color.r, m.color.g, m.color.b, m.color.a = 0.2, 0.8, 1.0, 0.9
            m.lifetime = lifetime
            ma.markers.append(m)

        self.marker_pub.publish(ma)

    # ------------ marker (vehicle) ------------
    def _publish_vehicle_marker(self, stamp_msg):
        if not self.have_pose:
            return

        lifetime = Duration(
            sec=int(self.marker_lifetime),
            nanosec=int((self.marker_lifetime % 1.0) * 1e9)
        )

        m = Marker()
        m.header.frame_id = self.grid_frame
        m.header.stamp = stamp_msg
        m.ns = "vehicle_pose"
        m.id = 0
        m.action = Marker.ADD

        m.pose.position.x = float(self.x_usv)
        m.pose.position.y = float(self.y_usv)
        m.pose.position.z = 0.2

        yaw = float(self.psi_usv)
        m.pose.orientation.x = 0.0
        m.pose.orientation.y = 0.0
        m.pose.orientation.z = math.sin(0.5 * yaw)
        m.pose.orientation.w = math.cos(0.5 * yaw)

        if self.vehicle_marker_type == "sphere":
            m.type = Marker.SPHERE
            d = max(0.2, float(self.vehicle_scale_xy))
            m.scale.x = d
            m.scale.y = d
            m.scale.z = max(0.1, float(self.vehicle_scale_z))
        else:
            m.type = Marker.ARROW
            m.scale.x = 0.25
            m.scale.y = 0.45
            m.scale.z = 0.60

        m.color.r = 1.0
        m.color.g = 0.5
        m.color.b = 0.0
        m.color.a = 0.95

        m.lifetime = lifetime
        self.vehicle_marker_pub.publish(m)

    # ------------ timer tick ------------
    def _timer_tick(self):
        stamp_msg = self.get_clock().now().to_msg()
        now = self._now()

        for tr in self.tracks:
            tr.kf.predict()

        if self.timeout > 0.0:
            self.tracks = [tr for tr in self.tracks if (now - tr.last_update) <= self.timeout]

        self._merge_close_tracks()

        out = ObjectPosition()
        for tr in self.tracks:
            if tr.hits < self.min_hits:
                continue
            x, y = tr.kf.get_position()
            self.get_logger().info(
                f"Obstacle {tr.class_name}_T{tr.id}: x={x:.2f}, y={y:.2f}, z={tr.z:.2f}, r={tr.radius:.2f}"
            )
            out.object_names.append(f"{tr.class_name}_T{tr.id}")
            out.x_object.append(float(x))
            out.y_object.append(float(y))
            out.z_object.append(float(tr.z))
            out.radii_object.append(float(tr.radius))
        self.pub_tracked.publish(out)

        grid = self._make_empty_grid()
        grid.header.stamp = stamp_msg
        for tr in self.tracks:
            if tr.hits < self.min_hits:
                continue
            x, y = tr.kf.get_position()
            self._paint_disk(grid, x, y, tr.radius + self.inflate_extra_m)
        self.grid_pub.publish(grid)

        self._publish_markers(stamp_msg)
        self._publish_vehicle_marker(stamp_msg)


def main(args=None):
    rclpy.init(args=args)
    node = BuoyObjectsKFTracker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()