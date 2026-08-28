#!/usr/bin/env python3
from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from collections import defaultdict, deque

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose, Pose2D
from builtin_interfaces.msg import Duration
from visualization_msgs.msg import Marker, MarkerArray
from fau_msgs.msg import ObjectPosition


# ============================================================
# Helpers
# ============================================================

def wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def body_to_global_ned(
    xb: float, yb: float, zb: float,
    x_usv: float, y_usv: float, psi_usv: float
) -> Tuple[float, float, float]:
    """
    Exact convention matching your current mapping code:
      x_body = xb
      y_body = -yb
      dN = c*x_body - s*y_body = c*xb + s*yb
      dE = s*x_body + c*y_body = s*xb - c*yb
    """
    c = math.cos(float(psi_usv))
    s = math.sin(float(psi_usv))

    dN = c * float(xb) + s * float(yb)
    dE = s * float(xb) - c * float(yb)

    return float(x_usv) + dN, float(y_usv) + dE, float(zb)


def global_to_body_measurement(
    mx: float, my: float,
    x_usv: float, y_usv: float, psi_usv: float
) -> np.ndarray:
    """
    Inverse of the convention above.
    Given landmark in global/map frame, predict the measurement [xb, yb].

    From:
      dN = mx - x_usv
      dE = my - y_usv

      xb = c*dN + s*dE
      yb = s*dN - c*dE
    """
    dN = float(mx) - float(x_usv)
    dE = float(my) - float(y_usv)

    c = math.cos(float(psi_usv))
    s = math.sin(float(psi_usv))

    xb = c * dN + s * dE
    yb = s * dN - c * dE

    return np.array([[xb], [yb]], dtype=float)


def measurement_jacobian_wrt_landmark(psi_usv: float) -> np.ndarray:
    """
    z = h(m) = [xb, yb]
    xb = c*(mx-x) + s*(my-y)
    yb = s*(mx-x) - c*(my-y)

    H = dz/dm
      = [[ c,  s],
         [ s, -c]]
    """
    c = math.cos(float(psi_usv))
    s = math.sin(float(psi_usv))
    return np.array([[c, s],
                     [s, -c]], dtype=float)


# ============================================================
# Landmark track
# ============================================================

@dataclass
class LandmarkTrack:
    track_id: int
    class_name: str
    mean: np.ndarray                  # shape (2,1) => [x, y]
    cov: np.ndarray                   # shape (2,2)
    z: float
    radius: float
    last_update: float
    hits: int = 1
    age: int = 1
    class_votes: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def get_position(self) -> Tuple[float, float]:
        return float(self.mean[0, 0]), float(self.mean[1, 0])


# ============================================================
# Landmark mapper
# ============================================================

class BuoyLandmarkMapper(Node):
    """
    Better mapping than the current 'transform-then-smooth' approach:

    - keeps landmarks in map frame
    - updates each landmark with body-frame measurement using EKF
    - uses Mahalanobis gating
    - range-dependent measurement covariance
    - same style of ROS I/O as your existing node
    """

    def __init__(self):
        super().__init__("vision_tracker")

        # ---------------- Parameters ----------------
        self.declare_parameter("wamv", "asv")
        self.declare_parameter("objects_topic", "vision/output/buoy_objects")
        self.declare_parameter("pose_topic", "vehicle_pose")

        self.declare_parameter("dt", 0.1)
        self.declare_parameter("timeout", 0.0)
        self.declare_parameter("min_hits", 1)

        self.declare_parameter("spawn_mahalanobis_gate", 16.0)
        self.declare_parameter("merge_dist", 0.7)

        self.declare_parameter("radius_alpha", 0.10)
        self.declare_parameter("inflate_extra_m", 0.0)

        self.declare_parameter("pose_buffer_size", 500)
        self.declare_parameter("max_pose_age_for_match", 0.05)

        self.declare_parameter("process_noise_xy", 0.005)

        self.declare_parameter("meas_sigma_near", 0.35)
        self.declare_parameter("meas_sigma_range_k", 0.06)

        self.declare_parameter("pose_sigma_xy", 0.30)
        self.declare_parameter("pose_sigma_yaw_deg", 6.0)

        self.declare_parameter("out_topic", "vision/output/buoy_tracked_global")
        self.declare_parameter("map_topic", "map/local_occupancy_2")
        self.declare_parameter("marker_topic", "viz/buoy_tracks")
        self.declare_parameter("vehicle_marker_topic", "viz/vehicle_pose")
        self.declare_parameter("marker_lifetime", 0.6)

        self.declare_parameter("grid_frame", "map")
        self.declare_parameter("grid_resolution", 0.2)
        self.declare_parameter("grid_width_m", 500.0)
        self.declare_parameter("grid_height_m", 500.0)
        self.declare_parameter("grid_unknown_by_default", False)

        self.declare_parameter("vehicle_marker_type", "arrow")
        self.declare_parameter("vehicle_scale_xy", 1.2)
        self.declare_parameter("vehicle_scale_z", 0.3)

        # ---------------- Read params ----------------
        self.wamv = str(self.get_parameter("wamv").value)
        self.objects_topic = str(self.get_parameter("objects_topic").value)
        self.pose_topic = str(self.get_parameter("pose_topic").value)

        self.dt = float(self.get_parameter("dt").value)
        self.timeout = float(self.get_parameter("timeout").value)
        self.min_hits = int(self.get_parameter("min_hits").value)

        self.spawn_mahalanobis_gate = float(self.get_parameter("spawn_mahalanobis_gate").value)
        self.merge_dist = float(self.get_parameter("merge_dist").value)

        self.radius_alpha = float(self.get_parameter("radius_alpha").value)
        self.inflate_extra_m = float(self.get_parameter("inflate_extra_m").value)

        self.pose_buffer_size = int(self.get_parameter("pose_buffer_size").value)
        self.max_pose_age_for_match = float(self.get_parameter("max_pose_age_for_match").value)

        self.process_noise_xy = float(self.get_parameter("process_noise_xy").value)

        self.meas_sigma_near = float(self.get_parameter("meas_sigma_near").value)
        self.meas_sigma_range_k = float(self.get_parameter("meas_sigma_range_k").value)

        self.pose_sigma_xy = float(self.get_parameter("pose_sigma_xy").value)
        self.pose_sigma_yaw = math.radians(float(self.get_parameter("pose_sigma_yaw_deg").value))

        self.out_topic = str(self.get_parameter("out_topic").value)
        self.map_topic = str(self.get_parameter("map_topic").value)
        self.marker_topic = str(self.get_parameter("marker_topic").value)
        self.vehicle_marker_topic = str(self.get_parameter("vehicle_marker_topic").value)
        self.marker_lifetime = float(self.get_parameter("marker_lifetime").value)

        self.grid_frame = str(self.get_parameter("grid_frame").value)
        self.grid_resolution = float(self.get_parameter("grid_resolution").value)
        self.grid_width_m = float(self.get_parameter("grid_width_m").value)
        self.grid_height_m = float(self.get_parameter("grid_height_m").value)
        self.grid_unknown_by_default = bool(self.get_parameter("grid_unknown_by_default").value)

        self.vehicle_marker_type = str(self.get_parameter("vehicle_marker_type").value).lower().strip()
        self.vehicle_scale_xy = float(self.get_parameter("vehicle_scale_xy").value)
        self.vehicle_scale_z = float(self.get_parameter("vehicle_scale_z").value)

        # ---------------- State ----------------
        self.tracks: List[LandmarkTrack] = []
        self.next_id = 0

        self.have_pose = False
        self.x_usv = 0.0
        self.y_usv = 0.0
        self.psi_usv = 0.0

        # store pose as (t_recv, x, y, yaw)
        self.pose_buffer = deque(maxlen=self.pose_buffer_size)

        # ---------------- QoS ----------------
        objects_qos = QoSProfile(depth=10)
        objects_qos.reliability = ReliabilityPolicy.RELIABLE
        objects_qos.durability = DurabilityPolicy.VOLATILE

        pose_qos = QoSProfile(depth=10)
        pose_qos.reliability = ReliabilityPolicy.RELIABLE
        pose_qos.durability = DurabilityPolicy.VOLATILE

        # ---------------- Subs ----------------
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

        # ---------------- Pubs ----------------
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
            "Buoy landmark mapper started\n"
            f"sub objects=/{self.wamv}/{self.objects_topic}\n"
            f"sub pose=/{self.wamv}/{self.pose_topic}\n"
            f"pub tracked=/{self.wamv}/{self.out_topic}\n"
            f"pub grid=/{self.wamv}/{self.map_topic}\n"
            f"max_pose_age_for_match={self.max_pose_age_for_match:.3f}s"
        )

    # ========================================================
    # Time helpers
    # ========================================================

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _warn_throttle(self, period_s: float, msg: str):
        now = self._now()
        if not hasattr(self, "_last_warn_t"):
            self._last_warn_t = {}
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

    # ========================================================
    # Pose callback
    # ========================================================

    def _pose_cb(self, msg: Pose2D):
        t_recv = self._now()

        self.x_usv = float(msg.x)
        self.y_usv = float(msg.y)
        self.psi_usv = float(msg.theta)
        self.have_pose = True

        self.pose_buffer.append((t_recv, self.x_usv, self.y_usv, self.psi_usv))

    # ========================================================
    # Covariance helpers
    # ========================================================

    def _measurement_cov_body(self, xb: float, yb: float) -> np.ndarray:
        rng = math.hypot(float(xb), float(yb))
        sigma = self.meas_sigma_near + self.meas_sigma_range_k * rng
        var = sigma * sigma
        return np.array([[var, 0.0],
                         [0.0, var]], dtype=float)

    def _spawn_covariance_global(
        self, xb: float, yb: float, psi: float
    ) -> np.ndarray:
        """
        Initial global covariance for a new landmark from:
        - body measurement noise
        - rough pose xy uncertainty
        - rough heading uncertainty
        """
        Rb = self._measurement_cov_body(xb, yb)

        c = math.cos(psi)
        s = math.sin(psi)

        # Jacobian from [xb, yb] to [xg, yg]
        Jz = np.array([[c, s],
                       [s, -c]], dtype=float)

        # Rotate measurement covariance into global
        Pg_meas = Jz @ Rb @ Jz.T

        # Pose position uncertainty
        Pg_pose_xy = np.array([[self.pose_sigma_xy ** 2, 0.0],
                               [0.0, self.pose_sigma_xy ** 2]], dtype=float)

        # Yaw uncertainty contribution
        # xg = x + c*xb + s*yb
        # yg = y + s*xb - c*yb
        # d/dpsi = [-s*xb + c*yb,  c*xb + s*yb]^T
        dpsi = np.array([[-s * xb + c * yb],
                         [ c * xb + s * yb]], dtype=float)
        Pg_yaw = dpsi @ dpsi.T * (self.pose_sigma_yaw ** 2)

        return Pg_meas + Pg_pose_xy + Pg_yaw + np.eye(2) * 1e-6

    # ========================================================
    # Object callback
    # ========================================================

    def _objects_cb(self, msg: ObjectPosition):
        if not self.have_pose:
            self._warn_throttle(2.0, "No /vehicle_pose yet; skipping detections.")
            return

        n = min(len(msg.object_names),
                len(msg.x_object),
                len(msg.y_object),
                len(msg.z_object),
                len(msg.radii_object))
        if n <= 0:
            return

        t_det = self._now()
        pose = self._get_pose_nearest(t_det)
        if pose is None:
            self._warn_throttle(2.0, "No sufficiently close pose by callback time; skipping detections.")
            return

        _, x_usv, y_usv, psi_usv = pose

        detections = []
        for i in range(n):
            name = str(msg.object_names[i])
            xb = float(msg.x_object[i])
            yb = float(msg.y_object[i])
            zb = float(msg.z_object[i])
            r = max(0.1, float(msg.radii_object[i]))

            if not (math.isfinite(xb) and math.isfinite(yb) and math.isfinite(zb) and math.isfinite(r)):
                continue

            detections.append((name, xb, yb, zb, r, x_usv, y_usv, psi_usv))

        if detections:
            self._associate_and_update(detections, self._now())

    # ========================================================
    # EKF update per landmark
    # ========================================================

    def _ekf_update_landmark(
        self,
        tr: LandmarkTrack,
        xb_meas: float,
        yb_meas: float,
        x_usv: float,
        y_usv: float,
        psi_usv: float
    ) -> Tuple[float, np.ndarray]:
        """
        Returns:
          d2: Mahalanobis distance squared
          innovation_cov: S
        """
        z = np.array([[xb_meas], [yb_meas]], dtype=float)
        zhat = global_to_body_measurement(
            tr.mean[0, 0], tr.mean[1, 0],
            x_usv, y_usv, psi_usv
        )

        H = measurement_jacobian_wrt_landmark(psi_usv)
        R = self._measurement_cov_body(xb_meas, yb_meas)

        y = z - zhat
        S = H @ tr.cov @ H.T + R

        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S = S + np.eye(2) * 1e-6
            S_inv = np.linalg.inv(S)

        d2 = float(y.T @ S_inv @ y)

        # EKF correction
        K = tr.cov @ H.T @ S_inv
        tr.mean = tr.mean + K @ y
        I = np.eye(2, dtype=float)
        tr.cov = (I - K @ H) @ tr.cov

        return d2, S

    # ========================================================
    # Association / spawn / merge
    # ========================================================

    def _spawn_track(
        self,
        name: str,
        xb: float,
        yb: float,
        zb: float,
        radius: float,
        x_usv: float,
        y_usv: float,
        psi_usv: float,
        now: float
    ):
        xg, yg, _ = body_to_global_ned(xb, yb, zb, x_usv, y_usv, psi_usv)
        mean = np.array([[xg], [yg]], dtype=float)
        cov = self._spawn_covariance_global(xb, yb, psi_usv)

        tr = LandmarkTrack(
            track_id=self.next_id,
            class_name=name,
            mean=mean,
            cov=cov,
            z=float(zb),
            radius=float(radius),
            last_update=float(now),
        )
        tr.class_votes[name] += 1
        self.tracks.append(tr)
        self.next_id += 1

    def _associate_and_update(self, detections, now: float):
        """
        detections item:
          (name, xb, yb, zb, r, x_usv, y_usv, psi_usv)
        """
        if len(self.tracks) == 0:
            for det in detections:
                self._spawn_track(*det, now)
            return

        candidates = []

        # Evaluate all possible associations using Mahalanobis distance
        for ti, tr in enumerate(self.tracks):
            for di, det in enumerate(detections):
                name, xb, yb, zb, r, x_usv, y_usv, psi_usv = det

                zhat = global_to_body_measurement(
                    tr.mean[0, 0], tr.mean[1, 0],
                    x_usv, y_usv, psi_usv
                )
                H = measurement_jacobian_wrt_landmark(psi_usv)
                R = self._measurement_cov_body(xb, yb)
                z = np.array([[xb], [yb]], dtype=float)
                y = z - zhat
                S = H @ tr.cov @ H.T + R

                try:
                    S_inv = np.linalg.inv(S)
                except np.linalg.LinAlgError:
                    S = S + np.eye(2) * 1e-6
                    S_inv = np.linalg.inv(S)

                d2 = float(y.T @ S_inv @ y)
                candidates.append((d2, ti, di))

        candidates.sort(key=lambda x: x[0])

        assigned_tracks = set()
        assigned_dets = set()

        for d2, ti, di in candidates:
            if d2 > self.spawn_mahalanobis_gate:
                continue
            if ti in assigned_tracks or di in assigned_dets:
                continue

            tr = self.tracks[ti]
            name, xb, yb, zb, r, x_usv, y_usv, psi_usv = detections[di]

            self._ekf_update_landmark(tr, xb, yb, x_usv, y_usv, psi_usv)

            tr.z = float(zb)
            tr.last_update = float(now)
            tr.hits += 1
            tr.age += 1

            tr.class_votes[name] += 1
            tr.class_name = max(tr.class_votes, key=tr.class_votes.get)

            a = max(0.0, min(1.0, self.radius_alpha))
            tr.radius = (1.0 - a) * tr.radius + a * float(r)

            assigned_tracks.add(ti)
            assigned_dets.add(di)

        # Spawn unmatched detections as new landmarks
        for di, det in enumerate(detections):
            if di in assigned_dets:
                continue
            self._spawn_track(*det, now)

    def _merge_close_tracks(self):
        if self.merge_dist <= 0.0 or len(self.tracks) < 2:
            return

        changed = True
        while changed:
            changed = False
            n = len(self.tracks)
            for i in range(n):
                if i >= len(self.tracks):
                    break
                ti = self.tracks[i]
                xi, yi = ti.get_position()

                for j in range(i + 1, len(self.tracks)):
                    tj = self.tracks[j]
                    xj, yj = tj.get_position()

                    if math.hypot(xi - xj, yi - yj) > self.merge_dist:
                        continue

                    # keep the more reliable one
                    keep_i = True
                    if tj.hits > ti.hits:
                        keep_i = False
                    elif tj.hits == ti.hits and np.trace(tj.cov) < np.trace(ti.cov):
                        keep_i = False

                    keeper = ti if keep_i else tj
                    removed = tj if keep_i else ti

                    # covariance-weighted fusion
                    try:
                        Pi_inv = np.linalg.inv(keeper.cov)
                        Pj_inv = np.linalg.inv(removed.cov)
                        Pf = np.linalg.inv(Pi_inv + Pj_inv)
                        mf = Pf @ (Pi_inv @ keeper.mean + Pj_inv @ removed.mean)
                        keeper.mean = mf
                        keeper.cov = Pf
                    except np.linalg.LinAlgError:
                        keeper.mean = 0.5 * (keeper.mean + removed.mean)
                        keeper.cov = keeper.cov + removed.cov

                    keeper.z = 0.5 * (keeper.z + removed.z)
                    keeper.radius = 0.5 * (keeper.radius + removed.radius)
                    keeper.hits = max(keeper.hits, removed.hits)
                    keeper.last_update = max(keeper.last_update, removed.last_update)

                    for k, v in removed.class_votes.items():
                        keeper.class_votes[k] += v
                    keeper.class_name = max(keeper.class_votes, key=keeper.class_votes.get)

                    self.tracks.remove(removed)
                    changed = True
                    break

                if changed:
                    break

    # ========================================================
    # Grid helpers
    # ========================================================

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

    # ========================================================
    # Marker publishing
    # ========================================================

    def _publish_markers(self, stamp_msg):
        ma = MarkerArray()
        lifetime = Duration(
            sec=int(self.marker_lifetime),
            nanosec=int((self.marker_lifetime % 1.0) * 1e9)
        )

        for tr in self.tracks:
            if tr.hits < self.min_hits:
                continue

            x, y = tr.get_position()

            m = Marker()
            m.header.frame_id = self.grid_frame
            m.header.stamp = stamp_msg
            m.ns = "buoy_tracks"
            m.id = tr.track_id
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

            # more opaque when covariance is smaller
            tr_cov = float(np.trace(tr.cov))
            alpha = 0.35 if tr_cov > 4.0 else 0.9
            m.color.r, m.color.g, m.color.b, m.color.a = 0.2, 0.8, 1.0, alpha
            m.lifetime = lifetime
            ma.markers.append(m)

        self.marker_pub.publish(ma)

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

    # ========================================================
    # Timer tick
    # ========================================================

    def _timer_tick(self):
        stamp_msg = self.get_clock().now().to_msg()
        now = self._now()

        # Static landmarks: only covariance inflation, no motion
        Q = np.eye(2, dtype=float) * self.process_noise_xy
        for tr in self.tracks:
            tr.cov = tr.cov + Q
            tr.age += 1

        if self.timeout > 0.0:
            self.tracks = [tr for tr in self.tracks if (now - tr.last_update) <= self.timeout]

        self._merge_close_tracks()

        # Publish tracked global objects
        out = ObjectPosition()
        for tr in self.tracks:
            if tr.hits < self.min_hits:
                continue
            x, y = tr.get_position()
            out.object_names.append(f"{tr.class_name}_T{tr.track_id}")
            out.x_object.append(float(x))
            out.y_object.append(float(y))
            out.z_object.append(float(tr.z))
            out.radii_object.append(float(tr.radius))

        self.pub_tracked.publish(out)

        # Publish occupancy grid
        grid = self._make_empty_grid()
        grid.header.stamp = stamp_msg
        for tr in self.tracks:
            if tr.hits < self.min_hits:
                continue
            x, y = tr.get_position()
            inflate = tr.radius + self.inflate_extra_m

            # Optionally enlarge disk slightly when uncertain
            unc = float(np.trace(tr.cov))
            inflate += min(1.0, 0.15 * math.sqrt(max(0.0, unc)))

            self._paint_disk(grid, x, y, inflate)

        self.grid_pub.publish(grid)
        self._publish_markers(stamp_msg)
        self._publish_vehicle_marker(stamp_msg)


def main(args=None):
    rclpy.init(args=args)
    node = BuoyLandmarkMapper()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()