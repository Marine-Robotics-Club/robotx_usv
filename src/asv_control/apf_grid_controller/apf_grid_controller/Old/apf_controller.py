#!/usr/bin/env python3
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from geometry_msgs.msg import Pose2D, PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from visualization_msgs.msg import Marker, MarkerArray

from fau_msgs.msg import ControlEffort


# ---------- utils ----------
def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def rot_wb(psi: float) -> np.ndarray:
    """body -> world (NED/map). v_w = R_wb @ v_b"""
    c, s = math.cos(psi), math.sin(psi)
    return np.array([[c, -s],
                     [s,  c]], dtype=np.float64)


# ---------- APF in BODY frame + wall-following escape ----------
def apf_tau_wall_following(
    *,
    pos_w: np.ndarray,          # (2,) world
    psi_w: float,               # yaw in world
    goal_w: np.ndarray,         # (2,) world
    obstacles_w: np.ndarray,    # (N,2) world
    ka=1.8, kr=40.0, rho=25.0,
    obs_rad=None, r_usv=1.0, c_safe=2.5,
    Fcap=200.0,
    d_slide=8.0, w_max=0.95, k_wall=0.0,
    f_threshold=0.08,
    d_wf=12.0,
    wf_gain=1.0,
    wf_blend=0.85,
    L_eff=2.5, alpha=0.5,
):
    """
    OUTPUT IS BODY FRAME:
      tau_apf = [Tx_body, Ty_body, Mz]
    dbg:
      (f_att_body, f_rep_body, f_xy_body, wf_active, dmin)
    """
    R_wb = rot_wb(float(psi_w))
    R_bw = R_wb.T

    goal_b = R_bw @ (goal_w - pos_w)

    if obstacles_w is None or obstacles_w.size == 0:
        obs_b = np.zeros((0, 2), dtype=np.float64)
    else:
        obs_rel_w = obstacles_w - pos_w.reshape(1, 2)
        obs_b = (R_bw @ obs_rel_w.T).T

    if obs_rad is not None:
        radii = np.asarray(obs_rad, dtype=np.float64)
        if radii.shape[0] != obs_b.shape[0]:
            radii = np.full(obs_b.shape[0], 0.0, dtype=np.float64)
    else:
        radii = np.full(obs_b.shape[0], 0.0, dtype=np.float64)

    # attraction
    d_goal = float(np.linalg.norm(goal_b))
    f_att = np.zeros(2, dtype=np.float64) if d_goal < 1e-9 else ka * (goal_b / d_goal)

    # closest obstacle
    n_hat_cl = None
    dmin = np.inf
    for o_b, r_o in zip(obs_b, radii):
        r_b = -o_b
        dc = float(np.linalg.norm(r_b))
        if dc < 1e-9:
            continue
        R_eff = float(r_o + r_usv + c_safe)
        gap = dc - R_eff
        if gap < dmin:
            dmin = gap
            n_hat_cl = r_b / dc

    # repulsion
    f_rep = np.zeros(2, dtype=np.float64)
    for o_b, r_o in zip(obs_b, radii):
        r_b = -o_b
        dc = float(np.linalg.norm(r_b))
        if dc < 1e-9:
            continue
        R_eff = float(r_o + r_usv + c_safe)
        d = dc - R_eff
        if d <= 0.0:
            f_rep += (kr * 1e3) * (r_b / dc)
        elif d < rho:
            n_hat = r_b / dc
            term = (1.0 / d - 1.0 / rho) * (1.0 / (d * d))
            f_rep += kr * term * n_hat

    norm_rep = float(np.linalg.norm(f_rep))
    if norm_rep > 1e-9:
        f_rep = (Fcap * np.tanh(norm_rep / Fcap)) * (f_rep / norm_rep)

    # slide
    f_slide = f_att.copy()
    if n_hat_cl is not None and np.isfinite(dmin):
        if dmin <= 0.0:
            w = w_max
        elif dmin < d_slide:
            w = w_max * (1.0 - (dmin / d_slide) ** 2)
        else:
            w = 0.0

        into = float(np.dot(f_att, n_hat_cl))
        f_slide = f_att - w * max(into, 0.0) * n_hat_cl

        if k_wall > 0.0 and dmin < d_slide:
            f_slide += k_wall * (1.0 - max(dmin, 0.0) / d_slide) * n_hat_cl

    f_xy = f_slide + f_rep

    # wall-following escape
    wf_active = False
    if n_hat_cl is not None and np.isfinite(dmin):
        near_wall = (dmin < d_wf)
        stuck = (float(np.linalg.norm(f_xy)) < f_threshold)
        if near_wall and stuck:
            t1 = np.array([-n_hat_cl[1], n_hat_cl[0]], dtype=np.float64)
            t2 = -t1
            g_dir = goal_b / (np.linalg.norm(goal_b) + 1e-9)
            t_hat = t1 if float(np.dot(t1, g_dir)) >= float(np.dot(t2, g_dir)) else t2

            w_wf = max(0.0, min(1.0, 1.0 - (max(dmin, 0.0) / d_wf)))
            f_wf = wf_gain * w_wf * t_hat

            f_xy = (1.0 - wf_blend) * f_xy + wf_blend * f_wf
            wf_active = True

    # yaw moment in BODY
    n_hatB = np.array([1.0, 0.0], dtype=np.float64)
    t_hatB = np.array([0.0, 1.0], dtype=np.float64)

    f_par  = np.dot(f_xy, n_hatB) * n_hatB
    f_perp = np.dot(f_xy, t_hatB) * t_hatB

    F_bow_b   = 0.5 * f_xy + 0.5 * alpha * f_perp
    F_stern_b = 0.5 * f_xy - 0.5 * alpha * f_perp

    r_bow_b   = np.array([ L_eff, 0.0], dtype=np.float64)
    r_stern_b = np.array([-L_eff, 0.0], dtype=np.float64)

    tau_bow   = r_bow_b[0]   * F_bow_b[1]   - r_bow_b[1]   * F_bow_b[0]
    tau_stern = r_stern_b[0] * F_stern_b[1] - r_stern_b[1] * F_stern_b[0]
    tau_psi   = float(tau_bow + tau_stern)

    tau_apf = np.array([float(f_xy[0]), float(f_xy[1]), tau_psi], dtype=np.float64)
    dbg = (f_att, f_rep, f_xy, wf_active, float(dmin if np.isfinite(dmin) else 1e9))
    return tau_apf, dbg


class APF_SKC_Controller(Node):
    def __init__(self):
        super().__init__("apf_skc_controller")

        # ✅ SIM TIME
        #self.declare_parameter("use_sim_time", True)

        # ---- topics ----
        self.declare_parameter("wamv", "wamv")
        self.declare_parameter("pose_topic", "vehicle_pose")
        self.declare_parameter("odom_topic", "p3d_wamv_ned")
        self.declare_parameter("map_topic", "map/local_occupancy_2")
        self.declare_parameter("goal_topic", "nav/goal")
        self.declare_parameter("control_effort_topic", "control_effort")

        # ---- map extraction ----
        self.declare_parameter("occ_threshold", 80)
        self.declare_parameter("treat_unknown_as_occupied", False)
        self.declare_parameter("window_radius_m", 30.0)
        self.declare_parameter("max_obstacles", 2500)
        self.declare_parameter("downsample_stride", 2)

        # ---- APF params ----
        self.declare_parameter("ka", 300.0)
        self.declare_parameter("kr", 50.0)
        self.declare_parameter("rho", 20.0)
        self.declare_parameter("r_usv", 1.0)
        self.declare_parameter("c_safe", 2.5)
        self.declare_parameter("Fcap", 500.0)
        self.declare_parameter("d_slide", 8.0)
        self.declare_parameter("w_max", 0.95)
        self.declare_parameter("k_wall", 1.0)

        # ---- wall-following escape ----
        self.declare_parameter("f_threshold", 0.15)
        self.declare_parameter("d_wf", 18.0)
        self.declare_parameter("wf_gain", 3.0)
        self.declare_parameter("wf_blend", 0.95)

        # ---- yaw moment geometry ----
        self.declare_parameter("L_eff", 2.5)
        self.declare_parameter("alpha", 1.0)

        # ---- USV3DOF parameters ----
        self.declare_parameter("m", 180.0)
        self.declare_parameter("Iz", 446.0)
        self.declare_parameter("xG", 0.0)
        self.declare_parameter("yG", 0.0)

        # added mass derivatives
        self.declare_parameter("Xudot", 0.0)
        self.declare_parameter("Yvdot", 0.0)
        self.declare_parameter("Yrdot", 0.0)
        self.declare_parameter("Nvdot", 0.0)
        self.declare_parameter("Nrdot", 0.0)

        # linear damping
        self.declare_parameter("Xu", 100.0)
        self.declare_parameter("Yv", 100.0)
        self.declare_parameter("Yr", 0.0)
        self.declare_parameter("Nv", 0.0)
        self.declare_parameter("Nr", 800.0)

        # quadratic damping
        self.declare_parameter("Xuu", 150.0)
        self.declare_parameter("Yvv", 100.0)
        self.declare_parameter("Nrr", 800.0)

        # force caps
        self.declare_parameter("F_max", 1000.0)
        self.declare_parameter("Mz_max", 2000.0)

        # current (optional)
        self.declare_parameter("nu_c", [0.0, 0.0, 0.0])

        # Kd
        self.declare_parameter("Kd_diag", [20.0, 0.0, 0.0])

        # loop timing
        self.declare_parameter("dt", 0.1)

        # ---- RViz desired path from APF rollout ----
        self.declare_parameter("apf_path_num_points", 80)     # number of points to publish
        self.declare_parameter("apf_path_ds", 0.6)            # step size (m) between rollout points
        self.declare_parameter("apf_path_max_yaw_rate", 0.6)  # rad/s used only for rollout heading update (visual)

        self.apf_path_num_points = int(self.get_parameter("apf_path_num_points").value)
        self.apf_path_ds = float(self.get_parameter("apf_path_ds").value)
        self.apf_path_max_yaw_rate = float(self.get_parameter("apf_path_max_yaw_rate").value)

        # ---- RViz topics ----
        self.declare_parameter("actual_path_topic", "viz/apf_actual_path")
        self.declare_parameter("desired_path_topic", "viz/apf_desired_path")
        self.declare_parameter("goal_marker_topic", "viz/apf_goal_marker")
        self.declare_parameter("path_max_len", 2000)
        self.declare_parameter("desired_path_num_points", 50)
        self.declare_parameter("viz_frame_id", "")

        # ---- read ----
        self.wamv = str(self.get_parameter("wamv").value)
        self.pose_topic = str(self.get_parameter("pose_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.map_topic  = str(self.get_parameter("map_topic").value)
        self.goal_topic = str(self.get_parameter("goal_topic").value)
        self.ctrl_topic = str(self.get_parameter("control_effort_topic").value)

        self.occ_threshold = int(self.get_parameter("occ_threshold").value)
        self.treat_unknown_as_occupied = bool(self.get_parameter("treat_unknown_as_occupied").value)
        self.window_radius_m = float(self.get_parameter("window_radius_m").value)
        self.max_obstacles = int(self.get_parameter("max_obstacles").value)
        self.downsample_stride = max(1, int(self.get_parameter("downsample_stride").value))

        self.ka = float(self.get_parameter("ka").value)
        self.kr = float(self.get_parameter("kr").value)
        self.rho = float(self.get_parameter("rho").value)
        self.r_usv = float(self.get_parameter("r_usv").value)
        self.c_safe = float(self.get_parameter("c_safe").value)
        self.Fcap = float(self.get_parameter("Fcap").value)
        self.d_slide = float(self.get_parameter("d_slide").value)
        self.w_max = float(self.get_parameter("w_max").value)
        self.k_wall = float(self.get_parameter("k_wall").value)

        self.f_threshold = float(self.get_parameter("f_threshold").value)
        self.d_wf        = float(self.get_parameter("d_wf").value)
        self.wf_gain     = float(self.get_parameter("wf_gain").value)
        self.wf_blend    = float(self.get_parameter("wf_blend").value)

        self.L_eff = float(self.get_parameter("L_eff").value)
        self.alpha = float(self.get_parameter("alpha").value)

        self.m  = float(self.get_parameter("m").value)
        self.Iz = float(self.get_parameter("Iz").value)
        self.xG = float(self.get_parameter("xG").value)
        self.yG = float(self.get_parameter("yG").value)

        self.Xudot = float(self.get_parameter("Xudot").value)
        self.Yvdot = float(self.get_parameter("Yvdot").value)
        self.Yrdot = float(self.get_parameter("Yrdot").value)
        self.Nvdot = float(self.get_parameter("Nvdot").value)
        self.Nrdot = float(self.get_parameter("Nrdot").value)

        self.Xu  = float(self.get_parameter("Xu").value)
        self.Yv  = float(self.get_parameter("Yv").value)
        self.Yr  = float(self.get_parameter("Yr").value)
        self.Nv  = float(self.get_parameter("Nv").value)
        self.Nr  = float(self.get_parameter("Nr").value)

        self.Xuu = float(self.get_parameter("Xuu").value)
        self.Yvv = float(self.get_parameter("Yvv").value)
        self.Nrr = float(self.get_parameter("Nrr").value)

        self.F_max = float(self.get_parameter("F_max").value)
        self.Mz_max = float(self.get_parameter("Mz_max").value)

        self.nu_c = np.array(self.get_parameter("nu_c").value, dtype=np.float64).reshape(3,)
        kd = list(self.get_parameter("Kd_diag").value)
        if len(kd) != 3:
            kd = [0.0, 0.0, 0.0]
        self.Kd = np.diag(np.array(kd, dtype=np.float64))

        self.dt = float(self.get_parameter("dt").value)

        self.actual_path_topic = str(self.get_parameter("actual_path_topic").value)
        self.desired_path_topic = str(self.get_parameter("desired_path_topic").value)
        self.goal_marker_topic = str(self.get_parameter("goal_marker_topic").value)
        self.path_max_len = int(self.get_parameter("path_max_len").value)
        self.desired_path_num_points = int(self.get_parameter("desired_path_num_points").value)
        self.viz_frame_id = str(self.get_parameter("viz_frame_id").value)

        # ---- state ----
        self.have_pose = False
        self.x = 0.0
        self.y = 0.0
        self.psi = 0.0

        self.have_nu = False
        self.nu = np.zeros(3, dtype=np.float64)

        self.have_goal = False
        self.goal = np.zeros(2, dtype=np.float64)

        self.map_msg: Optional[OccupancyGrid] = None

        # ---- QoS ----
        pose_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE)
        odom_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE)
        goal_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE)
        map_qos  = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)

        # ---- subs ----
        self.create_subscription(Pose2D, f"/{self.wamv}/{self.pose_topic}".replace("//", "/"), self._pose_cb, pose_qos)
        self.create_subscription(Odometry, f"/{self.wamv}/{self.odom_topic}".replace("//", "/"), self._odom_cb, odom_qos)
        self.create_subscription(OccupancyGrid, f"/{self.wamv}/{self.map_topic}".replace("//", "/"), self._map_cb, map_qos)
        self.create_subscription(Pose2D, f"/{self.wamv}/{self.goal_topic}".replace("//", "/"), self._goal_cb, goal_qos)

        # ---- pubs ----
        self.pub_tau = self.create_publisher(ControlEffort, f"/{self.wamv}/{self.ctrl_topic}".replace("//", "/"), 10)
        self.pub_actual_path  = self.create_publisher(Path, f"/{self.wamv}/{self.actual_path_topic}".replace("//", "/"), 10)
        self.pub_desired_path = self.create_publisher(Path, f"/{self.wamv}/{self.desired_path_topic}".replace("//", "/"), 10)
        self.pub_goal_marker  = self.create_publisher(MarkerArray, f"/{self.wamv}/{self.goal_marker_topic}".replace("//", "/"), 10)

        self.actual_path_msg = Path()
        self.desired_path_msg = Path()
        self.actual_path_msg.header.frame_id = "map"
        self.desired_path_msg.header.frame_id = "map"

        self.timer = self.create_timer(self.dt, self._tick)

    def _build_desired_path_from_apf(self, now_msg, obstacles_w: np.ndarray):
        """
        Roll out the APF field forward and publish as a Path in WORLD frame (map/NED).
        This is ONLY visualization (RViz), not used for control.
        """
        self.desired_path_msg.header.stamp = now_msg
        self.desired_path_msg.poses.clear()

        # start from current state (world)
        p_w = np.array([self.x, self.y], dtype=np.float64)
        psi = float(self.psi)

        n = max(2, int(self.apf_path_num_points))
        ds = float(self.apf_path_ds)

        # safety
        if ds <= 1e-3:
            ds = 0.3

        for i in range(n):
            ps = PoseStamped()
            ps.header.stamp = now_msg
            ps.header.frame_id = self.desired_path_msg.header.frame_id
            ps.pose.position.x = float(p_w[0])
            ps.pose.position.y = float(p_w[1])
            ps.pose.position.z = 0.0
            ps.pose.orientation.w = 1.0
            self.desired_path_msg.poses.append(ps)

            # stop if we're basically at goal
            gvec = np.array([self.goal[0], self.goal[1]], dtype=np.float64) - p_w
            if float(np.linalg.norm(gvec)) < 1.0:
                break

            # compute APF at this rollout point (BODY output + dbg has f_xy_body)
            tau_apf_i, dbg = apf_tau_wall_following(
                pos_w=p_w,
                psi_w=psi,
                goal_w=np.array([self.goal[0], self.goal[1]], dtype=np.float64),
                obstacles_w=obstacles_w,
                ka=self.ka, kr=self.kr, rho=self.rho,
                r_usv=self.r_usv, c_safe=self.c_safe,
                Fcap=self.Fcap,
                d_slide=self.d_slide, w_max=self.w_max, k_wall=self.k_wall,
                f_threshold=self.f_threshold, d_wf=self.d_wf,
                wf_gain=self.wf_gain, wf_blend=self.wf_blend,
                L_eff=self.L_eff, alpha=self.alpha,
            )

            # dbg = (f_att_body, f_rep_body, f_xy_body, wf_active, dmin)
            f_xy_b = dbg[2]
            fmag = float(np.linalg.norm(f_xy_b))
            if fmag < 1e-6:
                break

            # convert the BODY force direction to WORLD direction
            R_wb = rot_wb(psi)
            dir_w = R_wb @ (f_xy_b / fmag)

            # advance position by ds along the field direction
            p_w = p_w + ds * dir_w

            # optional: update psi for prettier path curvature (visual only)
            psi_des = float(math.atan2(dir_w[1], dir_w[0]))
            e = wrap_angle(psi_des - psi)

            # limit yaw change per step using apf_path_max_yaw_rate
            dpsi_max = float(self.apf_path_max_yaw_rate) * float(self.dt)
            psi = psi + float(np.clip(e, -dpsi_max, dpsi_max))

        self.pub_desired_path.publish(self.desired_path_msg)

    # ---- callbacks ----
    def _pose_cb(self, msg: Pose2D):
        self.x = float(msg.x)
        self.y = float(msg.y)
        self.psi = float(msg.theta)
        self.have_pose = True

    def _odom_cb(self, msg: Odometry):
        u = float(msg.twist.twist.linear.y)
        v = float(msg.twist.twist.linear.x)
        r = float(msg.twist.twist.angular.z)
        self.nu[:] = [u, v, r]
        self.have_nu = True

    def _map_cb(self, msg: OccupancyGrid):
        self.map_msg = msg
        frame = self.viz_frame_id.strip() if self.viz_frame_id.strip() else (msg.header.frame_id or "map")
        self.actual_path_msg.header.frame_id = frame
        self.desired_path_msg.header.frame_id = frame

    def _goal_cb(self, msg: Pose2D):
        self.goal[:] = [float(msg.x), float(msg.y)]
        self.have_goal = True

    # ---- grid helpers ----
    def _is_occupied(self, val: int) -> bool:
        if val < 0:
            return bool(self.treat_unknown_as_occupied)
        return val >= int(self.occ_threshold)

    def _world_to_cell(self, grid: OccupancyGrid, x: float, y: float) -> Optional[Tuple[int, int]]:
        res = float(grid.info.resolution)
        ox = float(grid.info.origin.position.x)
        oy = float(grid.info.origin.position.y)
        gx = int((x - ox) / res)
        gy = int((y - oy) / res)
        if gx < 0 or gy < 0 or gx >= int(grid.info.width) or gy >= int(grid.info.height):
            return None
        return gx, gy

    def _cell_center_world(self, grid: OccupancyGrid, gx: int, gy: int) -> Tuple[float, float]:
        res = float(grid.info.resolution)
        ox = float(grid.info.origin.position.x)
        oy = float(grid.info.origin.position.y)
        return (ox + (gx + 0.5) * res, oy + (gy + 0.5) * res)

    def _collect_obstacles(self, grid: OccupancyGrid) -> np.ndarray:
        boat_cell = self._world_to_cell(grid, self.x, self.y)
        if boat_cell is None:
            return np.zeros((0, 2), dtype=np.float64)

        res = float(grid.info.resolution)
        rad_cells = int(max(1, math.ceil(self.window_radius_m / res)))

        cx, cy = boat_cell
        w = int(grid.info.width)
        h = int(grid.info.height)

        gx0 = max(0, cx - rad_cells)
        gx1 = min(w - 1, cx + rad_cells)
        gy0 = max(0, cy - rad_cells)
        gy1 = min(h - 1, cy + rad_cells)

        R2 = float(self.window_radius_m) ** 2

        pts = []
        stride = self.downsample_stride

        for gy in range(gy0, gy1 + 1, stride):
            base = gy * w
            for gx in range(gx0, gx1 + 1, stride):
                val = int(grid.data[base + gx])
                if not self._is_occupied(val):
                    continue
                ox, oy = self._cell_center_world(grid, gx, gy)
                dx = ox - self.x
                dy = oy - self.y
                if (dx * dx + dy * dy) > R2:
                    continue
                pts.append((ox, oy))
                if len(pts) >= self.max_obstacles:
                    break
            if len(pts) >= self.max_obstacles:
                break

        if not pts:
            return np.zeros((0, 2), dtype=np.float64)
        return np.array(pts, dtype=np.float64)

    # ---- dynamics matrices ----
    def _M_A(self) -> np.ndarray:
        MA = -np.array([
            [self.Xudot, 0.0,        0.0],
            [0.0,        self.Yvdot, self.Yrdot],
            [0.0,        self.Nvdot, self.Nrdot],
        ], dtype=np.float64)
        MA[2, 1] = MA[1, 2]
        return MA

    def _C_RB(self, nu: np.ndarray) -> np.ndarray:
        u, v, r = float(nu[0]), float(nu[1]), float(nu[2])
        m = self.m
        xg = self.xG
        yg = self.yG
        return np.array([
            [0.0, 0.0, -m * (xg * r + v)],
            [0.0, 0.0,  m * (u - yg * r)],
            [m * (xg * r + v), -m * (u - yg * r), 0.0],
        ], dtype=np.float64)

    def _C_A(self, nu_r: np.ndarray) -> np.ndarray:
        u, v, r = float(nu_r[0]), float(nu_r[1]), float(nu_r[2])
        MA = self._M_A()
        a11 = MA[0, 0]
        a22 = MA[1, 1]
        a23 = MA[1, 2]
        return np.array([
            [0.0, 0.0, -(a22 * v + a23 * r)],
            [0.0, 0.0,  (a11 * u)],
            [(a22 * v + a23 * r), -(a11 * u), 0.0],
        ], dtype=np.float64)

    def _D(self, nu_r: np.ndarray) -> np.ndarray:
        u, v, r = float(nu_r[0]), float(nu_r[1]), float(nu_r[2])

        Xd = self.Xu + self.Xuu * abs(u)
        Yd = self.Yv + self.Yvv * abs(v)
        Nd = self.Nr + self.Nrr * abs(r)

        return np.array([
            [Xd, 0.0, 0.0],
            [0.0, Yd, self.Yr],
            [0.0, self.Nv, Nd],
        ], dtype=np.float64)

    # ---- RViz helpers ----
    def _build_desired_path_to_goal(self, now_msg):
        self.desired_path_msg.header.stamp = now_msg
        self.desired_path_msg.poses.clear()

        n = max(2, int(self.desired_path_num_points))
        x0, y0 = float(self.x), float(self.y)
        x1, y1 = float(self.goal[0]), float(self.goal[1])

        for i in range(n):
            t = i / (n - 1)
            ps = PoseStamped()
            ps.header.stamp = now_msg
            ps.header.frame_id = self.desired_path_msg.header.frame_id
            ps.pose.position.x = x0 + t * (x1 - x0)
            ps.pose.position.y = y0 + t * (y1 - y0)
            ps.pose.position.z = 0.0
            ps.pose.orientation.w = 1.0
            self.desired_path_msg.poses.append(ps)

        self.pub_desired_path.publish(self.desired_path_msg)

    def _publish_goal_marker(self, now_msg):
        frame = self.actual_path_msg.header.frame_id
        ma = MarkerArray()

        m = Marker()
        m.header.stamp = now_msg
        m.header.frame_id = frame
        m.ns = "apf_goal"
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = float(self.goal[0])
        m.pose.position.y = float(self.goal[1])
        m.pose.position.z = 0.2
        m.pose.orientation.w = 1.0
        m.scale.x = 1.0
        m.scale.y = 1.0
        m.scale.z = 1.0
        m.color.r = 1.0
        m.color.g = 0.2
        m.color.b = 0.2
        m.color.a = 0.9
        ma.markers.append(m)

        t = Marker()
        t.header.stamp = now_msg
        t.header.frame_id = frame
        t.ns = "apf_goal"
        t.id = 1
        t.type = Marker.TEXT_VIEW_FACING
        t.action = Marker.ADD
        t.pose.position.x = float(self.goal[0])
        t.pose.position.y = float(self.goal[1])
        t.pose.position.z = 1.2
        t.pose.orientation.w = 1.0
        t.scale.z = 0.6
        t.color.r = 1.0
        t.color.g = 1.0
        t.color.b = 1.0
        t.color.a = 1.0
        t.text = f"GOAL ({self.goal[0]:.1f}, {self.goal[1]:.1f})"
        ma.markers.append(t)

        self.pub_goal_marker.publish(ma)

    # ---- main loop ----
    def _tick(self):
        if not (self.have_pose and self.have_goal and self.have_nu):
            return
        if self.map_msg is None:
            return

        obstacles = self._collect_obstacles(self.map_msg)

        pos_w = np.array([self.x, self.y], dtype=np.float64)
        goal_w = np.array([self.goal[0], self.goal[1]], dtype=np.float64)

        tau_apf, _dbg = apf_tau_wall_following(
            pos_w=pos_w,
            psi_w=float(self.psi),
            goal_w=goal_w,
            obstacles_w=obstacles,
            ka=self.ka, kr=self.kr, rho=self.rho,
            r_usv=self.r_usv, c_safe=self.c_safe,
            Fcap=self.Fcap,
            d_slide=self.d_slide, w_max=self.w_max, k_wall=self.k_wall,
            f_threshold=self.f_threshold, d_wf=self.d_wf, wf_gain=self.wf_gain, wf_blend=self.wf_blend,
            L_eff=self.L_eff, alpha=self.alpha,
        )

        nu = self.nu.copy().reshape(3, 1)
        nu_r = (self.nu - self.nu_c).reshape(3, 1)

        CRB = self._C_RB(self.nu)
        CA  = self._C_A(self.nu - self.nu_c)
        D   = self._D(self.nu - self.nu_c)

        tau_apf_vec = tau_apf.reshape(3, 1)

        tau_des = (CRB @ nu) + (CA @ nu_r) + (D @ nu_r) + tau_apf_vec - (self.Kd @ nu)

        Tx = float(np.clip(tau_des[0, 0], -self.F_max, self.F_max))
        Ty = float(np.clip(tau_des[1, 0], -self.F_max, self.F_max))
        Mz = float(np.clip(tau_des[2, 0], -self.Mz_max, self.Mz_max))

        msg = ControlEffort()
        msg.tau = [Tx, Ty, Mz]
        self.pub_tau.publish(msg)

        now_msg = self.get_clock().now().to_msg()

        ps = PoseStamped()
        ps.header.stamp = now_msg
        ps.header.frame_id = self.actual_path_msg.header.frame_id
        ps.pose.position.x = float(self.x)
        ps.pose.position.y = float(self.y)
        ps.pose.position.z = 0.0
        ps.pose.orientation.w = 1.0

        self.actual_path_msg.header.stamp = now_msg
        self.actual_path_msg.poses.append(ps)
        if len(self.actual_path_msg.poses) > self.path_max_len:
            self.actual_path_msg.poses = self.actual_path_msg.poses[-self.path_max_len:]
        self.pub_actual_path.publish(self.actual_path_msg)

        self._build_desired_path_from_apf(now_msg, obstacles)
        self._publish_goal_marker(now_msg)


def main(args=None):
    rclpy.init(args=args)
    node = APF_SKC_Controller()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()