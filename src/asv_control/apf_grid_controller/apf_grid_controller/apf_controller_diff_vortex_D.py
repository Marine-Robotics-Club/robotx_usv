#!/usr/bin/env python3
# Direct dynamics-aware APF controller with semantic vortex guidance.
# The original APF, safety geometry, wall-following, dynamics, lookahead, and
# actuator limits from the supplied controller are preserved.
#
# The semantic vortex is first formed as a VELOCITY field using the classical
# point-vortex expression. A vehicle-dynamics mapping then converts that
# velocity field to a force before it is added to the original 2-D APF field.
# The resulting 2-D field is projected to differential-drive actuation:
#     tau = [Tx, 0, Mz].
# The APF lookahead/visual rollout uses the same forward-only differential-drive
# constraint, so a raw field pointing behind the vessel does not draw a reverse path.
#
# Body convention: +x forward, +y starboard/right.
# IALA Region B (intended direction of buoyage): RED -> CW, GREEN -> CCW.
from __future__ import annotations

import math
import json
from typing import Optional, Tuple, List, Dict, Any

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from geometry_msgs.msg import Pose2D, PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String

from fau_msgs.msg import ControlEffort


# ============================================================
# Utils
# ============================================================

def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def rot_wb(psi: float) -> np.ndarray:
    """
    Body -> world/map rotation.

    World/map convention used by this controller:
        x = North
        y = East

    Body convention:
        x = forward/surge
        y = right/sway
    """
    c = math.cos(psi)
    s = math.sin(psi)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def current_components_ned(current_speed: float, current_beta_deg: float) -> Tuple[float, float]:
    """
    Current/drift convention:
        0 deg   = North
        90 deg  = East
        180 deg = South
        270 deg = West

    Direction means current/drift GOES TO.
    """
    beta = math.radians(float(current_beta_deg))
    current_N = float(current_speed) * math.cos(beta)
    current_E = float(current_speed) * math.sin(beta)
    return current_N, current_E


def current_body_from_heading(
    psi: float,
    current_speed: float,
    current_beta_deg: float,
) -> np.ndarray:
    """
    Current is defined in world/NED frame, then rotated into body frame.
    """
    current_N, current_E = current_components_ned(
        current_speed=current_speed,
        current_beta_deg=current_beta_deg,
    )

    c = math.cos(psi)
    s = math.sin(psi)

    R_wb = np.array(
        [
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    v_c_world = np.array([current_N, current_E, 0.0], dtype=np.float64)
    return R_wb.T @ v_c_world


def yaw_to_quaternion(yaw: float) -> Tuple[float, float, float, float]:
    half = 0.5 * float(yaw)
    return 0.0, 0.0, math.sin(half), math.cos(half)




def semantic_vortex_velocity_body(
    *,
    pos_w: np.ndarray,
    psi_w: float,
    semantic_buoys_w: Optional[np.ndarray],
    semantic_buoy_colors: Optional[List[str]],
    use_semantic_vortex: bool,
    vortex_gain: float,
    vortex_u_inf: float,
    vortex_max_distance_m: float,
    iala_region: str,
    vortex_ahead_only: bool,
    vortex_behind_tol_m: float,
    r_usv: float,
    c_safe: float,
) -> np.ndarray:
    """Return the semantic point-vortex VELOCITY field in the current body frame.

    Classical point-vortex velocity magnitude:
        |v_Gamma| = |Gamma| / (2*pi*r)

    with circulation scale:
        |Gamma| = vortex_gain * vortex_u_inf * R_eff

    Units:
        Gamma       [m^2/s]
        v_Gamma     [m/s]

    Body convention:
        +x = forward, +y = starboard/right.

    IALA Region B rule used here (for the intended direction of buoyage):
        RED   -> CW  -> vessel passes to the left of the buoy
        GREEN -> CCW -> vessel passes to the right of the buoy
    """
    if (
        not use_semantic_vortex
        or semantic_buoys_w is None
        or np.asarray(semantic_buoys_w).size == 0
    ):
        return np.zeros(2, dtype=np.float64)

    sem_w = np.asarray(semantic_buoys_w, dtype=np.float64).reshape(-1, 2)
    if semantic_buoy_colors is not None and len(semantic_buoy_colors) == sem_w.shape[0]:
        sem_colors = [str(c).strip().lower() for c in semantic_buoy_colors]
    else:
        sem_colors = ["unknown"] * sem_w.shape[0]

    R_bw = rot_wb(float(psi_w)).T
    sem_rel_w = sem_w - np.asarray(pos_w, dtype=np.float64).reshape(1, 2)
    sem_b = (R_bw @ sem_rel_w.T).T

    R_eff = float(r_usv + c_safe)
    gamma_abs = max(0.0, float(vortex_gain)) * max(0.0, float(vortex_u_inf)) * R_eff

    v_gamma = np.zeros(2, dtype=np.float64)

    for o_b, color in zip(sem_b, sem_colors):
        if color not in ("green", "red"):
            continue

        # vehicle -> buoy = o_b, obstacle -> vehicle = -o_b
        r_b = -o_b
        dc = float(np.linalg.norm(r_b))
        if dc < 1e-9:
            continue
        if vortex_max_distance_m > 0.0 and dc > vortex_max_distance_m:
            continue
        if vortex_ahead_only and float(o_b[0]) < -float(vortex_behind_tol_m):
            continue
        if dc <= R_eff:
            # Let the validated APF radial repulsion own the emergency region.
            continue

        n_hat = r_b / dc
        t_cw = np.array([-n_hat[1], n_hat[0]], dtype=np.float64)
        t_ccw = -t_cw

        region = str(iala_region).strip().upper()
        if region == "B":
            t_hat = t_cw if color == "red" else t_ccw
        else:
            t_hat = t_cw if color == "green" else t_ccw

        v_gamma += (gamma_abs / (2.0 * math.pi * dc)) * t_hat

    return v_gamma


# ============================================================
# APF in body frame + wall-following escape
# ============================================================

def apf_tau_wall_following(
    *,
    pos_w: np.ndarray,
    psi_w: float,
    goal_w: np.ndarray,
    obstacles_w: np.ndarray,
    semantic_buoys_w: Optional[np.ndarray] = None,
    semantic_buoy_colors: Optional[List[str]] = None,
    use_semantic_vortex: bool = False,
    vortex_gain: float = 3.0,
    vortex_u_inf: float = 1.0,
    vortex_model: Optional[Dict[str, float]] = None,
    vehicle_nu_b: Optional[np.ndarray] = None,
    vortex_dt: float = 0.1,
    vortex_max_distance_m: float = 12.0,
    iala_region: str = "B",
    vortex_ahead_only: bool = True,
    vortex_behind_tol_m: float = 1.0,
    ka: float = 1.8,
    kr: float = 40.0,
    rho: float = 25.0,
    obs_rad=None,
    r_usv: float = 1.0,
    c_safe: float = 2.5,
    Fcap: float = 200.0,
    d_slide: float = 8.0,
    w_max: float = 0.95,
    k_wall: float = 0.0,
    f_threshold: float = 0.08,
    d_wf: float = 12.0,
    wf_gain: float = 1.0,
    wf_blend: float = 0.85,
    L_eff: float = 2.5,
    alpha: float = 0.5,
):
    """
    Compute APF force in body frame.

    Inputs are in world/map coordinates:
        x = North
        y = East

    Output:
        tau_apf = [Fx_body, Fy_body, Mz_virtual]
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

    # Attraction
    d_goal = float(np.linalg.norm(goal_b))
    if d_goal < 1e-9:
        f_att = np.zeros(2, dtype=np.float64)
    else:
        f_att = ka * (goal_b / d_goal)

    # Closest obstacle
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

    # Repulsion
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

    # ------------------------------------------------------------
    # Semantic vortex: velocity field -> hydrodynamic damping force
    # ------------------------------------------------------------
    v_gamma_b = semantic_vortex_velocity_body(
        pos_w=pos_w,
        psi_w=psi_w,
        semantic_buoys_w=semantic_buoys_w,
        semantic_buoy_colors=semantic_buoy_colors,
        use_semantic_vortex=use_semantic_vortex,
        vortex_gain=vortex_gain,
        vortex_u_inf=vortex_u_inf,
        vortex_max_distance_m=vortex_max_distance_m,
        iala_region=iala_region,
        vortex_ahead_only=vortex_ahead_only,
        vortex_behind_tol_m=vortex_behind_tol_m,
        r_usv=r_usv,
        c_safe=c_safe,
    )

    # D_t is used as a positive translational resistance operator. The
    # identified sway coefficients retain their hydrodynamic sign convention,
    # so their magnitudes are used here to map speed magnitude to force while
    # preserving the semantic vortex direction.
    vm = vortex_model or {}
    Xu_v = abs(float(vm.get("Xu", 0.0)))
    Xuu_v = abs(float(vm.get("Xuu", 0.0)))
    Yv_v = abs(float(vm.get("Yv", 0.0)))
    Yvv_v = abs(float(vm.get("Yvv", 0.0)))

    D_t = np.array(
        [
            [Xu_v + Xuu_v * abs(float(v_gamma_b[0])), 0.0],
            [0.0, Yv_v + Yvv_v * abs(float(v_gamma_b[1]))],
        ],
        dtype=np.float64,
    )
    f_vortex = D_t @ v_gamma_b

    # Sliding term near obstacles
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

    f_xy = f_slide + f_rep + f_vortex

    # Wall-following escape
    wf_active = False

    if n_hat_cl is not None and np.isfinite(dmin):
        near_wall = dmin < d_wf
        stuck = float(np.linalg.norm(f_xy)) < f_threshold

        if near_wall and stuck:
            t1 = np.array([-n_hat_cl[1], n_hat_cl[0]], dtype=np.float64)
            t2 = -t1
            g_dir = goal_b / (np.linalg.norm(goal_b) + 1e-9)
            t_hat = t1 if float(np.dot(t1, g_dir)) >= float(np.dot(t2, g_dir)) else t2

            w_wf = max(0.0, min(1.0, 1.0 - (max(dmin, 0.0) / d_wf)))
            f_wf = wf_gain * w_wf * t_hat
            f_xy = (1.0 - wf_blend) * f_xy + wf_blend * f_wf
            wf_active = True

    # Legacy virtual yaw moment for debug/compatibility
    f_perp = np.dot(f_xy, np.array([0.0, 1.0], dtype=np.float64)) * np.array(
        [0.0, 1.0], dtype=np.float64
    )

    F_bow_b = 0.5 * f_xy + 0.5 * alpha * f_perp
    F_stern_b = 0.5 * f_xy - 0.5 * alpha * f_perp

    r_bow_b = np.array([L_eff, 0.0], dtype=np.float64)
    r_stern_b = np.array([-L_eff, 0.0], dtype=np.float64)

    tau_bow = r_bow_b[0] * F_bow_b[1] - r_bow_b[1] * F_bow_b[0]
    tau_stern = r_stern_b[0] * F_stern_b[1] - r_stern_b[1] * F_stern_b[0]
    tau_psi_virtual = float(tau_bow + tau_stern)

    tau_apf = np.array([float(f_xy[0]), float(f_xy[1]), tau_psi_virtual], dtype=np.float64)

    dbg = (
        f_att,
        f_rep,
        f_xy,
        wf_active,
        float(dmin if np.isfinite(dmin) else 1e9),
    )

    return tau_apf, dbg


# ============================================================
# Differential-drive APF projection
# ============================================================

def apf_tau_diff_drive_alignment(
    *,
    pos_w: np.ndarray,
    psi_w: float,
    goal_w: np.ndarray,
    obstacles_w: np.ndarray,
    semantic_buoys_w: Optional[np.ndarray] = None,
    semantic_buoy_colors: Optional[List[str]] = None,
    use_semantic_vortex: bool = False,
    vortex_gain: float = 3.0,
    vortex_u_inf: float = 1.0,
    vortex_model: Optional[Dict[str, float]] = None,
    vehicle_nu_b: Optional[np.ndarray] = None,
    vortex_dt: float = 0.1,
    vortex_max_distance_m: float = 12.0,
    iala_region: str = "B",
    vortex_ahead_only: bool = True,
    vortex_behind_tol_m: float = 1.0,
    ka: float = 1.8,
    kr: float = 40.0,
    rho: float = 25.0,
    obs_rad=None,
    r_usv: float = 1.0,
    c_safe: float = 2.5,
    Fcap: float = 200.0,
    d_slide: float = 8.0,
    w_max: float = 0.95,
    k_wall: float = 0.0,
    f_threshold: float = 0.08,
    d_wf: float = 12.0,
    wf_gain: float = 1.0,
    wf_blend: float = 0.85,
    K_yaw_apf: float = 1200.0,
    Mz_cap: float = 2000.0,
):
    """
    Project APF force into differential-drive-compatible command:
        tau = [Tx, 0, Mz]
    """
    _tau_full, dbg = apf_tau_wall_following(
        pos_w=pos_w,
        psi_w=psi_w,
        goal_w=goal_w,
        obstacles_w=obstacles_w,
        semantic_buoys_w=semantic_buoys_w,
        semantic_buoy_colors=semantic_buoy_colors,
        use_semantic_vortex=use_semantic_vortex,
        vortex_gain=vortex_gain,
        vortex_u_inf=vortex_u_inf,
        vortex_model=vortex_model,
        vehicle_nu_b=vehicle_nu_b,
        vortex_dt=vortex_dt,
        vortex_max_distance_m=vortex_max_distance_m,
        iala_region=iala_region,
        vortex_ahead_only=vortex_ahead_only,
        vortex_behind_tol_m=vortex_behind_tol_m,
        ka=ka,
        kr=kr,
        rho=rho,
        obs_rad=obs_rad,
        r_usv=r_usv,
        c_safe=c_safe,
        Fcap=Fcap,
        d_slide=d_slide,
        w_max=w_max,
        k_wall=k_wall,
        f_threshold=f_threshold,
        d_wf=d_wf,
        wf_gain=wf_gain,
        wf_blend=wf_blend,
        L_eff=1.0,
        alpha=0.0,
    )

    f_xy_b = dbg[2]
    Fx = float(f_xy_b[0])
    Fy = float(f_xy_b[1])
    fmag = float(np.linalg.norm(f_xy_b))

    if fmag < 1e-6:
        tau_apf_dd = np.zeros(3, dtype=np.float64)
        dbg_dd = dbg + (0.0, 0.0, 0.0)
        return tau_apf_dd, dbg_dd

    e_apf = wrap_angle(math.atan2(Fy, Fx))

    Tx = fmag * max(0.0, math.cos(e_apf))
    Tx = float(np.clip(Tx, 0.0, Fcap))

    field_strength = min(1.0, fmag / max(Fcap, 1e-6))

    # NED/body convention here:
    # +body y = right, so e_apf > 0 means target field to the right.
    # +Mz is treated as clockwise/right turn in this controller stack.
    Mz = K_yaw_apf * field_strength * math.tanh(e_apf)
    Mz = float(np.clip(Mz, -Mz_cap, Mz_cap))

    tau_apf_dd = np.array([Tx, 0.0, Mz], dtype=np.float64)
    dbg_dd = dbg + (float(e_apf), float(Tx), float(Mz))

    return tau_apf_dd, dbg_dd


# ============================================================
# APF controller node
# ============================================================

class APF_SKC_Controller(Node):
    def __init__(self):
        super().__init__("apf_skc_controller_vortex_D")

        # ------------------------------------------------------------
        # Topics
        # ------------------------------------------------------------
        self.declare_parameter("wamv", "asv")
        self.declare_parameter("pose_topic", "vehicle_pose")
        self.declare_parameter("odom_topic", "p3d_wamv_ned")
        self.declare_parameter("map_topic", "map/local_occupancy_2")
        self.declare_parameter("goal_topic", "nav/goal")
        self.declare_parameter("control_effort_topic", "control_effort")
        self.declare_parameter("semantic_topic", "map/semantic_buoys")

        # Semantic vortex guidance
        self.declare_parameter("use_semantic_vortex", True)
        self.declare_parameter("vortex_gain", 3.0)
        self.declare_parameter("vortex_u_inf", 1.0)
        self.declare_parameter("vortex_max_distance_m", 12.0)
        self.declare_parameter("iala_region", "B")
        self.declare_parameter("vortex_ahead_only", True)
        self.declare_parameter("vortex_behind_tol_m", 1.0)
        self.declare_parameter("semantic_confirmed_only", True)

        # ------------------------------------------------------------
        # Map extraction
        # ------------------------------------------------------------
        self.declare_parameter("occ_threshold", 80)
        self.declare_parameter("treat_unknown_as_occupied", False)
        self.declare_parameter("window_radius_m", 15.0)
        self.declare_parameter("max_obstacles", 500)
        self.declare_parameter("downsample_stride", 5)

        # ------------------------------------------------------------
        # APF params
        # ------------------------------------------------------------
        self.declare_parameter("ka", 140.0)
        self.declare_parameter("kr", 10.0)
        self.declare_parameter("rho", 10.0)
        self.declare_parameter("r_usv", 2.5)
        self.declare_parameter("c_safe", 1.0)
        self.declare_parameter("Fcap", 240.0)
        self.declare_parameter("d_slide", 2.0)
        self.declare_parameter("w_max", 0.50)
        self.declare_parameter("k_wall", 0.0)

        # Differential-drive APF alignment
        self.declare_parameter("K_yaw_apf", 80.0)

        # Optional lookahead yaw for direct APF control.
        # If enabled, Tx is still computed from the local APF field, but Mz is
        # computed from a short APF rollout/lookahead point. This makes direct
        # APF less reactive and closer to the APF-path behavior without using
        # PID_HS or RL.
        self.declare_parameter("use_lookahead_yaw", False)
        self.declare_parameter("yaw_lookahead_m", 3.0)
        self.declare_parameter("yaw_lookahead_blend", 1.0)
        self.declare_parameter("yaw_lookahead_min_dist", 0.6)
        self.declare_parameter("yaw_lookahead_min_field_strength", 0.30)

        # Wall-following escape
        self.declare_parameter("f_threshold", 0.04)
        self.declare_parameter("d_wf", 4.0)
        self.declare_parameter("wf_gain", 1.0)
        self.declare_parameter("wf_blend", 0.35)

        # Legacy virtual yaw geometry, used only for visualization/debug
        self.declare_parameter("L_eff", 2.5)
        self.declare_parameter("alpha", 1.0)

        # ------------------------------------------------------------
        # USV 3-DOF parameters
        # ------------------------------------------------------------
        self.declare_parameter("m", 49.3)
        self.declare_parameter("Iz", 22.31)
        self.declare_parameter("xG", -0.15)
        self.declare_parameter("yG", 0.0)

        # Added mass derivatives
        self.declare_parameter("Xudot", -55.576760213798664)
        self.declare_parameter("Yvdot", -1.000006321227744)
        self.declare_parameter("Yrdot", 48.40551455718584)
        self.declare_parameter("Nvdot", -116.87232859310109)
        self.declare_parameter("Nrdot", -21.553699524355196)

        # Linear damping
        self.declare_parameter("Xu", 1.2726849448720725e-10)
        self.declare_parameter("Yv", -233.76775314498622)
        self.declare_parameter("Yr", -799.9998926338545)
        self.declare_parameter("Nv", 0.0)
        self.declare_parameter("Nr", -360.0205010215149)

        # Nonlinear / cross-coupling damping
        self.declare_parameter("Xuu", 28.544342526180266)
        self.declare_parameter("Yvv", 0.0)
        self.declare_parameter("Yvr", 0.0)
        self.declare_parameter("Yrv", 0.0)
        self.declare_parameter("Yrr", 0.0)
        self.declare_parameter("Nvv", 0.0)
        self.declare_parameter("Nvr", 0.0)
        self.declare_parameter("Nrv", 0.0)
        self.declare_parameter("Nrr", -842.3864909413818)

        # Current / drift
        self.declare_parameter("current_speed", 0.0)
        self.declare_parameter("current_beta", 161.0)

        # Force caps
        self.declare_parameter("F_max", 120.0)
        self.declare_parameter("Mz_max", 45.0)

        # Extra damping injection
        self.declare_parameter("Kd_diag", [80.0, 0.0, 0.0])

        # Loop timing
        self.declare_parameter("dt", 0.1)

        # RViz desired path from APF rollout
        self.declare_parameter("apf_path_num_points", 80)
        self.declare_parameter("apf_path_ds", 0.6)
        self.declare_parameter("apf_path_max_yaw_rate", 0.6)

        # RViz topics
        self.declare_parameter("actual_path_topic", "viz/apf_actual_path")
        self.declare_parameter("desired_path_topic", "viz/apf_desired_path")
        self.declare_parameter("goal_marker_topic", "viz/apf_goal_marker")
        self.declare_parameter("vehicle_pose_viz_topic", "viz/apf_vehicle_pose")
        self.declare_parameter("vehicle_marker_topic", "viz/apf_vehicle_marker")
        self.declare_parameter("vehicle_marker_length_m", 2.0)
        self.declare_parameter("vehicle_marker_width_m", 0.35)
        self.declare_parameter("vehicle_pose_publish_period_s", 0.10)
        self.declare_parameter("path_max_len", 2000)
        self.declare_parameter("desired_path_num_points", 50)
        self.declare_parameter("viz_frame_id", "")

        # ------------------------------------------------------------
        # Read topic params
        # ------------------------------------------------------------
        self.wamv = str(self.get_parameter("wamv").value)
        self.pose_topic = str(self.get_parameter("pose_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.map_topic = str(self.get_parameter("map_topic").value)
        self.goal_topic = str(self.get_parameter("goal_topic").value)
        self.ctrl_topic = str(self.get_parameter("control_effort_topic").value)
        self.semantic_topic = str(self.get_parameter("semantic_topic").value)

        self.use_semantic_vortex = bool(self.get_parameter("use_semantic_vortex").value)
        self.vortex_gain = float(self.get_parameter("vortex_gain").value)
        self.vortex_u_inf = float(self.get_parameter("vortex_u_inf").value)
        self.vortex_max_distance_m = float(self.get_parameter("vortex_max_distance_m").value)
        self.iala_region = str(self.get_parameter("iala_region").value).strip().upper()
        if self.iala_region not in ("A", "B"):
            self.get_logger().warn(
                f"Invalid iala_region={self.iala_region!r}; defaulting to 'B'."
            )
            self.iala_region = "B"
        self.vortex_ahead_only = bool(self.get_parameter("vortex_ahead_only").value)
        self.vortex_behind_tol_m = max(
            0.0, float(self.get_parameter("vortex_behind_tol_m").value)
        )
        self.semantic_confirmed_only = bool(self.get_parameter("semantic_confirmed_only").value)

        # ------------------------------------------------------------
        # Read map params
        # ------------------------------------------------------------
        self.occ_threshold = int(self.get_parameter("occ_threshold").value)
        self.treat_unknown_as_occupied = bool(
            self.get_parameter("treat_unknown_as_occupied").value
        )
        self.window_radius_m = float(self.get_parameter("window_radius_m").value)
        self.max_obstacles = int(self.get_parameter("max_obstacles").value)
        self.downsample_stride = max(1, int(self.get_parameter("downsample_stride").value))

        # ------------------------------------------------------------
        # Read APF params
        # ------------------------------------------------------------
        self.ka = float(self.get_parameter("ka").value)
        self.kr = float(self.get_parameter("kr").value)
        self.rho = float(self.get_parameter("rho").value)
        self.r_usv = float(self.get_parameter("r_usv").value)
        self.c_safe = float(self.get_parameter("c_safe").value)
        self.Fcap = float(self.get_parameter("Fcap").value)
        self.d_slide = float(self.get_parameter("d_slide").value)
        self.w_max = float(self.get_parameter("w_max").value)
        self.k_wall = float(self.get_parameter("k_wall").value)
        self.K_yaw_apf = float(self.get_parameter("K_yaw_apf").value)
        self.use_lookahead_yaw = bool(self.get_parameter("use_lookahead_yaw").value)
        self.yaw_lookahead_m = float(self.get_parameter("yaw_lookahead_m").value)
        self.yaw_lookahead_blend = float(self.get_parameter("yaw_lookahead_blend").value)
        self.yaw_lookahead_min_dist = float(self.get_parameter("yaw_lookahead_min_dist").value)
        self.yaw_lookahead_min_field_strength = float(
            self.get_parameter("yaw_lookahead_min_field_strength").value
        )
        self.yaw_lookahead_blend = float(np.clip(self.yaw_lookahead_blend, 0.0, 1.0))
        if self.yaw_lookahead_m <= 0.0:
            self.yaw_lookahead_m = 3.0
        if self.yaw_lookahead_min_dist <= 0.0:
            self.yaw_lookahead_min_dist = 0.6
        self.yaw_lookahead_min_field_strength = float(
            np.clip(self.yaw_lookahead_min_field_strength, 0.0, 1.0)
        )
        self.f_threshold = float(self.get_parameter("f_threshold").value)
        self.d_wf = float(self.get_parameter("d_wf").value)
        self.wf_gain = float(self.get_parameter("wf_gain").value)
        self.wf_blend = float(self.get_parameter("wf_blend").value)
        self.L_eff = float(self.get_parameter("L_eff").value)
        self.alpha = float(self.get_parameter("alpha").value)

        # ------------------------------------------------------------
        # Read vehicle params
        # ------------------------------------------------------------
        self.m = float(self.get_parameter("m").value)
        self.Iz = float(self.get_parameter("Iz").value)
        self.xG = float(self.get_parameter("xG").value)
        self.yG = float(self.get_parameter("yG").value)

        self.Xudot = float(self.get_parameter("Xudot").value)
        self.Yvdot = float(self.get_parameter("Yvdot").value)
        self.Yrdot = float(self.get_parameter("Yrdot").value)
        self.Nvdot = float(self.get_parameter("Nvdot").value)
        self.Nrdot = float(self.get_parameter("Nrdot").value)

        self.Xu = float(self.get_parameter("Xu").value)
        self.Yv = float(self.get_parameter("Yv").value)
        self.Yr = float(self.get_parameter("Yr").value)
        self.Nv = float(self.get_parameter("Nv").value)
        self.Nr = float(self.get_parameter("Nr").value)
        self.Xuu = float(self.get_parameter("Xuu").value)

        self.Yvv = float(self.get_parameter("Yvv").value)
        self.Yvr = float(self.get_parameter("Yvr").value)
        self.Yrv = float(self.get_parameter("Yrv").value)
        self.Yrr = float(self.get_parameter("Yrr").value)
        self.Nvv = float(self.get_parameter("Nvv").value)
        self.Nvr = float(self.get_parameter("Nvr").value)
        self.Nrv = float(self.get_parameter("Nrv").value)
        self.Nrr = float(self.get_parameter("Nrr").value)

        self.F_max = float(self.get_parameter("F_max").value)
        self.Mz_max = float(self.get_parameter("Mz_max").value)
        self.current_speed = float(self.get_parameter("current_speed").value)
        self.current_beta = float(self.get_parameter("current_beta").value)

        kd = list(self.get_parameter("Kd_diag").value)
        if len(kd) != 3:
            kd = [50.0, 0.0, 0.0]
        self.Kd = np.diag(np.array(kd, dtype=np.float64))

        self.dt = float(self.get_parameter("dt").value)

        # ------------------------------------------------------------
        # Read RViz params
        # ------------------------------------------------------------
        self.apf_path_num_points = int(self.get_parameter("apf_path_num_points").value)
        self.apf_path_ds = float(self.get_parameter("apf_path_ds").value)
        self.apf_path_max_yaw_rate = float(self.get_parameter("apf_path_max_yaw_rate").value)

        self.actual_path_topic = str(self.get_parameter("actual_path_topic").value)
        self.desired_path_topic = str(self.get_parameter("desired_path_topic").value)
        self.goal_marker_topic = str(self.get_parameter("goal_marker_topic").value)
        self.vehicle_pose_viz_topic = str(self.get_parameter("vehicle_pose_viz_topic").value)
        self.vehicle_marker_topic = str(self.get_parameter("vehicle_marker_topic").value)
        self.vehicle_marker_length_m = float(self.get_parameter("vehicle_marker_length_m").value)
        self.vehicle_marker_width_m = float(self.get_parameter("vehicle_marker_width_m").value)
        self.vehicle_pose_publish_period_s = float(
            self.get_parameter("vehicle_pose_publish_period_s").value
        )
        if self.vehicle_pose_publish_period_s <= 0.0:
            self.vehicle_pose_publish_period_s = 0.10
        self.path_max_len = int(self.get_parameter("path_max_len").value)
        self.desired_path_num_points = int(self.get_parameter("desired_path_num_points").value)
        self.viz_frame_id = str(self.get_parameter("viz_frame_id").value)

        # ------------------------------------------------------------
        # State
        # ------------------------------------------------------------
        self.have_pose = False
        self.x = 0.0
        self.y = 0.0
        self.psi = 0.0

        self.have_nu = False
        self.nu = np.zeros(3, dtype=np.float64)

        self.have_goal = False
        self.goal = np.zeros(2, dtype=np.float64)

        self.map_msg: Optional[OccupancyGrid] = None

        # Latest semantic buoy map from /<asv>/map/semantic_buoys.
        # Each entry: {"x": North, "y": East, "color": red/green, ...}
        self.semantic_buoys: List[Dict[str, Any]] = []

        # ------------------------------------------------------------
        # QoS
        # ------------------------------------------------------------
        pose_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        odom_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        goal_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        # ------------------------------------------------------------
        # Subscribers
        # ------------------------------------------------------------
        self.create_subscription(
            Pose2D,
            f"/{self.wamv}/{self.pose_topic}".replace("//", "/"),
            self._pose_cb,
            pose_qos,
        )

        self.create_subscription(
            Odometry,
            f"/{self.wamv}/{self.odom_topic}".replace("//", "/"),
            self._odom_cb,
            odom_qos,
        )

        self.create_subscription(
            OccupancyGrid,
            f"/{self.wamv}/{self.map_topic}".replace("//", "/"),
            self._map_cb,
            map_qos,
        )

        self.create_subscription(
            Pose2D,
            f"/{self.wamv}/{self.goal_topic}".replace("//", "/"),
            self._goal_cb,
            goal_qos,
        )

        semantic_topic_full = (
            self.semantic_topic
            if self.semantic_topic.startswith("/")
            else f"/{self.wamv}/{self.semantic_topic}".replace("//", "/")
        )

        self.create_subscription(
            String,
            semantic_topic_full,
            self._semantic_cb,
            10,
        )

        # ------------------------------------------------------------
        # Publishers
        # ------------------------------------------------------------
        self.pub_tau = self.create_publisher(
            ControlEffort,
            f"/{self.wamv}/{self.ctrl_topic}".replace("//", "/"),
            10,
        )

        self.pub_actual_path = self.create_publisher(
            Path,
            f"/{self.wamv}/{self.actual_path_topic}".replace("//", "/"),
            10,
        )

        self.pub_desired_path = self.create_publisher(
            Path,
            f"/{self.wamv}/{self.desired_path_topic}".replace("//", "/"),
            10,
        )

        self.pub_goal_marker = self.create_publisher(
            MarkerArray,
            f"/{self.wamv}/{self.goal_marker_topic}".replace("//", "/"),
            10,
        )

        self.pub_vehicle_pose_viz = self.create_publisher(
            PoseStamped,
            f"/{self.wamv}/{self.vehicle_pose_viz_topic}".replace("//", "/"),
            10,
        )

        self.pub_vehicle_marker = self.create_publisher(
            MarkerArray,
            f"/{self.wamv}/{self.vehicle_marker_topic}".replace("//", "/"),
            10,
        )

        self.actual_path_msg = Path()
        self.desired_path_msg = Path()
        self.actual_path_msg.header.frame_id = "map"
        self.desired_path_msg.header.frame_id = "map"

        self.timer = self.create_timer(self.dt, self._tick)

        # Vehicle pose visualization is published from its own timer so RViz
        # still sees the boat pose even when APF control is waiting for a goal,
        # odometry, or occupancy map.
        self.vehicle_pose_timer = self.create_timer(
            self.vehicle_pose_publish_period_s,
            self._vehicle_pose_timer_cb,
        )

        self.get_logger().info(
            "APF differential-drive controller (D vortex mapping) started. "
            "Publishing ControlEffort tau = [Tx, 0, Mz]. "
            f"K_yaw_apf={self.K_yaw_apf:.2f}, "
            f"Mz_max={self.Mz_max:.2f}, "
            f"use_lookahead_yaw={self.use_lookahead_yaw}, "
            f"yaw_lookahead_m={self.yaw_lookahead_m:.2f}, "
            f"semantic_vortex={self.use_semantic_vortex}, "
            f"vortex_gain={self.vortex_gain:.2f}, "
            f"vortex_u_inf={self.vortex_u_inf:.2f} m/s, "
            f"vortex_range={self.vortex_max_distance_m:.2f} m, "
            f"IALA_region={self.iala_region}, "
            f"ahead_only={self.vortex_ahead_only}"
        )
        self.get_logger().info(
            f"Vehicle RViz pose topics: /{self.wamv}/{self.vehicle_pose_viz_topic}, "
            f"/{self.wamv}/{self.vehicle_marker_topic}"
        )
        self.get_logger().info(
            f"Vehicle RViz pose publish period: {self.vehicle_pose_publish_period_s:.2f} s"
        )

    # ============================================================
    # Callbacks
    # ============================================================

    def _pose_cb(self, msg: Pose2D):
        self.x = float(msg.x)
        self.y = float(msg.y)
        self.psi = float(msg.theta)
        self.have_pose = True

    def _odom_cb(self, msg: Odometry):
        """
        /asv/p3d_wamv_ned publishes linear velocity in world/NED frame:
            linear.x = v_N = North velocity
            linear.y = v_E = East velocity

        The APF dynamic model needs body-frame velocity:
            u = surge / forward
            v = sway / right
            r = yaw rate
        """
        vN = float(msg.twist.twist.linear.x)
        vE = float(msg.twist.twist.linear.y)
        r = float(msg.twist.twist.angular.z)

        psi = float(self.psi)
        u_body = vN * math.cos(psi) + vE * math.sin(psi)
        v_body = -vN * math.sin(psi) + vE * math.cos(psi)

        self.nu[:] = [u_body, v_body, r]
        self.have_nu = True

    def _map_cb(self, msg: OccupancyGrid):
        self.map_msg = msg

        if self.viz_frame_id.strip():
            frame = self.viz_frame_id.strip()
        else:
            frame = msg.header.frame_id or "map"

        self.actual_path_msg.header.frame_id = frame
        self.desired_path_msg.header.frame_id = frame

    def _goal_cb(self, msg: Pose2D):
        self.goal[:] = [float(msg.x), float(msg.y)]
        self.have_goal = True

    def _semantic_cb(self, msg: String):
        """
        Parse /<asv>/map/semantic_buoys.

        Expected mapper payload:
            {
              "buoy_count": N,
              "buoys": [
                {
                  "color": "red" or "green",
                  "x": North,
                  "y": East,
                  "confirmed": true/false,
                  ...
                }
              ]
            }
        """
        try:
            payload = json.loads(msg.data)
        except Exception as exc:
            self.get_logger().warn(
                f"Could not parse semantic buoy JSON: {exc}"
            )
            return

        records = payload.get("buoys", []) if isinstance(payload, dict) else []
        parsed: List[Dict[str, Any]] = []

        for item in records:
            if not isinstance(item, dict):
                continue

            color = str(item.get("color", "")).strip().lower()
            if color not in ("red", "green"):
                continue

            if self.semantic_confirmed_only and not bool(item.get("confirmed", False)):
                continue

            try:
                x = float(item["x"])
                y = float(item["y"])
            except Exception:
                continue

            parsed.append(
                {
                    "x": x,
                    "y": y,
                    "color": color,
                    "confirmed": bool(item.get("confirmed", False)),
                    "track_id": item.get("track_id", item.get("id", -1)),
                }
            )

        self.semantic_buoys = parsed

    def _semantic_arrays(self) -> Tuple[np.ndarray, List[str]]:
        if not self.use_semantic_vortex or not self.semantic_buoys:
            return np.zeros((0, 2), dtype=np.float64), []

        xy = np.array(
            [[b["x"], b["y"]] for b in self.semantic_buoys],
            dtype=np.float64,
        )
        colors = [str(b["color"]).lower() for b in self.semantic_buoys]
        return xy, colors

    def _vortex_model_params(self) -> Dict[str, float]:
        return {
            "Xu": self.Xu,
            "Xuu": self.Xuu,
            "Yv": self.Yv,
            "Yvv": self.Yvv,
        }

    # ============================================================
    # Grid helpers
    # ============================================================

    def _is_occupied(self, val: int) -> bool:
        if val < 0:
            return bool(self.treat_unknown_as_occupied)
        return val >= int(self.occ_threshold)

    def _world_to_cell(self, grid: OccupancyGrid, x: float, y: float) -> Optional[Tuple[int, int]]:
        res = float(grid.info.resolution)
        ox = float(grid.info.origin.position.x)
        oy = float(grid.info.origin.position.y)

        if res <= 0.0:
            return None

        gx = int((x - ox) / res)
        gy = int((y - oy) / res)

        if gx < 0 or gy < 0:
            return None
        if gx >= int(grid.info.width) or gy >= int(grid.info.height):
            return None
        return gx, gy

    def _cell_center_world(self, grid: OccupancyGrid, gx: int, gy: int) -> Tuple[float, float]:
        res = float(grid.info.resolution)
        ox = float(grid.info.origin.position.x)
        oy = float(grid.info.origin.position.y)
        return ox + (gx + 0.5) * res, oy + (gy + 0.5) * res

    def _collect_obstacles(self, grid: OccupancyGrid) -> np.ndarray:
        boat_cell = self._world_to_cell(grid, self.x, self.y)
        if boat_cell is None:
            return np.zeros((0, 2), dtype=np.float64)

        res = float(grid.info.resolution)
        if res <= 0.0:
            return np.zeros((0, 2), dtype=np.float64)

        rad_cells = int(max(1, math.ceil(self.window_radius_m / res)))
        cx, cy = boat_cell
        w = int(grid.info.width)
        h = int(grid.info.height)

        gx0 = max(0, cx - rad_cells)
        gx1 = min(w - 1, cx + rad_cells)
        gy0 = max(0, cy - rad_cells)
        gy1 = min(h - 1, cy + rad_cells)

        R2 = float(self.window_radius_m) ** 2
        pts: List[Tuple[float, float]] = []
        stride = self.downsample_stride

        for gy in range(gy0, gy1 + 1, stride):
            base = gy * w
            for gx in range(gx0, gx1 + 1, stride):
                idx = base + gx
                if idx < 0 or idx >= len(grid.data):
                    continue

                val = int(grid.data[idx])
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

    # ============================================================
    # Dynamics matrices
    # ============================================================

    def _M_A(self) -> np.ndarray:
        return np.array(
            [
                [-self.Xudot, 0.0,         0.0],
                [0.0,        -self.Yvdot, -self.Yrdot],
                [0.0,        -self.Nvdot, -self.Nrdot],
            ],
            dtype=np.float64,
        )

    def _C_RB(self, nu: np.ndarray) -> np.ndarray:
        u = float(nu[0])
        v = float(nu[1])
        r = float(nu[2])

        m = self.m
        xg = self.xG
        yg = self.yG

        return np.array(
            [
                [0.0, 0.0, -m * (xg * r + v)],
                [0.0, 0.0,  m * (u - yg * r)],
                [m * (xg * r + v), -m * (u - yg * r), 0.0],
            ],
            dtype=np.float64,
        )

    def _C_A(self, nu_r: np.ndarray) -> np.ndarray:
        u = float(nu_r[0])
        v = float(nu_r[1])
        r = float(nu_r[2])

        return np.array(
            [
                [0.0, 0.0, self.Yvdot * v + 0.5 * (self.Yrdot + self.Nvdot) * r],
                [0.0, 0.0, -self.Xudot * u],
                [-self.Yvdot * v - 0.5 * (self.Yrdot + self.Nvdot) * r, self.Xudot * u, 0.0],
            ],
            dtype=np.float64,
        )

    def _D(self, nu_r: np.ndarray) -> np.ndarray:
        u = float(nu_r[0])
        v = float(nu_r[1])
        r = float(nu_r[2])

        Dl = np.array(
            [
                [self.Xu, 0.0,     0.0],
                [0.0,     self.Yv, self.Yr],
                [0.0,     self.Nv, self.Nr],
            ],
            dtype=np.float64,
        )

        Dln = np.array(
            [
                [self.Xuu * abs(u), 0.0, 0.0],
                [0.0, self.Yvv * abs(v) + self.Yvr * abs(r), self.Yrv * abs(v) + self.Yrr * abs(r)],
                [0.0, self.Nvv * abs(v) + self.Nvr * abs(r), self.Nrv * abs(v) + self.Nrr * abs(r)],
            ],
            dtype=np.float64,
        )

        return Dl + Dln

    # ============================================================
    # RViz helpers
    # ============================================================

    def _current_viz_frame(self) -> str:
        if self.viz_frame_id.strip():
            return self.viz_frame_id.strip()
        if self.actual_path_msg.header.frame_id:
            return self.actual_path_msg.header.frame_id
        return "map"

    def _publish_vehicle_pose_marker(self, now_msg):
        """
        Publish current vehicle pose for RViz.

        Topics:
            /asv/viz/apf_vehicle_pose    PoseStamped
            /asv/viz/apf_vehicle_marker  MarkerArray

        Coordinates:
            position.x = North
            position.y = East
        """
        frame = self._current_viz_frame()
        qx, qy, qz, qw = yaw_to_quaternion(self.psi)

        ps = PoseStamped()
        ps.header.stamp = now_msg
        ps.header.frame_id = frame
        ps.pose.position.x = float(self.x)
        ps.pose.position.y = float(self.y)
        ps.pose.position.z = 0.25
        ps.pose.orientation.x = qx
        ps.pose.orientation.y = qy
        ps.pose.orientation.z = qz
        ps.pose.orientation.w = qw
        self.pub_vehicle_pose_viz.publish(ps)

        ma = MarkerArray()

        clear = Marker()
        clear.header.stamp = now_msg
        clear.header.frame_id = frame
        clear.action = Marker.DELETEALL
        ma.markers.append(clear)

        arrow = Marker()
        arrow.header.stamp = now_msg
        arrow.header.frame_id = frame
        arrow.ns = "apf_vehicle_pose"
        arrow.id = 0
        arrow.type = Marker.ARROW
        arrow.action = Marker.ADD
        arrow.pose.position.x = float(self.x)
        arrow.pose.position.y = float(self.y)
        arrow.pose.position.z = 0.35
        arrow.pose.orientation.x = qx
        arrow.pose.orientation.y = qy
        arrow.pose.orientation.z = qz
        arrow.pose.orientation.w = qw
        arrow.scale.x = float(self.vehicle_marker_length_m)
        arrow.scale.y = float(self.vehicle_marker_width_m)
        arrow.scale.z = float(2.0 * self.vehicle_marker_width_m)
        arrow.color.r = 0.1
        arrow.color.g = 0.8
        arrow.color.b = 1.0
        arrow.color.a = 1.0
        ma.markers.append(arrow)

        text = Marker()
        text.header.stamp = now_msg
        text.header.frame_id = frame
        text.ns = "apf_vehicle_pose"
        text.id = 1
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position.x = float(self.x)
        text.pose.position.y = float(self.y)
        text.pose.position.z = 1.3
        text.pose.orientation.w = 1.0
        text.scale.z = 0.55
        text.color.r = 1.0
        text.color.g = 1.0
        text.color.b = 1.0
        text.color.a = 1.0
        text.text = (
            f"USV\n"
            f"N={self.x:.2f}, E={self.y:.2f}\n"
            f"psi={math.degrees(self.psi):.1f} deg"
        )
        ma.markers.append(text)

        self.pub_vehicle_marker.publish(ma)

    def _vehicle_pose_timer_cb(self):
        """
        Publish vehicle pose visualization independently from the APF control loop.

        The main _tick() waits for pose + goal + odometry. This timer only
        waits for pose, so RViz always shows the current vehicle marker as soon
        as /asv/vehicle_pose is available.
        """
        if not self.have_pose:
            return

        now_msg = self.get_clock().now().to_msg()
        self._publish_vehicle_pose_marker(now_msg)

    def _compute_lookahead_yaw_error(
        self,
        obstacles_w: np.ndarray,
        semantic_buoys_w: np.ndarray,
        semantic_buoy_colors: List[str],
    ) -> Optional[Tuple[float, np.ndarray, float]]:
        """
        Roll out the APF field ahead and return a yaw error to a lookahead point.

        This is used only for direct APF control when use_lookahead_yaw=True.
        It keeps surge Tx from the local APF force, but replaces/blends the yaw
        moment with a heading command toward a point along the short-horizon APF
        path. This makes direct APF start turning earlier around obstacles.

        Returns:
            e_lh: heading error from current heading to the APF lookahead point
            target_w: lookahead target [North, East]
            rollout_dist: arc length rolled out
        """
        p0 = np.array([self.x, self.y], dtype=np.float64)
        p_w = p0.copy()
        psi_roll = float(self.psi)
        goal_w = np.array([self.goal[0], self.goal[1]], dtype=np.float64)

        target_arc = max(float(self.yaw_lookahead_min_dist), float(self.yaw_lookahead_m))
        ds = float(self.apf_path_ds)
        if ds <= 1e-3:
            ds = 0.3

        n = max(2, int(self.apf_path_num_points))
        rollout_dist = 0.0

        for _ in range(n):
            if rollout_dist >= target_arc:
                break

            gvec = goal_w - p_w
            if float(np.linalg.norm(gvec)) < self.yaw_lookahead_min_dist:
                break

            _tau_i, dbg = apf_tau_wall_following(
                pos_w=p_w,
                psi_w=psi_roll,
                goal_w=goal_w,
                obstacles_w=obstacles_w,
                semantic_buoys_w=semantic_buoys_w,
                semantic_buoy_colors=semantic_buoy_colors,
                use_semantic_vortex=self.use_semantic_vortex,
                vortex_gain=self.vortex_gain,
                vortex_u_inf=self.vortex_u_inf,
                vortex_model=self._vortex_model_params(),
                vehicle_nu_b=self.nu,
                vortex_dt=self.dt,
                vortex_max_distance_m=self.vortex_max_distance_m,
                iala_region=self.iala_region,
                vortex_ahead_only=self.vortex_ahead_only,
                vortex_behind_tol_m=self.vortex_behind_tol_m,
                ka=self.ka,
                kr=self.kr,
                rho=self.rho,
                r_usv=self.r_usv,
                c_safe=self.c_safe,
                Fcap=self.Fcap,
                d_slide=self.d_slide,
                w_max=self.w_max,
                k_wall=self.k_wall,
                f_threshold=self.f_threshold,
                d_wf=self.d_wf,
                wf_gain=self.wf_gain,
                wf_blend=self.wf_blend,
                L_eff=self.L_eff,
                alpha=self.alpha,
            )

            f_xy_b = dbg[2]
            fmag = float(np.linalg.norm(f_xy_b))
            if fmag < 1e-6:
                break

            # Differential-drive-consistent APF rollout.  The real controller
            # does not command sway and does not command reverse APF surge:
            #     Tx = |F| max(0, cos(e_apf)),  Ty = 0.
            # Therefore the virtual lookahead vehicle turns toward the APF
            # field and only advances forward.  This prevents the rollout from
            # drawing a backwards segment when the raw 2-D field points more
            # than 90 deg away from the bow.
            e_apf = wrap_angle(math.atan2(float(f_xy_b[1]), float(f_xy_b[0])))
            forward_scale = max(0.0, math.cos(e_apf))

            dpsi_max = float(self.apf_path_max_yaw_rate) * float(self.dt)
            psi_roll = wrap_angle(
                psi_roll + float(np.clip(e_apf, -dpsi_max, dpsi_max))
            )

            dir_forward_w = np.array(
                [math.cos(psi_roll), math.sin(psi_roll)], dtype=np.float64
            )
            p_next = p_w + ds * forward_scale * dir_forward_w
            rollout_dist += float(np.linalg.norm(p_next - p_w))
            p_w = p_next

        if float(np.linalg.norm(p_w - p0)) < self.yaw_lookahead_min_dist:
            return None

        psi_lh = float(math.atan2(float(p_w[1] - self.y), float(p_w[0] - self.x)))
        e_lh = wrap_angle(psi_lh - float(self.psi))
        return e_lh, p_w, rollout_dist

    def _build_desired_path_from_apf(
        self,
        now_msg,
        obstacles_w: np.ndarray,
        semantic_buoys_w: np.ndarray,
        semantic_buoy_colors: List[str],
    ):
        """
        Roll out the APF field forward and publish as a Path in world/map frame.
        This is visualization only, not the actual command.
        """
        self.desired_path_msg.header.stamp = now_msg
        self.desired_path_msg.poses.clear()

        p_w = np.array([self.x, self.y], dtype=np.float64)
        psi = float(self.psi)

        n = max(2, int(self.apf_path_num_points))
        ds = float(self.apf_path_ds)
        if ds <= 1e-3:
            ds = 0.3

        goal_w = np.array([self.goal[0], self.goal[1]], dtype=np.float64)

        for _ in range(n):
            ps = PoseStamped()
            ps.header.stamp = now_msg
            ps.header.frame_id = self.desired_path_msg.header.frame_id
            ps.pose.position.x = float(p_w[0])
            ps.pose.position.y = float(p_w[1])
            ps.pose.position.z = 0.0
            ps.pose.orientation.w = 1.0
            self.desired_path_msg.poses.append(ps)

            gvec = goal_w - p_w
            if float(np.linalg.norm(gvec)) < 1.0:
                break

            _tau_i, dbg = apf_tau_wall_following(
                pos_w=p_w,
                psi_w=psi,
                goal_w=goal_w,
                obstacles_w=obstacles_w,
                semantic_buoys_w=semantic_buoys_w,
                semantic_buoy_colors=semantic_buoy_colors,
                use_semantic_vortex=self.use_semantic_vortex,
                vortex_gain=self.vortex_gain,
                vortex_u_inf=self.vortex_u_inf,
                vortex_model=self._vortex_model_params(),
                vehicle_nu_b=self.nu,
                vortex_dt=self.dt,
                vortex_max_distance_m=self.vortex_max_distance_m,
                iala_region=self.iala_region,
                vortex_ahead_only=self.vortex_ahead_only,
                vortex_behind_tol_m=self.vortex_behind_tol_m,
                ka=self.ka,
                kr=self.kr,
                rho=self.rho,
                r_usv=self.r_usv,
                c_safe=self.c_safe,
                Fcap=self.Fcap,
                d_slide=self.d_slide,
                w_max=self.w_max,
                k_wall=self.k_wall,
                f_threshold=self.f_threshold,
                d_wf=self.d_wf,
                wf_gain=self.wf_gain,
                wf_blend=self.wf_blend,
                L_eff=self.L_eff,
                alpha=self.alpha,
            )

            f_xy_b = dbg[2]
            fmag = float(np.linalg.norm(f_xy_b))
            if fmag < 1e-6:
                break

            # Differential-drive-consistent desired-path rollout.  Preserve
            # the full 2-D APF field for steering, but respect the actual
            # underactuated command structure: Ty = 0 and APF surge is never
            # commanded backwards.  The lateral field component turns the
            # virtual vessel; forward motion is scaled by cos(e_apf).
            e_apf = wrap_angle(math.atan2(float(f_xy_b[1]), float(f_xy_b[0])))
            forward_scale = max(0.0, math.cos(e_apf))

            dpsi_max = float(self.apf_path_max_yaw_rate) * float(self.dt)
            psi = wrap_angle(psi + float(np.clip(e_apf, -dpsi_max, dpsi_max)))

            dir_forward_w = np.array(
                [math.cos(psi), math.sin(psi)], dtype=np.float64
            )
            p_w = p_w + ds * forward_scale * dir_forward_w

        self.pub_desired_path.publish(self.desired_path_msg)

    def _publish_goal_marker(self, now_msg):
        frame = self._current_viz_frame()

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

    # ============================================================
    # Main loop
    # ============================================================

    def _tick(self):
        if not (self.have_pose and self.have_goal and self.have_nu):
            return

        if self.map_msg is None:
            obstacles = np.zeros((0, 2), dtype=np.float64)
        else:
            obstacles = self._collect_obstacles(self.map_msg)

        semantic_buoys_w, semantic_buoy_colors = self._semantic_arrays()

        pos_w = np.array([self.x, self.y], dtype=np.float64)
        goal_w = np.array([self.goal[0], self.goal[1]], dtype=np.float64)

        # ------------------------------------------------------------
        # Differential-drive APF projection
        # ------------------------------------------------------------
        tau_apf, _dbg = apf_tau_diff_drive_alignment(
            pos_w=pos_w,
            psi_w=float(self.psi),
            goal_w=goal_w,
            obstacles_w=obstacles,
            semantic_buoys_w=semantic_buoys_w,
            semantic_buoy_colors=semantic_buoy_colors,
            use_semantic_vortex=self.use_semantic_vortex,
            vortex_gain=self.vortex_gain,
            vortex_u_inf=self.vortex_u_inf,
            vortex_model=self._vortex_model_params(),
            vehicle_nu_b=self.nu,
            vortex_dt=self.dt,
            vortex_max_distance_m=self.vortex_max_distance_m,
            iala_region=self.iala_region,
            vortex_ahead_only=self.vortex_ahead_only,
            vortex_behind_tol_m=self.vortex_behind_tol_m,
            ka=self.ka,
            kr=self.kr,
            rho=self.rho,
            r_usv=self.r_usv,
            c_safe=self.c_safe,
            Fcap=self.Fcap,
            d_slide=self.d_slide,
            w_max=self.w_max,
            k_wall=self.k_wall,
            f_threshold=self.f_threshold,
            d_wf=self.d_wf,
            wf_gain=self.wf_gain,
            wf_blend=self.wf_blend,
            K_yaw_apf=self.K_yaw_apf,
            Mz_cap=self.Mz_max,
        )

        # Debug values currently not published, but kept for easy print/log additions.
        _e_apf = float(_dbg[5])
        _Tx_apf = float(_dbg[6])
        _Mz_apf = float(_dbg[7])

        # ------------------------------------------------------------
        # Optional lookahead yaw for direct APF
        # ------------------------------------------------------------
        if self.use_lookahead_yaw:
            lh = self._compute_lookahead_yaw_error(
                obstacles,
                semantic_buoys_w,
                semantic_buoy_colors,
            )
            if lh is not None:
                e_lh, _target_lh, _arc_lh = lh
                fmag = float(np.linalg.norm(_dbg[2]))
                field_strength = min(1.0, fmag / max(float(self.Fcap), 1e-6))
                field_strength = max(float(self.yaw_lookahead_min_field_strength), field_strength)

                Mz_lh = self.K_yaw_apf * field_strength * math.tanh(e_lh)
                Mz_lh = float(np.clip(Mz_lh, -self.Mz_max, self.Mz_max))

                # Blend with the original local APF yaw if desired. With blend=1.0,
                # direct APF yaw is fully based on the APF lookahead point.
                tau_apf[2] = (
                    (1.0 - self.yaw_lookahead_blend) * float(tau_apf[2])
                    + self.yaw_lookahead_blend * Mz_lh
                )

        # ------------------------------------------------------------
        # Dynamic compensation
        # ------------------------------------------------------------
        nu_g_vec = self.nu.copy()

        current_body = current_body_from_heading(
            psi=float(self.psi),
            current_speed=self.current_speed,
            current_beta_deg=self.current_beta,
        )

        nu_r_vec = nu_g_vec - current_body

        nu = nu_g_vec.reshape(3, 1)
        nu_r = nu_r_vec.reshape(3, 1)

        CRB = self._C_RB(nu_g_vec)
        CA = self._C_A(nu_r_vec)
        D = self._D(nu_r_vec)

        tau_apf_vec = tau_apf.reshape(3, 1)

        tau_model = (CRB @ nu) + (CA @ nu_r) + (D @ nu_r)

        # Surge: keep model compensation for Tx.
        Tx_raw = (
            float(tau_model[0, 0])
            + float(tau_apf_vec[0, 0])
            - float(self.Kd[0, 0]) * float(nu[0, 0])
        )
        Tx = float(np.clip(Tx_raw, -self.F_max, self.F_max))

        # Yaw: APF field alignment plus optional yaw-rate damping.
        r = float(nu[2, 0])
        Mz_apf = float(tau_apf_vec[2, 0])
        Kd_yaw = float(self.Kd[2, 2])
        Mz_raw = Mz_apf - Kd_yaw * r
        Mz = float(np.clip(Mz_raw, -self.Mz_max, self.Mz_max))

        Ty = 0.0

        msg = ControlEffort()
        msg.tau = [Tx, Ty, Mz]
        self.pub_tau.publish(msg)

        # ------------------------------------------------------------
        # RViz actual path, desired path, goal, and vehicle pose marker
        # ------------------------------------------------------------
        now_msg = self.get_clock().now().to_msg()

        # Vehicle pose marker is published by _vehicle_pose_timer_cb(),
        # independently of goal/odom/map availability.

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

        self._build_desired_path_from_apf(
            now_msg,
            obstacles,
            semantic_buoys_w,
            semantic_buoy_colors,
        )
        self._publish_goal_marker(now_msg)


# ============================================================
# Main
# ============================================================

def main(args=None):
    rclpy.init(args=args)
    node = APF_SKC_Controller()

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()