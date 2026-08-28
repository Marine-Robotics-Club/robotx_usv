#!/usr/bin/env python3
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from geometry_msgs.msg import Pose2D, PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import Float32MultiArray, Float64, String
from visualization_msgs.msg import Marker, MarkerArray


# ============================================================
# Helpers
# ============================================================

def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def yaw_to_quaternion(yaw: float) -> Tuple[float, float, float, float]:
    half = 0.5 * float(yaw)
    return 0.0, 0.0, math.sin(half), math.cos(half)


# ============================================================
# APF Path -> PID desired state bridge
# ============================================================

class APFPathToPIDWaypoint(Node):
    """
    Converts the APF desired Path into a simple desired-state command for PID_HS.

    Subscribes:
        /<asv>/viz/apf_desired_path      nav_msgs/Path
        /<asv>/vehicle_pose              geometry_msgs/Pose2D

    Publishes:
        /<asv>/traj_desired_state        std_msgs/Float32MultiArray, size 9
        /<asv>/velD                      std_msgs/Float64
        /<asv>/controller                std_msgs/String, optional
        /<asv>/viz/pid_apf_target        visualization_msgs/MarkerArray, optional

    PID_HS expects traj_desired_state size 9 and reads:
        data[2] = psi_goal
        data[3] = vx desired
        data[4] = vy desired

    This node publishes:
        data = [x_wp, y_wp, psi_des, vx_des, vy_des, 0, 0, 0, 0]

    Coordinates follow your APF/PID convention:
        x = North
        y = East
        psi = atan2(East, North), radians
    """

    def __init__(self):
        super().__init__("apf_path_to_pid_waypoint")

        # ---------------- Parameters ----------------
        self.declare_parameter("asv", "asv")

        self.declare_parameter("apf_path_topic", "viz/apf_desired_path")
        self.declare_parameter("pose_topic", "vehicle_pose")
        self.declare_parameter("desired_state_topic", "traj_desired_state")
        self.declare_parameter("velD_topic", "velD")
        self.declare_parameter("controller_topic", "controller")

        # Desired speed for PID_HS.
        self.declare_parameter("velD", 1.0)

        # Lookahead distance along the APF path.
        self.declare_parameter("lookahead_m", 2.0)

        # If close to final APF path point, target the final point.
        self.declare_parameter("goal_acceptance_m", 0.8)

        # Publishing rate.
        self.declare_parameter("dt", 0.1)

        # Optional controller selector publication.
        self.declare_parameter("publish_controller", True)
        self.declare_parameter("controller_name", "HS")

        # Optional visualization.
        self.declare_parameter("publish_markers", True)
        self.declare_parameter("marker_topic", "viz/pid_apf_target")
        self.declare_parameter("marker_frame_id", "")

        # Minimum distance before recomputing heading. Prevents atan2 noise.
        self.declare_parameter("min_heading_distance_m", 0.15)

        # Waypoint selection mode.
        #   continuous: recompute lookahead waypoint every tick. Good for PID_HS.
        #   discrete:   latch one waypoint and hold it until the vehicle reaches it. Good for RL.
        self.declare_parameter("waypoint_mode", "continuous")

        # In discrete mode, switch to the next APF-path waypoint only after
        # the boat gets this close to the current latched waypoint.
        self.declare_parameter("switch_radius_m", 1.0)

        # In discrete mode, if the APF path changes and the latched waypoint is
        # now far away from the new path, re-latch a new waypoint. This prevents
        # holding an obsolete target if obstacles/path change a lot.
        self.declare_parameter("target_replan_distance_m", 3.0)

        # ---------------- Read parameters ----------------
        self.asv = str(self.get_parameter("asv").value)

        self.apf_path_topic = str(self.get_parameter("apf_path_topic").value)
        self.pose_topic = str(self.get_parameter("pose_topic").value)
        self.desired_state_topic = str(self.get_parameter("desired_state_topic").value)
        self.velD_topic = str(self.get_parameter("velD_topic").value)
        self.controller_topic = str(self.get_parameter("controller_topic").value)

        self.velD = float(self.get_parameter("velD").value)
        self.lookahead_m = float(self.get_parameter("lookahead_m").value)
        self.goal_acceptance_m = float(self.get_parameter("goal_acceptance_m").value)
        self.dt = float(self.get_parameter("dt").value)
        self.publish_controller = bool(self.get_parameter("publish_controller").value)
        self.controller_name = str(self.get_parameter("controller_name").value)
        self.publish_markers = bool(self.get_parameter("publish_markers").value)
        self.marker_topic = str(self.get_parameter("marker_topic").value)
        self.marker_frame_id = str(self.get_parameter("marker_frame_id").value)
        self.min_heading_distance_m = float(self.get_parameter("min_heading_distance_m").value)
        self.waypoint_mode = str(self.get_parameter("waypoint_mode").value).strip().lower()
        self.switch_radius_m = float(self.get_parameter("switch_radius_m").value)
        self.target_replan_distance_m = float(self.get_parameter("target_replan_distance_m").value)

        if self.waypoint_mode in ["latch", "latched"]:
            self.waypoint_mode = "discrete"
        if self.waypoint_mode not in ["continuous", "discrete"]:
            self.get_logger().warn(
                f"Unknown waypoint_mode='{self.waypoint_mode}'. Using continuous."
            )
            self.waypoint_mode = "continuous"
        if self.switch_radius_m <= 0.0:
            self.switch_radius_m = 1.0
        if self.target_replan_distance_m <= 0.0:
            self.target_replan_distance_m = 3.0

        if self.dt <= 0.0:
            self.dt = 0.1
        if self.lookahead_m <= 0.0:
            self.lookahead_m = 1.0
        if self.velD < 0.0:
            self.get_logger().warn("velD was negative. Using abs(velD).")
            self.velD = abs(self.velD)

        # ---------------- State ----------------
        self.have_pose = False
        self.x = 0.0
        self.y = 0.0
        self.psi = 0.0

        self.have_path = False
        self.path_xy = np.zeros((0, 2), dtype=np.float64)
        self.path_frame_id = "map"

        self.last_psi_des = 0.0
        self.last_wp = np.zeros(2, dtype=np.float64)

        # Discrete/latch mode state for RL.
        self.have_latched_wp = False
        self.latched_wp = np.zeros(2, dtype=np.float64)
        self.latched_idx = -1

        # ---------------- QoS ----------------
        path_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        pose_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        # ---------------- Subscribers ----------------
        self.create_subscription(
            Path,
            f"/{self.asv}/{self.apf_path_topic}".replace("//", "/"),
            self._path_cb,
            path_qos,
        )

        self.create_subscription(
            Pose2D,
            f"/{self.asv}/{self.pose_topic}".replace("//", "/"),
            self._pose_cb,
            pose_qos,
        )

        # ---------------- Publishers ----------------
        self.pub_desired_state = self.create_publisher(
            Float32MultiArray,
            f"/{self.asv}/{self.desired_state_topic}".replace("//", "/"),
            10,
        )

        self.pub_velD = self.create_publisher(
            Float64,
            f"/{self.asv}/{self.velD_topic}".replace("//", "/"),
            10,
        )

        if self.publish_controller:
            self.pub_controller = self.create_publisher(
                String,
                f"/{self.asv}/{self.controller_topic}".replace("//", "/"),
                10,
            )
        else:
            self.pub_controller = None

        if self.publish_markers:
            self.pub_markers = self.create_publisher(
                MarkerArray,
                f"/{self.asv}/{self.marker_topic}".replace("//", "/"),
                10,
            )
        else:
            self.pub_markers = None

        self.timer = self.create_timer(self.dt, self._tick)

        self.get_logger().info(
            "APF path to PID waypoint bridge started.\n"
            f"  Path in: /{self.asv}/{self.apf_path_topic}\n"
            f"  Pose in: /{self.asv}/{self.pose_topic}\n"
            f"  Desired state out: /{self.asv}/{self.desired_state_topic}\n"
            f"  velD out: /{self.asv}/{self.velD_topic} = {self.velD:.2f} m/s\n"
            f"  lookahead_m: {self.lookahead_m:.2f}\n"
            f"  waypoint_mode: {self.waypoint_mode}\n"
            f"  switch_radius_m: {self.switch_radius_m:.2f}\n"
            f"  controller publish: {self.publish_controller} ({self.controller_name})"
        )

    # ============================================================
    # Callbacks
    # ============================================================

    def _pose_cb(self, msg: Pose2D):
        self.x = float(msg.x)
        self.y = float(msg.y)
        self.psi = float(msg.theta)
        self.have_pose = True

    def _path_cb(self, msg: Path):
        pts = []
        for ps in msg.poses:
            pts.append((float(ps.pose.position.x), float(ps.pose.position.y)))

        if len(pts) < 2:
            self.have_path = False
            self.path_xy = np.zeros((0, 2), dtype=np.float64)
            return

        self.path_xy = np.array(pts, dtype=np.float64)
        self.path_frame_id = msg.header.frame_id or "map"
        self.have_path = True

    # ============================================================
    # Path target selection
    # ============================================================

    def _select_continuous_waypoint(self) -> Tuple[np.ndarray, int, float, bool]:
        """
        Select a lookahead waypoint from the APF path.

        Returns:
            wp: selected waypoint [x, y]
            idx: selected waypoint index
            d_final: distance to final APF point
            near_final: True if close to final path point
        """
        p = np.array([self.x, self.y], dtype=np.float64)
        path = self.path_xy

        dists = np.linalg.norm(path - p.reshape(1, 2), axis=1)
        i_near = int(np.argmin(dists))

        final = path[-1]
        d_final = float(np.linalg.norm(final - p))
        near_final = d_final <= self.goal_acceptance_m

        if near_final:
            return final.copy(), len(path) - 1, d_final, True

        # Walk forward along the path until cumulative arc length exceeds lookahead.
        accum = 0.0
        idx = i_near
        for j in range(i_near, len(path) - 1):
            seg = float(np.linalg.norm(path[j + 1] - path[j]))
            accum += seg
            idx = j + 1
            if accum >= self.lookahead_m:
                break

        return path[idx].copy(), idx, d_final, False

    def _distance_point_to_current_path(self, q: np.ndarray) -> float:
        """Distance from point q to the latest APF path samples."""
        if self.path_xy.shape[0] == 0:
            return float("inf")
        return float(np.min(np.linalg.norm(self.path_xy - q.reshape(1, 2), axis=1)))

    def _select_discrete_waypoint(self) -> Tuple[np.ndarray, int, float, bool]:
        """
        RL-friendly waypoint selection.

        The APF path can update every tick, but this function holds one fixed
        waypoint until the boat reaches it. Then it selects a new lookahead
        waypoint from the latest APF path.
        """
        p = np.array([self.x, self.y], dtype=np.float64)
        path = self.path_xy
        final = path[-1]
        d_final = float(np.linalg.norm(final - p))
        near_final = d_final <= self.goal_acceptance_m

        # If near the final APF goal, hold the final point.
        if near_final:
            self.latched_wp[:] = final
            self.latched_idx = len(path) - 1
            self.have_latched_wp = True
            return final.copy(), len(path) - 1, d_final, True

        # If we already have a target, keep it until reached, unless the APF
        # path changed so much that this target is no longer near the new path.
        if self.have_latched_wp:
            d_to_latched = float(np.linalg.norm(self.latched_wp - p))
            d_target_to_path = self._distance_point_to_current_path(self.latched_wp)

            reached = d_to_latched <= self.switch_radius_m
            obsolete = d_target_to_path > self.target_replan_distance_m

            if not reached and not obsolete:
                return (
                    self.latched_wp.copy(),
                    int(self.latched_idx),
                    d_final,
                    False,
                )

        # No target, reached target, or target obsolete: select the next
        # lookahead waypoint from the latest APF path and latch it.
        wp, idx, d_final, near_final = self._select_continuous_waypoint()
        self.latched_wp[:] = wp
        self.latched_idx = int(idx)
        self.have_latched_wp = True

        self.get_logger().info(
            f"Latched new APF waypoint idx={idx}, "
            f"x={wp[0]:.2f}, y={wp[1]:.2f}, "
            f"switch_radius={self.switch_radius_m:.2f} m"
        )

        return wp, idx, d_final, near_final

    def _select_waypoint(self) -> Tuple[np.ndarray, int, float, bool]:
        if self.waypoint_mode == "discrete":
            return self._select_discrete_waypoint()
        return self._select_continuous_waypoint()

    # ============================================================
    # Main loop
    # ============================================================

    def _tick(self):
        if not (self.have_pose and self.have_path):
            return

        wp, idx, d_final, near_final = self._select_waypoint()
        p = np.array([self.x, self.y], dtype=np.float64)
        d_wp = float(np.linalg.norm(wp - p))

        if d_wp > self.min_heading_distance_m:
            psi_des = math.atan2(float(wp[1] - self.y), float(wp[0] - self.x))
            psi_des = wrap_angle(psi_des)
            self.last_psi_des = psi_des
        else:
            psi_des = self.last_psi_des

        # Desired world/NED velocity components. PID_HS only uses the magnitude
        # sqrt(vx^2 + vy^2) as velD, and psi as heading target.
        vx_des = self.velD * math.cos(psi_des)
        vy_des = self.velD * math.sin(psi_des)

        desired = Float32MultiArray()
        desired.data = [
            float(wp[0]),      # 0: x desired / North
            float(wp[1]),      # 1: y desired / East
            float(psi_des),    # 2: psi desired, used by PID_HS
            float(vx_des),     # 3: vx desired, used for velD magnitude
            float(vy_des),     # 4: vy desired, used for velD magnitude
            0.0,               # 5: unused
            0.0,               # 6: unused
            0.0,               # 7: unused
            0.0,               # 8: unused
        ]
        self.pub_desired_state.publish(desired)

        self.pub_velD.publish(Float64(data=float(self.velD)))

        if self.pub_controller is not None:
            self.pub_controller.publish(String(data=self.controller_name))

        self.last_wp[:] = wp

        if self.pub_markers is not None:
            self._publish_markers(wp, psi_des, idx, d_final, near_final)

    # ============================================================
    # Visualization
    # ============================================================

    def _publish_markers(self, wp: np.ndarray, psi_des: float, idx: int, d_final: float, near_final: bool):
        now = self.get_clock().now().to_msg()
        frame = self.marker_frame_id.strip() if self.marker_frame_id.strip() else self.path_frame_id

        ma = MarkerArray()

        clear = Marker()
        clear.header.stamp = now
        clear.header.frame_id = frame
        clear.action = Marker.DELETEALL
        ma.markers.append(clear)

        sphere = Marker()
        sphere.header.stamp = now
        sphere.header.frame_id = frame
        sphere.ns = "apf_pid_target"
        sphere.id = 0
        sphere.type = Marker.SPHERE
        sphere.action = Marker.ADD
        sphere.pose.position.x = float(wp[0])
        sphere.pose.position.y = float(wp[1])
        sphere.pose.position.z = 0.35
        sphere.pose.orientation.w = 1.0
        sphere.scale.x = 0.7
        sphere.scale.y = 0.7
        sphere.scale.z = 0.7
        sphere.color.r = 0.2
        sphere.color.g = 1.0
        sphere.color.b = 0.2
        sphere.color.a = 0.95
        ma.markers.append(sphere)

        qx, qy, qz, qw = yaw_to_quaternion(psi_des)
        arrow = Marker()
        arrow.header.stamp = now
        arrow.header.frame_id = frame
        arrow.ns = "apf_pid_target"
        arrow.id = 1
        arrow.type = Marker.ARROW
        arrow.action = Marker.ADD
        arrow.pose.position.x = float(self.x)
        arrow.pose.position.y = float(self.y)
        arrow.pose.position.z = 0.55
        arrow.pose.orientation.x = qx
        arrow.pose.orientation.y = qy
        arrow.pose.orientation.z = qz
        arrow.pose.orientation.w = qw
        arrow.scale.x = 1.5
        arrow.scale.y = 0.15
        arrow.scale.z = 0.25
        arrow.color.r = 0.2
        arrow.color.g = 1.0
        arrow.color.b = 0.2
        arrow.color.a = 0.95
        ma.markers.append(arrow)

        text = Marker()
        text.header.stamp = now
        text.header.frame_id = frame
        text.ns = "apf_pid_target"
        text.id = 2
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position.x = float(wp[0])
        text.pose.position.y = float(wp[1])
        text.pose.position.z = 1.2
        text.pose.orientation.w = 1.0
        text.scale.z = 0.45
        text.color.r = 1.0
        text.color.g = 1.0
        text.color.b = 1.0
        text.color.a = 1.0
        text.text = (
            f"APF target\n"
            f"idx={idx}\n"
            f"psi={math.degrees(psi_des):.1f} deg\n"
            f"velD={self.velD:.1f} m/s\n"
            f"d_final={d_final:.1f} m"
        )
        if near_final:
            text.text += "\nnear final"
        ma.markers.append(text)

        self.pub_markers.publish(ma)


def main(args=None):
    rclpy.init(args=args)
    node = APFPathToPIDWaypoint()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()