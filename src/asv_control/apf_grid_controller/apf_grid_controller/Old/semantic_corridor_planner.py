#!/usr/bin/env python3

import json
import math
import time
from typing import Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from geometry_msgs.msg import Pose2D, Point
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray


class SemanticCorridorPlanner(Node):
    """
    Semantic corridor planner for APF.

    Inputs:
      /asv/map/semantic_buoys   std_msgs/String JSON
      /asv/vehicle_pose         geometry_msgs/Pose2D

    Outputs:
      /asv/nav/goal             geometry_msgs/Pose2D
      /asv/viz/semantic_corridor_planner MarkerArray

    Purpose:
      Select a red-green buoy gate and publish a moving carrot goal along
      the gate centerline, forcing the APF attraction field to pass between
      the selected buoys while the APF occupancy grid handles collision avoidance.
    """

    def __init__(self):
        super().__init__("semantic_corridor_planner")

        # ----------------------------
        # Topics
        # ----------------------------
        self.declare_parameter("wamv", "asv")
        self.declare_parameter("semantic_buoys_topic", "map/semantic_buoys")
        self.declare_parameter("pose_topic", "/asv/vehicle_pose")
        self.declare_parameter("goal_topic", "/asv/nav/goal")
        self.declare_parameter("marker_topic", "/asv/viz/semantic_corridor_planner")
        self.declare_parameter("frame_id", "map")

        self.wamv = str(self.get_parameter("wamv").value).strip("/")
        semantic_param = str(self.get_parameter("semantic_buoys_topic").value).strip("/")
        self.semantic_topic = f"/{self.wamv}/{semantic_param}".replace("//", "/")

        self.pose_topic = str(self.get_parameter("pose_topic").value)
        self.goal_topic = str(self.get_parameter("goal_topic").value)
        self.marker_topic = str(self.get_parameter("marker_topic").value)
        self.frame_id = str(self.get_parameter("frame_id").value)

        # ----------------------------
        # Planner parameters
        # ----------------------------
        self.declare_parameter("semantic_timeout_s", 3.0)
        self.declare_parameter("publish_period_s", 0.10)
        self.declare_parameter("min_buoy_count", 2)

        self.declare_parameter("min_gate_width_m", 1.0)
        self.declare_parameter("max_gate_width_m", 8.0)
        self.declare_parameter("gate_behind_allow_m", 1.0)

        # Corridor geometry:
        # approach point ---- midpoint/gate ---- exit point
        self.declare_parameter("approach_distance_m", 3.0)
        self.declare_parameter("exit_distance_m", 7.0)
        self.declare_parameter("lookahead_m", 3.0)
        self.declare_parameter("goal_reach_threshold_m", 1.2)

        # Gate selection weights. Lower score is better.
        self.declare_parameter("score_lateral_weight", 1.0)
        self.declare_parameter("score_distance_weight", 0.25)
        self.declare_parameter("score_width_weight", 1.2)

        # Latching avoids jumping between the two possible red-green gates.
        self.declare_parameter("latch_gate", True)
        self.declare_parameter("latch_max_midpoint_jump_m", 3.0)
        self.declare_parameter("lost_gate_timeout_s", 3.0)

        # If true, publish final exit point after gate is completed.
        self.declare_parameter("hold_exit_goal", True)

        self.semantic_timeout_s = float(self.get_parameter("semantic_timeout_s").value)
        self.publish_period_s = float(self.get_parameter("publish_period_s").value)
        self.min_buoy_count = int(self.get_parameter("min_buoy_count").value)

        self.min_gate_width_m = float(self.get_parameter("min_gate_width_m").value)
        self.max_gate_width_m = float(self.get_parameter("max_gate_width_m").value)
        self.gate_behind_allow_m = float(self.get_parameter("gate_behind_allow_m").value)

        self.approach_distance_m = float(self.get_parameter("approach_distance_m").value)
        self.exit_distance_m = float(self.get_parameter("exit_distance_m").value)
        self.lookahead_m = float(self.get_parameter("lookahead_m").value)
        self.goal_reach_threshold_m = float(self.get_parameter("goal_reach_threshold_m").value)

        self.score_lateral_weight = float(self.get_parameter("score_lateral_weight").value)
        self.score_distance_weight = float(self.get_parameter("score_distance_weight").value)
        self.score_width_weight = float(self.get_parameter("score_width_weight").value)

        self.latch_gate = bool(self.get_parameter("latch_gate").value)
        self.latch_max_midpoint_jump_m = float(self.get_parameter("latch_max_midpoint_jump_m").value)
        self.lost_gate_timeout_s = float(self.get_parameter("lost_gate_timeout_s").value)
        self.hold_exit_goal = bool(self.get_parameter("hold_exit_goal").value)

        # ----------------------------
        # State
        # ----------------------------
        self.have_pose = False
        self.x_usv = 0.0
        self.y_usv = 0.0
        self.psi_usv = 0.0

        self.semantic_buoys: List[Dict] = []
        self.last_semantic_msg_time: Optional[float] = None

        self.active_gate_id: Optional[str] = None
        self.active_gate: Optional[Dict] = None
        self.last_gate_seen_time: Optional[float] = None
        self.completed = False
        self.final_exit_goal: Optional[Pose2D] = None

        self.last_status_time = 0.0
        self.status_period = 1.0
        self.last_warn_time = 0.0
        self.warn_period = 2.0

        # ----------------------------
        # QoS
        # ----------------------------
        semantic_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        pose_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        goal_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        marker_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.create_subscription(Pose2D, self.pose_topic, self.pose_cb, pose_qos)
        self.create_subscription(String, self.semantic_topic, self.semantic_cb, semantic_qos)

        self.goal_pub = self.create_publisher(Pose2D, self.goal_topic, goal_qos)
        self.marker_pub = self.create_publisher(MarkerArray, self.marker_topic, marker_qos)

        self.timer = self.create_timer(self.publish_period_s, self.timer_cb)

        self.get_logger().info("Semantic corridor planner started")
        self.get_logger().info(f"Sub semantic: {self.semantic_topic}")
        self.get_logger().info(f"Sub pose:     {self.pose_topic}")
        self.get_logger().info(f"Pub goal:     {self.goal_topic}")
        self.get_logger().info(f"Pub marker:   {self.marker_topic}")
        self.get_logger().info(
            f"Corridor: approach={self.approach_distance_m:.2f} m, "
            f"exit={self.exit_distance_m:.2f} m, lookahead={self.lookahead_m:.2f} m"
        )

    # ============================================================
    # Helpers
    # ============================================================

    @staticmethod
    def is_finite(*vals):
        for v in vals:
            try:
                if not math.isfinite(float(v)):
                    return False
            except Exception:
                return False
        return True

    @staticmethod
    def pi_wrap(a):
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a

    def warn_limited(self, text):
        now = time.time()
        if now - self.last_warn_time >= self.warn_period:
            self.last_warn_time = now
            self.get_logger().warn(text)

    @staticmethod
    def color_from_buoy(b):
        color = str(b.get("color", "")).strip().lower()
        cls = str(b.get("class", b.get("class_name", ""))).strip().lower()

        if color in ["red", "green"]:
            return color
        if "red" in cls:
            return "red"
        if "green" in cls:
            return "green"
        return "unknown"

    @staticmethod
    def body_xy(xg, yg, x0, y0, psi):
        dx = xg - x0
        dy = yg - y0
        c = math.cos(psi)
        s = math.sin(psi)
        xb = c * dx + s * dy
        yb = -s * dx + c * dy
        return xb, yb

    def semantic_fresh(self):
        if self.last_semantic_msg_time is None:
            return False
        return (time.time() - self.last_semantic_msg_time) <= self.semantic_timeout_s

    # ============================================================
    # Callbacks
    # ============================================================

    def pose_cb(self, msg):
        x = float(msg.x)
        y = float(msg.y)
        psi = float(msg.theta)

        if not self.is_finite(x, y, psi):
            self.warn_limited("Invalid vehicle pose")
            return

        self.x_usv = x
        self.y_usv = y
        self.psi_usv = psi
        self.have_pose = True

    def semantic_cb(self, msg):
        try:
            data = json.loads(msg.data)
        except Exception as exc:
            self.warn_limited(f"Could not parse semantic map JSON: {exc}")
            return

        raw = data.get("buoys", [])
        if not isinstance(raw, list):
            self.warn_limited("Semantic JSON does not contain 'buoys' list")
            return

        parsed = []
        now = time.time()

        for idx, b in enumerate(raw):
            if not isinstance(b, dict):
                continue

            color = self.color_from_buoy(b)
            if color not in ["red", "green"]:
                continue

            try:
                x = float(b.get("x", b.get("north_m")))
                y = float(b.get("y", b.get("east_m")))
            except Exception:
                continue

            if not self.is_finite(x, y):
                continue

            try:
                buoy_id = int(b.get("id", idx + 1))
            except Exception:
                buoy_id = idx + 1

            parsed.append({
                "id": buoy_id,
                "color": color,
                "x": x,
                "y": y,
                "raw": b,
                "last_seen": now,
            })

        self.semantic_buoys = parsed
        self.last_semantic_msg_time = now

    # ============================================================
    # Gate construction
    # ============================================================

    def gate_id(self, red, green):
        return f"red{int(red['id'])}_green{int(green['id'])}"

    def build_candidates(self):
        if not self.have_pose or not self.semantic_fresh():
            return []

        reds = [b for b in self.semantic_buoys if b["color"] == "red"]
        greens = [b for b in self.semantic_buoys if b["color"] == "green"]

        out = []

        heading_x = math.cos(self.psi_usv)
        heading_y = math.sin(self.psi_usv)

        for red in reds:
            for green in greens:
                rx = float(red["x"])
                ry = float(red["y"])
                gx = float(green["x"])
                gy = float(green["y"])

                gate_x = rx - gx
                gate_y = ry - gy
                width = math.hypot(gate_x, gate_y)

                if width < self.min_gate_width_m or width > self.max_gate_width_m:
                    continue

                mid_x = 0.5 * (rx + gx)
                mid_y = 0.5 * (ry + gy)

                mid_body_x, mid_body_y = self.body_xy(
                    mid_x, mid_y, self.x_usv, self.y_usv, self.psi_usv
                )

                if mid_body_x < -self.gate_behind_allow_m:
                    continue

                ux = gate_x / width
                uy = gate_y / width

                # Normals to the gate line.
                n1x, n1y = -uy, ux
                n2x, n2y = uy, -ux

                # Choose normal pointing most along current USV heading.
                dot1 = n1x * heading_x + n1y * heading_y
                dot2 = n2x * heading_x + n2y * heading_y

                if dot1 >= dot2:
                    nx, ny = n1x, n1y
                else:
                    nx, ny = n2x, n2y

                approach_x = mid_x - nx * self.approach_distance_m
                approach_y = mid_y - ny * self.approach_distance_m

                exit_x = mid_x + nx * self.exit_distance_m
                exit_y = mid_y + ny * self.exit_distance_m

                theta = math.atan2(ny, nx)

                # Lower score is better:
                # centered, close enough ahead, wider gate.
                dist_mid = math.hypot(mid_x - self.x_usv, mid_y - self.y_usv)
                score = (
                    self.score_lateral_weight * abs(mid_body_y)
                    + self.score_distance_weight * dist_mid
                    - self.score_width_weight * width
                )

                out.append({
                    "gate_id": self.gate_id(red, green),
                    "red": red,
                    "green": green,
                    "width": width,
                    "mid_x": mid_x,
                    "mid_y": mid_y,
                    "nx": nx,
                    "ny": ny,
                    "approach_x": approach_x,
                    "approach_y": approach_y,
                    "exit_x": exit_x,
                    "exit_y": exit_y,
                    "theta": self.pi_wrap(theta),
                    "mid_body_x": mid_body_x,
                    "mid_body_y": mid_body_y,
                    "score": score,
                })

        out.sort(key=lambda c: c["score"])
        return out

    def select_gate(self, candidates):
        now = time.time()

        if not candidates:
            if self.active_gate is not None and self.last_gate_seen_time is not None:
                if now - self.last_gate_seen_time > self.lost_gate_timeout_s:
                    self.get_logger().warn("Lost active corridor gate. Clearing latch.")
                    self.active_gate = None
                    self.active_gate_id = None
            return self.active_gate

        if self.latch_gate and self.active_gate is not None:
            # Prefer same ID.
            for c in candidates:
                if c["gate_id"] == self.active_gate_id:
                    self.active_gate = c
                    self.last_gate_seen_time = now
                    return c

            # If IDs change, keep nearest midpoint to previous gate.
            old = self.active_gate
            best = None
            best_d = 999.0
            for c in candidates:
                d = math.hypot(c["mid_x"] - old["mid_x"], c["mid_y"] - old["mid_y"])
                if d < best_d:
                    best_d = d
                    best = c

            if best is not None and best_d <= self.latch_max_midpoint_jump_m:
                self.active_gate = best
                self.active_gate_id = best["gate_id"]
                self.last_gate_seen_time = now
                return best

        chosen = candidates[0]
        self.active_gate = chosen
        self.active_gate_id = chosen["gate_id"]
        self.last_gate_seen_time = now
        self.completed = False
        self.final_exit_goal = None

        self.get_logger().info(
            f"Selected corridor gate | id={chosen['gate_id']} "
            f"red={chosen['red']['id']} green={chosen['green']['id']} "
            f"width={chosen['width']:.2f} m "
            f"mid=({chosen['mid_x']:.2f},{chosen['mid_y']:.2f}) "
            f"exit=({chosen['exit_x']:.2f},{chosen['exit_y']:.2f})"
        )

        return chosen

    # ============================================================
    # Corridor carrot goal
    # ============================================================

    def goal_from_gate(self, gate):
        ax = float(gate["approach_x"])
        ay = float(gate["approach_y"])
        mx = float(gate["mid_x"])
        my = float(gate["mid_y"])
        ex = float(gate["exit_x"])
        ey = float(gate["exit_y"])
        nx = float(gate["nx"])
        ny = float(gate["ny"])

        total_len = self.approach_distance_m + self.exit_distance_m

        # Projection of USV onto corridor axis measured from approach point.
        s_usv = (self.x_usv - ax) * nx + (self.y_usv - ay) * ny

        # Moving carrot ahead of the USV along the corridor.
        s_goal = max(0.0, min(total_len, s_usv + self.lookahead_m))

        gx = ax + nx * s_goal
        gy = ay + ny * s_goal

        # If we are almost through, publish the exit point.
        dist_to_exit = math.hypot(self.x_usv - ex, self.y_usv - ey)
        if s_usv >= total_len - self.goal_reach_threshold_m or dist_to_exit <= self.goal_reach_threshold_m:
            self.completed = True
            gx = ex
            gy = ey

        goal = Pose2D()
        goal.x = float(gx)
        goal.y = float(gy)
        goal.theta = float(gate["theta"])

        if self.completed:
            self.final_exit_goal = Pose2D()
            self.final_exit_goal.x = ex
            self.final_exit_goal.y = ey
            self.final_exit_goal.theta = float(gate["theta"])

        return goal

    def publish_goal(self, goal):
        self.goal_pub.publish(goal)

    # ============================================================
    # Markers
    # ============================================================

    def add_sphere(self, ma, mid, ns, x, y, z, diameter, rgba):
        m = Marker()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = self.frame_id
        m.ns = ns
        m.id = mid
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = float(x)
        m.pose.position.y = float(y)
        m.pose.position.z = float(z)
        m.pose.orientation.w = 1.0
        m.scale.x = float(diameter)
        m.scale.y = float(diameter)
        m.scale.z = min(float(diameter), 0.6)
        m.color.r = float(rgba[0])
        m.color.g = float(rgba[1])
        m.color.b = float(rgba[2])
        m.color.a = float(rgba[3])
        ma.markers.append(m)
        return mid + 1

    def add_text(self, ma, mid, ns, x, y, z, text):
        m = Marker()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = self.frame_id
        m.ns = ns
        m.id = mid
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        m.pose.position.x = float(x)
        m.pose.position.y = float(y)
        m.pose.position.z = float(z)
        m.pose.orientation.w = 1.0
        m.scale.z = 0.45
        m.color.r = 1.0
        m.color.g = 1.0
        m.color.b = 1.0
        m.color.a = 1.0
        m.text = text
        ma.markers.append(m)
        return mid + 1

    def add_line(self, ma, mid, ns, pts, rgba, width):
        m = Marker()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = self.frame_id
        m.ns = ns
        m.id = mid
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = float(width)
        m.color.r = float(rgba[0])
        m.color.g = float(rgba[1])
        m.color.b = float(rgba[2])
        m.color.a = float(rgba[3])

        for x, y in pts:
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = 0.35
            m.points.append(p)

        ma.markers.append(m)
        return mid + 1

    def publish_markers(self, gate, goal, candidates):
        ma = MarkerArray()
        now_msg = self.get_clock().now().to_msg()

        clear = Marker()
        clear.header.stamp = now_msg
        clear.header.frame_id = self.frame_id
        clear.action = Marker.DELETEALL
        ma.markers.append(clear)

        mid = 0

        for b in self.semantic_buoys:
            rgba = (1.0, 0.0, 0.0, 0.9) if b["color"] == "red" else (0.0, 1.0, 0.0, 0.9)
            mid = self.add_sphere(ma, mid, "semantic_buoys", b["x"], b["y"], 0.25, 0.55, rgba)
            mid = self.add_text(ma, mid, "semantic_buoy_labels", b["x"], b["y"], 1.0, f"{b['color']} {b['id']}")

        for c in candidates:
            mid = self.add_line(
                ma, mid, "candidate_gates",
                [(c["red"]["x"], c["red"]["y"]), (c["green"]["x"], c["green"]["y"])],
                (0.2, 0.6, 1.0, 0.35),
                0.04,
            )

        if gate is not None:
            # Selected gate line.
            mid = self.add_line(
                ma, mid, "selected_gate",
                [(gate["red"]["x"], gate["red"]["y"]), (gate["green"]["x"], gate["green"]["y"])],
                (1.0, 1.0, 0.0, 1.0),
                0.12,
            )

            # Corridor centerline.
            mid = self.add_line(
                ma, mid, "corridor_centerline",
                [(gate["approach_x"], gate["approach_y"]), (gate["mid_x"], gate["mid_y"]), (gate["exit_x"], gate["exit_y"])],
                (1.0, 0.5, 0.0, 1.0),
                0.10,
            )

            mid = self.add_sphere(ma, mid, "corridor_approach", gate["approach_x"], gate["approach_y"], 0.45, 0.50, (0.0, 0.6, 1.0, 1.0))
            mid = self.add_sphere(ma, mid, "corridor_mid", gate["mid_x"], gate["mid_y"], 0.55, 0.65, (1.0, 1.0, 0.0, 1.0))
            mid = self.add_sphere(ma, mid, "corridor_exit", gate["exit_x"], gate["exit_y"], 0.45, 0.60, (1.0, 0.5, 0.0, 1.0))

            if goal is not None:
                mid = self.add_sphere(ma, mid, "current_carrot_goal", goal.x, goal.y, 0.75, 0.80, (1.0, 0.0, 1.0, 1.0))
                mid = self.add_text(ma, mid, "current_goal_label", goal.x, goal.y, 1.45, "corridor\ncarrot goal")

        self.marker_pub.publish(ma)

    # ============================================================
    # Main timer
    # ============================================================

    def timer_cb(self):
        if not self.have_pose:
            self.warn_limited("Waiting for vehicle pose.")
            return

        if not self.semantic_fresh():
            self.warn_limited(f"Waiting for semantic buoys on {self.semantic_topic}.")
            return

        if len(self.semantic_buoys) < self.min_buoy_count:
            self.warn_limited("Not enough semantic buoys for corridor planning.")
            return

        candidates = self.build_candidates()
        gate = self.select_gate(candidates)

        if gate is None:
            self.warn_limited("No usable red-green corridor gate.")
            return

        if self.completed and self.hold_exit_goal and self.final_exit_goal is not None:
            goal = self.final_exit_goal
        else:
            goal = self.goal_from_gate(gate)

        self.publish_goal(goal)
        self.publish_markers(gate, goal, candidates)

        now = time.time()
        if now - self.last_status_time >= self.status_period:
            self.last_status_time = now

            reds = sum(1 for b in self.semantic_buoys if b["color"] == "red")
            greens = sum(1 for b in self.semantic_buoys if b["color"] == "green")

            self.get_logger().info(
                f"corridor status | red={reds} green={greens} "
                f"candidates={len(candidates)} active={gate['gate_id']} "
                f"width={gate['width']:.2f} "
                f"goal=({goal.x:.2f},{goal.y:.2f}) "
                f"completed={self.completed}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = SemanticCorridorPlanner()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
