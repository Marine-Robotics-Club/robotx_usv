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


class ZedGateGoalObstaclePublisher(Node):
    """
    Semantic-map gate goal publisher.

    This replaces the old direct-ZED-detection gate selector.

    Inputs:
      /asv/map/semantic_buoys   std_msgs/String JSON from semantic map node
      /asv/vehicle_pose         geometry_msgs/Pose2D

    Outputs:
      /asv/nav/goal             geometry_msgs/Pose2D
      /asv/viz/semantic_gate_goal visualization_msgs/MarkerArray

    Coordinate convention:
      x = North [m]
      y = East  [m]
      theta = yaw/heading in NED map frame

    Gate logic:
      - Build candidate gates only from red + green buoy pairs.
      - Goal 1 can be the gate midpoint.
      - Goal 2 can be the through-gate point.
      - Through-gate direction is selected as the side of the gate that is more
        forward relative to the current USV heading.
    """

    def __init__(self):
        super().__init__("semantic_gate_goal_publisher")

        # ============================================================
        # Topics
        # ============================================================
        self.declare_parameter("wamv", "asv")
        self.declare_parameter("semantic_buoys_topic", "map/semantic_buoys")
        self.declare_parameter("pose_topic", "/asv/vehicle_pose")
        self.declare_parameter("goal_topic", "/asv/nav/goal")
        self.declare_parameter("marker_topic", "/asv/viz/semantic_gate_goal")
        self.declare_parameter("frame_id", "map")

        self.wamv = str(self.get_parameter("wamv").value).strip("/")
        self.semantic_buoys_topic_param = str(
            self.get_parameter("semantic_buoys_topic").value
        ).strip("/")

        self.semantic_buoys_topic = f"/{self.wamv}/{self.semantic_buoys_topic_param}".replace("//", "/")
        self.pose_topic = str(self.get_parameter("pose_topic").value)
        self.goal_topic = str(self.get_parameter("goal_topic").value)
        self.marker_topic = str(self.get_parameter("marker_topic").value)
        self.frame_id = str(self.get_parameter("frame_id").value)

        # ============================================================
        # Semantic buoy filtering
        # ============================================================
        self.declare_parameter("semantic_timeout_s", 3.0)
        self.declare_parameter("min_buoy_count", 2)

        self.semantic_timeout_s = float(self.get_parameter("semantic_timeout_s").value)
        self.min_buoy_count = int(self.get_parameter("min_buoy_count").value)

        # ============================================================
        # Gate and goal parameters
        # ============================================================
        self.declare_parameter("min_gate_width_m", 1.0)
        self.declare_parameter("max_gate_width_m", 8.0)
        self.declare_parameter("through_gate_distance_m", 3.0)
        self.declare_parameter("reach_threshold_m", 1.5)
        self.declare_parameter("publish_period_s", 0.10)
        self.declare_parameter("gate_behind_allow_m", 2.0)

        # If true:
        #   waypoint 1 = gate midpoint
        #   waypoint 2 = through-gate point
        # If false:
        #   publish only through-gate point
        self.declare_parameter("use_midpoint_first", True)

        # Number of red-green gates to complete.
        # Use 1 for a single red-green gate.
        # Use 2 if you want two separate gates.
        self.declare_parameter("total_gates", 2)

        # Prevent selecting the same physical gate again.
        self.declare_parameter("completed_gate_match_distance_m", 6.0)
        self.declare_parameter("min_new_gate_separation_m", 7.0)

        # Keep refining active goal while semantic map updates.
        self.declare_parameter("update_active_goal_from_semantic_map", True)
        self.declare_parameter("active_gate_update_max_midpoint_jump_m", 6.0)

        # Look ahead for the next red-green gate while current gate is active.
        self.declare_parameter("lookahead_next_gate_enabled", True)
        self.declare_parameter("pending_gate_update_max_midpoint_jump_m", 8.0)

        # If the next gate is detected while we are driving to the midpoint
        # of the current gate, do not command the current gate through-point.
        # Instead, mark the current gate complete at the midpoint and switch
        # immediately to the next gate. This is useful when the next gate is
        # visible early and the through-point would make the boat turn back or
        # stop before continuing.
        self.declare_parameter("skip_through_point_if_next_gate_detected", True)

        self.min_gate_width_m = float(self.get_parameter("min_gate_width_m").value)
        self.max_gate_width_m = float(self.get_parameter("max_gate_width_m").value)
        self.through_gate_distance_m = float(self.get_parameter("through_gate_distance_m").value)
        self.reach_threshold_m = float(self.get_parameter("reach_threshold_m").value)
        self.publish_period_s = float(self.get_parameter("publish_period_s").value)
        self.gate_behind_allow_m = float(self.get_parameter("gate_behind_allow_m").value)
        self.use_midpoint_first = bool(self.get_parameter("use_midpoint_first").value)

        self.total_gates = int(self.get_parameter("total_gates").value)
        self.completed_gate_match_distance_m = float(
            self.get_parameter("completed_gate_match_distance_m").value
        )
        self.min_new_gate_separation_m = float(
            self.get_parameter("min_new_gate_separation_m").value
        )
        self.update_active_goal_from_semantic_map = bool(
            self.get_parameter("update_active_goal_from_semantic_map").value
        )
        self.active_gate_update_max_midpoint_jump_m = float(
            self.get_parameter("active_gate_update_max_midpoint_jump_m").value
        )
        self.lookahead_next_gate_enabled = bool(
            self.get_parameter("lookahead_next_gate_enabled").value
        )
        self.pending_gate_update_max_midpoint_jump_m = float(
            self.get_parameter("pending_gate_update_max_midpoint_jump_m").value
        )
        self.skip_through_point_if_next_gate_detected = bool(
            self.get_parameter("skip_through_point_if_next_gate_detected").value
        )

        self.waypoints_per_gate = 2 if self.use_midpoint_first else 1
        self.total_expected_waypoints = self.total_gates * self.waypoints_per_gate

        # ============================================================
        # Vehicle pose state
        # ============================================================
        self.have_pose = False

        # Global NED:
        # x = North
        # y = East
        # theta = heading/yaw
        self.x_usv_ned = 0.0
        self.y_usv_ned = 0.0
        self.psi_usv_ned = 0.0

        # ============================================================
        # Semantic buoys from map
        # ============================================================
        self.semantic_buoys: List[Dict] = []
        self.last_semantic_msg_time: Optional[float] = None

        # ============================================================
        # Gate mission state
        # ============================================================
        self.active_gate_id: Optional[str] = None
        self.active_gate_number = 0

        self.completed_gate_ids = set()
        self.completed_gate_midpoints: List[Tuple[float, float]] = []

        self.current_waypoints: List[Pose2D] = []
        self.current_waypoint_idx = 0
        self.mission_active = False
        self.mission_complete_logged = False

        self.current_gate_debug: Optional[Dict] = None
        self.last_refined_goal: Optional[Pose2D] = None

        self.pending_gate_id: Optional[str] = None
        self.pending_gate_debug: Optional[Dict] = None
        self.pending_gate_waypoints: List[Pose2D] = []
        self.pending_gate_number = 0

        # ============================================================
        # Logging timers
        # ============================================================
        self.last_no_pose_warn_time = 0.0
        self.no_pose_warn_period = 2.0

        self.last_no_semantic_warn_time = 0.0
        self.no_semantic_warn_period = 2.0

        self.last_invalid_warn_time = 0.0
        self.invalid_warn_period = 1.0

        self.last_status_time = 0.0
        self.status_period = 2.0

        # ============================================================
        # QoS
        # ============================================================
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

        # ============================================================
        # Subscribers
        # ============================================================
        self.pose_sub = self.create_subscription(
            Pose2D,
            self.pose_topic,
            self.pose_callback,
            pose_qos,
        )

        self.semantic_sub = self.create_subscription(
            String,
            self.semantic_buoys_topic,
            self.semantic_buoys_callback,
            semantic_qos,
        )

        # ============================================================
        # Publishers
        # ============================================================
        self.goal_pub = self.create_publisher(
            Pose2D,
            self.goal_topic,
            goal_qos,
        )

        self.marker_pub = self.create_publisher(
            MarkerArray,
            self.marker_topic,
            marker_qos,
        )

        self.timer = self.create_timer(
            self.publish_period_s,
            self.timer_callback,
        )

        self.get_logger().info("Semantic-map red-green gate goal publisher started")
        self.get_logger().info(f"Subscribing semantic buoys: {self.semantic_buoys_topic}")
        self.get_logger().info(f"Subscribing pose:           {self.pose_topic}")
        self.get_logger().info(f"Publishing goal:            {self.goal_topic}")
        self.get_logger().info(f"Publishing markers:         {self.marker_topic}")
        self.get_logger().info(f"Frame ID:                   {self.frame_id}")
        self.get_logger().info(f"Expected gates:             {self.total_gates}")
        self.get_logger().info(f"Expected waypoints:         {self.total_expected_waypoints}")
        self.get_logger().info(f"Min gate width:             {self.min_gate_width_m:.2f} m")
        self.get_logger().info(f"Max gate width:             {self.max_gate_width_m:.2f} m")
        self.get_logger().info(f"Through-gate distance:      {self.through_gate_distance_m:.2f} m")
        self.get_logger().info(f"Min new gate separation:    {self.min_new_gate_separation_m:.2f} m")
        self.get_logger().info(
            f"Skip through-point if next gate detected: "
            f"{self.skip_through_point_if_next_gate_detected}"
        )
        self.get_logger().info("Gate source: semantic map colors red + green")

    # ============================================================
    # Basic helpers
    # ============================================================

    @staticmethod
    def is_finite(*values) -> bool:
        for v in values:
            try:
                if not math.isfinite(float(v)):
                    return False
            except Exception:
                return False
        return True

    @staticmethod
    def pi_wrap(angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def warn_limited(self, text: str):
        now = time.time()
        if (now - self.last_invalid_warn_time) >= self.invalid_warn_period:
            self.last_invalid_warn_time = now
            self.get_logger().warn(text)

    @staticmethod
    def normalize_label(label: str) -> str:
        cls = str(label).strip().lower()
        cls = cls.replace(" ", "_")
        cls = cls.replace("-", "_")
        return cls

    @staticmethod
    def color_from_buoy_dict(buoy: Dict) -> str:
        color = str(buoy.get("color", "")).strip().lower()
        cls = str(buoy.get("class", buoy.get("class_name", ""))).strip().lower()

        if color in ("red", "green", "yellow", "black", "blue"):
            return color

        if "red" in cls:
            return "red"
        if "green" in cls:
            return "green"
        if "yellow" in cls:
            return "yellow"
        if "black" in cls:
            return "black"
        if "blue" in cls:
            return "blue"

        return "unknown"

    @staticmethod
    def is_red_buoy(buoy: Dict) -> bool:
        return ZedGateGoalObstaclePublisher.color_from_buoy_dict(buoy) == "red"

    @staticmethod
    def is_green_buoy(buoy: Dict) -> bool:
        return ZedGateGoalObstaclePublisher.color_from_buoy_dict(buoy) == "green"

    # ============================================================
    # Callbacks
    # ============================================================

    def pose_callback(self, msg: Pose2D):
        x = float(msg.x)
        y = float(msg.y)
        psi = float(msg.theta)

        if not self.is_finite(x, y, psi):
            self.warn_limited(
                f"Received invalid vehicle pose: x={x}, y={y}, theta={psi}"
            )
            return

        self.x_usv_ned = x
        self.y_usv_ned = y
        self.psi_usv_ned = psi
        self.have_pose = True

    def semantic_buoys_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
        except Exception as exc:
            self.warn_limited(f"Could not parse semantic buoy JSON: {exc}")
            return

        raw_buoys = data.get("buoys", [])
        if not isinstance(raw_buoys, list):
            self.warn_limited("semantic_buoys JSON does not contain a valid 'buoys' list")
            return

        now = time.time()
        parsed: List[Dict] = []

        for idx, b in enumerate(raw_buoys):
            if not isinstance(b, dict):
                continue

            color = self.color_from_buoy_dict(b)
            if color not in ("red", "green"):
                continue

            try:
                # Semantic map convention:
                # x / north_m = North
                # y / east_m  = East
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

            class_name = str(b.get("class", f"{color}_buoy"))

            parsed.append(
                {
                    "id": buoy_id,
                    "class_name": class_name,
                    "color": color,
                    "x": x,
                    "y": y,
                    "north_m": x,
                    "east_m": y,
                    "last_seen": now,
                    "raw": b,
                }
            )

        self.semantic_buoys = parsed
        self.last_semantic_msg_time = now

    # ============================================================
    # Coordinate transforms
    # ============================================================

    @staticmethod
    def global_ned_to_body(
        xg: float,
        yg: float,
        x0: float,
        y0: float,
        psi: float,
    ) -> Tuple[float, float]:
        """
        GLOBAL NED -> BODY.

        Global:
          x = North
          y = East

        Body:
          xb = forward
          yb = right
        """
        dx = xg - x0
        dy = yg - y0

        c = math.cos(psi)
        s = math.sin(psi)

        xb = c * dx + s * dy
        yb = -s * dx + c * dy

        return xb, yb

    # ============================================================
    # Semantic buoy helpers
    # ============================================================

    def semantic_is_fresh(self) -> bool:
        if self.last_semantic_msg_time is None:
            return False
        return (time.time() - self.last_semantic_msg_time) <= self.semantic_timeout_s

    def valid_semantic_buoys(self) -> List[Dict]:
        if not self.semantic_is_fresh():
            return []

        valid = []
        for b in self.semantic_buoys:
            if not self.is_finite(b.get("x"), b.get("y")):
                continue
            if b.get("color") not in ("red", "green"):
                continue
            valid.append(b)

        return valid

    # ============================================================
    # Gate selection and waypoint generation
    # ============================================================

    def gate_id_from_buoys(self, red: Dict, green: Dict) -> str:
        return f"red{int(red['id'])}_green{int(green['id'])}"

    def build_gate_candidates(self) -> List[Dict]:
        if not self.have_pose:
            return []

        buoys = self.valid_semantic_buoys()

        reds = [b for b in buoys if self.is_red_buoy(b)]
        greens = [b for b in buoys if self.is_green_buoy(b)]

        candidates: List[Dict] = []

        for red in reds:
            for green in greens:
                rx = float(red["x"])
                ry = float(red["y"])
                gx = float(green["x"])
                gy = float(green["y"])

                gate_vec_x = rx - gx
                gate_vec_y = ry - gy
                width = math.hypot(gate_vec_x, gate_vec_y)

                if width < self.min_gate_width_m or width > self.max_gate_width_m:
                    continue

                mid_x = 0.5 * (rx + gx)
                mid_y = 0.5 * (ry + gy)

                mid_body_x, mid_body_y = self.global_ned_to_body(
                    mid_x,
                    mid_y,
                    self.x_usv_ned,
                    self.y_usv_ned,
                    self.psi_usv_ned,
                )

                # Do not select gates far behind the boat.
                if mid_body_x < -self.gate_behind_allow_m:
                    continue

                # Unit vector along line between red and green.
                ux = gate_vec_x / width
                uy = gate_vec_y / width

                # Two possible normals to the gate line.
                n1_x = -uy
                n1_y = ux

                n2_x = uy
                n2_y = -ux

                p1_x = mid_x + n1_x * self.through_gate_distance_m
                p1_y = mid_y + n1_y * self.through_gate_distance_m

                p2_x = mid_x + n2_x * self.through_gate_distance_m
                p2_y = mid_y + n2_y * self.through_gate_distance_m

                p1_body_x, p1_body_y = self.global_ned_to_body(
                    p1_x,
                    p1_y,
                    self.x_usv_ned,
                    self.y_usv_ned,
                    self.psi_usv_ned,
                )

                p2_body_x, p2_body_y = self.global_ned_to_body(
                    p2_x,
                    p2_y,
                    self.x_usv_ned,
                    self.y_usv_ned,
                    self.psi_usv_ned,
                )

                # Select the through point that is more forward.
                if p1_body_x >= p2_body_x:
                    through_x = p1_x
                    through_y = p1_y
                    through_body_x = p1_body_x
                    through_body_y = p1_body_y
                    normal_x = n1_x
                    normal_y = n1_y
                else:
                    through_x = p2_x
                    through_y = p2_y
                    through_body_x = p2_body_x
                    through_body_y = p2_body_y
                    normal_x = n2_x
                    normal_y = n2_y

                theta_goal = math.atan2(
                    through_y - mid_y,
                    through_x - mid_x,
                )
                theta_goal = self.pi_wrap(theta_goal)

                # Prefer gates in front, centered laterally, and reasonable width.
                score = mid_body_x + 0.25 * abs(mid_body_y) + 0.03 * width

                gate_id = self.gate_id_from_buoys(red, green)

                candidates.append(
                    {
                        "gate_id": gate_id,
                        "red_buoy": red,
                        "green_buoy": green,
                        "red_track": red,
                        "green_track": green,
                        "mid_x": mid_x,
                        "mid_y": mid_y,
                        "through_x": through_x,
                        "through_y": through_y,
                        "theta_goal": theta_goal,
                        "normal_x": normal_x,
                        "normal_y": normal_y,
                        "width": width,
                        "mid_body_x": mid_body_x,
                        "mid_body_y": mid_body_y,
                        "through_body_x": through_body_x,
                        "through_body_y": through_body_y,
                        "score": score,
                    }
                )

        candidates.sort(key=lambda item: item["score"])
        return candidates

    def completed_gate_count(self) -> int:
        return len(self.completed_gate_midpoints)

    def midpoint_distance_between_candidates(self, a: Dict, b: Dict) -> float:
        return math.hypot(
            float(a["mid_x"]) - float(b["mid_x"]),
            float(a["mid_y"]) - float(b["mid_y"]),
        )

    def is_candidate_completed_by_position(self, cand: Dict) -> bool:
        mid_x = float(cand["mid_x"])
        mid_y = float(cand["mid_y"])

        separation_limit = max(
            self.completed_gate_match_distance_m,
            self.min_new_gate_separation_m,
        )

        for done_x, done_y in self.completed_gate_midpoints:
            d = math.hypot(mid_x - done_x, mid_y - done_y)
            if d <= separation_limit:
                return True

        return False

    def is_candidate_too_close_to_active_gate(self, cand: Dict) -> bool:
        if self.current_gate_debug is None:
            return False

        d = self.midpoint_distance_between_candidates(cand, self.current_gate_debug)
        return d < self.min_new_gate_separation_m

    def is_candidate_usable_for_new_gate(self, cand: Dict) -> bool:
        if cand["gate_id"] in self.completed_gate_ids:
            return False

        if self.is_candidate_completed_by_position(cand):
            return False

        if self.active_gate_id is not None and cand["gate_id"] == self.active_gate_id:
            return False

        if self.is_candidate_too_close_to_active_gate(cand):
            return False

        return True

    def candidate_to_waypoints(self, cand: Dict) -> List[Pose2D]:
        midpoint_goal = Pose2D()
        midpoint_goal.x = float(cand["mid_x"])
        midpoint_goal.y = float(cand["mid_y"])
        midpoint_goal.theta = float(cand["theta_goal"])

        through_goal = Pose2D()
        through_goal.x = float(cand["through_x"])
        through_goal.y = float(cand["through_y"])
        through_goal.theta = float(cand["theta_goal"])

        if self.use_midpoint_first:
            return [midpoint_goal, through_goal]

        return [through_goal]

    def clear_pending_gate(self):
        self.pending_gate_id = None
        self.pending_gate_debug = None
        self.pending_gate_waypoints = []
        self.pending_gate_number = 0

    def set_pending_gate_from_candidate(self, cand: Dict, log_first_detection: bool):
        self.pending_gate_id = cand["gate_id"]
        self.pending_gate_debug = cand
        self.pending_gate_waypoints = self.candidate_to_waypoints(cand)
        self.pending_gate_number = self.completed_gate_count() + 2

        if log_first_detection:
            self.get_logger().info(
                f"Prepared next red-green gate {self.pending_gate_number}/{self.total_gates} | "
                f"red_id={cand['red_buoy']['id']} green_id={cand['green_buoy']['id']} | "
                f"mid=({cand['mid_x']:.2f}, {cand['mid_y']:.2f}) | "
                f"through=({cand['through_x']:.2f}, {cand['through_y']:.2f}) | "
                f"width={cand['width']:.2f} m"
            )

    def update_pending_next_gate(self):
        if not self.lookahead_next_gate_enabled:
            return

        if not self.mission_active:
            return

        if self.completed_gate_count() + 1 >= self.total_gates:
            return

        if self.current_gate_debug is None:
            return

        candidates = self.build_gate_candidates()

        if not candidates:
            return

        if self.pending_gate_debug is not None:
            for cand in candidates:
                if cand["gate_id"] == self.pending_gate_id and self.is_candidate_usable_for_new_gate(cand):
                    self.set_pending_gate_from_candidate(cand, log_first_detection=False)
                    return

            old_pending = self.pending_gate_debug
            best_cand = None
            best_dist = float("inf")

            for cand in candidates:
                if not self.is_candidate_usable_for_new_gate(cand):
                    continue

                d = self.midpoint_distance_between_candidates(cand, old_pending)

                if d < best_dist:
                    best_dist = d
                    best_cand = cand

            if best_cand is not None and best_dist <= self.pending_gate_update_max_midpoint_jump_m:
                self.set_pending_gate_from_candidate(best_cand, log_first_detection=False)

            return

        for cand in candidates:
            if not self.is_candidate_usable_for_new_gate(cand):
                continue

            self.set_pending_gate_from_candidate(cand, log_first_detection=True)
            return

    def start_pending_gate_if_available(self) -> bool:
        if self.pending_gate_debug is None:
            return False

        if self.completed_gate_count() >= self.total_gates:
            self.clear_pending_gate()
            return False

        pending = self.pending_gate_debug

        if self.is_candidate_completed_by_position(pending):
            self.clear_pending_gate()
            return False

        self.active_gate_id = self.pending_gate_id
        self.active_gate_number = self.completed_gate_count() + 1
        self.current_gate_debug = pending
        self.current_waypoints = self.pending_gate_waypoints
        self.current_waypoint_idx = 0
        self.mission_active = True
        self.last_refined_goal = self.current_waypoints[self.current_waypoint_idx]

        self.clear_pending_gate()

        self.get_logger().info(
            f"Switching immediately to prepared red-green gate "
            f"{self.active_gate_number}/{self.total_gates}."
        )
        self.announce_current_waypoint()
        self.publish_current_goal()
        return True

    def find_latest_active_gate_candidate(self) -> Optional[Dict]:
        if self.current_gate_debug is None:
            return None

        candidates = self.build_gate_candidates()

        if not candidates:
            return None

        for cand in candidates:
            if cand["gate_id"] == self.active_gate_id:
                return cand

        old_mid_x = float(self.current_gate_debug["mid_x"])
        old_mid_y = float(self.current_gate_debug["mid_y"])

        best_cand = None
        best_dist = float("inf")

        for cand in candidates:
            mid_x = float(cand["mid_x"])
            mid_y = float(cand["mid_y"])
            d = math.hypot(mid_x - old_mid_x, mid_y - old_mid_y)

            if d < best_dist:
                best_dist = d
                best_cand = cand

        if best_cand is not None and best_dist <= self.active_gate_update_max_midpoint_jump_m:
            return best_cand

        return None

    def refresh_active_gate_waypoints(self):
        if not self.update_active_goal_from_semantic_map:
            return

        if not self.mission_active:
            return

        if self.current_waypoint_idx >= len(self.current_waypoints):
            return

        latest = self.find_latest_active_gate_candidate()

        if latest is None:
            return

        self.current_gate_debug = latest
        self.active_gate_id = latest["gate_id"]
        self.current_waypoints = self.candidate_to_waypoints(latest)

        new_goal = self.current_waypoints[self.current_waypoint_idx]
        self.last_refined_goal = Pose2D()
        self.last_refined_goal.x = float(new_goal.x)
        self.last_refined_goal.y = float(new_goal.y)
        self.last_refined_goal.theta = float(new_goal.theta)

    def select_new_gate_if_needed(self):
        if self.mission_active:
            return

        if self.completed_gate_count() >= self.total_gates:
            if not self.mission_complete_logged:
                self.mission_complete_logged = True
                self.get_logger().info(
                    f"Gate mission complete: {self.completed_gate_count()}/"
                    f"{self.total_gates} gates, "
                    f"{self.total_expected_waypoints}/{self.total_expected_waypoints} waypoints."
                )
            return

        candidates = self.build_gate_candidates()

        if not candidates:
            return

        chosen = None

        for cand in candidates:
            if not self.is_candidate_usable_for_new_gate(cand):
                continue
            chosen = cand
            break

        if chosen is None:
            return

        self.active_gate_id = chosen["gate_id"]
        self.active_gate_number = self.completed_gate_count() + 1
        self.current_gate_debug = chosen
        self.current_waypoints = self.candidate_to_waypoints(chosen)
        self.current_waypoint_idx = 0
        self.mission_active = True
        self.last_refined_goal = self.current_waypoints[self.current_waypoint_idx]

        self.get_logger().info(
            f"Selected red-green gate {self.active_gate_number}/{self.total_gates} | "
            f"red_id={chosen['red_buoy']['id']} "
            f"green_id={chosen['green_buoy']['id']} | "
            f"red=({chosen['red_buoy']['x']:.2f}, {chosen['red_buoy']['y']:.2f}) | "
            f"green=({chosen['green_buoy']['x']:.2f}, {chosen['green_buoy']['y']:.2f}) | "
            f"mid=({chosen['mid_x']:.2f}, {chosen['mid_y']:.2f}) | "
            f"through=({chosen['through_x']:.2f}, {chosen['through_y']:.2f}) | "
            f"width={chosen['width']:.2f} m | "
            f"mid_body=({chosen['mid_body_x']:.2f}, {chosen['mid_body_y']:.2f})"
        )

        self.announce_current_waypoint()

    def current_global_waypoint_number(self) -> int:
        gate_offset = max(0, self.active_gate_number - 1) * self.waypoints_per_gate
        return gate_offset + self.current_waypoint_idx + 1

    def waypoint_kind(self, local_idx: int) -> str:
        if self.use_midpoint_first and local_idx == 0:
            return "mid-gate"
        return "through-gate"

    def announce_current_waypoint(self):
        if not self.mission_active:
            return

        if self.current_waypoint_idx >= len(self.current_waypoints):
            return

        goal = self.current_waypoints[self.current_waypoint_idx]
        global_wp = self.current_global_waypoint_number()
        kind = self.waypoint_kind(self.current_waypoint_idx)

        self.get_logger().info(
            f"Going to waypoint {global_wp}/{self.total_expected_waypoints} "
            f"({kind}) | gate {self.active_gate_number}/{self.total_gates} | "
            f"goal=({goal.x:.2f}, {goal.y:.2f}, theta={goal.theta:.2f})"
        )

    def distance_to_current_goal(self) -> float:
        if not self.mission_active:
            return float("inf")

        if self.current_waypoint_idx >= len(self.current_waypoints):
            return float("inf")

        goal = self.current_waypoints[self.current_waypoint_idx]

        dx = float(goal.x) - self.x_usv_ned
        dy = float(goal.y) - self.y_usv_ned

        return math.hypot(dx, dy)

    def publish_current_goal(self):
        if not self.mission_active:
            return

        if self.current_waypoint_idx >= len(self.current_waypoints):
            return

        goal = self.current_waypoints[self.current_waypoint_idx]
        self.goal_pub.publish(goal)

    def mark_active_gate_complete(self, reason: str = ""):
        """
        Mark the current active gate as completed and clear active mission state.

        This helper is used in two cases:
          1. Normal behavior: the boat reaches the final waypoint of the gate.
          2. Fast gate-chain behavior: the boat reaches the midpoint and the next
             gate is already detected, so we skip the current through-point and
             switch to the next gate immediately.
        """
        if self.active_gate_id is not None:
            self.completed_gate_ids.add(self.active_gate_id)

        if self.current_gate_debug is not None:
            self.completed_gate_midpoints.append(
                (
                    float(self.current_gate_debug["mid_x"]),
                    float(self.current_gate_debug["mid_y"]),
                )
            )

        suffix = f" {reason}" if reason else ""
        self.get_logger().info(
            f"Completed red-green gate {self.active_gate_number}/{self.total_gates}. "
            f"Completed gates: {self.completed_gate_count()}/{self.total_gates}."
            f"{suffix}"
        )

        self.active_gate_id = None
        self.active_gate_number = 0
        self.current_waypoints = []
        self.current_waypoint_idx = 0
        self.mission_active = False
        self.current_gate_debug = None
        self.last_refined_goal = None

    def should_skip_current_through_point(self) -> bool:
        """
        Return True when we should skip the through-point of the current gate.

        Condition:
          - Parameter enabled.
          - The planner uses midpoint first.
          - We just reached the midpoint waypoint of the active gate.
          - A pending next gate has already been detected.
          - We still need more gates after the current one.
        """
        if not self.skip_through_point_if_next_gate_detected:
            return False

        if not self.use_midpoint_first:
            return False

        if self.current_waypoint_idx != 0:
            return False

        if self.pending_gate_debug is None:
            return False

        if self.completed_gate_count() + 1 >= self.total_gates:
            return False

        return True

    def update_waypoint_progress(self):
        if not self.mission_active:
            return

        if not self.have_pose:
            return

        dist = self.distance_to_current_goal()

        if dist > self.reach_threshold_m:
            return

        reached_global_wp = self.current_global_waypoint_number()
        reached_kind = self.waypoint_kind(self.current_waypoint_idx)

        self.get_logger().info(
            f"Reached waypoint {reached_global_wp}/{self.total_expected_waypoints} "
            f"({reached_kind}) within {self.reach_threshold_m:.2f} m."
        )

        # New behavior requested for chained gates:
        # If we reach the midpoint of the current gate and the next gate is
        # already detected, do not command the current through-point. Switch to
        # the next gate immediately.
        if self.should_skip_current_through_point():
            pending = self.pending_gate_debug
            self.get_logger().info(
                "Next gate already detected; skipping current gate through-point "
                "and switching to the next gate. "
                f"next_red_id={pending['red_buoy']['id']} "
                f"next_green_id={pending['green_buoy']['id']} | "
                f"next_mid=({pending['mid_x']:.2f}, {pending['mid_y']:.2f})"
            )
            self.mark_active_gate_complete(
                reason="Skipped through-point because next gate was detected."
            )
            if self.start_pending_gate_if_available():
                return
            return

        if self.current_waypoint_idx < len(self.current_waypoints) - 1:
            self.current_waypoint_idx += 1
            self.announce_current_waypoint()
            self.publish_current_goal()
            return

        self.mark_active_gate_complete()

        if self.start_pending_gate_if_available():
            return

    # ============================================================
    # RViz markers
    # ============================================================

    def color_for_buoy(self, buoy: Dict) -> Tuple[float, float, float]:
        color = buoy.get("color", "unknown")

        if color == "red":
            return 1.0, 0.0, 0.0

        if color == "green":
            return 0.0, 1.0, 0.0

        return 1.0, 1.0, 1.0

    def add_sphere_marker(
        self,
        marker_array: MarkerArray,
        marker_id: int,
        ns: str,
        x: float,
        y: float,
        z: float,
        diameter: float,
        color: Tuple[float, float, float],
        alpha: float,
    ) -> int:
        r, g, b = color

        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = self.frame_id
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        marker.pose.position.z = float(z)
        marker.pose.orientation.w = 1.0

        marker.scale.x = float(diameter)
        marker.scale.y = float(diameter)
        marker.scale.z = 0.35

        marker.color.r = float(r)
        marker.color.g = float(g)
        marker.color.b = float(b)
        marker.color.a = float(alpha)

        marker_array.markers.append(marker)
        return marker_id + 1

    def add_text_marker(
        self,
        marker_array: MarkerArray,
        marker_id: int,
        ns: str,
        x: float,
        y: float,
        z: float,
        text: str,
    ) -> int:
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = self.frame_id
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD

        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        marker.pose.position.z = float(z)
        marker.pose.orientation.w = 1.0

        marker.scale.z = 0.45

        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        marker.text = text
        marker_array.markers.append(marker)
        return marker_id + 1

    def add_line_marker(
        self,
        marker_array: MarkerArray,
        marker_id: int,
        ns: str,
        points_xy: List[Tuple[float, float]],
        color: Tuple[float, float, float],
        alpha: float,
        width: float,
    ) -> int:
        r, g, b = color

        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = self.frame_id
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        marker.scale.x = float(width)

        marker.color.r = float(r)
        marker.color.g = float(g)
        marker.color.b = float(b)
        marker.color.a = float(alpha)

        for x, y in points_xy:
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = 0.35
            marker.points.append(p)

        marker_array.markers.append(marker)
        return marker_id + 1

    def build_marker_msg(self) -> MarkerArray:
        marker_array = MarkerArray()
        now_msg = self.get_clock().now().to_msg()

        clear_marker = Marker()
        clear_marker.header.stamp = now_msg
        clear_marker.header.frame_id = self.frame_id
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)

        marker_id = 0

        for buoy in self.valid_semantic_buoys():
            color = self.color_for_buoy(buoy)

            marker_id = self.add_sphere_marker(
                marker_array=marker_array,
                marker_id=marker_id,
                ns="semantic_buoys",
                x=float(buoy["x"]),
                y=float(buoy["y"]),
                z=0.25,
                diameter=0.60,
                color=color,
                alpha=0.90,
            )

            marker_id = self.add_text_marker(
                marker_array=marker_array,
                marker_id=marker_id,
                ns="semantic_buoy_labels",
                x=float(buoy["x"]),
                y=float(buoy["y"]),
                z=1.0,
                text=(
                    f"{buoy['color']} id={buoy['id']}\n"
                    f"N={float(buoy['x']):.2f}, E={float(buoy['y']):.2f}"
                ),
            )

        candidates = self.build_gate_candidates()

        for cand in candidates:
            marker_id = self.add_line_marker(
                marker_array=marker_array,
                marker_id=marker_id,
                ns="candidate_red_green_gate_lines",
                points_xy=[
                    (float(cand["red_buoy"]["x"]), float(cand["red_buoy"]["y"])),
                    (float(cand["green_buoy"]["x"]), float(cand["green_buoy"]["y"])),
                ],
                color=(0.2, 0.6, 1.0),
                alpha=0.55,
                width=0.06,
            )

        if self.current_gate_debug is not None:
            cand = self.current_gate_debug

            marker_id = self.add_line_marker(
                marker_array=marker_array,
                marker_id=marker_id,
                ns="active_red_green_gate_line",
                points_xy=[
                    (float(cand["red_buoy"]["x"]), float(cand["red_buoy"]["y"])),
                    (float(cand["green_buoy"]["x"]), float(cand["green_buoy"]["y"])),
                ],
                color=(1.0, 1.0, 0.0),
                alpha=1.0,
                width=0.12,
            )

            marker_id = self.add_line_marker(
                marker_array=marker_array,
                marker_id=marker_id,
                ns="active_gate_forward_line",
                points_xy=[
                    (float(cand["mid_x"]), float(cand["mid_y"])),
                    (float(cand["through_x"]), float(cand["through_y"])),
                ],
                color=(1.0, 0.5, 0.0),
                alpha=1.0,
                width=0.10,
            )

            marker_id = self.add_sphere_marker(
                marker_array=marker_array,
                marker_id=marker_id,
                ns="active_gate_midpoint",
                x=float(cand["mid_x"]),
                y=float(cand["mid_y"]),
                z=0.45,
                diameter=0.6,
                color=(1.0, 1.0, 0.0),
                alpha=1.0,
            )

            marker_id = self.add_sphere_marker(
                marker_array=marker_array,
                marker_id=marker_id,
                ns="active_gate_through_point",
                x=float(cand["through_x"]),
                y=float(cand["through_y"]),
                z=0.45,
                diameter=0.8,
                color=(1.0, 0.5, 0.0),
                alpha=1.0,
            )

        for idx, wp in enumerate(self.current_waypoints):
            if idx == self.current_waypoint_idx:
                color = (1.0, 0.0, 1.0)
                diameter = 0.9
            else:
                color = (0.6, 0.0, 0.8)
                diameter = 0.6

            marker_id = self.add_sphere_marker(
                marker_array=marker_array,
                marker_id=marker_id,
                ns="gate_goal_waypoints",
                x=float(wp.x),
                y=float(wp.y),
                z=0.65,
                diameter=diameter,
                color=color,
                alpha=1.0,
            )

            marker_id = self.add_text_marker(
                marker_array=marker_array,
                marker_id=marker_id,
                ns="gate_goal_labels",
                x=float(wp.x),
                y=float(wp.y),
                z=1.35,
                text=f"goal {idx + 1}\nN={wp.x:.2f}, E={wp.y:.2f}",
            )

        return marker_array

    # ============================================================
    # Main timer
    # ============================================================

    def timer_callback(self):
        if not self.have_pose:
            now = time.time()
            if (now - self.last_no_pose_warn_time) >= self.no_pose_warn_period:
                self.last_no_pose_warn_time = now
                self.get_logger().warn("Waiting for /asv/vehicle_pose.")
            return

        if not self.semantic_is_fresh():
            now = time.time()
            if (now - self.last_no_semantic_warn_time) >= self.no_semantic_warn_period:
                self.last_no_semantic_warn_time = now
                self.get_logger().warn(
                    f"Waiting for fresh semantic buoys on {self.semantic_buoys_topic}."
                )
            return

        if len(self.valid_semantic_buoys()) < self.min_buoy_count:
            return

        self.select_new_gate_if_needed()
        self.refresh_active_gate_waypoints()
        self.update_pending_next_gate()
        self.publish_current_goal()
        self.update_waypoint_progress()

        marker_msg = self.build_marker_msg()
        self.marker_pub.publish(marker_msg)

        now = time.time()
        if (now - self.last_status_time) >= self.status_period:
            self.last_status_time = now
            red_count = sum(1 for b in self.valid_semantic_buoys() if b["color"] == "red")
            green_count = sum(1 for b in self.valid_semantic_buoys() if b["color"] == "green")
            candidate_count = len(self.build_gate_candidates())
            self.get_logger().info(
                f"semantic gate status | red={red_count}, green={green_count}, "
                f"candidates={candidate_count}, active={self.mission_active}, "
                f"completed={self.completed_gate_count()}/{self.total_gates}"
            )


def main(args=None):
    rclpy.init(args=args)

    node = ZedGateGoalObstaclePublisher()

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()