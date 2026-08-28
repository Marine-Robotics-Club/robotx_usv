#!/usr/bin/env python3
from __future__ import annotations

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from geometry_msgs.msg import Pose2D, Point
from std_msgs.msg import Bool, Float32, String
from visualization_msgs.msg import Marker, MarkerArray


def yaw_to_quaternion(yaw: float):
    half = 0.5 * float(yaw)
    return 0.0, 0.0, math.sin(half), math.cos(half)


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class FixedTestPoints(Node):
    """
    Continuous 8-waypoint publisher.

    Sequence:

        0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7
        7 -> 6 -> 5 -> 4 -> 3 -> 2 -> 1 -> 0
        0 -> 1 -> 2 -> ...

    Repeats forever.

    Main topics kept the same:

    Published:
      /<asv>/nav/goal                         geometry_msgs/Pose2D
      /<asv>/nav/test_start                   geometry_msgs/Pose2D
      /<asv>/nav/test_middle                  geometry_msgs/Pose2D
      /<asv>/nav/test_final                   geometry_msgs/Pose2D
      /<asv>/test/phase                       std_msgs/String
      /<asv>/test/start_error_m               std_msgs/Float32
      /<asv>/test/middle_error_m              std_msgs/Float32
      /<asv>/test/final_error_m               std_msgs/Float32
      /<asv>/test/start_heading_error_rad     std_msgs/Float32
      /<asv>/test/at_start                    std_msgs/Bool
      /<asv>/test/at_middle                   std_msgs/Bool
      /<asv>/test/start_reached               std_msgs/Bool
      /<asv>/test/middle_reached              std_msgs/Bool
      /<asv>/viz/test_start_goal              visualization_msgs/MarkerArray

    Subscribed:
      /<asv>/vehicle_pose                     geometry_msgs/Pose2D

    Additional waypoint topics:
      /<asv>/nav/test_waypoint_0
      ...
      /<asv>/nav/test_waypoint_7
    """

    def __init__(self):
        super().__init__("fixed_test_points")

        self.declare_parameter("asv", "asv")
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("rate_hz", 5.0)

        # WP0
        self.declare_parameter("waypoint0_x", -20.309)
        self.declare_parameter("waypoint0_y", 51.245)
        self.declare_parameter("waypoint0_theta", 0.0)

        # WP1
        self.declare_parameter("waypoint1_x", -19.770)
        self.declare_parameter("waypoint1_y", 47.282)
        self.declare_parameter("waypoint1_theta", 0.0)

        # WP2
        self.declare_parameter("waypoint2_x", -19.230)
        self.declare_parameter("waypoint2_y", 43.318)
        self.declare_parameter("waypoint2_theta", 0.0)

        # WP3
        self.declare_parameter("waypoint3_x", -18.691)
        self.declare_parameter("waypoint3_y", 39.355)
        self.declare_parameter("waypoint3_theta", 0.0)

        # WP4
        self.declare_parameter("waypoint4_x", -18.151)
        self.declare_parameter("waypoint4_y", 35.391)
        self.declare_parameter("waypoint4_theta", 0.0)

        # WP5
        self.declare_parameter("waypoint5_x", -17.612)
        self.declare_parameter("waypoint5_y", 31.428)
        self.declare_parameter("waypoint5_theta", 0.0)

        # WP6
        self.declare_parameter("waypoint6_x", -17.073)
        self.declare_parameter("waypoint6_y", 27.464)
        self.declare_parameter("waypoint6_theta", 0.0)

        # WP7
        self.declare_parameter("waypoint7_x", -16.533)
        self.declare_parameter("waypoint7_y", 23.501)
        self.declare_parameter("waypoint7_theta", 0.0)

        # Distance at which we consider a waypoint reached.
        self.declare_parameter("switch_radius_m", 1.0)

        # Keep original START heading option.
        # This only applies to waypoint 0.
        self.declare_parameter("require_start_heading", False)
        self.declare_parameter(
            "switch_heading_tolerance_deg",
            180.0,
        )

        # ------------------------------------------------------------
        # General parameters
        # ------------------------------------------------------------
        self.asv = str(self.get_parameter("asv").value)
        self.frame_id = str(self.get_parameter("frame_id").value)

        self.rate_hz = float(self.get_parameter("rate_hz").value)
        if self.rate_hz <= 0.0:
            self.rate_hz = 5.0

        self.switch_radius_m = float(
            self.get_parameter("switch_radius_m").value
        )

        self.require_start_heading = bool(
            self.get_parameter("require_start_heading").value
        )

        self.switch_heading_tol = math.radians(
            float(
                self.get_parameter(
                    "switch_heading_tolerance_deg"
                ).value
            )
        )

        # ------------------------------------------------------------
        # Build waypoint list
        # ------------------------------------------------------------
        self.waypoints = []

        for i in range(8):
            pose = Pose2D()

            pose.x = float(
                self.get_parameter(f"waypoint{i}_x").value
            )

            pose.y = float(
                self.get_parameter(f"waypoint{i}_y").value
            )

            pose.theta = float(
                self.get_parameter(f"waypoint{i}_theta").value
            )

            self.waypoints.append(pose)

        # Legacy aliases.
        #
        # Keeps the original topics available:
        #   test_start  -> WP0
        #   test_middle -> WP1
        #   test_final  -> WP7
        self.start = self.waypoints[0]
        self.middle = self.waypoints[1]
        self.goal = self.waypoints[7]

        # ------------------------------------------------------------
        # Navigation state
        # ------------------------------------------------------------

        # Begin at waypoint 0.
        self.current_index = 0

        # +1 = moving 0 -> 7
        # -1 = moving 7 -> 0
        self.direction = 1

        self.pose = None

        # Legacy reached flags.
        self.start_reached = False
        self.middle_reached = False

        self.last_phase = "WAYPOINT_0"

        # Number of individual waypoint arrivals.
        self.total_waypoints_reached = 0

        # ------------------------------------------------------------
        # QoS
        # ------------------------------------------------------------
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        # ------------------------------------------------------------
        # Subscriber
        # ------------------------------------------------------------
        self.create_subscription(
            Pose2D,
            f"/{self.asv}/vehicle_pose",
            self.pose_cb,
            qos,
        )

        # ------------------------------------------------------------
        # Main publishers -- same topics as before
        # ------------------------------------------------------------
        self.pub_goal = self.create_publisher(
            Pose2D,
            f"/{self.asv}/nav/goal",
            qos,
        )

        self.pub_start = self.create_publisher(
            Pose2D,
            f"/{self.asv}/nav/test_start",
            qos,
        )

        self.pub_middle = self.create_publisher(
            Pose2D,
            f"/{self.asv}/nav/test_middle",
            qos,
        )

        self.pub_final = self.create_publisher(
            Pose2D,
            f"/{self.asv}/nav/test_final",
            qos,
        )

        self.pub_phase = self.create_publisher(
            String,
            f"/{self.asv}/test/phase",
            qos,
        )

        self.pub_markers = self.create_publisher(
            MarkerArray,
            f"/{self.asv}/viz/test_start_goal",
            qos,
        )

        # ------------------------------------------------------------
        # Original diagnostic topics
        # ------------------------------------------------------------
        self.pub_start_error = self.create_publisher(
            Float32,
            f"/{self.asv}/test/start_error_m",
            qos,
        )

        self.pub_middle_error = self.create_publisher(
            Float32,
            f"/{self.asv}/test/middle_error_m",
            qos,
        )

        self.pub_final_error = self.create_publisher(
            Float32,
            f"/{self.asv}/test/final_error_m",
            qos,
        )

        self.pub_start_heading_error = self.create_publisher(
            Float32,
            f"/{self.asv}/test/start_heading_error_rad",
            qos,
        )

        self.pub_at_start = self.create_publisher(
            Bool,
            f"/{self.asv}/test/at_start",
            qos,
        )

        self.pub_at_middle = self.create_publisher(
            Bool,
            f"/{self.asv}/test/at_middle",
            qos,
        )

        self.pub_start_reached = self.create_publisher(
            Bool,
            f"/{self.asv}/test/start_reached",
            qos,
        )

        self.pub_middle_reached = self.create_publisher(
            Bool,
            f"/{self.asv}/test/middle_reached",
            qos,
        )

        # ------------------------------------------------------------
        # New publishers for all 8 fixed waypoint positions
        # ------------------------------------------------------------
        self.pub_waypoints = []

        for i in range(8):
            pub = self.create_publisher(
                Pose2D,
                f"/{self.asv}/nav/test_waypoint_{i}",
                qos,
            )
            self.pub_waypoints.append(pub)

        # ------------------------------------------------------------
        # Timer
        # ------------------------------------------------------------
        self.timer = self.create_timer(
            1.0 / self.rate_hz,
            self.tick,
        )

        # ------------------------------------------------------------
        # Startup information
        # ------------------------------------------------------------
        self.get_logger().info(
            "Continuous 8-waypoint fixed test node started"
        )

        for i, wp in enumerate(self.waypoints):
            self.get_logger().info(
                f"  WP{i}: "
                f"x={wp.x:.3f}, "
                f"y={wp.y:.3f}, "
                f"theta={wp.theta:.3f} rad"
            )

        self.get_logger().info(
            "Sequence: "
            "0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> "
            "6 -> 5 -> 4 -> 3 -> 2 -> 1 -> 0 -> ... FOREVER"
        )

        self.get_logger().info(
            f"switch_radius_m={self.switch_radius_m:.2f}"
        )

        self.get_logger().info(
            f"active target published on /{self.asv}/nav/goal"
        )

    # ================================================================
    # CALLBACK
    # ================================================================

    def pose_cb(self, msg: Pose2D):
        self.pose = msg

    # ================================================================
    # DISTANCE
    # ================================================================

    @staticmethod
    def dist_pose(a: Pose2D, b: Pose2D) -> float:
        return math.hypot(
            float(a.x) - float(b.x),
            float(a.y) - float(b.y),
        )

    # ================================================================
    # ADVANCE WAYPOINT
    # ================================================================

    def advance_waypoint(self):
        """
        Advance through:

          0 -> 1 -> ... -> 7 -> 6 -> ... -> 0 -> 1 -> ...

        forever.
        """

        old_index = self.current_index

        # ------------------------------------------------------------
        # Reached WP7 while going forward.
        # Reverse direction and target WP6.
        # ------------------------------------------------------------
        if (
            self.current_index == 7
            and self.direction == 1
        ):
            self.direction = -1
            self.current_index = 6

        # ------------------------------------------------------------
        # Reached WP0 while going backward.
        # Reverse direction and target WP1.
        # ------------------------------------------------------------
        elif (
            self.current_index == 0
            and self.direction == -1
        ):
            self.direction = 1
            self.current_index = 1

        # ------------------------------------------------------------
        # Normal progression.
        # ------------------------------------------------------------
        else:
            self.current_index += self.direction

        direction_text = (
            "FORWARD 0->7"
            if self.direction > 0
            else "REVERSE 7->0"
        )

        self.get_logger().info(
            f"WP{old_index} reached. "
            f"Next target: WP{self.current_index} "
            f"[{direction_text}]"
        )

    # ================================================================
    # MAIN TIMER
    # ================================================================

    def tick(self):

        # ------------------------------------------------------------
        # Diagnostic errors for legacy topics
        # ------------------------------------------------------------
        start_err = float("inf")
        middle_err = float("inf")
        final_err = float("inf")

        start_herr = 0.0

        at_start = False
        at_middle = False

        # ------------------------------------------------------------
        # Current target
        # ------------------------------------------------------------
        active_goal = self.waypoints[self.current_index]

        if self.pose is not None:

            # Legacy diagnostics.
            start_err = self.dist_pose(
                self.pose,
                self.waypoints[0],
            )

            middle_err = self.dist_pose(
                self.pose,
                self.waypoints[1],
            )

            final_err = self.dist_pose(
                self.pose,
                self.waypoints[7],
            )

            start_herr = wrap_angle(
                float(self.pose.theta)
                - self.waypoints[0].theta
            )

            at_start = (
                start_err <= self.switch_radius_m
            )

            at_middle = (
                middle_err <= self.switch_radius_m
            )

            # --------------------------------------------------------
            # Distance from vehicle to CURRENT waypoint
            # --------------------------------------------------------
            current_error = self.dist_pose(
                self.pose,
                active_goal,
            )

            waypoint_reached = (
                current_error <= self.switch_radius_m
            )

            # --------------------------------------------------------
            # Preserve original optional heading requirement for WP0.
            # --------------------------------------------------------
            if (
                self.current_index == 0
                and self.require_start_heading
            ):
                heading_error = wrap_angle(
                    float(self.pose.theta)
                    - self.waypoints[0].theta
                )

                heading_ok = (
                    abs(heading_error)
                    <= self.switch_heading_tol
                )

                waypoint_reached = (
                    waypoint_reached
                    and heading_ok
                )

            # --------------------------------------------------------
            # Current waypoint reached
            # --------------------------------------------------------
            if waypoint_reached:

                reached_index = self.current_index

                self.total_waypoints_reached += 1

                if reached_index == 0:
                    self.start_reached = True

                if reached_index == 1:
                    self.middle_reached = True

                self.get_logger().info(
                    f"Reached WP{reached_index}: "
                    f"error={current_error:.2f} m"
                )

                # Change to next waypoint.
                self.advance_waypoint()

                # Update active target immediately.
                active_goal = self.waypoints[
                    self.current_index
                ]

        # ------------------------------------------------------------
        # Phase
        # ------------------------------------------------------------
        phase = f"WAYPOINT_{self.current_index}"

        if phase != self.last_phase:
            self.get_logger().info(
                f"Phase changed to {phase}"
            )
            self.last_phase = phase

        # ------------------------------------------------------------
        # ACTIVE GOAL
        #
        # This is the important topic seen by your controller/APF.
        # ------------------------------------------------------------
        self.pub_goal.publish(active_goal)

        # ------------------------------------------------------------
        # Keep legacy fixed-point topics
        # ------------------------------------------------------------
        self.pub_start.publish(
            self.waypoints[0]
        )

        self.pub_middle.publish(
            self.waypoints[1]
        )

        self.pub_final.publish(
            self.waypoints[7]
        )

        # ------------------------------------------------------------
        # Publish all 8 fixed waypoint coordinates
        # ------------------------------------------------------------
        for i in range(8):
            self.pub_waypoints[i].publish(
                self.waypoints[i]
            )

        # ------------------------------------------------------------
        # Phase
        # ------------------------------------------------------------
        self.pub_phase.publish(
            String(data=phase)
        )

        # ------------------------------------------------------------
        # Legacy state outputs
        # ------------------------------------------------------------
        self.pub_start_reached.publish(
            Bool(data=bool(self.start_reached))
        )

        self.pub_middle_reached.publish(
            Bool(data=bool(self.middle_reached))
        )

        self.pub_at_start.publish(
            Bool(data=bool(at_start))
        )

        self.pub_at_middle.publish(
            Bool(data=bool(at_middle))
        )

        # ------------------------------------------------------------
        # Legacy error outputs
        # ------------------------------------------------------------
        if math.isfinite(start_err):
            self.pub_start_error.publish(
                Float32(data=float(start_err))
            )

            self.pub_start_heading_error.publish(
                Float32(data=float(start_herr))
            )

        if math.isfinite(middle_err):
            self.pub_middle_error.publish(
                Float32(data=float(middle_err))
            )

        if math.isfinite(final_err):
            self.pub_final_error.publish(
                Float32(data=float(final_err))
            )

        # ------------------------------------------------------------
        # RViz
        # ------------------------------------------------------------
        self.pub_markers.publish(
            self.make_markers()
        )

    # ================================================================
    # RVIZ SPHERE
    # ================================================================

    def make_sphere(
        self,
        marker_id: int,
        ns: str,
        pose: Pose2D,
        active: bool,
    ):

        m = Marker()

        m.ns = ns
        m.id = marker_id

        m.type = Marker.SPHERE
        m.action = Marker.ADD

        m.pose.position.x = float(pose.x)
        m.pose.position.y = float(pose.y)
        m.pose.position.z = 0.35

        m.pose.orientation.w = 1.0

        # Active target is slightly larger.
        if active:
            m.scale.x = 1.25
            m.scale.y = 1.25
            m.scale.z = 1.25

            m.color.r = 1.0
            m.color.g = 0.25
            m.color.b = 0.1
            m.color.a = 1.0

        else:
            m.scale.x = 0.90
            m.scale.y = 0.90
            m.scale.z = 0.90

            m.color.r = 0.1
            m.color.g = 0.8
            m.color.b = 1.0
            m.color.a = 0.55

        return m

    # ================================================================
    # RVIZ TEXT
    # ================================================================

    def make_text(
        self,
        marker_id: int,
        ns: str,
        pose: Pose2D,
        waypoint_number: int,
        active: bool,
    ):

        t = Marker()

        t.ns = ns
        t.id = marker_id

        t.type = Marker.TEXT_VIEW_FACING
        t.action = Marker.ADD

        t.pose.position.x = float(pose.x)
        t.pose.position.y = float(pose.y)
        t.pose.position.z = 1.45

        t.pose.orientation.w = 1.0

        t.scale.z = (
            0.70
            if active
            else 0.50
        )

        t.color.r = 1.0
        t.color.g = 1.0
        t.color.b = 1.0
        t.color.a = 1.0

        if active:
            t.text = (
                f"WP{waypoint_number} - ACTIVE\n"
                f"({pose.x:.1f}, {pose.y:.1f})"
            )
        else:
            t.text = (
                f"WP{waypoint_number}\n"
                f"({pose.x:.1f}, {pose.y:.1f})"
            )

        return t

    # ================================================================
    # RVIZ MARKERS
    # ================================================================

    def make_markers(self) -> MarkerArray:

        now = self.get_clock().now().to_msg()

        ma = MarkerArray()

        # Clear previous markers.
        clear = Marker()
        clear.header.stamp = now
        clear.header.frame_id = self.frame_id
        clear.action = Marker.DELETEALL

        ma.markers.append(clear)

        ns = "fixed_test_points"

        marker_id = 0

        # ------------------------------------------------------------
        # Draw all 8 waypoints
        # ------------------------------------------------------------
        for i, pose in enumerate(self.waypoints):

            active = (
                i == self.current_index
            )

            sphere = self.make_sphere(
                marker_id,
                ns,
                pose,
                active,
            )
            marker_id += 1

            text = self.make_text(
                marker_id,
                ns,
                pose,
                i,
                active,
            )
            marker_id += 1

            sphere.header.stamp = now
            sphere.header.frame_id = self.frame_id

            text.header.stamp = now
            text.header.frame_id = self.frame_id

            ma.markers.append(sphere)
            ma.markers.append(text)

        # ------------------------------------------------------------
        # Draw path connecting WP0 -> WP7
        # ------------------------------------------------------------
        line = Marker()

        line.ns = ns
        line.id = marker_id

        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD

        line.scale.x = 0.08

        line.color.r = 1.0
        line.color.g = 1.0
        line.color.b = 1.0
        line.color.a = 0.75

        for pose in self.waypoints:

            p = Point()

            p.x = float(pose.x)
            p.y = float(pose.y)
            p.z = 0.10

            line.points.append(p)

        line.header.stamp = now
        line.header.frame_id = self.frame_id

        ma.markers.append(line)

        # ------------------------------------------------------------
        # Arrow on CURRENT active waypoint
        # ------------------------------------------------------------
        active_pose = self.waypoints[
            self.current_index
        ]

        qx, qy, qz, qw = yaw_to_quaternion(
            active_pose.theta
        )

        arrow = Marker()

        arrow.ns = ns
        arrow.id = marker_id + 1

        arrow.type = Marker.ARROW
        arrow.action = Marker.ADD

        arrow.pose.position.x = active_pose.x
        arrow.pose.position.y = active_pose.y
        arrow.pose.position.z = 0.55

        arrow.pose.orientation.x = qx
        arrow.pose.orientation.y = qy
        arrow.pose.orientation.z = qz
        arrow.pose.orientation.w = qw

        arrow.scale.x = 2.0
        arrow.scale.y = 0.18
        arrow.scale.z = 0.25

        arrow.color.r = 1.0
        arrow.color.g = 0.25
        arrow.color.b = 0.1
        arrow.color.a = 0.9

        arrow.header.stamp = now
        arrow.header.frame_id = self.frame_id

        ma.markers.append(arrow)

        return ma


def main(args=None):

    rclpy.init(args=args)

    node = FixedTestPoints()

    try:
        rclpy.spin(node)

    finally:

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()