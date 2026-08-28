#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose2D


class WaypointGoalPublisher(Node):
    def __init__(self):
        super().__init__("waypoint_goal_publisher")

        # ----------------------------
        # Parameters
        # ----------------------------
        self.declare_parameter("pose_topic", "/asv/vehicle_pose")
        self.declare_parameter("goal_topic", "/asv/nav/goal")

        # IMPORTANT:
        # With many smooth waypoints, 4.0 m is too large and can skip points.
        # 1.0 to 2.0 m is better for this radius-5 figure eight.
        self.declare_parameter("reach_threshold", 2.0)

        self.declare_parameter("publish_period", 0.1)

        # Figure-eight parameters
        self.declare_parameter("radius", 7.0)
        self.declare_parameter("points_per_loop", 5)

        self.pose_topic = str(self.get_parameter("pose_topic").value)
        self.goal_topic = str(self.get_parameter("goal_topic").value)
        self.reach_threshold = float(self.get_parameter("reach_threshold").value)
        self.publish_period = float(self.get_parameter("publish_period").value)

        self.radius = float(self.get_parameter("radius").value)
        self.points_per_loop = int(self.get_parameter("points_per_loop").value)

        # ----------------------------
        # Waypoints: x = North, y = East
        #
        # Figure eight:
        #   - Intersection at x=0, y=0
        #   - North loop center at x=+radius, y=0
        #   - South loop center at x=-radius, y=0
        #   - Radius = 5 m by default
        #   - Path repeats continuously
        # ----------------------------
        self.waypoints = self.generate_figure_eight_waypoints(
            radius=self.radius,
            points_per_loop=self.points_per_loop,
        )

        self.current_idx = 0
        self.have_pose = False
        self.loop_count = 0

        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        # ----------------------------
        # ROS interfaces
        # ----------------------------
        self.pose_sub = self.create_subscription(
            Pose2D,
            self.pose_topic,
            self.pose_callback,
            10,
        )

        self.goal_pub = self.create_publisher(
            Pose2D,
            self.goal_topic,
            10,
        )

        self.timer = self.create_timer(
            self.publish_period,
            self.timer_callback,
        )

        self.get_logger().info("Continuous figure-eight waypoint publisher started.")
        self.get_logger().info(f"Subscribing pose: {self.pose_topic}")
        self.get_logger().info(f"Publishing goal: {self.goal_topic}")
        self.get_logger().info(f"Reach threshold: {self.reach_threshold:.2f} m")
        self.get_logger().info(f"Figure-eight radius: {self.radius:.2f} m")
        self.get_logger().info(f"Points per loop: {self.points_per_loop}")
        self.get_logger().info(f"Total waypoints: {len(self.waypoints)}")

        self.publish_current_goal()

    def generate_figure_eight_waypoints(self, radius: float, points_per_loop: int):
        """
        Generate a smooth north-south figure eight.

        Coordinate convention:
            x = North
            y = East

        Shape:
            North loop center: x = +radius, y = 0
            South loop center: x = -radius, y = 0
            Intersection:      x = 0,       y = 0

        The waypoint sequence starts at the intersection, goes around the
        north loop, passes through the intersection, goes around the south
        loop, and returns to the intersection.
        """

        radius = abs(float(radius))
        points_per_loop = max(int(points_per_loop), 8)

        waypoints = []

        # ----------------------------
        # North loop
        # Center: x=+radius, y=0
        #
        # Parameterization:
        #   x = radius + radius*cos(a)
        #   y = radius*sin(a)
        #
        # Start at intersection:
        #   a = pi -> x=0, y=0
        #
        # Go clockwise/counter-clockwise smoothly around the top loop.
        # ----------------------------
        north_center_x = radius
        north_center_y = 0.0

        for i in range(points_per_loop + 1):
            a = math.pi - (2.0 * math.pi * i / points_per_loop)

            x = north_center_x + radius * math.cos(a)
            y = north_center_y + radius * math.sin(a)

            waypoints.append(Pose2D(x=x, y=y, theta=0.0))

        # ----------------------------
        # South loop
        # Center: x=-radius, y=0
        #
        # Start again from intersection:
        #   a = 0 -> x=0, y=0
        # ----------------------------
        south_center_x = -radius
        south_center_y = 0.0

        for i in range(1, points_per_loop + 1):
            a = 0.0 - (2.0 * math.pi * i / points_per_loop)

            x = south_center_x + radius * math.cos(a)
            y = south_center_y + radius * math.sin(a)

            waypoints.append(Pose2D(x=x, y=y, theta=0.0))

        return waypoints

    def pose_callback(self, msg: Pose2D):
        self.x = float(msg.x)
        self.y = float(msg.y)
        self.theta = float(msg.theta)
        self.have_pose = True

    def distance_to_current_goal(self) -> float:
        goal = self.waypoints[self.current_idx]

        dx = goal.x - self.x
        dy = goal.y - self.y

        return math.sqrt(dx * dx + dy * dy)

    def publish_current_goal(self):
        goal = self.waypoints[self.current_idx]
        self.goal_pub.publish(goal)

        self.get_logger().info(
            f"Published waypoint {self.current_idx + 1}/{len(self.waypoints)} "
            f"| loop={self.loop_count} | "
            f"x={goal.x:.2f}, y={goal.y:.2f}, theta={goal.theta:.2f}"
        )

    def advance_to_next_waypoint(self):
        self.current_idx += 1

        # Continuous figure eight:
        # When the last waypoint is reached, go back to the first waypoint.
        if self.current_idx >= len(self.waypoints):
            self.current_idx = 0
            self.loop_count += 1
            self.get_logger().info(
                f"Completed figure-eight loop {self.loop_count}. Repeating continuously."
            )

        self.publish_current_goal()

    def timer_callback(self):
        # Keep publishing current goal so APF/controller always has it.
        self.goal_pub.publish(self.waypoints[self.current_idx])

        if not self.have_pose:
            self.get_logger().warn("Waiting for vehicle pose...")
            return

        dist = self.distance_to_current_goal()

        if dist <= self.reach_threshold:
            self.get_logger().info(
                f"Reached waypoint {self.current_idx + 1}/{len(self.waypoints)} "
                f"within {self.reach_threshold:.2f} m. Distance={dist:.2f} m."
            )

            self.advance_to_next_waypoint()


def main(args=None):
    rclpy.init(args=args)

    node = WaypointGoalPublisher()

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()