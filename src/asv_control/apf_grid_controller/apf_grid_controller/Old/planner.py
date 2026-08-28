#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose2D


class TwoWaypointGoalPublisher(Node):
    def __init__(self):
        super().__init__("two_waypoint_goal_publisher")

        self.declare_parameter("pose_topic", "/asv/vehicle_pose")
        self.declare_parameter("goal_topic", "/asv/nav/goal")
        self.declare_parameter("reach_threshold", 3.0)
        self.declare_parameter("publish_period", 0.1)
        self.declare_parameter("loop", False)

        self.pose_topic = str(self.get_parameter("pose_topic").value)
        self.goal_topic = str(self.get_parameter("goal_topic").value)
        self.reach_threshold = float(self.get_parameter("reach_threshold").value)
        self.publish_period = float(self.get_parameter("publish_period").value)
        self.loop = bool(self.get_parameter("loop").value)

        # x = North, y = East
        self.waypoints = [
            Pose2D(x=-13.0, y=26.0, theta=0.0),
            Pose2D(x=-17.0, y=38.0, theta=0.0),
        ]

        self.current_idx = 0
        self.have_pose = False
        self.finished = False

        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

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

        self.get_logger().info("Two-waypoint goal publisher started.")
        self.get_logger().info(f"Pose topic: {self.pose_topic}")
        self.get_logger().info(f"Goal topic: {self.goal_topic}")
        self.get_logger().info(f"Reach threshold: {self.reach_threshold:.2f} m")
        self.get_logger().info(f"Loop: {self.loop}")

        self.publish_current_goal()

    def pose_callback(self, msg: Pose2D):
        self.x = float(msg.x)
        self.y = float(msg.y)
        self.theta = float(msg.theta)
        self.have_pose = True

    def distance_to_current_goal(self):
        goal = self.waypoints[self.current_idx]
        dx = goal.x - self.x
        dy = goal.y - self.y
        return math.sqrt(dx * dx + dy * dy)

    def publish_current_goal(self):
        goal = self.waypoints[self.current_idx]
        self.goal_pub.publish(goal)

        self.get_logger().info(
            f"Publishing waypoint {self.current_idx + 1}/{len(self.waypoints)}: "
            f"x={goal.x:.2f}, y={goal.y:.2f}, theta={goal.theta:.2f}"
        )

    def timer_callback(self):
        # Keep publishing the current goal so APF always has it.
        self.goal_pub.publish(self.waypoints[self.current_idx])

        if not self.have_pose:
            self.get_logger().warn("Waiting for /asv/vehicle_pose...")
            return

        dist = self.distance_to_current_goal()

        self.get_logger().info(
            f"Current waypoint {self.current_idx + 1}/{len(self.waypoints)} "
            f"| distance={dist:.2f} m",
            throttle_duration_sec=1.0,
        )

        if dist > self.reach_threshold:
            return

        self.get_logger().info(
            f"Reached waypoint {self.current_idx + 1} within "
            f"{self.reach_threshold:.2f} m."
        )

        if self.current_idx < len(self.waypoints) - 1:
            self.current_idx += 1
            self.publish_current_goal()
            return

        # Last waypoint reached.
        if self.loop:
            self.current_idx = 0
            self.get_logger().info("Last waypoint reached. Looping back to waypoint 1.")
            self.publish_current_goal()
        else:
            self.finished = True
            self.get_logger().info("Final waypoint reached. Holding final goal.")


def main(args=None):
    rclpy.init(args=args)
    node = TwoWaypointGoalPublisher()

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()