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
        self.declare_parameter("reach_threshold", 3.0)
        self.declare_parameter("publish_period", 0.1)

        self.pose_topic = str(self.get_parameter("pose_topic").value)
        self.goal_topic = str(self.get_parameter("goal_topic").value)
        self.reach_threshold = float(self.get_parameter("reach_threshold").value)
        self.publish_period = float(self.get_parameter("publish_period").value)

        # ----------------------------
        # Waypoints: x = North, y = East
        # Edit these four points as needed
        # ----------------------------
        self.waypoints = [
            # Intersection
            Pose2D(x=0.00, y=0.00, theta=0.0),

            # North loop, radius 7 m, center at x=7, y=0
            Pose2D(x=5.00, y=5.0, theta=0.0),
            Pose2D(x=10.00, y=0.00, theta=0.0),
            Pose2D(x=5.00, y=-5.00, theta=0.0),

            Pose2D(x=0.00, y=0.00, theta=0.0),

            # South loop, radius 7 m, center at x=-7, y=0
            Pose2D(x=-5.0, y=5.0, theta=0.0),
            Pose2D(x=-10.00, y=-0.00, theta=0.0),
            Pose2D(x=-5.00, y=-5.00, theta=0.0),

            # Finish at intersection
            Pose2D(x=0.00, y=0.00, theta=0.0),
        ]

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

        self.get_logger().info("Continuous waypoint goal publisher started.")
        self.get_logger().info(f"Subscribing pose: {self.pose_topic}")
        self.get_logger().info(f"Publishing goal: {self.goal_topic}")
        self.get_logger().info(f"Reach threshold: {self.reach_threshold:.2f} m")

        self.publish_current_goal()

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
            f"| loop={self.loop_count}: "
            f"x={goal.x:.2f}, y={goal.y:.2f}, theta={goal.theta:.2f}"
        )

    def timer_callback(self):
        # Keep publishing current goal so APF/controller always has it
        self.goal_pub.publish(self.waypoints[self.current_idx])

        if not self.have_pose:
            self.get_logger().warn("Waiting for vehicle pose...")
            return

        dist = self.distance_to_current_goal()

        if dist <= self.reach_threshold:
            self.get_logger().info(
                f"Reached waypoint {self.current_idx + 1} "
                f"within {self.reach_threshold:.2f} m."
            )

            self.current_idx += 1

            # Continuous loop:
            # after the last waypoint, go back to the first waypoint
            if self.current_idx >= len(self.waypoints):
                self.current_idx = 0
                self.loop_count += 1
                self.get_logger().info(
                    f"Figure-eight loop {self.loop_count} complete. Restarting waypoints."
                )

            self.publish_current_goal()


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