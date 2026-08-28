#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
)

from geometry_msgs.msg import Pose2D
from nav_msgs.msg import Odometry, OccupancyGrid
from std_msgs.msg import String

from rosidl_runtime_py.utilities import get_message
from rosidl_runtime_py.convert import message_to_ordereddict


def quaternion_to_yaw(x, y, z, w):
    """Convert quaternion to yaw in radians."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def safe_json(msg: Any) -> str:
    """
    Convert an arbitrary ROS message to JSON.

    Useful for topics such as:
      /asv/control_effort
      /asv/motor_cmds
    where we do not want to hard-code the message type.
    """
    try:
        data = message_to_ordereddict(msg)
        return json.dumps(data, separators=(",", ":"))
    except Exception as exc:
        return json.dumps({"error": str(exc)})


class APFCSVLogger(Node):

    def __init__(self):
        super().__init__("apf_csv_logger")

        # ============================================================
        # PARAMETERS
        # ============================================================

        self.declare_parameter("asv", "asv")

        self.declare_parameter(
            "output_directory",
            "~/roboboat_logs/csv",
        )

        # CSV logging frequency.
        self.declare_parameter("rate_hz", 10.0)

        self.declare_parameter(
            "test_name",
            "apf_test",
        )

        self.asv = str(
            self.get_parameter("asv").value
        )

        output_directory = str(
            self.get_parameter(
                "output_directory"
            ).value
        )

        self.rate_hz = float(
            self.get_parameter("rate_hz").value
        )

        self.test_name = str(
            self.get_parameter("test_name").value
        )

        if self.rate_hz <= 0.0:
            self.rate_hz = 10.0

        # ============================================================
        # CREATE UNIQUE OUTPUT FILE
        # ============================================================

        directory = Path(
            output_directory
        ).expanduser()

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        self.csv_path = directory / (
            f"{self.test_name}_{timestamp}.csv"
        )

        # ============================================================
        # LATEST DATA
        # ============================================================

        self.vehicle_pose = None
        self.odom = None
        self.goal = None
        self.phase = ""

        self.waypoints = {
            i: None
            for i in range(8)
        }

        self.map_msg = None

        # Dynamically detected messages.
        self.control_effort_msg = None
        self.motor_cmds_msg = None
        self.desired_path_msg = None
        self.actual_path_msg = None

        self.control_effort_type = ""
        self.motor_cmds_type = ""
        self.desired_path_type = ""
        self.actual_path_type = ""

        # Prevent duplicate dynamic subscriptions.
        self.dynamic_subscriptions = {}

        # Keep subscription objects alive.
        self.subscription_handles = []

        # ============================================================
        # QOS
        # ============================================================

        qos = QoSProfile(
            depth=20,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        sensor_qos = QoSProfile(
            depth=20,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        # ============================================================
        # KNOWN TOPICS
        # ============================================================

        self.subscription_handles.append(
            self.create_subscription(
                Pose2D,
                f"/{self.asv}/vehicle_pose",
                self.vehicle_pose_cb,
                qos,
            )
        )

        self.subscription_handles.append(
            self.create_subscription(
                Odometry,
                f"/{self.asv}/p3d_wamv_ned",
                self.odom_cb,
                sensor_qos,
            )
        )

        self.subscription_handles.append(
            self.create_subscription(
                Pose2D,
                f"/{self.asv}/nav/goal",
                self.goal_cb,
                qos,
            )
        )

        self.subscription_handles.append(
            self.create_subscription(
                String,
                f"/{self.asv}/test/phase",
                self.phase_cb,
                qos,
            )
        )

        # ============================================================
        # 8 WAYPOINTS
        # ============================================================

        for i in range(8):

            sub = self.create_subscription(
                Pose2D,
                f"/{self.asv}/nav/test_waypoint_{i}",
                lambda msg, idx=i: self.waypoint_cb(
                    idx,
                    msg,
                ),
                qos,
            )

            self.subscription_handles.append(sub)

        # ============================================================
        # OCCUPANCY GRID
        # ============================================================

        self.subscription_handles.append(
            self.create_subscription(
                OccupancyGrid,
                f"/{self.asv}/map/local_occupancy_2",
                self.map_cb,
                sensor_qos,
            )
        )

        # ============================================================
        # CSV HEADER
        # ============================================================

        self.fields = [

            # --------------------------------------------------------
            # TIME
            # --------------------------------------------------------
            "elapsed_s",
            "ros_time_s",
            "wall_time",

            # --------------------------------------------------------
            # VEHICLE POSE
            # --------------------------------------------------------
            "vehicle_x",
            "vehicle_y",
            "vehicle_yaw_rad",
            "vehicle_yaw_deg",

            # --------------------------------------------------------
            # ODOMETRY POSITION
            # --------------------------------------------------------
            "odom_x",
            "odom_y",
            "odom_z",
            "odom_yaw_rad",
            "odom_yaw_deg",

            # --------------------------------------------------------
            # ODOMETRY VELOCITY
            # --------------------------------------------------------
            "velocity_x",
            "velocity_y",
            "velocity_z",
            "speed_mps",

            # --------------------------------------------------------
            # ANGULAR VELOCITY
            # --------------------------------------------------------
            "angular_velocity_x",
            "angular_velocity_y",
            "angular_velocity_z",

            # --------------------------------------------------------
            # GOAL
            # --------------------------------------------------------
            "goal_x",
            "goal_y",
            "goal_theta_rad",
            "goal_theta_deg",

            # --------------------------------------------------------
            # TRACKING ERROR
            # --------------------------------------------------------
            "goal_error_m",
            "goal_heading_error_rad",
            "goal_heading_error_deg",

            # --------------------------------------------------------
            # CURRENT WAYPOINT
            # --------------------------------------------------------
            "phase",
            "active_waypoint",

            # --------------------------------------------------------
            # OCCUPANCY GRID INFORMATION
            # --------------------------------------------------------
            "grid_resolution",
            "grid_width",
            "grid_height",
            "grid_origin_x",
            "grid_origin_y",
            "grid_occupied_cells",
            "grid_free_cells",
            "grid_unknown_cells",

            # --------------------------------------------------------
            # CONTROLLER
            # --------------------------------------------------------
            "control_effort_type",
            "control_effort",

            # --------------------------------------------------------
            # THRUSTERS
            # --------------------------------------------------------
            "motor_cmds_type",
            "motor_cmds",

            # --------------------------------------------------------
            # APF VISUALIZATION DATA
            # --------------------------------------------------------
            "desired_path_type",
            "desired_path",

            "actual_path_type",
            "actual_path",
        ]

        # Add waypoint positions to CSV.
        for i in range(8):

            self.fields.extend([
                f"wp{i}_x",
                f"wp{i}_y",
                f"wp{i}_theta",
            ])

        # ============================================================
        # OPEN CSV
        # ============================================================

        self.csv_file = open(
            self.csv_path,
            "w",
            newline="",
            buffering=1,
        )

        self.writer = csv.DictWriter(
            self.csv_file,
            fieldnames=self.fields,
        )

        self.writer.writeheader()

        # ============================================================
        # START TIME
        # ============================================================

        self.start_ros_time = (
            self.get_clock().now().nanoseconds
            / 1e9
        )

        # ============================================================
        # DYNAMIC TOPIC DISCOVERY
        # ============================================================

        # Some message types were not hard-coded because this logger
        # can discover their actual ROS message type at runtime.
        self.dynamic_topics = {

            f"/{self.asv}/control_effort":
                self.control_effort_cb,

            f"/{self.asv}/motor_cmds":
                self.motor_cmds_cb,

            f"/{self.asv}/viz/apf_desired_path":
                self.desired_path_cb,

            f"/{self.asv}/viz/apf_actual_path":
                self.actual_path_cb,
        }

        # Look for these topics periodically until all are discovered.
        self.discovery_timer = self.create_timer(
            1.0,
            self.discover_dynamic_topics,
        )

        # ============================================================
        # CSV LOGGING TIMER
        # ============================================================

        self.log_timer = self.create_timer(
            1.0 / self.rate_hz,
            self.log_row,
        )

        # ============================================================
        # START MESSAGE
        # ============================================================

        self.get_logger().info(
            "APF CSV LOGGER STARTED"
        )

        self.get_logger().info(
            f"ASV: {self.asv}"
        )

        self.get_logger().info(
            f"Logging at {self.rate_hz:.1f} Hz"
        )

        self.get_logger().info(
            f"CSV file:\n{self.csv_path}"
        )

    # ================================================================
    # CALLBACKS
    # ================================================================

    def vehicle_pose_cb(self, msg):
        self.vehicle_pose = msg

    def odom_cb(self, msg):
        self.odom = msg

    def goal_cb(self, msg):
        self.goal = msg

    def phase_cb(self, msg):
        self.phase = msg.data

    def waypoint_cb(self, index, msg):
        self.waypoints[index] = msg

    def map_cb(self, msg):
        self.map_msg = msg

    def control_effort_cb(self, msg):
        self.control_effort_msg = msg

    def motor_cmds_cb(self, msg):
        self.motor_cmds_msg = msg

    def desired_path_cb(self, msg):
        self.desired_path_msg = msg

    def actual_path_cb(self, msg):
        self.actual_path_msg = msg

    # ================================================================
    # DYNAMIC TOPIC DISCOVERY
    # ================================================================

    def discover_dynamic_topics(self):

        available = dict(
            self.get_topic_names_and_types()
        )

        for topic, callback in self.dynamic_topics.items():

            if topic in self.dynamic_subscriptions:
                continue

            if topic not in available:
                continue

            types = available[topic]

            if not types:
                continue

            type_string = types[0]

            try:

                message_class = get_message(
                    type_string
                )

                # Best effort is more tolerant for experimental topics.
                qos = QoSProfile(
                    depth=20,
                    reliability=ReliabilityPolicy.BEST_EFFORT,
                    durability=DurabilityPolicy.VOLATILE,
                )

                sub = self.create_subscription(
                    message_class,
                    topic,
                    callback,
                    qos,
                )

                self.dynamic_subscriptions[
                    topic
                ] = sub

                self.subscription_handles.append(
                    sub
                )

                if topic.endswith(
                    "/control_effort"
                ):
                    self.control_effort_type = (
                        type_string
                    )

                elif topic.endswith(
                    "/motor_cmds"
                ):
                    self.motor_cmds_type = (
                        type_string
                    )

                elif topic.endswith(
                    "/apf_desired_path"
                ):
                    self.desired_path_type = (
                        type_string
                    )

                elif topic.endswith(
                    "/apf_actual_path"
                ):
                    self.actual_path_type = (
                        type_string
                    )

                self.get_logger().info(
                    f"Discovered {topic}: "
                    f"{type_string}"
                )

            except Exception as exc:

                self.get_logger().warning(
                    f"Could not subscribe to "
                    f"{topic}: {exc}"
                )

    # ================================================================
    # ACTIVE WAYPOINT NUMBER
    # ================================================================

    def get_active_waypoint(self):

        if not self.phase:
            return ""

        # WAYPOINT_0 -> 0
        if self.phase.startswith(
            "WAYPOINT_"
        ):

            try:
                return int(
                    self.phase.split("_")[-1]
                )
            except ValueError:
                pass

        return self.phase

    # ================================================================
    # LOG ONE ROW
    # ================================================================

    def log_row(self):

        now = self.get_clock().now()

        ros_time_s = (
            now.nanoseconds / 1e9
        )

        elapsed_s = (
            ros_time_s
            - self.start_ros_time
        )

        row = {
            field: ""
            for field in self.fields
        }

        # ============================================================
        # TIME
        # ============================================================

        row["elapsed_s"] = (
            f"{elapsed_s:.6f}"
        )

        row["ros_time_s"] = (
            f"{ros_time_s:.9f}"
        )

        row["wall_time"] = (
            datetime.now().isoformat(
                timespec="milliseconds"
            )
        )

        # ============================================================
        # VEHICLE POSE
        # ============================================================

        if self.vehicle_pose is not None:

            row["vehicle_x"] = (
                self.vehicle_pose.x
            )

            row["vehicle_y"] = (
                self.vehicle_pose.y
            )

            row["vehicle_yaw_rad"] = (
                self.vehicle_pose.theta
            )

            row["vehicle_yaw_deg"] = (
                math.degrees(
                    self.vehicle_pose.theta
                )
            )

        # ============================================================
        # ODOMETRY
        # ============================================================

        if self.odom is not None:

            p = self.odom.pose.pose.position
            q = self.odom.pose.pose.orientation

            row["odom_x"] = p.x
            row["odom_y"] = p.y
            row["odom_z"] = p.z

            yaw = quaternion_to_yaw(
                q.x,
                q.y,
                q.z,
                q.w,
            )

            row["odom_yaw_rad"] = yaw
            row["odom_yaw_deg"] = (
                math.degrees(yaw)
            )

            linear = (
                self.odom.twist.twist.linear
            )

            angular = (
                self.odom.twist.twist.angular
            )

            row["velocity_x"] = linear.x
            row["velocity_y"] = linear.y
            row["velocity_z"] = linear.z

            row["speed_mps"] = math.sqrt(
                linear.x ** 2
                + linear.y ** 2
                + linear.z ** 2
            )

            row["angular_velocity_x"] = (
                angular.x
            )

            row["angular_velocity_y"] = (
                angular.y
            )

            row["angular_velocity_z"] = (
                angular.z
            )

        # ============================================================
        # GOAL
        # ============================================================

        if self.goal is not None:

            row["goal_x"] = self.goal.x
            row["goal_y"] = self.goal.y

            row["goal_theta_rad"] = (
                self.goal.theta
            )

            row["goal_theta_deg"] = (
                math.degrees(
                    self.goal.theta
                )
            )

        # ============================================================
        # GOAL POSITION / HEADING ERROR
        # ============================================================

        if (
            self.vehicle_pose is not None
            and self.goal is not None
        ):

            dx = (
                self.goal.x
                - self.vehicle_pose.x
            )

            dy = (
                self.goal.y
                - self.vehicle_pose.y
            )

            row["goal_error_m"] = (
                math.hypot(dx, dy)
            )

            heading_error = (
                self.goal.theta
                - self.vehicle_pose.theta
            )

            heading_error = (
                heading_error + math.pi
            ) % (2.0 * math.pi) - math.pi

            row[
                "goal_heading_error_rad"
            ] = heading_error

            row[
                "goal_heading_error_deg"
            ] = math.degrees(
                heading_error
            )

        # ============================================================
        # WAYPOINT PHASE
        # ============================================================

        row["phase"] = self.phase

        row["active_waypoint"] = (
            self.get_active_waypoint()
        )

        # ============================================================
        # ALL 8 WAYPOINTS
        # ============================================================

        for i in range(8):

            wp = self.waypoints[i]

            if wp is not None:

                row[f"wp{i}_x"] = wp.x
                row[f"wp{i}_y"] = wp.y
                row[f"wp{i}_theta"] = (
                    wp.theta
                )

        # ============================================================
        # OCCUPANCY GRID
        #
        # Do NOT put every grid cell into the CSV.
        # Instead store map dimensions + useful statistics.
        # Raw grid remains available in the rosbag.
        # ============================================================

        if self.map_msg is not None:

            info = self.map_msg.info

            row["grid_resolution"] = (
                info.resolution
            )

            row["grid_width"] = (
                info.width
            )

            row["grid_height"] = (
                info.height
            )

            row["grid_origin_x"] = (
                info.origin.position.x
            )

            row["grid_origin_y"] = (
                info.origin.position.y
            )

            data = self.map_msg.data

            if data:

                occupied = 0
                free = 0
                unknown = 0

                for value in data:

                    if value < 0:
                        unknown += 1

                    elif value >= 50:
                        occupied += 1

                    else:
                        free += 1

                row[
                    "grid_occupied_cells"
                ] = occupied

                row[
                    "grid_free_cells"
                ] = free

                row[
                    "grid_unknown_cells"
                ] = unknown

        # ============================================================
        # CONTROL EFFORT
        # ============================================================

        row[
            "control_effort_type"
        ] = self.control_effort_type

        if self.control_effort_msg is not None:

            row["control_effort"] = (
                safe_json(
                    self.control_effort_msg
                )
            )

        # ============================================================
        # MOTOR COMMANDS
        # ============================================================

        row[
            "motor_cmds_type"
        ] = self.motor_cmds_type

        if self.motor_cmds_msg is not None:

            row["motor_cmds"] = (
                safe_json(
                    self.motor_cmds_msg
                )
            )

        # ============================================================
        # DESIRED APF PATH
        # ============================================================

        row[
            "desired_path_type"
        ] = self.desired_path_type

        if self.desired_path_msg is not None:

            row["desired_path"] = (
                safe_json(
                    self.desired_path_msg
                )
            )

        # ============================================================
        # ACTUAL APF PATH
        # ============================================================

        row[
            "actual_path_type"
        ] = self.actual_path_type

        if self.actual_path_msg is not None:

            row["actual_path"] = (
                safe_json(
                    self.actual_path_msg
                )
            )

        # ============================================================
        # WRITE CSV
        # ============================================================

        self.writer.writerow(row)

    # ================================================================
    # SHUTDOWN
    # ================================================================

    def destroy_node(self):

        try:

            if hasattr(
                self,
                "csv_file",
            ):

                self.csv_file.flush()
                self.csv_file.close()

                self.get_logger().info(
                    f"CSV saved:\n"
                    f"{self.csv_path}"
                )

        finally:

            super().destroy_node()


def main(args=None):

    rclpy.init(args=args)

    node = APFCSVLogger()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()