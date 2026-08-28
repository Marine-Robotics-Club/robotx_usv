#!/usr/bin/env python3

import csv
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


class CsvWriter:
    def __init__(self, filepath: Path, fieldnames: List[str]):
        self.filepath = filepath
        self.fieldnames = fieldnames

        self.filepath.parent.mkdir(parents=True, exist_ok=True)

        self.f = open(self.filepath, "w", newline="")
        self.writer = csv.DictWriter(self.f, fieldnames=self.fieldnames)
        self.writer.writeheader()

    def write(self, row: Dict[str, Any]):
        clean = {}

        for key in self.fieldnames:
            val = row.get(key, "")

            if isinstance(val, float):
                if math.isnan(val) or math.isinf(val):
                    val = ""

            clean[key] = val

        self.writer.writerow(clean)

    def close(self):
        self.f.flush()
        self.f.close()


class ApfBagToCsvAnalyzer(Node):
    def __init__(self):
        super().__init__("analyze_apf_bag_to_csv")

        # ------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------
        self.declare_parameter("bag_path", "")
        self.declare_parameter("output_dir", "")
        self.declare_parameter("storage_id", "sqlite3")

        # If True, exports every point in Path messages.
        # This can create large CSVs because Path messages often contain
        # many repeated points.
        self.declare_parameter("export_full_paths", False)

        # If True, writes a generic CSV for topics not specially handled.
        # This can be useful, but usually not needed.
        self.declare_parameter("export_unknown_topics", False)

        self.bag_path = str(self.get_parameter("bag_path").value)
        self.output_dir_param = str(self.get_parameter("output_dir").value)
        self.storage_id = str(self.get_parameter("storage_id").value)
        self.export_full_paths = bool(self.get_parameter("export_full_paths").value)
        self.export_unknown_topics = bool(
            self.get_parameter("export_unknown_topics").value
        )

        if not self.bag_path:
            raise RuntimeError(
                "Missing bag_path parameter. Example:\n"
                "ros2 run apf_grid_controller analyze_apf_bag_to_csv "
                "--ros-args -p bag_path:=/home/highlevel/bags/apf_zed_buoy_gate/test_01_light"
            )

        self.bag_path = os.path.expanduser(self.bag_path)
        self.bag_path = os.path.abspath(self.bag_path)

        if self.output_dir_param.strip():
            self.output_dir = Path(os.path.expanduser(self.output_dir_param)).absolute()
        else:
            self.output_dir = Path(self.bag_path).absolute() / "csv_export"

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.get_logger().info(f"Bag path:   {self.bag_path}")
        self.get_logger().info(f"Output dir: {self.output_dir}")
        self.get_logger().info(f"Storage ID: {self.storage_id}")

        # ------------------------------------------------------------
        # CSV writers
        # ------------------------------------------------------------
        self.writers: Dict[str, CsvWriter] = {}

        self.setup_writers()

        # Statistics
        self.topic_counts: Dict[str, int] = {}
        self.first_time_ns: Optional[int] = None
        self.last_time_ns: Optional[int] = None

    # ============================================================
    # Writer setup
    # ============================================================

    def setup_writers(self):
        self.writers["topics_summary"] = CsvWriter(
            self.output_dir / "topics_summary.csv",
            [
                "topic",
                "type",
                "message_count",
            ],
        )

        self.writers["vehicle_pose"] = CsvWriter(
            self.output_dir / "vehicle_pose.csv",
            [
                "bag_time_ns",
                "time_s",
                "x_north_m",
                "y_east_m",
                "yaw_rad",
                "yaw_deg",
            ],
        )

        self.writers["odom"] = CsvWriter(
            self.output_dir / "odometry.csv",
            [
                "bag_time_ns",
                "time_s",
                "topic",
                "frame_id",
                "child_frame_id",
                "x_m",
                "y_m",
                "z_m",
                "qx",
                "qy",
                "qz",
                "qw",
                "yaw_rad",
                "yaw_deg",
                "vx_mps",
                "vy_mps",
                "vz_mps",
                "wx_radps",
                "wy_radps",
                "wz_radps",
            ],
        )

        self.writers["control_effort"] = CsvWriter(
            self.output_dir / "control_effort.csv",
            [
                "bag_time_ns",
                "time_s",
                "topic",
                "tau_x",
                "tau_y",
                "tau_z",
                "tau_raw",
            ],
        )

        self.writers["motor_cmds"] = CsvWriter(
            self.output_dir / "motor_cmds.csv",
            [
                "bag_time_ns",
                "time_s",
                "topic",
                "data",
                "left",
                "right",
                "port",
                "stbd",
                "raw",
            ],
        )

        self.writers["thrusters"] = CsvWriter(
            self.output_dir / "thrusters.csv",
            [
                "bag_time_ns",
                "time_s",
                "topic",
                "value",
                "raw",
            ],
        )

        self.writers["zed_detections"] = CsvWriter(
            self.output_dir / "zed_detections.csv",
            [
                "bag_time_ns",
                "time_s",
                "det_index",
                "class_name",
                "confidence",
                "x_min",
                "y_min",
                "x_max",
                "y_max",
                "x_center",
                "y_center",
                "x_loc_nwu_m",
                "y_loc_nwu_m",
                "z_loc_nwu_m",
                "x_local_ned_m",
                "y_local_ned_m",
                "z_local_ned_m",
                "range_local_m",
            ],
        )

        self.writers["map_stats"] = CsvWriter(
            self.output_dir / "map_stats.csv",
            [
                "bag_time_ns",
                "time_s",
                "topic",
                "frame_id",
                "resolution_m",
                "width_cells",
                "height_cells",
                "origin_x",
                "origin_y",
                "occupied_count",
                "occupied_area_m2",
                "occupied_centroid_x",
                "occupied_centroid_y",
                "occupied_min_x",
                "occupied_max_x",
                "occupied_min_y",
                "occupied_max_y",
            ],
        )

        self.writers["marker_obstacles"] = CsvWriter(
            self.output_dir / "marker_obstacles.csv",
            [
                "bag_time_ns",
                "time_s",
                "topic",
                "frame_id",
                "ns",
                "id",
                "type",
                "action",
                "x",
                "y",
                "z",
                "scale_x",
                "scale_y",
                "scale_z",
                "color_r",
                "color_g",
                "color_b",
                "color_a",
                "text",
            ],
        )

        self.writers["paths_summary"] = CsvWriter(
            self.output_dir / "paths_summary.csv",
            [
                "bag_time_ns",
                "time_s",
                "topic",
                "frame_id",
                "num_points",
                "first_x",
                "first_y",
                "first_z",
                "last_x",
                "last_y",
                "last_z",
            ],
        )

        self.writers["path_points"] = CsvWriter(
            self.output_dir / "path_points.csv",
            [
                "bag_time_ns",
                "time_s",
                "topic",
                "frame_id",
                "msg_index",
                "point_index",
                "x",
                "y",
                "z",
                "qx",
                "qy",
                "qz",
                "qw",
                "yaw_rad",
                "yaw_deg",
            ],
        )

        self.writers["goal_pose"] = CsvWriter(
            self.output_dir / "goal_pose.csv",
            [
                "bag_time_ns",
                "time_s",
                "topic",
                "x",
                "y",
                "theta",
                "theta_deg",
            ],
        )

        self.writers["imu"] = CsvWriter(
            self.output_dir / "imu.csv",
            [
                "bag_time_ns",
                "time_s",
                "topic",
                "frame_id",
                "qx",
                "qy",
                "qz",
                "qw",
                "roll_rad",
                "pitch_rad",
                "yaw_rad",
                "roll_deg",
                "pitch_deg",
                "yaw_deg",
                "angular_velocity_x",
                "angular_velocity_y",
                "angular_velocity_z",
                "linear_acceleration_x",
                "linear_acceleration_y",
                "linear_acceleration_z",
            ],
        )

        self.writers["gps_fix"] = CsvWriter(
            self.output_dir / "gps_fix.csv",
            [
                "bag_time_ns",
                "time_s",
                "topic",
                "frame_id",
                "latitude",
                "longitude",
                "altitude",
                "status",
                "service",
            ],
        )

        self.writers["image_index"] = CsvWriter(
            self.output_dir / "image_index.csv",
            [
                "bag_time_ns",
                "time_s",
                "topic",
                "frame_id",
                "format",
                "height",
                "width",
                "encoding",
                "data_size_bytes",
            ],
        )

        self.writers["unknown"] = CsvWriter(
            self.output_dir / "unknown_topics.csv",
            [
                "bag_time_ns",
                "time_s",
                "topic",
                "type",
                "raw",
            ],
        )

    # ============================================================
    # Utility functions
    # ============================================================

    @staticmethod
    def safe_get(obj: Any, attr: str, default: Any = "") -> Any:
        return getattr(obj, attr, default)

    @staticmethod
    def list_get(values: Any, idx: int, default: Any = "") -> Any:
        try:
            if values is None:
                return default
            if idx < len(values):
                return values[idx]
        except Exception:
            return default
        return default

    @staticmethod
    def quat_to_euler(qx: float, qy: float, qz: float, qw: float) -> Tuple[float, float, float]:
        # ROS quaternion to roll, pitch, yaw
        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (qw * qy - qz * qx)
        if abs(sinp) >= 1.0:
            pitch = math.copysign(math.pi / 2.0, sinp)
        else:
            pitch = math.asin(sinp)

        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    @staticmethod
    def finite_or_blank(value: Any) -> Any:
        try:
            v = float(value)
            if math.isfinite(v):
                return v
            return ""
        except Exception:
            return value

    @staticmethod
    def scalar_from_msg(msg: Any) -> Optional[float]:
        # Handles std_msgs Float32/Float64/Int/etc.
        if hasattr(msg, "data"):
            try:
                return float(msg.data)
            except Exception:
                return None
        return None

    def time_s(self, bag_time_ns: int) -> float:
        if self.first_time_ns is None:
            self.first_time_ns = bag_time_ns

        return float(bag_time_ns - self.first_time_ns) * 1e-9

    def close_writers(self):
        for writer in self.writers.values():
            writer.close()

    # ============================================================
    # Rosbag processing
    # ============================================================

    def run(self):
        reader = rosbag2_py.SequentialReader()

        storage_options = rosbag2_py.StorageOptions(
            uri=self.bag_path,
            storage_id=self.storage_id,
        )

        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        )

        reader.open(storage_options, converter_options)

        topics_and_types = reader.get_all_topics_and_types()

        topic_type_map = {
            topic_metadata.name: topic_metadata.type
            for topic_metadata in topics_and_types
        }

        msg_type_map = {}

        self.get_logger().info("Topics in bag:")

        for topic, type_name in topic_type_map.items():
            self.get_logger().info(f"  {topic} [{type_name}]")

            try:
                msg_type_map[topic] = get_message(type_name)
            except Exception as e:
                self.get_logger().warn(
                    f"Could not load message type for topic {topic}: {type_name}. Error: {e}"
                )

        path_msg_counter: Dict[str, int] = {}

        total_messages = 0

        while reader.has_next():
            topic, data, bag_time_ns = reader.read_next()

            total_messages += 1
            self.last_time_ns = bag_time_ns
            self.topic_counts[topic] = self.topic_counts.get(topic, 0) + 1

            if topic not in msg_type_map:
                continue

            msg_type = msg_type_map[topic]
            type_name = topic_type_map.get(topic, "")

            try:
                msg = deserialize_message(data, msg_type)
            except Exception as e:
                self.get_logger().warn(
                    f"Could not deserialize message on {topic} [{type_name}]: {e}"
                )
                continue

            t = self.time_s(bag_time_ns)

            try:
                self.dispatch_message(
                    topic=topic,
                    type_name=type_name,
                    msg=msg,
                    bag_time_ns=bag_time_ns,
                    time_s=t,
                    path_msg_counter=path_msg_counter,
                )
            except Exception as e:
                self.get_logger().warn(
                    f"Error processing message on {topic} [{type_name}]: {e}"
                )

        # Write summary
        for topic, type_name in sorted(topic_type_map.items()):
            self.writers["topics_summary"].write(
                {
                    "topic": topic,
                    "type": type_name,
                    "message_count": self.topic_counts.get(topic, 0),
                }
            )

        self.close_writers()

        self.get_logger().info(f"Processed total messages: {total_messages}")
        self.get_logger().info(f"CSV export complete: {self.output_dir}")

    # ============================================================
    # Dispatcher
    # ============================================================

    def dispatch_message(
        self,
        topic: str,
        type_name: str,
        msg: Any,
        bag_time_ns: int,
        time_s: float,
        path_msg_counter: Dict[str, int],
    ):
        # Vehicle pose
        if type_name == "geometry_msgs/msg/Pose2D":
            self.handle_pose2d(topic, msg, bag_time_ns, time_s)
            return

        # Odometry
        if type_name == "nav_msgs/msg/Odometry":
            self.handle_odometry(topic, msg, bag_time_ns, time_s)
            return

        # IMU
        if type_name == "sensor_msgs/msg/Imu":
            self.handle_imu(topic, msg, bag_time_ns, time_s)
            return

        # GPS
        if type_name == "sensor_msgs/msg/NavSatFix":
            self.handle_navsatfix(topic, msg, bag_time_ns, time_s)
            return

        # APF ControlEffort
        if type_name.endswith("/ControlEffort") or topic == "/asv/control_effort":
            self.handle_control_effort(topic, msg, bag_time_ns, time_s)
            return

        # ZED custom detections
        if type_name.endswith("/ZedDetection") or topic == "/zed_custom_detections":
            self.handle_zed_detection(topic, msg, bag_time_ns, time_s)
            return

        # Occupancy grid
        if type_name == "nav_msgs/msg/OccupancyGrid":
            self.handle_occupancy_grid(topic, msg, bag_time_ns, time_s)
            return

        # MarkerArray
        if type_name == "visualization_msgs/msg/MarkerArray":
            self.handle_marker_array(topic, msg, bag_time_ns, time_s)
            return

        # Path
        if type_name == "nav_msgs/msg/Path":
            self.handle_path(topic, msg, bag_time_ns, time_s, path_msg_counter)
            return

        # Images / compressed images
        if type_name in (
            "sensor_msgs/msg/CompressedImage",
            "sensor_msgs/msg/Image",
            "sensor_msgs/msg/CameraInfo",
        ):
            self.handle_image_like(topic, type_name, msg, bag_time_ns, time_s)
            return

        # Thrusters and motor command style topics
        if (
            "thruster" in topic.lower()
            or "motor" in topic.lower()
            or topic in ("/asv/motor_cmds", "/asv/motor_pwm")
        ):
            self.handle_motor_or_thruster(topic, msg, bag_time_ns, time_s)
            return

        if self.export_unknown_topics:
            self.writers["unknown"].write(
                {
                    "bag_time_ns": bag_time_ns,
                    "time_s": time_s,
                    "topic": topic,
                    "type": type_name,
                    "raw": str(msg),
                }
            )

    # ============================================================
    # Handlers
    # ============================================================

    def handle_pose2d(self, topic: str, msg: Any, bag_time_ns: int, time_s: float):
        x = float(msg.x)
        y = float(msg.y)
        theta = float(msg.theta)

        # Route /asv/nav/goal to goal CSV
        if "goal" in topic.lower():
            self.writers["goal_pose"].write(
                {
                    "bag_time_ns": bag_time_ns,
                    "time_s": time_s,
                    "topic": topic,
                    "x": x,
                    "y": y,
                    "theta": theta,
                    "theta_deg": math.degrees(theta),
                }
            )
            return

        self.writers["vehicle_pose"].write(
            {
                "bag_time_ns": bag_time_ns,
                "time_s": time_s,
                "x_north_m": x,
                "y_east_m": y,
                "yaw_rad": theta,
                "yaw_deg": math.degrees(theta),
            }
        )

    def handle_odometry(self, topic: str, msg: Any, bag_time_ns: int, time_s: float):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        v = msg.twist.twist.linear
        w = msg.twist.twist.angular

        roll, pitch, yaw = self.quat_to_euler(q.x, q.y, q.z, q.w)

        self.writers["odom"].write(
            {
                "bag_time_ns": bag_time_ns,
                "time_s": time_s,
                "topic": topic,
                "frame_id": msg.header.frame_id,
                "child_frame_id": msg.child_frame_id,
                "x_m": p.x,
                "y_m": p.y,
                "z_m": p.z,
                "qx": q.x,
                "qy": q.y,
                "qz": q.z,
                "qw": q.w,
                "yaw_rad": yaw,
                "yaw_deg": math.degrees(yaw),
                "vx_mps": v.x,
                "vy_mps": v.y,
                "vz_mps": v.z,
                "wx_radps": w.x,
                "wy_radps": w.y,
                "wz_radps": w.z,
            }
        )

    def handle_imu(self, topic: str, msg: Any, bag_time_ns: int, time_s: float):
        q = msg.orientation
        av = msg.angular_velocity
        la = msg.linear_acceleration

        roll, pitch, yaw = self.quat_to_euler(q.x, q.y, q.z, q.w)

        self.writers["imu"].write(
            {
                "bag_time_ns": bag_time_ns,
                "time_s": time_s,
                "topic": topic,
                "frame_id": msg.header.frame_id,
                "qx": q.x,
                "qy": q.y,
                "qz": q.z,
                "qw": q.w,
                "roll_rad": roll,
                "pitch_rad": pitch,
                "yaw_rad": yaw,
                "roll_deg": math.degrees(roll),
                "pitch_deg": math.degrees(pitch),
                "yaw_deg": math.degrees(yaw),
                "angular_velocity_x": av.x,
                "angular_velocity_y": av.y,
                "angular_velocity_z": av.z,
                "linear_acceleration_x": la.x,
                "linear_acceleration_y": la.y,
                "linear_acceleration_z": la.z,
            }
        )

    def handle_navsatfix(self, topic: str, msg: Any, bag_time_ns: int, time_s: float):
        self.writers["gps_fix"].write(
            {
                "bag_time_ns": bag_time_ns,
                "time_s": time_s,
                "topic": topic,
                "frame_id": msg.header.frame_id,
                "latitude": msg.latitude,
                "longitude": msg.longitude,
                "altitude": msg.altitude,
                "status": msg.status.status,
                "service": msg.status.service,
            }
        )

    def handle_control_effort(
        self,
        topic: str,
        msg: Any,
        bag_time_ns: int,
        time_s: float,
    ):
        tau = getattr(msg, "tau", [])

        tau_x = self.list_get(tau, 0, "")
        tau_y = self.list_get(tau, 1, "")
        tau_z = self.list_get(tau, 2, "")

        self.writers["control_effort"].write(
            {
                "bag_time_ns": bag_time_ns,
                "time_s": time_s,
                "topic": topic,
                "tau_x": tau_x,
                "tau_y": tau_y,
                "tau_z": tau_z,
                "tau_raw": list(tau) if tau is not None else "",
            }
        )

    def handle_motor_or_thruster(
        self,
        topic: str,
        msg: Any,
        bag_time_ns: int,
        time_s: float,
    ):
        scalar = self.scalar_from_msg(msg)

        if "thruster" in topic.lower():
            self.writers["thrusters"].write(
                {
                    "bag_time_ns": bag_time_ns,
                    "time_s": time_s,
                    "topic": topic,
                    "value": scalar if scalar is not None else "",
                    "raw": str(msg),
                }
            )
            return

        # Try common field names
        data = getattr(msg, "data", "")
        left = getattr(msg, "left", "")
        right = getattr(msg, "right", "")
        port = getattr(msg, "port", "")
        stbd = getattr(msg, "stbd", "")

        self.writers["motor_cmds"].write(
            {
                "bag_time_ns": bag_time_ns,
                "time_s": time_s,
                "topic": topic,
                "data": data,
                "left": left,
                "right": right,
                "port": port,
                "stbd": stbd,
                "raw": str(msg),
            }
        )

    def handle_zed_detection(
        self,
        topic: str,
        msg: Any,
        bag_time_ns: int,
        time_s: float,
    ):
        class_name = getattr(msg, "class_name", [])
        confidence = getattr(msg, "confidence", [])

        x_min = getattr(msg, "x_min", [])
        y_min = getattr(msg, "y_min", [])
        x_max = getattr(msg, "x_max", [])
        y_max = getattr(msg, "y_max", [])
        x_center = getattr(msg, "x_center", [])
        y_center = getattr(msg, "y_center", [])

        x_loc = getattr(msg, "x_loc", [])
        y_loc = getattr(msg, "y_loc", [])
        z_loc = getattr(msg, "z_loc", [])

        n = min(
            len(class_name),
            len(confidence),
            len(x_loc),
            len(y_loc),
            len(z_loc),
        )

        for i in range(n):
            x_nwu = float(self.list_get(x_loc, i, 0.0))
            y_nwu = float(self.list_get(y_loc, i, 0.0))
            z_nwu = float(self.list_get(z_loc, i, 0.0))

            # Same convention as your map node:
            # ZED local NWU -> local/body NED
            x_ned = x_nwu
            y_ned = -y_nwu
            z_ned = -z_nwu

            rng = math.sqrt(x_ned * x_ned + y_ned * y_ned)

            self.writers["zed_detections"].write(
                {
                    "bag_time_ns": bag_time_ns,
                    "time_s": time_s,
                    "det_index": i,
                    "class_name": self.list_get(class_name, i, ""),
                    "confidence": self.list_get(confidence, i, ""),
                    "x_min": self.list_get(x_min, i, ""),
                    "y_min": self.list_get(y_min, i, ""),
                    "x_max": self.list_get(x_max, i, ""),
                    "y_max": self.list_get(y_max, i, ""),
                    "x_center": self.list_get(x_center, i, ""),
                    "y_center": self.list_get(y_center, i, ""),
                    "x_loc_nwu_m": x_nwu,
                    "y_loc_nwu_m": y_nwu,
                    "z_loc_nwu_m": z_nwu,
                    "x_local_ned_m": x_ned,
                    "y_local_ned_m": y_ned,
                    "z_local_ned_m": z_ned,
                    "range_local_m": rng,
                }
            )

    def handle_occupancy_grid(
        self,
        topic: str,
        msg: Any,
        bag_time_ns: int,
        time_s: float,
    ):
        data = list(msg.data)
        res = float(msg.info.resolution)
        width = int(msg.info.width)
        height = int(msg.info.height)
        ox = float(msg.info.origin.position.x)
        oy = float(msg.info.origin.position.y)

        occupied_indices = [
            idx for idx, val in enumerate(data)
            if int(val) >= 80
        ]

        occupied_count = len(occupied_indices)
        occupied_area = occupied_count * res * res

        if occupied_count == 0:
            centroid_x = ""
            centroid_y = ""
            min_x = ""
            max_x = ""
            min_y = ""
            max_y = ""
        else:
            xs = []
            ys = []

            for idx in occupied_indices:
                gy = idx // width
                gx = idx % width

                x = ox + (gx + 0.5) * res
                y = oy + (gy + 0.5) * res

                xs.append(x)
                ys.append(y)

            centroid_x = sum(xs) / len(xs)
            centroid_y = sum(ys) / len(ys)
            min_x = min(xs)
            max_x = max(xs)
            min_y = min(ys)
            max_y = max(ys)

        self.writers["map_stats"].write(
            {
                "bag_time_ns": bag_time_ns,
                "time_s": time_s,
                "topic": topic,
                "frame_id": msg.header.frame_id,
                "resolution_m": res,
                "width_cells": width,
                "height_cells": height,
                "origin_x": ox,
                "origin_y": oy,
                "occupied_count": occupied_count,
                "occupied_area_m2": occupied_area,
                "occupied_centroid_x": centroid_x,
                "occupied_centroid_y": centroid_y,
                "occupied_min_x": min_x,
                "occupied_max_x": max_x,
                "occupied_min_y": min_y,
                "occupied_max_y": max_y,
            }
        )

    def handle_marker_array(
        self,
        topic: str,
        msg: Any,
        bag_time_ns: int,
        time_s: float,
    ):
        for marker in msg.markers:
            p = marker.pose.position
            s = marker.scale
            c = marker.color

            self.writers["marker_obstacles"].write(
                {
                    "bag_time_ns": bag_time_ns,
                    "time_s": time_s,
                    "topic": topic,
                    "frame_id": marker.header.frame_id,
                    "ns": marker.ns,
                    "id": marker.id,
                    "type": marker.type,
                    "action": marker.action,
                    "x": p.x,
                    "y": p.y,
                    "z": p.z,
                    "scale_x": s.x,
                    "scale_y": s.y,
                    "scale_z": s.z,
                    "color_r": c.r,
                    "color_g": c.g,
                    "color_b": c.b,
                    "color_a": c.a,
                    "text": marker.text,
                }
            )

    def handle_path(
        self,
        topic: str,
        msg: Any,
        bag_time_ns: int,
        time_s: float,
        path_msg_counter: Dict[str, int],
    ):
        msg_index = path_msg_counter.get(topic, 0)
        path_msg_counter[topic] = msg_index + 1

        poses = list(msg.poses)
        n = len(poses)

        if n == 0:
            self.writers["paths_summary"].write(
                {
                    "bag_time_ns": bag_time_ns,
                    "time_s": time_s,
                    "topic": topic,
                    "frame_id": msg.header.frame_id,
                    "num_points": 0,
                    "first_x": "",
                    "first_y": "",
                    "first_z": "",
                    "last_x": "",
                    "last_y": "",
                    "last_z": "",
                }
            )
            return

        first = poses[0].pose.position
        last = poses[-1].pose.position

        self.writers["paths_summary"].write(
            {
                "bag_time_ns": bag_time_ns,
                "time_s": time_s,
                "topic": topic,
                "frame_id": msg.header.frame_id,
                "num_points": n,
                "first_x": first.x,
                "first_y": first.y,
                "first_z": first.z,
                "last_x": last.x,
                "last_y": last.y,
                "last_z": last.z,
            }
        )

        if not self.export_full_paths:
            return

        for idx, ps in enumerate(poses):
            p = ps.pose.position
            q = ps.pose.orientation

            _, _, yaw = self.quat_to_euler(q.x, q.y, q.z, q.w)

            self.writers["path_points"].write(
                {
                    "bag_time_ns": bag_time_ns,
                    "time_s": time_s,
                    "topic": topic,
                    "frame_id": msg.header.frame_id,
                    "msg_index": msg_index,
                    "point_index": idx,
                    "x": p.x,
                    "y": p.y,
                    "z": p.z,
                    "qx": q.x,
                    "qy": q.y,
                    "qz": q.z,
                    "qw": q.w,
                    "yaw_rad": yaw,
                    "yaw_deg": math.degrees(yaw),
                }
            )

    def handle_image_like(
        self,
        topic: str,
        type_name: str,
        msg: Any,
        bag_time_ns: int,
        time_s: float,
    ):
        frame_id = ""

        if hasattr(msg, "header"):
            frame_id = msg.header.frame_id

        row = {
            "bag_time_ns": bag_time_ns,
            "time_s": time_s,
            "topic": topic,
            "frame_id": frame_id,
            "format": "",
            "height": "",
            "width": "",
            "encoding": "",
            "data_size_bytes": "",
        }

        if type_name == "sensor_msgs/msg/CompressedImage":
            row["format"] = getattr(msg, "format", "")
            row["data_size_bytes"] = len(getattr(msg, "data", []))

        elif type_name == "sensor_msgs/msg/Image":
            row["height"] = getattr(msg, "height", "")
            row["width"] = getattr(msg, "width", "")
            row["encoding"] = getattr(msg, "encoding", "")
            row["data_size_bytes"] = len(getattr(msg, "data", []))

        elif type_name == "sensor_msgs/msg/CameraInfo":
            row["height"] = getattr(msg, "height", "")
            row["width"] = getattr(msg, "width", "")
            row["encoding"] = "camera_info"

        self.writers["image_index"].write(row)


def main(args=None):
    rclpy.init(args=args)

    node = ApfBagToCsvAnalyzer()

    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()