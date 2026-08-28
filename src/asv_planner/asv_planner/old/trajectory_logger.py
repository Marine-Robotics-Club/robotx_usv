#!/usr/bin/env python3

import os
import csv
import math

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped


def quat_to_yaw(qx, qy, qz, qw) -> float:
    # yaw from quaternion (Z axis)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


class TrajectoryLogger(Node):
    def __init__(self):
        super().__init__('trajectory_logger')

        # NOTE: do NOT declare use_sim_time here; launch/ROS declares it already.

        self.declare_parameter('wamv', 'wamv')
        self.declare_parameter('odom_topic', 'p3d_wamv_ned')          # relative topic under /<wamv>/
        self.declare_parameter('path_topic', 'trajectory/path')
        self.declare_parameter('save_path', '')              # e.g. "/home/xavi/logs/run1.csv"
        self.declare_parameter('min_dist', 0.10)             # meters between samples
        self.declare_parameter('max_points', 20000)          # cap memory (optional)

        self.wamv = str(self.get_parameter('wamv').value)
        self.odom_topic = str(self.get_parameter('odom_topic').value)
        self.path_topic = str(self.get_parameter('path_topic').value)
        self.save_path = str(self.get_parameter('save_path').value).strip()
        self.min_dist = float(self.get_parameter('min_dist').value)
        self.max_points = int(self.get_parameter('max_points').value)

        # Storage
        self.rows = []         # list of [t, x, y, z, yaw]
        self.last_xy = None

        # Path for RViz
        self.path_msg = Path()
        self.path_msg.header.frame_id = 'odom'

        self.path_pub = self.create_publisher(
            Path,
            f'/{self.wamv}/{self.path_topic}'.replace('//', '/'),
            10
        )

        self.create_subscription(
            Odometry,
            f'/{self.wamv}/{self.odom_topic}'.replace('//', '/'),
            self.odom_cb,
            50
        )

        self.get_logger().info(
            f"TrajectoryLogger | sub=/{self.wamv}/{self.odom_topic} "
            f"pub=/{self.wamv}/{self.path_topic} min_dist={self.min_dist}m "
            f"max_points={self.max_points} "
            f"save_path='{self.save_path or 'DISABLED'}'"
        )

    def odom_cb(self, msg: Odometry):
        # frame id from odom
        frame = msg.header.frame_id if msg.header.frame_id else 'odom'
        self.path_msg.header.frame_id = frame
        self.path_msg.header.stamp = msg.header.stamp

        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        z = float(msg.pose.pose.position.z)

        q = msg.pose.pose.orientation
        yaw = quat_to_yaw(q.x, q.y, q.z, q.w)

        # Downsample by distance
        if self.last_xy is not None:
            dx = x - self.last_xy[0]
            dy = y - self.last_xy[1]
            if math.hypot(dx, dy) < self.min_dist:
                return

        # Timestamp (sec.nanosec)
        t = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9

        # Save row
        self.rows.append([t, x, y, z, yaw])
        self.last_xy = (x, y)

        # Cap memory (optional)
        if len(self.rows) > self.max_points:
            self.rows.pop(0)
            if self.path_msg.poses:
                self.path_msg.poses.pop(0)

        # Append to Path for RViz
        ps = PoseStamped()
        ps.header = msg.header
        ps.pose = msg.pose.pose
        self.path_msg.poses.append(ps)

        self.path_pub.publish(self.path_msg)

    def destroy_node(self):
        # Save on shutdown
        if self.save_path:
            out_dir = os.path.dirname(self.save_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)

            with open(self.save_path, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['t', 'x', 'y', 'z', 'yaw'])
                w.writerows(self.rows)

            self.get_logger().info(f"Saved trajectory CSV: {self.save_path} ({len(self.rows)} samples)")

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
