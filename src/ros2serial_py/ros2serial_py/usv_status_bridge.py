"""Publish the RoboBoat USV state as the shared SeaOwls VehicleStatus message."""
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Pose2D, Vector3, Vector3Stamped
from sensor_msgs.msg import NavSatFix
from seaowls_interfaces.msg import VehicleStatus


class UsvStatusBridge(Node):
    def __init__(self):
        super().__init__('seaowls_usv_status_bridge')
        self.declare_parameter('asv', 'asv')
        self.declare_parameter('output_topic', '/seaowls/usv1/status')
        self.declare_parameter('gps_topic', '/sbg_legacy/gps/fix')
        self.declare_parameter('rpy_topic', '/sbg_legacy/rpy')
        asv = self.get_parameter('asv').value
        self.lat = self.lon = self.speed = self.heading = 0.0
        self.roll = self.pitch = 0.0
        self.last_rx = 0.0
        self.pub = self.create_publisher(
            VehicleStatus, self.get_parameter('output_topic').value, 10)
        self.create_subscription(
            NavSatFix, self.get_parameter('gps_topic').value,
            self.gps_cb, qos_profile_sensor_data)
        self.create_subscription(
            Vector3, f'/{asv}/bfvelo_ned', self.velocity_cb, 10)
        self.create_subscription(
            Pose2D, f'/{asv}/vehicle_pose', self.pose_cb, 10)
        self.create_subscription(
            Vector3Stamped, self.get_parameter('rpy_topic').value,
            self.rpy_cb, qos_profile_sensor_data)
        self.create_timer(0.2, self.publish_status)

    def gps_cb(self, msg):
        self.lat, self.lon = msg.latitude, msg.longitude
        self.last_rx = time.monotonic()

    def velocity_cb(self, msg):
        self.speed = math.hypot(msg.x, msg.y)
        self.last_rx = time.monotonic()

    def pose_cb(self, msg):
        self.heading = math.degrees(msg.theta) % 360.0
        self.last_rx = time.monotonic()

    def rpy_cb(self, msg):
        # state_to_ned receives SBG RPY in radians and preserves x/y as roll/pitch.
        self.roll = math.degrees(msg.vector.x)
        self.pitch = math.degrees(msg.vector.y)
        self.last_rx = time.monotonic()

    def publish_status(self):
        if self.last_rx == 0.0 or time.monotonic() - self.last_rx > 2.0:
            self.get_logger().warning('USV telemetry is stale; status suppressed')
            return
        msg = VehicleStatus()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'usv1'
        msg.vehicle_id = 'USV1'
        msg.state = VehicleStatus.STATE_AUTO
        msg.vehicle_type = VehicleStatus.TYPE_USV
        msg.current_task = VehicleStatus.TASK_NONE
        msg.latitude, msg.longitude = self.lat, self.lon
        msg.speed_mps, msg.heading_deg = float(self.speed), float(self.heading)
        msg.roll_deg, msg.pitch_deg = float(self.roll), float(self.pitch)
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = UsvStatusBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
