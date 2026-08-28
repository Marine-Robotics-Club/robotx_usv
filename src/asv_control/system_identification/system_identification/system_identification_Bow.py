#!/usr/bin/env python3
import math
import csv
import time
import datetime
from pathlib import Path

import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64
from geometry_msgs.msg import TwistStamped, Pose2D
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix

from fau_msgs.msg import MotorCmds, TeensyStatus


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


class WamVSysID(Node):
    def __init__(self):
        super().__init__('system_identification')

        # ---------------- params ----------------
        self.declare_parameter('wamv', 'wamv')
        self.declare_parameter('test_type', 'bollard_pull')
        self.declare_parameter('test_axis', 'surge')   # surge | sway | yaw
        self.declare_parameter('yaw_source', 'stern')  # stern | bow | mixed

        # logical commands [%] in your controller/sign convention
        self.declare_parameter('stern_port_cmd', 0)
        self.declare_parameter('stern_stbd_cmd', 0)
        self.declare_parameter('bow_port_cmd', 0)
        self.declare_parameter('bow_stbd_cmd', 0)

        # actuator signs for hardware/sim publishing
        self.declare_parameter('stern_port_sign', 1)
        self.declare_parameter('stern_stbd_sign', 1)
        self.declare_parameter('bow_port_sign', -1)   # inverted bow thrusters
        self.declare_parameter('bow_stbd_sign', 1)   # inverted bow thrusters

        self.declare_parameter('sim_thrust_max_n', 1000.0)
        self.declare_parameter('Simulation', True)

        # ---------------- read params ----------------
        self.wamv_name = self.get_parameter('wamv').get_parameter_value().string_value
        self.test_type = self.get_parameter('test_type').get_parameter_value().string_value
        self.test_axis = self.get_parameter('test_axis').get_parameter_value().string_value
        self.yaw_source = self.get_parameter('yaw_source').get_parameter_value().string_value

        self.stern_port = int(self.get_parameter('stern_port_cmd').get_parameter_value().integer_value)
        self.stern_stbd = int(self.get_parameter('stern_stbd_cmd').get_parameter_value().integer_value)
        self.bow_port = int(self.get_parameter('bow_port_cmd').get_parameter_value().integer_value)
        self.bow_stbd = int(self.get_parameter('bow_stbd_cmd').get_parameter_value().integer_value)

        self.stern_port_sign = int(self.get_parameter('stern_port_sign').get_parameter_value().integer_value)
        self.stern_stbd_sign = int(self.get_parameter('stern_stbd_sign').get_parameter_value().integer_value)
        self.bow_port_sign = int(self.get_parameter('bow_port_sign').get_parameter_value().integer_value)
        self.bow_stbd_sign = int(self.get_parameter('bow_stbd_sign').get_parameter_value().integer_value)

        self.sim_thrust_max_n = float(self.get_parameter('sim_thrust_max_n').get_parameter_value().double_value)
        self.Simulation = bool(self.get_parameter('Simulation').get_parameter_value().bool_value)

        # ---------------- publishers ----------------
        self.Motor_cmds_pub = self.create_publisher(
            MotorCmds, f'/{self.wamv_name}/jetson_cmds', 1
        )

        self.left_thrust_angle_pub = self.create_publisher(
            Float64, f'/{self.wamv_name}/thrusters/left/pos', 10
        )
        self.left_thrust_cmd_pub = self.create_publisher(
            Float64, f'/{self.wamv_name}/thrusters/left/thrust', 10
        )
        self.right_thrust_angle_pub = self.create_publisher(
            Float64, f'/{self.wamv_name}/thrusters/right/pos', 10
        )
        self.right_thrust_cmd_pub = self.create_publisher(
            Float64, f'/{self.wamv_name}/thrusters/right/thrust', 10
        )
        self.left_bow_thrust_cmd_pub = self.create_publisher(
            Float64, f'/{self.wamv_name}/thrusters/left_bow/thrust', 10
        )
        self.right_bow_thrust_cmd_pub = self.create_publisher(
            Float64, f'/{self.wamv_name}/thrusters/right_bow/thrust', 10
        )

        # ---------------- subscribers ----------------
        # Pose source
        self.curr_pose_sub = self.create_subscription(
            Pose2D, f'/{self.wamv_name}/vehicle_pose', self.pose_callback, 10
        )

        # Body-frame velocity source
        self.odom_sub = self.create_subscription(
            Odometry, f'/{self.wamv_name}/p3d_wamv_ned', self.odom_callback, 10
        )

        # Aux GPS
        self.vel_sub = self.create_subscription(
            TwistStamped, f'/{self.wamv_name}/global_position/raw/gps_vel',
            self.gps_vel_callback, 10
        )
        # GPS fix topic depends on sim vs hardware
        if self.Simulation:
            gps_fix_topic = f'/{self.wamv_name}/sensors/gps/gps/fix'
        else:
            gps_fix_topic = f'/{self.wamv_name}/global_position/global'

        self.gps_fix_sub = self.create_subscription(
            NavSatFix, gps_fix_topic, self.gps_fix_callback, 10
        )
        # Aux weather
        self.create_subscription(
            TeensyStatus, f'/{self.wamv_name}/pcu_status',
            self.teensy_status_callback, 10
        )

        # ---------------- state ----------------
        self.timer_test = time.time()
        self.start_time = False
        self.finished = False
        self.done_msg_printed = False

        self.sim_t0 = None

        # Pose from /vehicle_pose
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        # Body velocities from /p3d_wamv_ned
        self.u = 0.0
        self.v = 0.0
        self.vz = 0.0
        self.wx = 0.0
        self.wy = 0.0
        self.r = 0.0

        # Numerical derivatives
        self.ax = 0.0
        self.ay = 0.0
        self.r_dot = 0.0

        self.prev_odom_time = None
        self.prev_u = None
        self.prev_v = None
        self.prev_r = None

        # GPS aux
        self.lat = 0.0
        self.lon = 0.0
        self.gps_u = 0.0
        self.gps_v = 0.0
        self.gps_vz = 0.0
        self.gps_speed = 0.0
        self.gps_yaw_ned = 0.0

        # Weather aux
        self.wind_dir_local_deg = 0.0
        self.wind_dir_global_deg = 0.0
        self.wind_speed = 0.0
        self.temp = 0.0
        self.pressure = 0.0

        # last logical commands
        self.last_stern_port_pct = self.stern_port
        self.last_stern_stbd_pct = self.stern_stbd
        self.last_bow_port_pct = self.bow_port
        self.last_bow_stbd_pct = self.bow_stbd

        # last hardware-applied commands after sign correction
        self.last_stern_port_hw_pct = self.stern_port * self.stern_port_sign
        self.last_stern_stbd_hw_pct = self.stern_stbd * self.stern_stbd_sign
        self.last_bow_port_hw_pct = self.bow_port * self.bow_port_sign
        self.last_bow_stbd_hw_pct = self.bow_stbd * self.bow_stbd_sign

        # ---------------- logging ----------------
        self.log_enabled = True
        self.log_dir = Path.home() / 'sysid_logs'
        self.log_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_path = self.log_dir / (
            f"sysid_{self.wamv_name}_{self.test_type}_{self.test_axis}_{ts}.csv"
        )

        self._log_fh = open(self.log_path, 'w', newline='', encoding='utf-8')
        self._log_csv = csv.writer(self._log_fh)
        self._log_csv.writerow([
            'sim_time_s', 'ros_time_s', 'wall_time_s',
            'test_type', 'test_axis', 'yaw_source',
            'x', 'y', 'yaw',
            'u_bf', 'v_bf', 'w_bf',
            'wx', 'wy', 'r',
            'ax_bf', 'ay_bf', 'r_dot',
            'gps_lat', 'gps_lon',
            'gps_u', 'gps_v', 'gps_vz', 'gps_speed', 'gps_yaw_ned',
            'cmd_stern_port_pct', 'cmd_stern_stbd_pct', 'cmd_bow_port_pct', 'cmd_bow_stbd_pct',
            'cmd_stern_port_hw_pct', 'cmd_stern_stbd_hw_pct', 'cmd_bow_port_hw_pct', 'cmd_bow_stbd_hw_pct',
            'wind_dir_local_deg', 'wind_dir_global_deg', 'wind_speed', 'temp_C', 'pressure_hPa'
        ])

        self.get_logger().info(f"Logging to {self.log_path}")

        # 10 Hz main loop
        self.timer = self.create_timer(0.1, self.loop)

    # ---------------- helpers ----------------
    def _pct_to_pwm(self, pct: int) -> int:
        pct = clamp(int(pct), -100, 100)
        return clamp(1500 + 5 * pct, 1000, 2000)

    def _pct_to_thrust_N(self, pct: int) -> float:
        pct = clamp(int(pct), -100, 100)
        return (pct / 100.0) * self.sim_thrust_max_n

    def _reset_test_state(self):
        self.timer_test = time.time()
        self.start_time = False
        self.finished = False
        self.done_msg_printed = False

    # ---------------- callbacks ----------------
    def pose_callback(self, msg: Pose2D):
        # Pose comes from /vehicle_pose
        self.x = float(msg.x)
        self.y = float(msg.y)
        self.yaw = float(msg.theta)

    def odom_callback(self, msg: Odometry):
        # IMPORTANT:
        # /p3d_wamv_ned already contains BODY-FRAME linear velocity:
        #   linear.x = u_body
        #   linear.y = v_body
        #   linear.z = w_body
        #
        # We use /vehicle_pose for x, y, yaw.
        self.u = float(msg.twist.twist.linear.x)
        self.v = float(msg.twist.twist.linear.y)
        self.vz = float(msg.twist.twist.linear.z)

        self.wx = float(msg.twist.twist.angular.x)
        self.wy = float(msg.twist.twist.angular.y)
        self.r = float(msg.twist.twist.angular.z)

        # Numerical body-frame derivatives
        now = self.get_clock().now().nanoseconds * 1e-9

        if self.prev_odom_time is None:
            self.ax = 0.0
            self.ay = 0.0
            self.r_dot = 0.0
        else:
            dt = now - self.prev_odom_time
            if dt > 1e-6:
                self.ax = (self.u - self.prev_u) / dt
                self.ay = (self.v - self.prev_v) / dt
                self.r_dot = (self.r - self.prev_r) / dt
            else:
                self.ax = 0.0
                self.ay = 0.0
                self.r_dot = 0.0

        self.prev_odom_time = now
        self.prev_u = self.u
        self.prev_v = self.v
        self.prev_r = self.r

    def gps_vel_callback(self, msg: TwistStamped):
        vx = float(msg.twist.linear.x)
        vy = float(msg.twist.linear.y)
        vz = float(msg.twist.linear.z)

        self.gps_u = vx
        self.gps_v = vy
        self.gps_vz = vz
        self.gps_speed = math.hypot(vx, vy)

        yaw_enu = math.atan2(vy, vx)
        self.gps_yaw_ned = (math.pi / 2.0 - yaw_enu + math.pi) % (2.0 * math.pi) - math.pi

    def gps_fix_callback(self, msg: NavSatFix):
        self.lat = float(msg.latitude)
        self.lon = float(msg.longitude)

    def teensy_status_callback(self, msg: TeensyStatus):
        self.wind_dir_local_deg = float(getattr(msg, 'wind_direction', 0.0))
        self.wind_speed = float(getattr(msg, 'wind_speed', 0.0))
        self.temp = float(getattr(msg, 'temp', 0.0))
        self.pressure = float(getattr(msg, 'pressure', 0.0))

        yaw_deg = math.degrees(self.yaw)
        self.wind_dir_global_deg = (self.wind_dir_local_deg + yaw_deg) % 360.0

    # ---------------- actuation ----------------
    def send_motor_cmds(self, stern_port: int, stern_stbd: int, bow_port: int = 0, bow_stbd: int = 0):
        # Save logical commands
        self.last_stern_port_pct = int(stern_port)
        self.last_stern_stbd_pct = int(stern_stbd)
        self.last_bow_port_pct = int(bow_port)
        self.last_bow_stbd_pct = int(bow_stbd)

        # Apply actuator sign convention
        stern_port_hw = clamp(self.stern_port_sign * int(stern_port), -100, 100)
        stern_stbd_hw = clamp(self.stern_stbd_sign * int(stern_stbd), -100, 100)
        bow_port_hw = clamp(self.bow_port_sign * int(bow_port), -100, 100)
        bow_stbd_hw = clamp(self.bow_stbd_sign * int(bow_stbd), -100, 100)

        self.last_stern_port_hw_pct = stern_port_hw
        self.last_stern_stbd_hw_pct = stern_stbd_hw
        self.last_bow_port_hw_pct = bow_port_hw
        self.last_bow_stbd_hw_pct = bow_stbd_hw

        # Simulation thrust topics
        self.left_thrust_cmd_pub.publish(Float64(data=self._pct_to_thrust_N(stern_port_hw)))
        self.right_thrust_cmd_pub.publish(Float64(data=self._pct_to_thrust_N(stern_stbd_hw)))
        self.left_bow_thrust_cmd_pub.publish(Float64(data=self._pct_to_thrust_N(bow_port_hw)))
        self.right_bow_thrust_cmd_pub.publish(Float64(data=self._pct_to_thrust_N(bow_stbd_hw)))

        # Keep stern azimuth forward
        self.left_thrust_angle_pub.publish(Float64(data=0.0))
        self.right_thrust_angle_pub.publish(Float64(data=0.0))

        # Hardware PWM via MotorCmds
        msg = MotorCmds()
        msg.port_thrust = int(self._pct_to_pwm(stern_port_hw))
        msg.stbd_thrust = int(self._pct_to_pwm(stern_stbd_hw))
        msg.port_bow_thrust = int(self._pct_to_pwm(bow_port_hw))
        msg.stbd_bow_thrust = int(self._pct_to_pwm(bow_stbd_hw))
        msg.port_azimuth = 1540
        msg.stbd_azimuth = 1560
        msg.red = 0
        msg.yellow = 0
        msg.green = 0
        self.Motor_cmds_pub.publish(msg)

    def stop_all(self):
        self.send_motor_cmds(0, 0, 0, 0)

    # ---------------- logging ----------------
    def _log_sample(self):
        if not self.log_enabled:
            return

        def f(x, p):
            try:
                return f"{float(x):.{p}f}"
            except Exception:
                return str(x)

        now = self.get_clock().now()
        ros_t = now.nanoseconds * 1e-9

        if self.sim_t0 is None:
            self.sim_t0 = now
        sim_t = (now - self.sim_t0).nanoseconds * 1e-9

        wall_t = time.time()

        self._log_csv.writerow([
            f(sim_t, 3), f(ros_t, 3), f(wall_t, 3),
            self.test_type, self.test_axis, self.yaw_source,
            f(self.x, 3), f(self.y, 3), f(self.yaw, 4),
            f(self.u, 3), f(self.v, 3), f(self.vz, 3),
            f(self.wx, 4), f(self.wy, 4), f(self.r, 4),
            f(self.ax, 4), f(self.ay, 4), f(self.r_dot, 4),
            f(self.lat, 8), f(self.lon, 8),
            f(self.gps_u, 3), f(self.gps_v, 3), f(self.gps_vz, 3),
            f(self.gps_speed, 3), f(self.gps_yaw_ned, 4),
            int(self.last_stern_port_pct), int(self.last_stern_stbd_pct),
            int(self.last_bow_port_pct), int(self.last_bow_stbd_pct),
            int(self.last_stern_port_hw_pct), int(self.last_stern_stbd_hw_pct),
            int(self.last_bow_port_hw_pct), int(self.last_bow_stbd_hw_pct),
            f(self.wind_dir_local_deg, 1), f(self.wind_dir_global_deg, 1),
            f(self.wind_speed, 3), f(self.temp, 2), f(self.pressure, 2)
        ])
        self._log_fh.flush()

    # ---------------- tests ----------------
    def _run_bollard_pull(self):
        self.send_motor_cmds(
            self.stern_port,
            self.stern_stbd,
            self.bow_port,
            self.bow_stbd
        )
        self._log_sample()

    def _run_acceleration(self):
        if not self.start_time:
            self.get_logger().info(
                f'Running Acceleration test | axis={self.test_axis} | yaw_source={self.yaw_source}'
            )
            self.timer_test = time.time()
            self.start_time = True
            self.finished = False
            self.done_msg_printed = False

        elapsed = time.time() - self.timer_test

        if elapsed > 60.0:
            self.stop_all()
            if not self.done_msg_printed:
                self.get_logger().info('Done Acceleration test.')
                self.done_msg_printed = True
            self.finished = True
            self._log_sample()
            return

        if self.test_axis == 'surge':
            # Surge from stern
            self.send_motor_cmds(self.stern_port, self.stern_stbd, 0, 0)

        elif self.test_axis == 'sway':
            # Sway from bow
            self.send_motor_cmds(0, 0, self.bow_port, self.bow_stbd)

        elif self.test_axis == 'yaw':
            if self.yaw_source == 'stern':
                # Example: +40 / -40
                self.send_motor_cmds(self.stern_port, self.stern_stbd, 0, 0)

            elif self.yaw_source == 'bow':
                # Example logical: +40 / -40; sign inversion handled automatically
                self.send_motor_cmds(0, 0, self.bow_port, self.bow_stbd)

            elif self.yaw_source == 'mixed':
                self.send_motor_cmds(self.stern_port, self.stern_stbd, self.bow_port, self.bow_stbd)

            else:
                self.get_logger().warn(f'Unknown yaw_source: {self.yaw_source}')
                self.stop_all()

        else:
            self.get_logger().warn(f'Unknown test_axis: {self.test_axis}')
            self.stop_all()

        self._log_sample()

    def _run_circle(self):
        if not self.start_time:
            self.get_logger().info(f'Running Circle test | yaw_source={self.yaw_source}')
            self.timer_test = time.time()
            self.start_time = True
            self.finished = False
            self.done_msg_printed = False

        elapsed = time.time() - self.timer_test

        if elapsed <= 20.0:
            # Accelerate straight first
            self.send_motor_cmds(100, 100, 0, 0)

        elif elapsed <= 60.0:
            if self.yaw_source == 'stern':
                # Keep whatever stern asymmetry you pass
                self.send_motor_cmds(self.stern_port, self.stern_stbd, 0, 0)

            elif self.yaw_source == 'bow':
                # Forward from stern + yaw from bow
                self.send_motor_cmds(60, 60, self.bow_port, self.bow_stbd)

            elif self.yaw_source == 'mixed':
                self.send_motor_cmds(self.stern_port, self.stern_stbd, self.bow_port, self.bow_stbd)

            else:
                self.get_logger().warn(f'Unknown yaw_source: {self.yaw_source}')
                self.stop_all()

        else:
            self.stop_all()
            if not self.done_msg_printed:
                self.get_logger().info('Done Circle test.')
                self.done_msg_printed = True
            self.finished = True

        self._log_sample()

    def _run_zigzag(self):
        if not self.start_time:
            self.get_logger().info('Zig-Zag: start')
            self.timer_test = time.time()
            self.start_time = True
            self.finished = False
            self.done_msg_printed = False

        t = time.time() - self.timer_test
        leg = 10.0

        if t <= 20.0:
            self.send_motor_cmds(60, 60, 0, 0)
        elif t <= 20.0 + 1 * leg:
            self.send_motor_cmds(70, 30, 0, 0)
        elif t <= 20.0 + 2 * leg:
            self.send_motor_cmds(30, 70, 0, 0)
        elif t <= 20.0 + 3 * leg:
            self.send_motor_cmds(70, 30, 0, 0)
        elif t <= 20.0 + 4 * leg:
            self.send_motor_cmds(30, 70, 0, 0)
        else:
            self.stop_all()
            if not self.done_msg_printed:
                self.get_logger().info('Zig-Zag: done')
                self.done_msg_printed = True
            self.finished = True

        self._log_sample()

    def _run_wind(self):
        if not self.start_time:
            self.get_logger().info('Logging Wind Data...')
            self.start_time = True
        self._log_sample()

    # ---------------- main loop ----------------
    def loop(self):
        # Runtime param reload
        new_test_type = self.get_parameter('test_type').get_parameter_value().string_value
        new_test_axis = self.get_parameter('test_axis').get_parameter_value().string_value
        new_yaw_source = self.get_parameter('yaw_source').get_parameter_value().string_value

        if (
            new_test_type != self.test_type
            or new_test_axis != self.test_axis
            or new_yaw_source != self.yaw_source
        ):
            self.test_type = new_test_type
            self.test_axis = new_test_axis
            self.yaw_source = new_yaw_source
            self._reset_test_state()
        else:
            self.test_type = new_test_type
            self.test_axis = new_test_axis
            self.yaw_source = new_yaw_source

        self.stern_port = int(self.get_parameter('stern_port_cmd').get_parameter_value().integer_value)
        self.stern_stbd = int(self.get_parameter('stern_stbd_cmd').get_parameter_value().integer_value)
        self.bow_port = int(self.get_parameter('bow_port_cmd').get_parameter_value().integer_value)
        self.bow_stbd = int(self.get_parameter('bow_stbd_cmd').get_parameter_value().integer_value)

        self.stern_port_sign = int(self.get_parameter('stern_port_sign').get_parameter_value().integer_value)
        self.stern_stbd_sign = int(self.get_parameter('stern_stbd_sign').get_parameter_value().integer_value)
        self.bow_port_sign = int(self.get_parameter('bow_port_sign').get_parameter_value().integer_value)
        self.bow_stbd_sign = int(self.get_parameter('bow_stbd_sign').get_parameter_value().integer_value)

        if self.test_type == 'bollard_pull':
            self._run_bollard_pull()

        elif self.test_type == 'acceleration':
            self._run_acceleration()

        elif self.test_type == 'circle':
            self._run_circle()

        elif self.test_type == 'zigzag':
            self._run_zigzag()

        elif self.test_type == 'dynamic_pull':
            self.get_logger().info('Running Dynamic Pull test... TODO')
            self._log_sample()

        elif self.test_type == 'wind':
            self._run_wind()

        else:
            self.get_logger().warn(f'Unknown test type: {self.test_type}')
            self.stop_all()

    # ---------------- shutdown ----------------
    def destroy_node(self):
        try:
            self.stop_all()
            if hasattr(self, '_log_fh') and not self._log_fh.closed:
                self._log_fh.flush()
                self._log_fh.close()
        finally:
            super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = WamVSysID()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()