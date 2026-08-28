#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Float32
from fau_msgs.msg import MotorCmds, TeensyStatus
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion
from sensor_msgs.msg import NavSatFix
import math
import csv, time, datetime
from pathlib import Path

def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

class WamVSysID(Node):
    def __init__(self):
        super().__init__('system_identification')

        # --- params ---
        self.declare_parameter('wamv', 'wamv1')
        self.declare_parameter('test_type', 'bollard_pull')
        self.declare_parameter('stbd_cmd', 100)      # int %
        self.declare_parameter('port_cmd', 100)      # int %
        self.declare_parameter('sim_thrust_max_n', 1000.0)
        self.declare_parameter('Simulation', True)

        self.wind_direction = None
        self.wind_speed = None
        self.temp = None
        self.pressure = None

        # --- read params ---
        self.wamv_name = self.get_parameter('wamv').get_parameter_value().string_value
        self.test_type = self.get_parameter('test_type').get_parameter_value().string_value
        self.stbd = int(self.get_parameter('stbd_cmd').get_parameter_value().integer_value)
        self.port = int(self.get_parameter('port_cmd').get_parameter_value().integer_value)
        self.sim_thrust_max_n = float(self.get_parameter('sim_thrust_max_n').get_parameter_value().double_value)
        self.Simulation = bool(self.get_parameter('Simulation').get_parameter_value().bool_value)

        # command presets (kept)
        self.port_cmds = [-100,-90,-80,-70,-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60,70,80,90,100]
        self.stbd_cmds = [-100,-90,-80,-70,-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60,70,80,90,100]

        # --- publishers ---
        self.Motor_cmds_pub = self.create_publisher(MotorCmds, f'/{self.wamv_name}/jetson_cmds', 1)
        self.left_thrust_angle_pub   = self.create_publisher(Float64, f'/{self.wamv_name}/thrusters/left/pos', 10)
        self.left_thrust_cmd_pub     = self.create_publisher(Float64, f'/{self.wamv_name}/thrusters/left/thrust', 10)
        self.right_thrust_angle_pub  = self.create_publisher(Float64, f'/{self.wamv_name}/thrusters/right/pos', 10)
        self.right_thrust_cmd_pub    = self.create_publisher(Float64, f'/{self.wamv_name}/thrusters/right/thrust', 10)
        self.right_bow_thrust_cmd_pub = self.create_publisher(Float64, f'/{self.wamv_name}/thrusters/right_bow/thrust', 10)
        self.left_bow_thrust_cmd_pub  = self.create_publisher(Float64, f'/{self.wamv_name}/thrusters/left_bow/thrust', 10)

        # --- subscribers ---
        # Use Odometry, not Pose2D, since your callback uses pose.pose & twist.twist
        self.odom_sub = self.create_subscription(
            Odometry, f'/{self.wamv_name}/p3d_wamv_ned', self.odom_callback, 10
        )
        self.vel_sub = self.create_subscription(
            TwistStamped, f'/{self.wamv_name}/global_position/raw/gps_vel', self.gps_vel_callback, 10
        )
        self.gps_fix_sub = self.create_subscription(
            NavSatFix, f'/{self.wamv_name}/global_position/global', self._gps_fix_cb, 10
        )

        # Subscriptions
        self.create_subscription(TeensyStatus, f'/{self.wamv_name}/pcu_status', self.teensy_status_callback, 10)

        # --- timers & state ---
        self.timer_test = time.time()
        self.start_time = False

        # --- logging (no ROS params, all in-code) ---
        self.log_enabled = True
        self.log_dir = Path.home() / 'sysid_logs'
        self.log_base = 'sysid'
        self.log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_path = self.log_dir / f"{self.log_base}_{self.wamv_name}_{self.test_type}_{ts}.csv"
        self._log_fh = open(self.log_path, 'w', newline='', encoding='utf-8')
        self._log_csv = csv.writer(self._log_fh)
        self._log_csv.writerow([
            'ros_time_s','wall_time_s','test_type',
            'x','y','yaw',
            'u','v','vz',
            'wx','wy','r','ax','ay','r_dot',
            'gps_lat','gps_lon','gps_u','gps_v','gps_vz','gps_speed','gps_yaw_ned',
            'cmd_port_pct','cmd_stbd_pct',
            'wind_dir_local_deg','wind_dir_global_deg','wind_speed','temp_C','pressure_hPa'
        ])

        self.get_logger().info(f"Logging to {self.log_path}")

        # defaults used by the logger
        self.x = self.y = self.yaw = 0.0
        self.u = self.v = self.vz = 0.0
        self.wx = self.wy = self.r = 0.0
        self.ax = self.ay = self.r_dot = 0.0
        self.lat = self.lon = 0.0
        self.gps_u = self.gps_v = self.gps_vz = 0.0
        self.gps_speed = self.gps_yaw_ned = 0.0
        self.last_port_pct = int(self.port)
        self.last_stbd_pct = int(self.stbd)

        self.wind_dir_local_deg = 0.0
        self.wind_dir_global_deg = 0.0
        self.wind_speed = 0.0
        self.temp = 0.0
        self.pressure = 0.0


        # 10 Hz loop
        self.timer = self.create_timer(0.1, self.loop)

    # -------------- helpers --------------
    def _pct_to_pwm(self, pct: int) -> int:
        """[-100..100]% -> PWM [1000..2000], 0%->1500."""
        pct = clamp(int(pct), -100, 100)
        return clamp(1500 + 5 * pct, 1000, 2000)

    def _pct_to_thrust_N(self, pct: int) -> float:
        """[-100..100]% -> [-sim_thrust_max_n .. +sim_thrust_max_n] N."""
        pct = clamp(int(pct), -100, 100)
        return (pct / 100.0) * self.sim_thrust_max_n

    # -------------- actuation --------------
    def send_motor_cmds(self, port: int, stbd: int):
        self.last_port_pct = int(port)
        self.last_stbd_pct = int(stbd)


        left_msg = Float64()
        right_msg = Float64()
        left_msg.data  = self._pct_to_thrust_N(port)
        right_msg.data = self._pct_to_thrust_N(stbd)
        self.left_thrust_cmd_pub.publish(left_msg)
        self.right_thrust_cmd_pub.publish(right_msg)

        # Hardware: PWM via MotorCmds
        msg = MotorCmds()
        msg.port_thrust = int(self._pct_to_pwm(port))
        msg.stbd_thrust = int(self._pct_to_pwm(stbd))
        msg.port_azimuth = 1500
        msg.stbd_azimuth = 1500
        msg.stbd_bow_thrust = 1500
        msg.port_bow_thrust = 1500
        msg.red = 0
        msg.yellow = 0
        msg.green = 0
        self.Motor_cmds_pub.publish(msg)

    # -------------- callbacks --------------
    def odom_callback(self, msg: Odometry):
        # position
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y

        # yaw: ENU -> NED (ψ_NED = wrap(pi/2 - ψ_ENU))
        q = msg.pose.pose.orientation
        q_list = [q.x, q.y, q.z, q.w]
        _, _, yaw_enu = euler_from_quaternion(q_list)
        def wrap_pi(a: float) -> float:
            return (a + math.pi) % (2.0 * math.pi) - math.pi
        self.yaw = wrap_pi(math.pi/2.0 - yaw_enu)

        # linear velocities
        self.u  = msg.twist.twist.linear.x
        self.v  = msg.twist.twist.linear.y
        self.vz = msg.twist.twist.linear.z

        # angular rates
        self.wx = msg.twist.twist.angular.x
        self.wy = msg.twist.twist.angular.y
        self.r  = msg.twist.twist.angular.z   # yaw rate

        # optional "accelerations" stashed in covariance[0:2] and [2] for yaw accel
        cov = msg.twist.covariance
        try:
            self.ax   = float(cov[0])
            self.ay   = float(cov[1])
            self.r_dot = float(cov[2])
        except Exception:
            self.ax = self.ay = self.r_dot = 0.0


    def gps_vel_callback(self, msg: TwistStamped):
        vx = msg.twist.linear.x
        vy = msg.twist.linear.y
        vz = msg.twist.linear.z

        self.gps_u  = vx
        self.gps_v  = vy
        self.gps_vz = vz
        self.gps_speed = math.hypot(vx, vy)

        yaw_enu = math.atan2(vy, vx)
        def wrap_pi(a: float) -> float:
            return (a + math.pi) % (2.0 * math.pi) - math.pi
        self.gps_yaw_ned = wrap_pi(math.pi/2.0 - yaw_enu)

    def _gps_fix_cb(self, msg: NavSatFix):
        self.lat = float(msg.latitude)
        self.lon = float(msg.longitude)

    def teensy_status_callback(self, msg: TeensyStatus):
        # Local (sensor) wind direction in degrees (NED)
        self.wind_dir_local_deg = float(getattr(msg, 'wind_direction', 0.0))
        self.wind_speed = float(getattr(msg, 'wind_speed', 0.0))
        self.temp = float(getattr(msg, 'temp', 0.0))
        self.pressure = float(getattr(msg, 'pressure', 0.0))

        # Use NED yaw you already compute in odom_callback (radians)
        yaw_deg = math.degrees(self.yaw) if self.yaw is not None else 0.0

        # Transform to global and wrap to [0, 360)
        self.wind_dir_global_deg = (self.wind_dir_local_deg + yaw_deg) % 360.0

    # -------------- logging --------------
    def _log_sample(self):
        if not self.log_enabled:
            return
        

        # helpers for consistent precision
        def f(x, p):  # format float with p decimals
            try:
                return f"{float(x):.{p}f}"
            except Exception:
                return str(x)
        ros_t = self.get_clock().now().nanoseconds * 1e-9
        wall_t = time.time()
        row = [
            f(ros_t, 3), f(wall_t, 3), self.test_type,
            f(self.x, 3), f(self.y, 3), f(self.yaw, 4),
            f(self.u, 3), f(self.v, 3), f(self.vz, 3),
            f(self.wx, 4), f(self.wy, 4), f(self.r, 4),
            f(self.ax, 4), f(self.ay, 4), f(self.r_dot, 4),
            f(self.lat, 8), f(self.lon, 8),
            f(self.gps_u, 3), f(self.gps_v, 3), f(self.gps_vz, 3),
            f(self.gps_speed, 3), f(self.gps_yaw_ned, 4),
            int(self.last_port_pct), int(self.last_stbd_pct),
            f(self.wind_dir_local_deg, 1), f(self.wind_dir_global_deg, 1),
            f(self.wind_speed, 3), f(self.temp, 2), f(self.pressure, 2)
        ]


        self._log_csv.writerow(row)
        self._log_fh.flush()

    # -------------- main loop --------------
    def loop(self):
        # allow runtime param change of test_type
        self.test_type = self.get_parameter('test_type').get_parameter_value().string_value

        if self.test_type == 'bollard_pull':
            self.get_logger().info("Running Bollard Pull test...")
            self.send_motor_cmds(self.port, self.stbd)
            self._log_sample()

        elif self.test_type == 'acceleration':
            if not self.start_time:
                self.get_logger().info("Running Acceleration test...")
                self.timer_test = time.time()
                self.start_time = True
            
            if (time.time() - self.timer_test) > 60.0:
                self.send_motor_cmds(0, 0)
                self.get_logger().info("Done Acceleration test.")
            else:
                self.send_motor_cmds(self.port, self.stbd)
            self._log_sample()

        elif self.test_type == 'circle':
            if not self.start_time:
                self.get_logger().info("Running Circle test... Accelerating")
                self.timer_test = time.time()
                self.start_time = True

            elapsed = time.time() - self.timer_test
            if elapsed <= 20.0:
                self.send_motor_cmds(100, 100)
            elif elapsed <= 50.0:
                self.send_motor_cmds(self.port, self.stbd)
            else:
                self.send_motor_cmds(0, 0)
                self.get_logger().info("Done Circle test.")
            self._log_sample()

        elif self.test_type == 'zigzag':
            if not self.start_time:
                self.get_logger().info("Zig-Zag: start")
                self.timer_test = time.time()
                self.start_time = True

            t = time.time() - self.timer_test
            LEG = 10.0
            if t <= 20.0:
                self.send_motor_cmds(100, 100)
            elif t <= 20.0 + 1*LEG:
                self.send_motor_cmds(100, 0)
            elif t <= 20.0 + 2*LEG:
                self.send_motor_cmds(0, 100)
            elif t <= 20.0 + 3*LEG:
                self.send_motor_cmds(100, 0)
            elif t <= 20.0 + 4*LEG:
                self.send_motor_cmds(0, 100)
            else:
                self.send_motor_cmds(0, 0)
                self.get_logger().info("Zig-Zag: done")
            self._log_sample()

        elif self.test_type == 'dynamic_pull':
            self.get_logger().info("Running Dynamic Pull test...")
            # TODO: add your code here

        elif self.test_type == 'wind':
            if not self.start_time:
                self.get_logger().info("Logging Wind Data...")
                self.start_time = True
            # no motor commands; just log at timer rate
            self._log_sample()


        else:
            self.get_logger().warn(f"Unknown test type: {self.test_type}")

    # -------------- shutdown --------------
    def destroy_node(self):
        try:
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
