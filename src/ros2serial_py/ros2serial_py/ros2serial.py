#!/usr/bin/env python3
import math
import threading
import time
from dataclasses import dataclass
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
)

from std_msgs.msg import Bool, Float64
from sensor_msgs.msg import NavSatFix, Imu
from geometry_msgs.msg import Twist
from fau_msgs.msg import MotorCmds, TeensyStatus

import serial  # pyserial


def deg2rad(d: float) -> float:
    return d * math.pi / 180.0


@dataclass
class TelemetryFrame:
    auto_flag: int
    kill_flag: int
    batt: float
    rc_port_pwm: int
    rc_stbd_pwm: int
    lat: float
    lon: float
    alt: float
    hdg_deg: float
    vel_mps: float
    ax: float
    ay: float
    az: float
    qx: float
    qy: float
    qz: float
    qw: float
    wx: float
    wy: float
    wz: float
    temperature: float
    checksum: int


class Ros2SerialPy(Node):
    """
    ROS2 <-> Teensy serial bridge.

    Teensy -> Jetson telemetry frame (one per line):
      <auto,kill,batt,port_pwm,stbd_pwm,lat,lon,alt,hdg,vel,ax,ay,az,qx,qy,qz,qw,wx,wy,wz,temperature,*hh>

    Jetson -> Teensy command frame (keepalive):
      <port,stbd,estop,auto_send,kill_send,pump,*hh>

      pump is a single character: 'A' or 'a'
    """

    def __init__(self):
        super().__init__("ros2serial_py")

        # ---------------- Params ----------------
        self.rosboat_name = self.declare_parameter("asv", "asv").get_parameter_value().string_value
        self.port = self.declare_parameter("port", "/dev/ttyACM0").get_parameter_value().string_value
        self.baud = self.declare_parameter("baud_rate", 9600).get_parameter_value().integer_value
        self.gps_frame_id = self.declare_parameter("gps_frame_id", "gps_link").get_parameter_value().string_value
        self.imu_frame_id = self.declare_parameter("imu_frame_id", "imu_link").get_parameter_value().string_value
        self.check_checksum = self.declare_parameter("check_checksum", True).get_parameter_value().bool_value

        self.keepalive = self.declare_parameter("keepalive", True).get_parameter_value().bool_value
        self.keepalive_ms = self.declare_parameter("keepalive_ms", 100).get_parameter_value().integer_value

        # Debug toggles
        self.debug_rx = self.declare_parameter("debug_rx", False).get_parameter_value().bool_value
        self.debug_rx_bytes = self.declare_parameter("debug_rx_bytes", False).get_parameter_value().bool_value
        self.debug_tx = self.declare_parameter("debug_tx", False).get_parameter_value().bool_value
        self.debug_throttle_sec = float(
            self.declare_parameter("debug_throttle_sec", 0.5).get_parameter_value().double_value
        )

        # Pump defaults/params (NEW)
        self.pump_on_char = self.declare_parameter("pump_on_char", "A").get_parameter_value().string_value or "A"
        self.pump_off_char = self.declare_parameter("pump_off_char", "a").get_parameter_value().string_value or "a"

        # ---------------- QoS ----------------
        sensor_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        # ---------------- Publishers ----------------
        base = f"/{self.rosboat_name}"
        self.status_pub = self.create_publisher(TeensyStatus, f"{base}/teensy_status", 1)
        self.gps_pub = self.create_publisher(NavSatFix, f"{base}/gps/fix", sensor_qos)
        self.imu_pub = self.create_publisher(Imu, f"{base}/imu/data", sensor_qos)
        self.batt_pub = self.create_publisher(Float64, f"{base}/battery/voltage", 1)
        self.temp_pub = self.create_publisher(Float64, f"{base}/teensy/temperature", 1)
        self.vel_pub = self.create_publisher(Twist, f"{base}/gps/velocity", sensor_qos)
        self.motor_pub = self.create_publisher(MotorCmds, f"{base}/motor_pwm", sensor_qos)

        # ---------------- Subscribers ----------------
        self.last_port_thrust = 1500
        self.last_stbd_thrust = 1500
        self.have_motor_cmd = False

        self.last_auto_flag = False
        self.last_kill_flag = False
        self.d_estop = False

        # Pump state (NEW)
        self.pump_enabled = False  # False -> 'a', True -> 'A'

        self.motor_sub = self.create_subscription(
            MotorCmds,
            f"{base}/motor_cmds",
            self.on_motor_cmd,
            1,
        )
        self.estop_sub = self.create_subscription(
            Bool,
            f"{base}/digital_estop",
            self.on_estop,
            1,
        )

        # Pump subscriber (NEW)
        # Publish Bool on: /<rosboat>/pump_cmd  (True -> 'A', False -> 'a')
        self.pump_sub = self.create_subscription(
            Bool,
            f"{base}/pump_cmd",
            self.on_pump_cmd,
            1,
        )

        # ---------------- Serial open ----------------
        self.get_logger().info(f"Opening {self.port} @ {self.baud}")
        self.ser = serial.Serial(
            self.port,
            self.baud,
            timeout=0.0,        # non-blocking read
            write_timeout=0.2,  # avoid hanging forever
        )

        # RX thread state
        self._rx_buf = bytearray()
        self._last_rx_time = self.get_clock().now()

        # Start RX thread
        self._running = True
        self._rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
        self._rx_thread.start()

        # Main timer (100 Hz)
        self.timer = self.create_timer(0.01, self.loop)
        self._last_tx_time = time.monotonic()

        self.get_logger().info("TX+RX started (ros2serial_py)")
        self.get_logger().info(f"Publishing under base: {base}")

    # ---------------- Sub callbacks ----------------
    def on_estop(self, msg: Bool):
        self.d_estop = bool(msg.data)

    def on_pump_cmd(self, msg: Bool):
        self.pump_enabled = bool(msg.data)

    def on_motor_cmd(self, msg: MotorCmds):
        self.last_port_thrust = int(msg.port_thrust)
        self.last_stbd_thrust = int(msg.stbd_thrust)
        self.have_motor_cmd = True
        self._send_motor_frame(self.last_port_thrust, self.last_stbd_thrust)

    # ---------------- Main loop ----------------
    def loop(self):
        # watchdog: checks whether any serial line is arriving (not whether parsing succeeded)
        if (self.get_clock().now() - self._last_rx_time).nanoseconds > int(1e9):
            self.get_logger().warn("No RX for > 1s", throttle_duration_sec=2.0)

        # TX periodic keepalive
        if not self.keepalive:
            return

        now = time.monotonic()
        if (now - self._last_tx_time) * 1000.0 >= float(self.keepalive_ms):
            self._last_tx_time = now
            if self.have_motor_cmd:
                self._send_motor_frame(self.last_port_thrust, self.last_stbd_thrust)
            else:
                self._send_motor_frame(1500, 1500)

    # ---------------- Serial RX ----------------
    def _rx_loop(self):
        """Background thread: read bytes, split lines by '\\n', parse frames."""
        while self._running:
            try:
                chunk = self.ser.read(256)
            except Exception as e:
                self.get_logger().warn(f"Serial read failed: {e}")
                time.sleep(0.05)
                continue

            if chunk:
                # DEBUG: raw bytes
                if self.debug_rx_bytes:
                    self.get_logger().info(
                        f"[RX BYTES] {chunk!r}",
                        throttle_duration_sec=self.debug_throttle_sec,
                    )

                self._rx_buf.extend(chunk)

                while True:
                    nl = self._rx_buf.find(b"\n")
                    if nl < 0:
                        break

                    raw_line_bytes = bytes(self._rx_buf[:nl])
                    del self._rx_buf[:nl + 1]

                    line = raw_line_bytes.decode(errors="ignore").strip()
                    if not line:
                        continue

                    if self.debug_rx:
                        self.get_logger().info(
                            f"[RX RAW] {line}",
                            throttle_duration_sec=self.debug_throttle_sec,
                        )

                    self._handle_rx_line(line)
            else:
                time.sleep(0.002)

    def _handle_rx_line(self, line: str):
        self._last_rx_time = self.get_clock().now()

        fr = self._parse_telemetry_frame(line)
        if fr is None:
            if self.debug_rx:
                self.get_logger().warn(f"[RX PARSE FAIL] {line}")
            return

        if self.check_checksum:
            calc = self._calc_checksum_rx(fr)
            if calc != fr.checksum:
                if self.debug_rx:
                    self.get_logger().warn(f"[RX BAD CHECKSUM] {line} rx={fr.checksum} calc={calc}")
                return

        self.last_auto_flag = bool(fr.auto_flag)
        self.last_kill_flag = bool(fr.kill_flag)

        self._publish_from_frame(fr)

    def _parse_telemetry_frame(self, s: str) -> Optional[TelemetryFrame]:
        lt = s.find("<")
        star = s.find("*")
        gt = s.find(">")
        if lt < 0 or star < 0 or gt < 0 or not (lt < star < gt):
            return None

        payload = s[lt + 1: star]
        cs_text = s[star + 1: gt].strip()
        if len(cs_text) < 2:
            return None

        try:
            rx_cs = int(cs_text, 16)
        except Exception:
            return None

        payload = payload.rstrip(",")
        parts = payload.split(",")

        # 21 fields (temperature at the end)
        if len(parts) != 21:
            return None

        try:
            auto_flag = int(parts[0])
            kill_flag = int(parts[1])
            batt = float(parts[2])

            rc_port_pwm = int(parts[3])
            rc_stbd_pwm = int(parts[4])

            lat = float(parts[5])
            lon = float(parts[6])
            alt = float(parts[7])

            hdg = float(parts[8])
            vel = float(parts[9])

            ax = float(parts[10])
            ay = float(parts[11])
            az = float(parts[12])

            qx = float(parts[13])
            qy = float(parts[14])
            qz = float(parts[15])
            qw = float(parts[16])

            wx = float(parts[17])
            wy = float(parts[18])
            wz = float(parts[19])

            temperature = float(parts[20])

        except Exception:
            self.get_logger().warn(f"[PARSE] exception converting fields | payload='{payload}'")
            return None

        return TelemetryFrame(
            auto_flag=auto_flag,
            kill_flag=kill_flag,
            batt=batt,
            rc_port_pwm=rc_port_pwm,
            rc_stbd_pwm=rc_stbd_pwm,
            lat=lat,
            lon=lon,
            alt=alt,
            hdg_deg=hdg,
            vel_mps=vel,
            ax=ax,
            ay=ay,
            az=az,
            qx=qx,
            qy=qy,
            qz=qz,
            qw=qw,
            wx=wx,
            wy=wy,
            wz=wz,
            temperature=temperature,
            checksum=rx_cs,
        )

    def _calc_checksum_rx(self, fr: TelemetryFrame) -> int:
        s = (
            int(fr.auto_flag) + int(fr.kill_flag) + fr.batt +
            fr.rc_port_pwm + fr.rc_stbd_pwm +
            fr.lat + fr.lon + fr.alt + fr.hdg_deg + fr.vel_mps +
            fr.ax + fr.ay + fr.az +
            fr.qx + fr.qy + fr.qz + fr.qw +
            fr.wx + fr.wy + fr.wz +
            fr.temperature
        )
        return abs(int(s)) % 256

    # ---------------- Publishers ----------------
    def _publish_from_frame(self, fr: TelemetryFrame):
        now = self.get_clock().now().to_msg()

        st = TeensyStatus()
        st.auto_flag = bool(fr.auto_flag)
        st.kill_flag = bool(fr.kill_flag)
        self.status_pub.publish(st)

        b = Float64()
        b.data = float(fr.batt)
        self.batt_pub.publish(b)

        t = Float64()
        t.data = float(fr.temperature)
        self.temp_pub.publish(t)

        fix = NavSatFix()
        fix.header.stamp = now
        fix.header.frame_id = self.gps_frame_id
        fix.latitude = float(fr.lat)
        fix.longitude = float(fr.lon)
        fix.altitude = float(fr.alt)
        self.gps_pub.publish(fix)

        imu = Imu()
        imu.header.stamp = now
        imu.header.frame_id = self.imu_frame_id
        imu.orientation.x = float(fr.qx)
        imu.orientation.y = float(fr.qy)
        imu.orientation.z = float(fr.qz)
        imu.orientation.w = float(fr.qw)
        imu.angular_velocity.x = float(fr.wx)
        imu.angular_velocity.y = float(fr.wy)
        imu.angular_velocity.z = float(fr.wz)
        imu.linear_acceleration.x = float(fr.ax)
        imu.linear_acceleration.y = float(fr.ay)
        imu.linear_acceleration.z = float(fr.az)
        self.imu_pub.publish(imu)

        tw = Twist()
        tw.linear.x = float(fr.vel_mps)
        self.vel_pub.publish(tw)

        m = MotorCmds()
        m.port_thrust = int(fr.rc_port_pwm)
        m.stbd_thrust = int(fr.rc_stbd_pwm)
        self.motor_pub.publish(m)

    # ---------------- Serial TX ----------------
    @staticmethod
    def _checksum_cmd(port: int, stbd: int, estop: int, auto_send: int, kill_send: int, pump: str) -> int:
        pump_val = ord(pump) if pump else 0  # include pump in checksum deterministically
        s = int(port) + int(stbd) + int(estop) + int(auto_send) + int(kill_send) + int(pump_val)
        return abs(int(s)) % 256

    def _send_motor_frame(self, port_thrust: int, stbd_thrust: int):
        if not self.ser or not self.ser.is_open:
            return

        estop = 1 if self.d_estop else 0
        auto_send = 0 if estop else (1 if self.last_auto_flag else 0)
        kill_send = 1 if estop else (1 if self.last_kill_flag else 0)

        pump_char = self.pump_on_char[0] if self.pump_enabled else self.pump_off_char[0]

        cs = self._checksum_cmd(port_thrust, stbd_thrust, estop, auto_send, kill_send, pump_char)
        frame = f"<{port_thrust},{stbd_thrust},{estop},{auto_send},{kill_send},{pump_char},*{cs:02x}>"

        if self.debug_tx:
            self.get_logger().info(
                f"[TX RAW] {frame}",
                throttle_duration_sec=self.debug_throttle_sec
            )

        try:
            self.ser.write((frame + "\n").encode("ascii"))
        except Exception as e:
            self.get_logger().warn(f"Serial write failed: {e}")

    def destroy_node(self):
        self._running = False
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = Ros2SerialPy()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
