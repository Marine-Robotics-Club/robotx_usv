#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64, String
from fau_msgs.msg import ControlEffort, MotorCmds
#

class ControlAllocatorDiffDrivePWM(Node):
    """
    Differential-drive allocator using thrust(N) -> PWM calibration.

    Input:
        /{wamv}/control_effort
        ControlEffort.tau = [T_x, T_y, M_z]

    Differential-drive allocation:

        Np_cmd = T_x / 2 + M_z / B
        Ns_cmd = T_x / 2 - M_z / B

    where:
        Np_cmd = desired port/left motor thrust in Newtons
        Ns_cmd = desired starboard/right motor thrust in Newtons

    Notes:
        - T_y is ignored because differential drive cannot command sway.
        - No normalized [-1, 1] clipping is used.
        - The thrust-to-PWM calibration handles deadband and physical saturation.
        - Output is published to /{wamv}/jetson_cmds as MotorCmds.

    Important calibration convention:
        The calibration table below is TOTAL VEHICLE THRUST.
        Since this allocator commands each motor independently, the table is
        converted internally to PER-MOTOR thrust by dividing Newtons by 2.
    """

    def __init__(self):
        super().__init__("control_allocator_diff_drive_pwm")

        # ---------------- Parameters ----------------
        self.declare_parameter("asv", "asv")

        # Controller gate
        self.declare_parameter("use_controller_gate", True)
        self.declare_parameter("controller_name", "HS")

        # Allocation:
        # Np = Tx/2 + Mz/B
        # Ns = Tx/2 - Mz/B
        #
        # Keep B consistent with your controller convention.
        self.declare_parameter("B", 0.62)

        # Motor PWM values
        self.declare_parameter("port_azimuth_pwm", 1540)
        self.declare_parameter("stbd_azimuth_pwm", 1560)
        self.declare_parameter("neutral_pwm", 1500)

        # Topics
        self.declare_parameter("control_effort_topic", "control_effort")
        self.declare_parameter("controller_topic", "controller")
        self.declare_parameter("jetson_cmds_topic", "motor_cmds")

        # Optional debug force topics
        self.declare_parameter("publish_force_debug", True)
        self.declare_parameter("left_force_topic", "thrusters/left/thrust")
        self.declare_parameter("right_force_topic", "thrusters/right/thrust")

        self.wamv = str(self.get_parameter("asv").value)

        self.use_gate = bool(self.get_parameter("use_controller_gate").value)
        self.controller_name = str(self.get_parameter("controller_name").value)
        self.controller = self.controller_name

        self.B = float(self.get_parameter("B").value)

        self.port_azimuth_pwm = int(self.get_parameter("port_azimuth_pwm").value)
        self.stbd_azimuth_pwm = int(self.get_parameter("stbd_azimuth_pwm").value)
        self.neutral_pwm = int(self.get_parameter("neutral_pwm").value)

        self.control_effort_topic = str(self.get_parameter("control_effort_topic").value)
        self.controller_topic = str(self.get_parameter("controller_topic").value)
        self.jetson_cmds_topic = str(self.get_parameter("jetson_cmds_topic").value)

        self.publish_force_debug = bool(self.get_parameter("publish_force_debug").value)
        self.left_force_topic = str(self.get_parameter("left_force_topic").value)
        self.right_force_topic = str(self.get_parameter("right_force_topic").value)

        if abs(self.B) < 1e-6:
            self.get_logger().warn("B is too small. Resetting B to 0.62.")
            self.B = 0.62

        # ---------------- Calibration table ----------------
        # Column 0: motor command percentage
        # Column 1: measured TOTAL VEHICLE thrust in Newtons
        #
        # This table matches the optimizer convention:
        #   total_thrust = table[pct]
        #   motor_thrust = 0.5 * total_thrust
        #
        # The allocator commands individual motors, so we convert to
        # per-motor thrust below.
        calib_total_vehicle = np.array([
            [-100, -158.7936],
            [-90,  -155.6800],
            [-80,  -155.6800],
            [-70,  -129.8816],
            [-60,  -104.39456],
            [-50,   -73.52544],
            [-40,   -58.80256],
            [-30,   -37.27424],
            [-20,   -19.61568],
            [-10,    -6.84992],
            [0,       0.0],
            [10,     14.72288],
            [20,     35.76192],
            [30,     65.20768],
            [40,     98.03392],
            [50,    132.81728],
            [60,    166.66656],
            [70,    202.8288],
            [80,    240.192],
            [90,    240.192],
            [100,   240.192],
        ], dtype=float)

        # Convert TOTAL VEHICLE thrust to PER-MOTOR thrust.
        calib = calib_total_vehicle.copy()
        calib[:, 1] *= 0.5

        self._pct = calib[:, 0]
        self._thrN = calib[:, 1]

        # Monotone segments for inverse interpolation: thrust(N) -> percent.
        #
        # New calibration has nonzero thrust at +10% and -10%,
        # so use all positive and all negative percentage points.
        fwd_mask = self._pct > 0.0
        rev_mask = self._pct < 0.0

        self._thrN_fwd = self._thrN[fwd_mask]
        self._pct_fwd = self._pct[fwd_mask]

        # Reverse side must be sorted by thrust value for np.interp.
        order = np.argsort(self._thrN[rev_mask])
        self._thrN_rev = self._thrN[rev_mask][order]
        self._pct_rev = self._pct[rev_mask][order]

        self._N_fwd_min = float(self._thrN_fwd[0])
        self._N_fwd_max = float(self._thrN_fwd[-1])

        self._N_rev_min = float(self._thrN_rev[0])
        self._N_rev_max = float(self._thrN_rev[-1])

        # ---------------- Publishers ----------------
        self.pub_pwm = self.create_publisher(
            MotorCmds,
            f"/{self.wamv}/{self.jetson_cmds_topic}".replace("//", "/"),
            10,
        )

        if self.publish_force_debug:
            self.pub_left_force = self.create_publisher(
                Float64,
                f"/{self.wamv}/{self.left_force_topic}".replace("//", "/"),
                10,
            )

            self.pub_right_force = self.create_publisher(
                Float64,
                f"/{self.wamv}/{self.right_force_topic}".replace("//", "/"),
                10,
            )
        else:
            self.pub_left_force = None
            self.pub_right_force = None

        # ---------------- Subscribers ----------------
        self.sub_tau = self.create_subscription(
            ControlEffort,
            f"/{self.wamv}/{self.control_effort_topic}".replace("//", "/"),
            self._tau_cb,
            10,
        )

        self.sub_controller = self.create_subscription(
            String,
            f"/{self.wamv}/{self.controller_topic}".replace("//", "/"),
            self._controller_cb,
            10,
        )

        self.get_logger().info(
            "Differential-drive PWM allocator started.\n"
            f"  Namespace: /{self.wamv}\n"
            f"  Controller gate: {self.use_gate}\n"
            f"  Controller name: {self.controller_name}\n"
            f"  B: {self.B:.3f}\n"
            f"  Calibration: total vehicle thrust table converted to per-motor thrust\n"
            f"  Forward per-motor thrust range: {self._N_fwd_min:.2f} to {self._N_fwd_max:.2f} N\n"
            f"  Reverse per-motor thrust range: {self._N_rev_min:.2f} to {self._N_rev_max:.2f} N\n"
            f"  Jetson topic: /{self.wamv}/{self.jetson_cmds_topic}"
        )

    # ---------------- Callbacks ----------------
    def _controller_cb(self, msg: String):
        self.controller = str(msg.data).strip()

    def _tau_cb(self, msg: ControlEffort):
        if len(msg.tau) < 3:
            self.get_logger().warn("Received tau vector with less than 3 elements.")
            return

        if self.use_gate and self.controller != self.controller_name:
            return

        T_x = float(msg.tau[0])
        T_y = float(msg.tau[1])
        M_z = float(msg.tau[2])

        # Differential-drive ignores sway force.
        if abs(T_y) > 1e-6:
            self.get_logger().debug(
                f"Ignoring T_y={T_y:+.3f} because this is differential drive."
            )

        # Differential-drive allocation in Newtons.
        # No [-1, 1] clipping here.
        Np_cmd = 0.5 * T_x - M_z / self.B
        Ns_cmd = 0.5 * T_x + M_z / self.B

        # Convert desired per-motor thrust in Newtons to PWM.
        pwm_p = self._thrustN_to_pwm(Np_cmd)
        pwm_s = self._thrustN_to_pwm(Ns_cmd)

        self._publish_pwm(pwm_p, pwm_s)

        # Optional debug: publish requested forces in Newtons.
        if self.publish_force_debug:
            self.pub_left_force.publish(Float64(data=float(Np_cmd)))
            self.pub_right_force.publish(Float64(data=float(Ns_cmd)))

        self.get_logger().debug(
            f"tau=[Tx={T_x:+.2f}, Ty={T_y:+.2f}, Mz={M_z:+.2f}] -> "
            f"Np={Np_cmd:+.2f} N, Ns={Ns_cmd:+.2f} N -> "
            f"PWMp={pwm_p}, PWMs={pwm_s}"
        )

    # ---------------- Helpers: thrust(N) -> PWM ----------------
    @staticmethod
    def _percent_to_pwm(percent: float) -> int:
        return int(round(1500.0 - 5.0 * percent))

    def _thrustN_to_pwm(self, N: float) -> int:
        """
        Convert desired per-motor thrust in Newtons to PWM using calibration.

        The new calibration has small nonzero thrust already at:
            +10% forward
            -10% reverse

        Therefore:
            forward deadband is 0 to self._N_fwd_min
            reverse deadband is self._N_rev_max to 0

        The function saturates to the available calibrated thrust range.
        """

        if abs(N) < 1e-6:
            return self.neutral_pwm

        if N > 0.0:
            # Inside forward deadband.
            if N < self._N_fwd_min:
                return self.neutral_pwm

            # Saturate to maximum calibrated forward thrust.
            Nq = min(N, self._N_fwd_max)

            pct = float(np.interp(Nq, self._thrN_fwd, self._pct_fwd))
            return self._percent_to_pwm(pct)

        # N < 0.0

        # Inside reverse deadband.
        # Example: a very small negative thrust should still be neutral.
        if N > self._N_rev_max:
            return self.neutral_pwm

        # Saturate to maximum calibrated reverse thrust.
        Nq = max(N, self._N_rev_min)

        pct = float(np.interp(Nq, self._thrN_rev, self._pct_rev))
        return self._percent_to_pwm(pct)

    def _publish_pwm(self, pwm_port: int, pwm_stbd: int):
        msg = MotorCmds()

        msg.port_azimuth = self.port_azimuth_pwm
        msg.port_bow_thrust = self.neutral_pwm
        msg.port_thrust = int(pwm_port)

        msg.stbd_azimuth = self.stbd_azimuth_pwm
        msg.stbd_bow_thrust = self.neutral_pwm
        msg.stbd_thrust = int(pwm_stbd)

        self.pub_pwm.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ControlAllocatorDiffDrivePWM()

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()