#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, String
from fau_msgs.msg import ControlEffort


class ControlAllocatorSKPy(Node):
    """
    Allocator using your forward mapping:

        tau = B f
        B = [[1,   0,   1,   0],
             [0,   1,   0,   1],
             [Lyp,-Lxp,-Lys, Lxs]]

        f = [Fxp, Fyp, Fxs, Fys]^T  (forces in Newtons)

    Publishes:
      /{wamv}/thrusters/left/thrust        -> Fxp
      /{wamv}/thrusters/right/thrust       -> Fxs
      /{wamv}/thrusters/left_bow/thrust    -> Fyp (or sign flip if your hardware expects opposite)
      /{wamv}/thrusters/right_bow/thrust   -> Fys
    """

    def __init__(self):
        super().__init__("control_allocator_sk_py")

        # --- params ---
        self.declare_parameter("asv", "wamv")
        self.wamv = str(self.get_parameter("asv").value)

        # Geometry (meters) - set these to your real lever arms
        # NOTE: In your C++ you used Ly_Stern=1.03, Lx_Bow=1.37.
        # Here we split them into the matrix entries you requested:
        # Stern lateral offsets (distance from centerline)
        self.declare_parameter("Lyp",  1.027135)   # port stern y
        self.declare_parameter("Lys", 1.027135)   # starboard stern y

        # Bow longitudinal offsets
        self.declare_parameter("Lxp",  1.373776)   # port bow x
        self.declare_parameter("Lxs",  1.373776)   # starboard bow x

        # If you want stern longitudinal too (recommended)
        self.declare_parameter("Lx_stern", -2.373776)

        # Motor force saturation (Newtons)
        self.declare_parameter("Fmax", 1000.0)

        # optional gate
        self.declare_parameter("use_controller_gate", True)
        self.controller = "SK"

        self.Lyp = float(self.get_parameter("Lyp").value)
        self.Lxp = float(self.get_parameter("Lxp").value)
        self.Lys = float(self.get_parameter("Lys").value)
        self.Lxs = float(self.get_parameter("Lxs").value)

        self.Fmax = float(self.get_parameter("Fmax").value)
        self.use_gate = bool(self.get_parameter("use_controller_gate").value)

        # --- pubs ---
        self.pub_left = self.create_publisher(Float64, f"/{self.wamv}/thrusters/left/thrust", 10)          # Fxp
        self.pub_right = self.create_publisher(Float64, f"/{self.wamv}/thrusters/right/thrust", 10)        # Fxs
        self.pub_left_bow = self.create_publisher(Float64, f"/{self.wamv}/thrusters/left_bow/thrust", 10)  # Fyp
        self.pub_right_bow = self.create_publisher(Float64, f"/{self.wamv}/thrusters/right_bow/thrust", 10)# Fys

        # --- subs ---
        self.sub_tau = self.create_subscription(
            ControlEffort, f"/{self.wamv}/control_effort", self._tau_cb, 10
        )
        self.sub_ctrl = self.create_subscription(
            String, f"/{self.wamv}/controller", self._ctrl_cb, 10
        )

        self.get_logger().info(
            "ControlAllocatorSKPy (B-matrix version)\n"
            f"  B row3 = [{self.Lyp:+.3f}, {-self.Lxp:+.3f}, {-self.Lys:+.3f}, {self.Lxs:+.3f}]\n"
            f"  Fmax per motor = {self.Fmax:.1f} N\n"
        )

    def _ctrl_cb(self, msg: String):
        self.controller = str(msg.data).strip()

    # ---------- math ----------
    def _B(self) -> np.ndarray:
        # B = [[1,0,1,0],
        #      [0,1,0,1],
        #      [Lyp,-Lxp,-Lys,Lxs]]
        return np.array([
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [self.Lyp, -self.Lxp, -self.Lys, self.Lxs],
        ], dtype=np.float64)

    def _pinv_min_norm(self, B: np.ndarray) -> np.ndarray:
        """
        Minimum-norm pseudoinverse for full row-rank B (3x4):
          B^+ = B^T (B B^T)^(-1)
        """
        BBt = B @ B.T
        inv = np.linalg.inv(BBt)
        return B.T @ inv

    def _allocate(self, tau: np.ndarray) -> np.ndarray:
        B = self._B()
        Bpinv = self._pinv_min_norm(B)
        f = Bpinv @ tau  # [Fxp,Fyp,Fxs,Fys]
        return f

    def _scale_tau_to_motor_limits(self, tau: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Scale tau so that allocated thrusters satisfy |f_i| <= Fmax.
        Keeps allocation direction (same idea we discussed).
        """
        f = self._allocate(tau)
        max_abs = float(np.max(np.abs(f)))
        if max_abs <= self.Fmax or max_abs < 1e-9:
            return tau, 1.0
        s = self.Fmax / max_abs
        return tau * s, s

    # ---------- callback ----------
    def _tau_cb(self, msg: ControlEffort):
        if len(msg.tau) < 3:
            self.get_logger().warn("tau must have at least 3 elements [Tx, Ty, Mz]")
            return

        if self.use_gate and self.controller and self.controller != "SK":
            return

        Tx = float(msg.tau[0])
        Ty = float(msg.tau[1])
        Mz = float(msg.tau[2])

        tau = np.array([Tx, Ty, Mz], dtype=np.float64)

        # Scale tau so that after allocation every motor <= Fmax
        tau_scaled, s = self._scale_tau_to_motor_limits(tau)

        # Allocate thrusters
        f = self._allocate(tau_scaled)
        Fxp, Fyp, Fxs, Fys = [float(x) for x in f]

        # ---- publish ----
        self.pub_left.publish(Float64(data=Fxp))
        self.pub_right.publish(Float64(data=Fxs))
        self.pub_left_bow.publish(Float64(data=Fyp))
        self.pub_right_bow.publish(Float64(data=Fys))

        # optional debug
        self.get_logger().debug(
            f"tau=[{Tx:+.1f},{Ty:+.1f},{Mz:+.1f}]  scale={s:.3f}  "
            f"f=[Fxp={Fxp:+.1f},Fyp={Fyp:+.1f},Fxs={Fxs:+.1f},Fys={Fys:+.1f}]"
        )


def main(args=None):
    rclpy.init(args=args)
    node = ControlAllocatorSKPy()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()