from __future__ import annotations

import time
import threading
from dataclasses import dataclass

import rclpy
from rclpy.node import Node

from std_msgs.msg import UInt8, UInt32MultiArray, Float64MultiArray, String
from fau_msgs.msg import TeensyStatus


@dataclass
class LastObjectLL:
    lat: float | None = None
    lon: float | None = None
    t: float = 0.0


class EventOnlyMissionPlanner(Node):
    """
    Mission planner (event-only):
      - publishes /{asv}/current_task for heartbeat
      - republishes task events to /{asv}/robocommand/* for the bridge to send
      - does lightweight debounce + sanity checks
      - does NOT do path planning or goal execution
    """

    def __init__(self):
        super().__init__("mission_planner_events_only")

        # Params
        self.declare_parameter("asv_name", "asv")
        self.asv = self.get_parameter("asv_name").value

        # Optional: debouncing (seconds)
        self.declare_parameter("debounce_gate_s", 1.0)
        self.declare_parameter("debounce_obj_s", 0.3)
        self.declare_parameter("debounce_delivery_s", 1.0)
        self.declare_parameter("debounce_docking_s", 2.0)
        self.declare_parameter("debounce_sound_s", 2.0)

        self.debounce_gate_s = float(self.get_parameter("debounce_gate_s").value)
        self.debounce_obj_s = float(self.get_parameter("debounce_obj_s").value)
        self.debounce_delivery_s = float(self.get_parameter("debounce_delivery_s").value)
        self.debounce_docking_s = float(self.get_parameter("debounce_docking_s").value)
        self.debounce_sound_s = float(self.get_parameter("debounce_sound_s").value)

        # Outputs (to heartbeat bridge)
        self.current_task_pub = self.create_publisher(UInt8, f"/{self.asv}/current_task", 10)

        self.rc_gate_pub = self.create_publisher(UInt8, f"/{self.asv}/robocommand/gate_pass", 10)
        self.rc_obj_pub = self.create_publisher(UInt32MultiArray, f"/{self.asv}/robocommand/object_detected", 10)
        self.rc_obj_ll_pub = self.create_publisher(Float64MultiArray, f"/{self.asv}/robocommand/object_detected_ll", 10)
        self.rc_delivery_pub = self.create_publisher(UInt32MultiArray, f"/{self.asv}/robocommand/object_delivery", 10)
        self.rc_docking_pub = self.create_publisher(String, f"/{self.asv}/robocommand/docking", 10)
        self.rc_sound_pub = self.create_publisher(UInt32MultiArray, f"/{self.asv}/robocommand/sound_signal", 10)
        self.mode_pub = self.create_publisher(UInt8,f"/{self.asv}/mode",10)

        # Inputs (from autonomy/path planner)
        self.create_subscription(UInt8, f"/{self.asv}/mission/set_task", self._on_set_task, 10)
        self.create_subscription(UInt8, f"/{self.asv}/mission/gate_pass", self._on_gate_pass, 10)
        self.create_subscription(UInt32MultiArray, f"/{self.asv}/mission/object_detected", self._on_object_detected, 10)
        self.create_subscription(Float64MultiArray, f"/{self.asv}/mission/object_detected_ll", self._on_object_detected_ll, 10)
        self.create_subscription(UInt32MultiArray, f"/{self.asv}/mission/object_delivery", self._on_object_delivery, 10)
        self.create_subscription(String, f"/{self.asv}/mission/docking", self._on_docking, 10)
        self.create_subscription(UInt32MultiArray, f"/{self.asv}/mission/sound_signal", self._on_sound_signal, 10)
        self.create_subscription(TeensyStatus,f"/{self.asv}/teensy_status", self.pcu_status_callback,10)

        # State
        self._lock = threading.Lock()
        self.current_task: int = 2 # TASK_NONE default in your enum
        self._last_gate_at = 0.0
        self._last_obj_at = 0.0
        self._last_delivery_at = 0.0
        self._last_docking_at = 0.0
        self._last_sound_at = 0.0

        self._last_obj_ll = LastObjectLL()

        self._last_mode = None


        # publish current task at 1 Hz so heartbeat bridge always has it
        self.task_timer = self.create_timer(1.0, self._publish_task)

        self.get_logger().info(f"Event-only mission planner up for {self.asv}")

    # -----------------
    # Task bookkeeping
    # -----------------
    def _on_set_task(self, msg: UInt8):
        with self._lock:
            self.current_task = int(msg.data)
        self.get_logger().info(f"[MISSION] current_task set -> {self.current_task}")
        self._publish_task()

    def _publish_task(self):
        with self._lock:
            t = int(self.current_task)
        self.current_task_pub.publish(UInt8(data=t))

    # -----------------
    # Events -> RoboCommand topics
    # -----------------
    def _on_gate_pass(self, msg: UInt8):
        now = time.time()
        if (now - self._last_gate_at) < self.debounce_gate_s:
            return
        self._last_gate_at = now

        gate_type = int(msg.data)
        self.rc_gate_pub.publish(UInt8(data=gate_type))
        self.get_logger().info(f"[RC] GatePass type={gate_type}")

    def _on_object_detected_ll(self, msg: Float64MultiArray):
        if len(msg.data) < 2:
            self.get_logger().warn("object_detected_ll expects [lat, lon]")
            return
        with self._lock:
            self._last_obj_ll.lat = float(msg.data[0])
            self._last_obj_ll.lon = float(msg.data[1])
            self._last_obj_ll.t = time.time()

    def _on_object_detected(self, msg: UInt32MultiArray):
        # expects [object_type, color, object_id, task_context]
        if len(msg.data) < 4:
            self.get_logger().warn("object_detected expects [object_type, color, object_id, task_context]")
            return

        now = time.time()
        if (now - self._last_obj_at) < self.debounce_obj_s:
            return
        self._last_obj_at = now

        meta = [int(x) for x in msg.data[:4]]

        # Try to attach LL if recently provided
        with self._lock:
            lat = self._last_obj_ll.lat
            lon = self._last_obj_ll.lon
            age = now - self._last_obj_ll.t

        if lat is None or lon is None or age > 5.0:
            # still publish meta, but warn about missing LL
            self.get_logger().warn("[RC] ObjectDetected missing recent lat/lon (publish object_detected_ll within 5s).")
        else:
            ll = Float64MultiArray()
            ll.data = [float(lat), float(lon)]
            self.rc_obj_ll_pub.publish(ll)

        mm = UInt32MultiArray()
        mm.data = meta
        self.rc_obj_pub.publish(mm)

        if lat is not None and lon is not None and age <= 5.0:
            self.get_logger().info(f"[RC] ObjectDetected meta={meta} ll={[lat, lon]}")
        else:
            self.get_logger().info(f"[RC] ObjectDetected meta={meta}")

    def _on_object_delivery(self, msg: UInt32MultiArray):
        # expects [vessel_color, delivery_type]
        if len(msg.data) < 2:
            self.get_logger().warn("object_delivery expects [vessel_color, delivery_type]")
            return

        now = time.time()
        if (now - self._last_delivery_at) < self.debounce_delivery_s:
            return
        self._last_delivery_at = now

        out = UInt32MultiArray()
        out.data = [int(msg.data[0]), int(msg.data[1])]
        self.rc_delivery_pub.publish(out)
        self.get_logger().info(f"[RC] ObjectDelivery {out.data}")

    def _on_docking(self, msg: String):
        now = time.time()
        if (now - self._last_docking_at) < self.debounce_docking_s:
            return
        self._last_docking_at = now

        s = msg.data.strip()
        # basic sanity: "N,1" etc.
        if "," not in s:
            self.get_logger().warn("docking expects 'N,1' or 'S,3'")
            return

        self.rc_docking_pub.publish(String(data=s))
        self.get_logger().info(f"[RC] Docking {s}")

    def _on_sound_signal(self, msg: UInt32MultiArray):
        # expects [signal_type, frequency_hz, assigned_task]
        if len(msg.data) < 3:
            self.get_logger().warn("sound_signal expects [signal_type, frequency_hz, assigned_task]")
            return

        #self.current_task = 7

        out = UInt32MultiArray()
        out.data = [int(msg.data[0]), int(msg.data[1]), int(msg.data[2]+1)]
        self.rc_sound_pub.publish(out)
        #self.get_logger().info(f"[RC] SoundSignal {out.data}")

    def pcu_status_callback(self, msg):
        is_auto_flag = bool(msg.auto_flag)
        is_killed_flag = bool(msg.kill_flag)

        if is_killed_flag:
            mode = 1          # KILLED
            self.run_pid = False
        elif is_auto_flag:
            mode = 3          # AUTO
            self.run_pid = True
        else:
            mode = 2          # MANUAL
            self.run_pid = False

        # OPTIONAL: publish only when it changes
        if getattr(self, "_last_mode", None) != mode:
            self.mode_pub.publish(UInt8(data=mode))
            self._last_mode = mode




def main(args=None):
    rclpy.init(args=args)
    node = EventOnlyMissionPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

