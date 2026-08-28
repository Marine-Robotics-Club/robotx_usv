#!/usr/bin/env python3
from __future__ import annotations
import math
from dataclasses import dataclass
import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt8, Bool, UInt32MultiArray, Float32MultiArray, Float64, String, Header, Int32, Float64MultiArray, Float32
from geometry_msgs.msg import Point, Twist, TwistStamped, Pose2D, PoseStamped
from nav_msgs.msg import Odometry, Path
from fau_msgs.msg import PropulsionSystem, ObjectPositionArray
from yolov26_msgs.msg import YoloDetection
import numpy as np
from tf_transformations import euler_from_quaternion
from typing import Dict, List, Tuple
import time
from visualization_msgs.msg import Marker, MarkerArray
import time


# RobotState (if you publish it from boat; bridge maps mode->pb anyway)
STATE_UNKNOWN = 0
STATE_KILLED = 1
STATE_MANUAL = 2
STATE_AUTO = 3

# TaskType
TASK_UNKNOWN = 0
TASK_NONE = 1
TASK_ENTRY_EXIT = 2
TASK_NAV_CHANNEL = 3
TASK_SPEED_CHALLENGE = 4
TASK_OBJECT_DELIVERY = 5
TASK_DOCKING = 6
TASK_SOUND_SIGNAL = 7

QUALIFYING = 9

# GateType
GATE_UNKNOWN = 0
GATE_ENTRY = 1
GATE_EXIT = 2
GATE_SPEED_START = 3
GATE_SPEED_END = 4

# ObjectType
OBJECT_UNKNOWN = 0
OBJECT_BOAT = 1
OBJECT_LIGHT_BEACON = 2
OBJECT_BUOY = 3

# Color
COLOR_UNKNOWN = 0
COLOR_YELLOW = 1
COLOR_BLACK = 2
COLOR_RED = 3
COLOR_GREEN = 4

# SignalType
SIGNAL_UNKNOWN = 0
SIGNAL_ONE_BLAST = 1
SIGNAL_TWO_BLAST = 2

# DeliveryType
DELIVERY_UNKNOWN = 0
DELIVERY_WATER = 1
DELIVERY_BALL = 2

ERROR_WP = 1.0


def _unit(vx, vy, eps=1e-9):
    n = math.sqrt(vx*vx + vy*vy)
    if n < eps:
        return 0.0, 0.0
    return vx/n, vy/n



class ASVPathPlanner(Node):
    """
    BOAT NODE (minimal skeleton):
    - consumes nav + perception outputs
    - publishes goal to controller
    - republishes mission cues for mission planner if needed
    """

    def __init__(self):
        super().__init__("asv_path_planner")

        self.declare_parameter("asv_name", "asv")
        self.asv_name = str(self.get_parameter("asv_name").value)

    
        self.position_sub = self.create_subscription(
            Pose2D,
            f'/{self.asv_name}/vehicle_pose',
            self.position_callback,
            10
        )

        # Current task (owned by mission planner typically)
        self.current_task = TASK_NONE
        self.create_subscription(UInt8, f"/{self.asv_name}/current_task", self._on_task, 10)

        # Perception inputs (examples)
        # gate midpoint lat/lon: [gate_type, lat, lon]
        self.create_subscription(Float64MultiArray,
                                 f"/{self.asv_name}/perception/gate_midpoint_ll",
                                 self._on_gate_midpoint_ll, 10)

        # object meta: [object_type, color, object_id, task_context]
        self.create_subscription(UInt32MultiArray,
                                 f"/{self.asv_name}/perception/object_detected",
                                 self._on_object_detected_meta, 10)

        # object ll: [lat, lon]
        self.create_subscription(Float64MultiArray,
                                 f"/{self.asv_name}/perception/object_detected_ll",
                                 self._on_object_detected_ll, 10)
        
        self.create_subscription(
            ObjectPositionArray,
            f"/{self.asv_name}/vision/output/fusion_tracked",
            self.fusion_tracked_callback,
            10
        )

        self.create_subscription(
            ObjectPositionArray,
            f"/vision/output/fusion",
            self.fusion_callback,
            10
        )

        self.yolo_center = self.create_subscription(
            YoloDetection,
            '/yolov26/detections/center',
            self.yolo_center_callback,
            10
        )

        self.sub_buoys = self.create_subscription(
            ObjectPositionArray,
            "/vision/tracked/obstacles",
            self.cb_buoys,
            10
        )

        self.object_delivery_topic = f"/{self.asv_name}/robocommand/object_delivery"

        # Publisher (same message type as your subscriber)
        self._object_delivery_pub = self.create_publisher(
            UInt32MultiArray,
            self.object_delivery_topic,
            10
        )

        self.create_subscription(UInt32MultiArray, f"/{self.asv_name}/mission/sound_signal", self._on_sound_signal, 10)

        # Mission cues outputs (optional, for mission planner)
        self.gate_midpoint_ll_pub = self.create_publisher(Float64MultiArray, f"/{self.asv_name}/mission/gate_midpoint_ll", 10)
        self.goal_reached_pub = self.create_publisher(Bool, f"/{self.asv_name}/mission/goal_reached", 10)
        self.obj_meta_pub = self.create_publisher(UInt32MultiArray, f"/{self.asv_name}/mission/object_detected", 10)
        self.obj_ll_pub = self.create_publisher(Float64MultiArray, f"/{self.asv_name}/mission/object_detected_ll", 10)
        self.heading_deg_pub = self.create_publisher(Float32,f"/{self.asv_name}/heading_deg",10)
        self.desired_state_pub = self.create_publisher(Float32MultiArray,f"/{self.asv_name}/traj_desired_state",10)
        self.gate_pass_pub = self.create_publisher(
            Float64MultiArray,
            f"/{self.asv_name}/mission/gate_pass",
            10
        )
        self.pub_object_detected = self.create_publisher(
            Float64MultiArray,
            f"/{self.asv_name}/robocommand/object_detected",
            10
        )
        self.pub_controller = self.create_publisher(
            String,
            f"/{self.asv_name}/controller",
            10
        )

        self.pee_pub = self.create_publisher(Bool,f"/{self.asv_name}/pump_cmd",1)
        self.task2_path_sub = self.create_subscription(Path, '/path_vision_2', self.task2_path_callback, 1)
        self.task1_path_sub = self.create_subscription(Path, '/path_vision_1', self.task1_path_callback, 1)
        self.create_subscription(
            UInt32MultiArray,
            f"/{self.asv_name}/robocommand/sound_signal",
            self._on_sound_signal,
            10
        )

        # goal reached detection
        self._active_goal_xy = None  # (x,y)
        self._goal_tol = 2.0

        # buffers
        self._last_gate = None  # (gate_type, lat, lon)
        self._last_obj_meta = None
        self._last_obj_ll = None

                # -----------------------------
        # Task 1 (Entry/Exit) state
        # -----------------------------
        self.drive_config = 0

        self.point = 0
        self.gates_passed = 0
        self.found_gates = False

        self.gate_passed = False
        self.last_gate_mid = None
        self.ignore_radius = 8.0     # meters (increase if it still reselects same gate)
        self.gate_markers_pub = self.create_publisher(MarkerArray, "/gate_markers", 10)
        self._marker_seq = 0  # for unique IDs

        # entry/exit gate storage
        self.x_entry_gate = 0.0
        self.y_entry_gate = 0.0
        self.x_exit_gate  = 0.0
        self.y_exit_gate  = 0.0

        self.x_entry_gate_2 = 0.0
        self.y_entry_gate_2 = 0.0
        self.x_exit_gate_2  = 0.0
        self.y_exit_gate_2  = 0.0
        self.lat_entry_gate = 0.0 
        self.lon_entry_gate = 0.0
        self.lat_exit_gate = 0.0 
        self.lon_exit_gate = 0.0

        # ---- Gate state flags ----
        self.entry_pass = False
        self.exit_pass = False
        

        # gate detection flags
        self.found_1st_gate = False
        self.found_2nd_gate = False

        #Task 2
        self.stop_counting = False
        self.turn_360 = False
        self._published_buoy_keys = set()
        self._published_buoy_key_time = {}  # key -> last_time
        self._published_ttl_sec = 0.0       # keep 0.0 to mean "never republish"
        self.return_task2 = False
        self.yellow_buoy_x_ned = 0.0
        self.yellow_buoy_y_ned = 0.0

        #pee task
        self.peeing = False
        self.peeing_timer_ = None
        self.x_before_pee = 0.0
        self.y_before_pee = 0.0
        self.psi_before_pee = 0.0
        self.yellow_boat_x = 0.0
        self.yellow_boat_y = 0.0
        self.found_yellow_boat = False
        self.boat_angle = 0.0

        #sound task
        self.go2task = 0
        self.sound_cnt = 0
        self.direction_task3 = 'NONE'
        self.gotDirection = False
        

        #speed challenge
        self.speed_cnt_red = 0
        self.speed_cnt_green = 0

        self.x_goal_poses = [0.0 for _ in range(100)]
        self.y_goal_poses = [0.0 for _ in range(100)]
        self.psi_goal_poses = [0.0 for _ in range(100)]

        self.x_usv_NED = float(0.0)
        self.y_usv_NED = float(0.0)
        self.psi_usv_NED = float(0.0)

        self.lat0 = 27.375139
        self.long0 = -82.45256
        self.goal_pose_qty = 0

        self.velD = 1.3

        self.qualy = False
        self.gotTask = False

        self.saved_pose = False


        self.image_width_px = 1218
        self.horizontal_fov_deg = 108.0

        # pose error
        self.e_xy = float("inf")
        self.heard_sound = False

        # OPTIONAL: if you still want planner to request moving to next task
        self.move_task_pub = self.create_publisher(UInt8, f"/{self.asv_name}/mission/set_task", 10)
        self.move_task_msg = UInt8()

        self.timer = self.create_timer(0.1, self._loop)
        self.get_logger().info(f"ASVPathPlanner running for {self.asv_name}")


    def position_callback(self, msg:Pose2D):
        self.x_usv_NED = msg.x
        self.y_usv_NED = msg.y
        self.psi_usv_NED = msg.theta

    def ned_to_geodetic(self, x_north_m: float,
                    y_east_m: float,
                    lat0_deg: float,
                    lon0_deg: float):
        """
        Convert local NED coordinates (meters) to latitude/longitude (degrees).

        x_north_m : +North (meters)
        y_east_m  : +East (meters)
        lat0_deg  : origin latitude (degrees)
        lon0_deg  : origin longitude (degrees)

        Returns:
            (lat_deg, lon_deg)
        """

        R = 6378137.0  # WGS-84 Earth radius (meters)

        lat0_rad = math.radians(lat0_deg)
        lon0_rad = math.radians(lon0_deg)

        dlat = x_north_m / R
        dlon = y_east_m / (R * math.cos(lat0_rad))

        lat = lat0_rad + dlat
        lon = lon0_rad + dlon

        return math.degrees(lat), math.degrees(lon)


    def calculate_object_angle(self, x_object_px, image_width_px=704, horizontal_fov_deg=108.0):
        cx = image_width_px / 2.0
        hfov = math.radians(horizontal_fov_deg)
        f = (image_width_px / 2.0) / math.tan(hfov / 2.0)  # focal length in pixels

        dx = x_object_px - cx
        return math.atan(dx / f)  # radians

    def yolo_center_callback(self, msg: YoloDetection):
        n = len(msg.class_name)
        if n == 0:
            return

        for i in range(n):
            cls = msg.class_name[i]
            conf = msg.confidence[i]
            x = msg.x_center[i]

            '''

            if self.current_task == TASK_OBJECT_DELIVERY:
                if cls == "black_triangle":
                    self.found_yellow_boat = True
                    self.boat_angle = self.calculate_object_angle(
                        x_object_px=x,
                        image_width_px=self.image_width_px,
                        horizontal_fov_deg=self.horizontal_fov_deg
                    )

                    self.get_logger().info(
                        f"{cls} conf={conf:.2f} x_center={x:.1f} -> angle={self.boat_angle:.3f} rad ({math.degrees(self.boat_angle):.2f} deg)"
                    )
            '''
            if self.current_task == TASK_SPEED_CHALLENGE:
                if cls == "red_indicator" and not self.gotDirection:
                    self.speed_cnt_red +=1
                    if self.speed_cnt_red == 5:
                        self.direction_task3 = 'CCW'
                elif cls == 'green_indicator' and not self.gotDirection:
                    self.speed_cnt_green+=1
                    if self.speed_cnt_green == 5:
                        self.direction_task3 = 'CW'


    def start_peeing(self):
        """Start peeing, send 'A' to /pump_command, and create a timer."""
        if self.peeing_timer_ is None:
            self.peeing = True
            msg = Bool()
            msg.data = True
            self.pee_pub.publish(msg)
            self.get_logger().info("🚰 Peeing started!")
    def stop_peeing(self):
        """Stop peeing, send 'B' to /pump_command, and cancel the timer."""
        if self.peeing:
            msg = Bool()
            msg.data = False
            self.pee_pub.publish(msg)
            self.get_logger().info("🚽 Peeing stopped!")

            self.point = 0
            self.move_task_msg.data = 4
            self.move_task_pub.publish(self.move_task_msg)
            
            # Destroy the timer after execution
            if self.peeing_timer_ is not None:
                self.peeing_timer_.cancel()
                self.peeing_timer_ = None
    def transform_nwu_ned(self, x_local, y_local):

        x_local = float(x_local)  # Ensure x_local is a number
        y_local = float(y_local)
        d_L_G = np.array([self.x_usv_NED, self.y_usv_NED, 0])
        
        R_L_G = np.array([
            [math.cos(self.psi_usv_NED), -math.sin(self.psi_usv_NED), 0],
            [math.sin(self.psi_usv_NED), math.cos(self.psi_usv_NED), 0],
            [0, 0, 1]
        ])
        
        d_P_L = np.array([x_local, -y_local, 0])
        R_P_L = np.eye(3)
        
        H_L_G = np.eye(4)
        H_L_G[:3, :3] = R_L_G
        H_L_G[:3, 3] = d_L_G
        
        H_P_L = np.eye(4)
        H_P_L[:3, :3] = R_P_L
        H_P_L[:3, 3] = d_P_L
        
        H_P_G = np.dot(H_L_G, H_P_L)
        
        return H_P_G[0, 3], H_P_G[1, 3]

    def task1_path_callback(self,msg):
        if (self.current_task == TASK_ENTRY_EXIT):
            num_waypoints = len(msg.poses)
            if self.current_task != TASK_ENTRY_EXIT:
                #self.get_logger().warn(f"Ignoring path because PP_state = {self.PP_state}")
                return
            if num_waypoints < 3:
                    #self.get_logger().warn("Received path with less than two waypoints, cannot extract second point.")
                    return
            second_pose = msg.poses[1].pose.position
            third_pose = msg.poses[2].pose.position
            self.x_goal_poses[0], self.y_goal_poses[0] = self.transform_nwu_ned(second_pose.x, second_pose.y)
            self.x_goal_poses[1], self.y_goal_poses[1] = self.transform_nwu_ned(third_pose.x, third_pose.y)
            if not self.found_1st_gate:
                self.x_entry_gate, self.y_entry_gate = self.transform_nwu_ned(second_pose.x, second_pose.y)
                self.x_entry_gate_2, self.y_entry_gate_2 = self.transform_nwu_ned(third_pose.x, third_pose.y)
                self.found_1st_gate = True
                self.get_logger().info(f"First Gate at {self.x_entry_gate}, {self.y_entry_gate}")
                self.lat_entry_gate, self.lon_entry_gate = self.ned_to_geodetic(self.x_entry_gate, self.y_entry_gate, self.lat0, self.long0)

            else:
                x_new, y_new = self.transform_nwu_ned(second_pose.x, second_pose.y)
                distance = math.sqrt((self.x_entry_gate - x_new)**2 + (self.y_entry_gate - y_new)**2)
                if not self.found_2nd_gate and distance > 5.0:
                    self.x_exit_gate, self.y_exit_gate = self.transform_nwu_ned(second_pose.x, second_pose.y)
                    self.x_exit_gate_2, self.y_exit_gate_2 = self.transform_nwu_ned(third_pose.x, third_pose.y)
                    self.found_2nd_gate = True
                    self.get_logger().info(f"Second Gate at {self.x_exit_gate}, {self.y_exit_gate}")
                    self.lat_exit_gate, self.lon_exit_gate = self.ned_to_geodetic(self.x_exit_gate, self.y_exit_gate, self.lat0, self.long0)


    def task2_path_callback(self, msg):
        if (self.current_task == TASK_NAV_CHANNEL):
            num_waypoints = len(msg.poses)
            self.found_gates = True
            if self.current_task != TASK_NAV_CHANNEL:
                #self.get_logger().warn(f"Ignoring path because PP_state = {self.PP_state}")
                return
            if num_waypoints < 2:
                    #self.get_logger().warn("Received path with less than two waypoints, cannot extract second point.")
                    return
            second_pose = msg.poses[1].pose.position
            self.x_goal_poses[0], self.y_goal_poses[0] = self.transform_nwu_ned(second_pose.x, second_pose.y)
            self.point = 0


    # ---------------- NAV ---------------

    def _on_task(self, msg: UInt8):
        self.current_task = int(msg.data)

    # ---------------- Perception ----------------
    def _on_gate_midpoint_ll(self, msg: Float64MultiArray):
        if len(msg.data) < 3:
            self.get_logger().warn("gate_midpoint_ll expects [gate_type, lat, lon]")
            return
        gate_type = int(msg.data[0])
        lat = float(msg.data[1])
        lon = float(msg.data[2])
        self._last_gate = (gate_type, lat, lon)

        out = Float64MultiArray()
        out.data = [float(gate_type), lat, lon]
        self.gate_midpoint_ll_pub.publish(out)

    def _on_object_detected_meta(self, msg: UInt32MultiArray):
        if len(msg.data) < 4:
            self.get_logger().warn("object_detected meta expects [object_type, color, object_id, task_context]")
            return
        self._last_obj_meta = msg
        self.obj_meta_pub.publish(msg)

    def _on_object_detected_ll(self, msg: Float64MultiArray):
        if len(msg.data) < 2:
            self.get_logger().warn("object_detected_ll expects [lat, lon]")
            return
        self._last_obj_ll = msg
        self.obj_ll_pub.publish(msg)
    
    def fusion_callback(self, msg):
        for i in range(len(msg.x_object)):          # Keep your original class check
            if msg.object_names[i] == "black_triangle":   # orange boat (CHANGE THIS FOR SEMIFINAL!!!)

                # --- Keep your original "save pose once" behavior ---
                if not self.saved_pose:
                    self.x_before_pee = self.x_usv_NED
                    self.y_before_pee = self.y_usv_NED
                    self.psi_before_pee = self.psi_usv_NED
                    self.saved_pose = True
                    self.get_logger().info("Saved from Sensor fusion")
                    self.get_logger().info("New Boat Detected")

                # --- Keep your original flags + local storage ---
                x_local = float(msg.x_object[i])
                y_local = float(msg.y_object[i])

                distance = math.sqrt(x_local * x_local + y_local * y_local)  # keep (you had it)

                if not self.found_yellow_boat:
                    self.found_yellow_boat = True

                self.yellow_boat_x = x_local
                self.yellow_boat_y = y_local

                # --- Keep your original goal update (LOCAL -> GLOBAL) ---
                # You had: self.transform_nwu_ned(self.yellow_boat_x, self.yellow_boat_y)
                # (this should output GLOBAL NED)
                xg, yg = self.transform_nwu_ned(self.yellow_boat_x, self.yellow_boat_y)
                self.x_goal_poses[0] = float(xg)
                self.y_goal_poses[0] = float(yg)

                # Convert vessel NED -> lat/lon (IF your robocommand wants geo)
                vessel_lat, vessel_lon = self.ned_to_geodetic(xg, yg, self.lat0, self.long0)

                # Stable vessel ID (if only one, keep it 1)
                vessel_id = 1

                if self.current_task != TASK_OBJECT_DELIVERY:
                    # ======== ObjectDetected ONCE per vessel ========
                    if not getattr(self, "found_yellow_boat", False):
                        self._publish_object_detected(
                            obj_type=1,          # or int 1 if you don’t import enums here
                            color_id=1,               # set to the correct stationary vessel color
                            obj_id=int(vessel_id),
                            task_ctx=int(1),
                            lat=float(vessel_lat),
                            lon=float(vessel_lon)
                        )
                        self.found_yellow_boat = True
                        self.yellow_boat_x, yellow_boat_y = x_local, y_local    
                        self.get_logger().info(
                            f"ObjectDetected SENT: vessel_id={vessel_id} "
                            f"NED=({xg:.2f},{yg:.2f}) LL=({vessel_lat:.7f},{vessel_lon:.7f})"
                        )

            return

    def _publish_object_delivery_once(
        self,
        vessel_color: int,
        delivery_type: int,
        obj_id: int = 0,
        task_ctx: int = 0,
        lat: float | None = None,
        lon: float | None = None,
    ) -> bool:
        if not hasattr(self, "_delivery_reported"):
            self._delivery_reported = False
        if self._delivery_reported:
            return False

        lat_e7 = int(round(float(lat) * 1e7)) if lat is not None else 0
        lon_e7 = int(round(float(lon) * 1e7)) if lon is not None else 0

        msg = Int32MultiArray()
        msg.data = [
            int(vessel_color),
            int(delivery_type),
            int(obj_id),
            int(task_ctx),
            int(lat_e7),
            int(lon_e7),
        ]
        self._object_delivery_pub.publish(msg)
        self._delivery_reported = True
        return True

    def fusion_tracked_callback(self, msg):
        if self.current_task != TASK_ENTRY_EXIT and self.current_task != TASK_NAV_CHANNEL:
            return
        # If you DON'T want to change gates while executing a gate, enable:
        # if self.found_gates:
        #     return

        # ---------- Ensure state vars exist ----------
        if not hasattr(self, "last_gate_mid"):
            self.last_gate_mid = None
        if not hasattr(self, "ignore_radius"):
            self.ignore_radius = 8.0
        if not hasattr(self, "last_gate_change_t"):
            self.last_gate_change_t = 0.0
        if not hasattr(self, "reacquire_cooldown"):
            self.reacquire_cooldown = 0.4
        if not hasattr(self, "dbg_gate_every"):
            self.dbg_gate_every = 10
        if not hasattr(self, "dbg_gate_k"):
            self.dbg_gate_k = 0

        # cooldown after passing gate
        if (time.time() - self.last_gate_change_t) < self.reacquire_cooldown:
            return

        # ---------- Vehicle pose in GLOBAL NED ----------
        x0 = float(self.x_usv_NED)
        y0 = float(self.y_usv_NED)
        psi = float(self.psi_usv_NED)

        # ---------- Parameters ----------
        MIN_RANGE = 4.0
        MAX_RANGE = 20.0

        MIN_WIDTH = 5.0
        MAX_WIDTH = 15.0

        BODY_AHEAD_MIN = 4.0     # must be ahead in body +X
        THROUGH_DIST = 5.0       # meters beyond gate (perpendicular)

        # OPTIONAL: enforce red-left/green-right (OFF by default)
        ENFORCE_RED_LEFT = False

        n = min(len(msg.object_names), len(msg.x_object), len(msg.y_object), len(msg.z_object))
        if n == 0:
            return

        # ---------- Helpers ----------
        def ned_to_body(xg, yg, x0, y0, psi):
            """
            GLOBAL -> BODY
            xb = forward
            yb = left (ASSUMPTION)
            """
            dx = xg - x0
            dy = yg - y0
            c = math.cos(psi)
            s = math.sin(psi)
            xb =  c * dx + s * dy
            yb = -s * dx + c * dy
            return xb, yb

        # ---------- Collect buoy points ----------
        red_buoys = []
        green_buoys = []

        for i in range(n):
            name = str(msg.object_names[i]).lower()
            px = float(msg.x_object[i])
            py = float(msg.y_object[i])
            pz = float(msg.z_object[i])

            dx = px - x0
            dy = py - y0
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < MIN_RANGE or dist > MAX_RANGE:
                continue

            p = Point()
            p.x, p.y, p.z = px, py, pz

            if "red_buoy" in name or name.startswith("red"):
                red_buoys.append(p)
            elif "green_buoy" in name or name.startswith("green"):
                green_buoys.append(p)

        if len(red_buoys) < 1 or len(green_buoys) < 1:
            return
        # ---------- Debug counters ----------
        rej_width = 0
        rej_lastgate = 0
        rej_ahead = 0
        rej_left = 0
        considered = 0

        # ---------- Pair selection (red-green) ----------
        pair = []
        best_mid = None
        best_body_mid = None
        best_width = None
        best_score = float("inf")

        for i in range(len(red_buoys)):
            for j in range(len(green_buoys)):
                r = red_buoys[i]
                g = green_buoys[j]
                considered += 1

                vx = r.x - g.x
                vy = r.y - g.y
                width = math.sqrt(vx * vx + vy * vy)
                if width < MIN_WIDTH or width > MAX_WIDTH:
                    rej_width += 1
                    continue

                mid_x = 0.5 * (r.x + g.x)
                mid_y = 0.5 * (r.y + g.y)

                # ignore last passed gate
                if self.last_gate_mid is not None:
                    dxl = mid_x - self.last_gate_mid[0]
                    dyl = mid_y - self.last_gate_mid[1]
                    if math.sqrt(dxl * dxl + dyl * dyl) < self.ignore_radius:
                        rej_lastgate += 1
                        continue

                # must be in front in BODY frame (+X)
                mx_b, my_b = ned_to_body(mid_x, mid_y, x0, y0, psi)
                if mx_b <= BODY_AHEAD_MIN:
                    rej_ahead += 1
                    continue

                # optional red-left constraint
                if ENFORCE_RED_LEFT:
                    r_bx, r_by = ned_to_body(r.x, r.y, x0, y0, psi)
                    g_bx, g_by = ned_to_body(g.x, g.y, x0, y0, psi)
                    # if yb is left: red-left means r_by > g_by
                    if r_by <= g_by:
                        rej_left += 1
                        continue

                # score: closest ahead in body frame (small mx_b), prefer centerline
                score = mx_b + 0.25 * abs(my_b) + 0.03 * width
                if score < best_score:
                    best_score = score
                    pair = [r, g]
                    best_mid = (mid_x, mid_y)
                    best_body_mid = (mx_b, my_b)
                    best_width = width

        # ---------- Print why we failed ----------
        self.dbg_gate_k += 1
        if not pair:
            if (self.dbg_gate_k % self.dbg_gate_every) == 0:
                self.get_logger().warn(
                    f"[gate dbg] no gate. reds={len(red_buoys)} greens={len(green_buoys)} "
                    f"pairs={considered} rej(width={rej_width}, last={rej_lastgate}, "
                    f"ahead={rej_ahead}, left={rej_left}) "
                    f"psi(deg)={math.degrees(psi):.1f}"
                )
            return

        # ---------- Build goals (midpoint + perpendicular forward choice) ----------
        mid_x, mid_y = best_mid

        vx = pair[0].x - pair[1].x
        vy = pair[0].y - pair[1].y
        mag = math.sqrt(vx * vx + vy * vy)
        if mag < 1e-6:
            return

        ux = vx / mag
        uy = vy / mag

        perp_x1 = -uy * THROUGH_DIST
        perp_y1 =  ux * THROUGH_DIST
        perp_x2 =  uy * THROUGH_DIST
        perp_y2 = -ux * THROUGH_DIST

        point1_x, point1_y = mid_x + perp_x1, mid_y + perp_y1
        point2_x, point2_y = mid_x + perp_x2, mid_y + perp_y2

        # choose the point that goes more forward in BODY (+X)
        p1_bx, _ = ned_to_body(point1_x, point1_y, x0, y0, psi)
        p2_bx, _ = ned_to_body(point2_x, point2_y, x0, y0, psi)

        self.x_goal_poses[0], self.y_goal_poses[0] = mid_x, mid_y
        if p1_bx >= p2_bx:
            self.x_goal_poses[1], self.y_goal_poses[1] = point1_x, point1_y
        else:
            self.x_goal_poses[1], self.y_goal_poses[1] = point2_x, point2_y

        self.point = 0
        self.found_gates = True

        mx_b, my_b = best_body_mid
        self.get_logger().info(
            f"Selected gate: mid(world)=({mid_x:.2f},{mid_y:.2f}) "
            f"mid(body)=({mx_b:.2f},{my_b:.2f}) width={best_width:.2f}"
        )



    # ---------------- Main Loop ----------------
    def _loop(self):
        if self.current_task == TASK_ENTRY_EXIT:
            if self.qualy:
                self._task1_entry_exit_qualy()
            else:
                self._task1_entry_exit()
        elif self.current_task == TASK_NAV_CHANNEL:
            self._task2_nav_channel()
        elif self.current_task == TASK_SPEED_CHALLENGE:
            self._task3_speed()
        elif self.current_task == TASK_OBJECT_DELIVERY:
            self._task4_delivery()
        elif self.current_task == TASK_DOCKING:
            self._task5_docking()
        elif self.current_task == TASK_SOUND_SIGNAL:
            self._task6_sound()
        elif self.current_task == QUALIFYING:
            self.qualifying_tasks()
        elif self.current_task == 10:
            self.PID_Tuning()
        else:
            pass
        

    def waypoints_task1(self):

        self.x_goal_poses[0] = self.x_exit_gate
        self.y_goal_poses[0] = self.y_exit_gate

        self.x_goal_poses[1] = self.x_entry_gate_2
        self.y_goal_poses[1] = self.y_entry_gate_2

        self.x_goal_poses[2] = self.x_entry_gate
        self.y_goal_poses[2] = self.y_entry_gate

        self.goal_pose_qty = 3

    def PID_Tuning(self):
        self.publish_desired_state(
            x=self.x_goal_poses[self.point],
            y=self.y_goal_poses[self.point],
            psi= -1.7,
            # Desired velocities / accelerations
            vx=1.5,   
            vy=0.0,
            w=0.0,
            ax=0.0,
            ay=0.0,
            alpha=0.0,
        )

    def _on_sound_signal(self, msg: UInt32MultiArray):
        # expects [signal_type, frequency_hz, assigned_task]

        if not self.gotTask:
            if len(msg.data) < 3:
                self.get_logger().warn("sound_signal expects [signal_type, frequency_hz, assigned_task]")
                return

            self.go2task = int(msg.data[2])
    def _publish_object_detected(self, obj_type: int, color_id: int, obj_id: int, task_ctx: int, lat: float, lon: float):
        msg = Float64MultiArray()
        # One topic: [object_type, color, object_id, task_context, lat, lon]
        msg.data = [
            float(obj_type),
            float(color_id),
            float(obj_id),
            float(task_ctx),
            float(lat),
            float(lon),
        ]
        self.pub_object_detected.publish(msg)
    def _buoy_publish_key(self, color: str, x_ned: float, y_ned: float) -> tuple:
        """
        Key used to decide if this is a "new buoy".
        Quantize position so small KF jitter doesn't create new keys.

        If you want stricter uniqueness, change 0.5 -> 0.2
        If you want looser, change 0.5 -> 1.0
        """
        q = 0.5  # meters quantization
        return (color, int(round(x_ned / q)), int(round(y_ned / q)))
    def _color_str_to_id(self, color: str) -> int:
        # expects: "red_buoy", "green_buoy", "yellow_buoy", "black_buoy"
        m = {
            "yellow_buoy": 1,  # COLOR_YELLOW
            "black_buoy":  2,  # COLOR_BLACK
            "red_buoy":    3,  # COLOR_RED
            "green_buoy":  4,  # COLOR_GREEN
        }
        return int(m.get(color, 0))  # COLOR_UNKNOWN

    def cb_buoys(self, msg: ObjectPositionArray):
        if not (self.current_task == TASK_NAV_CHANNEL or self.current_task == TASK_ENTRY_EXIT):
            return

        if getattr(self, "stop_counting", False):
            return

        DEFAULT_UNIQUE_RADIUS = 2.0
        YELLOW_UNIQUE_RADIUS = 5.0

        def unique_radius_for_color(color: str) -> float:
            c = (color or "").strip().lower()
            if c == "yellow_buoy":
                return YELLOW_UNIQUE_RADIUS
            return DEFAULT_UNIQUE_RADIUS

        unique_by_color = {}

        for name, x, y, z in zip(msg.object_names, msg.x_object, msg.y_object, msg.z_object):
            x = float(x)
            y = float(y)
            z = float(z)

            parts = (name or "").split("_")
            if len(parts) >= 2:
                color = f"{parts[0]}_{parts[1]}"
            else:
                color = (name or "").strip()

            R = unique_radius_for_color(color)
            R2 = R * R

            if color not in unique_by_color:
                unique_by_color[color] = [(x, y, z)]
                continue

            is_duplicate = False
            for (ux, uy, uz) in unique_by_color[color]:
                dx = x - ux
                dy = y - uy
                if (dx * dx + dy * dy) <= R2:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_by_color[color].append((x, y, z))

        unique_buoys = []
        for color in unique_by_color:
            for (x, y, z) in unique_by_color[color]:
                unique_buoys.append((color, x, y, z))

        self.current_buoys = unique_buoys

        if not hasattr(self, "_known_buoys"):
            self._known_buoys = {}
        if not hasattr(self, "_next_known_id"):
            self._next_known_id = 1
        if not hasattr(self, "_published_known_ids"):
            self._published_known_ids = set()

        if not hasattr(self, "yellow_buoy_x_ned"):
            self.yellow_buoy_x_ned = None
        if not hasattr(self, "yellow_buoy_y_ned"):
            self.yellow_buoy_y_ned = None

        OBJ_TYPE_BUOY = 3
        TASK_CTX_NAV_CHANNEL = int(TASK_NAV_CHANNEL)

        ALPHA = 0.2

        for (color, x_ned, y_ned, _z) in unique_buoys:
            if color not in self._known_buoys:
                self._known_buoys[color] = []

            R = unique_radius_for_color(color)
            R2 = R * R

            best = None
            best_d2 = float("inf")
            for kb in self._known_buoys[color]:
                dx = float(x_ned) - float(kb["x"])
                dy = float(y_ned) - float(kb["y"])
                d2 = dx * dx + dy * dy
                if d2 < best_d2:
                    best_d2 = d2
                    best = kb

            if best is not None and best_d2 <= R2:
                best["x"] = (1.0 - ALPHA) * float(best["x"]) + ALPHA * float(x_ned)
                best["y"] = (1.0 - ALPHA) * float(best["y"]) + ALPHA * float(y_ned)
                best["hits"] += 1

                if color == "yellow_buoy":
                    self.yellow_buoy_x_ned = float(best["x"])
                    self.yellow_buoy_y_ned = float(best["y"])

                continue

            new_id = self._next_known_id
            self._next_known_id += 1

            kb = {"id": int(new_id), "x": float(x_ned), "y": float(y_ned), "hits": 1}
            self._known_buoys[color].append(kb)

            if color == "yellow_buoy":
                self.yellow_buoy_x_ned = float(kb["x"])
                self.yellow_buoy_y_ned = float(kb["y"])

            if kb["id"] in self._published_known_ids:
                continue

            lat, lon = self.ned_to_geodetic(kb["x"], kb["y"], self.lat0, self.long0)
            color_id = self._color_str_to_id(color)

            self._publish_object_detected(
                obj_type=OBJ_TYPE_BUOY,
                color_id=color_id,
                obj_id=int(kb["id"]),
                task_ctx=TASK_CTX_NAV_CHANNEL,
                lat=lat,
                lon=lon
            )

            self._published_known_ids.add(kb["id"])

            self.get_logger().info(
                f"NEW BUOY -> published: {color} "
                f"NED=({kb['x']:.2f},{kb['y']:.2f}) LL=({lat:.7f},{lon:.7f}) id={kb['id']}"
            )

    def waypoints_task6(self):
        if self.go2task == 3:
            self.get_logger().info(f"TASK 3")
            self.x_goal_poses[0] = 26.0
            self.y_goal_poses[0] = -25.0

            self.goal_pose_qty = 1
            self.gotTask = True
        elif self.go2task == 5:
            self.get_logger().info(f"TASK 5")

            self.x_goal_poses[0] = -3.5
            self.y_goal_poses[0] = -3.85

            self.goal_pose_qty = 1

            self.gotTask = True

        
    def waypoints_task3(self):
        self.get_logger().info(f'CW')

        self.x_goal_poses[0] = -11.0
        self.y_goal_poses[0] = -16.0

        self.x_goal_poses[1] = -62.0
        self.y_goal_poses[1] = -19.0


        self.x_goal_poses[2] = -62.0
        self.y_goal_poses[2] = -26.0

        self.x_goal_poses[3] = -62.0
        self.y_goal_poses[3] = -33.0

        self.x_goal_poses[4] = -64.5
        self.y_goal_poses[4] = -38.5

        self.x_goal_poses[5] = -67.4
        self.y_goal_poses[5] = -42.4

        self.x_goal_poses[6] = -72.0
        self.y_goal_poses[6] = -47.0

        self.x_goal_poses[6] = 2.0
        self.y_goal_poses[6] = -24.0

        self.x_goal_poses[7] = -2.0
        self.y_goal_poses[7] = -27.0

        self.x_goal_poses[8] = -7.0
        self.y_goal_poses[8] = -28.0

        self.goal_pose_qty = 9

        self.gotDirection = True
        
    def pi_wrap(self,angle_rad):
        return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi

    def _task6_sound(self):
        self.send_controller_command("HS")
        self.waypoints_task6()
        if self.go2task == 0 and not self.gotTask:
            self.get_logger().info(f"NO TASK")
            return
        
        self.calculate_pose_errors(
                self.x_goal_poses[self.point],
                self.y_goal_poses[self.point]
            )
        self.psi_goal_poses[self.point] = self.pi_wrap(math.atan2(self.e_y, self.e_x))
        self.publish_desired_state(
            x=self.x_goal_poses[self.point],
            y=self.y_goal_poses[self.point],
            psi=self.psi_goal_poses[self.point],
            # Desired velocities / accelerations
            vx=1.0,   
            vy=0.0,
            w=0.0,
            ax=0.0,
            ay=0.0,
            alpha=0.0,
        )
        if self.e_xy < ERROR_WP:
            self.point += 1
            self.get_logger().info(f"Waypoint {self.point}/{self.goal_pose_qty}")


        if self.point == self.goal_pose_qty:
            self.psi_goal_poses[0] = self.psi_goal_poses[self.point-1]
            self.get_logger().info(f"DONE")

    def _task1_entry_exit_qualy(self):
        self.waypoints_task1()
        self.send_controller_command("HS")
        #if self.direction_task3 == 'NONE' and not self.gotDirection:
        #    self.get_logger().info(f"NO DIRECTION -- TASK3")
        #    return
        
        self.calculate_pose_errors(
                self.x_goal_poses[self.point],
                self.y_goal_poses[self.point]
            )
        self.psi_goal_poses[self.point] = self.pi_wrap(math.atan2(self.e_y, self.e_x))
        self.publish_desired_state(
            x=self.x_goal_poses[self.point],
            y=self.y_goal_poses[self.point],
            psi=self.psi_goal_poses[self.point],
            # Desired velocities / accelerations
            vx=1.5,   
            vy=0.0,
            w=0.0,
            ax=0.0,
            ay=0.0,
            alpha=0.0,
        )
        if self.e_xy < ERROR_WP:
            self.point += 1
            self.get_logger().info(f"Waypoint {self.point}/{self.goal_pose_qty}")


        if self.point == self.goal_pose_qty:
            self.psi_goal_poses[0] = self.psi_goal_poses[self.point-1]
            self.get_logger().info(f"DONE")
            msg = Float64MultiArray()
            msg.data = [
                2.0,                  # gate type = ENTRY
                self.lat_entry_gate,       # latitude
                self.lon_entry_gate        # longitude
            ]
            self.gate_pass_pub.publish(msg)

    def _task1_entry_exit(self):
        self.send_controller_command("HS")
        if not self.entry_pass:
            self.point = 0
        elif self.entry_pass and not self.found_2nd_gate:
            self.point = 1
        elif self.entry_pass and self.found_2nd_gate and not self.exit_pass:
            self.point = 0
        elif self.entry_pass and self.exit_pass:
            self.point = 1
        
        self.calculate_pose_errors(
                self.x_goal_poses[self.point],
                self.y_goal_poses[self.point]
            )
        self.psi_goal_poses[self.point] = self.pi_wrap(math.atan2(self.e_y, self.e_x))
        if self.entry_pass and not self.found_2nd_gate:
            self.psi_goal_poses[1] = self.psi_goal_poses[0]
        self.publish_desired_state(
            x=self.x_goal_poses[self.point],
            y=self.y_goal_poses[self.point],
            psi=self.psi_goal_poses[self.point],
            # Desired velocities / accelerations
            vx=1.5,   
            vy=0.0,
            w=0.0,
            ax=0.0,
            ay=0.0,
            alpha=0.0,
        )
        #self.get_logger().info(f"Entry = {self.entry_pass}, Exit ={self.exit_pass}") 
        #self.get_logger().info(f"Moved to Task {self.move_task_msg.data}") 
        #self.get_logger().info(f"x_0 = {self.x_goal_poses[0]}, y_0={self.y_goal_poses[0]}, x_1 = {self.x_goal_poses[1]}, y_1 = {self.y_goal_poses[1]}") 
        #self.get_logger().info(f"Point {self.point}")

        if self.entry_pass and self.exit_pass:
            self.move_task_msg.data = 3
            self.move_task_pub.publish(self.move_task_msg)
            self.get_logger().info(f"Passed EXIT GATE") 
            self.get_logger().info(f"Moved to Task {self.move_task_msg.data}") 

        if not self.entry_pass and self.found_1st_gate:
            
            error = self.calculate_gate_errors(self.x_entry_gate, self.y_entry_gate)
            if error < 1.5:

                self.entry_pass = True
                self.get_logger().info(f"Passed ENTRY GATE") 
        elif self.entry_pass and not self.exit_pass and self.found_2nd_gate:
            error = self.calculate_gate_errors(self.x_exit_gate, self.y_exit_gate)
            if error < 2.0:
                msg = Float64MultiArray()
                msg.data = [
                    1.0,                  # gate type = ENTRY
                    self.lat_exit_gate,       # latitude
                    self.lon_exit_gate        # longitude
                ]

                self.gate_pass_pub.publish(msg)
                self.exit_pass = True           


    # -----------------------------
    # Task 1: Entry/Exit gate driving
    # Uses two waypoints:
    #   point=0 -> midpoint
    #   point=1 -> through-point (beyond gate)
    # Gate is counted as passed after reaching point=1
    # -----------------------------

    def ned_to_body(self, xg, yg, x0, y0, psi):
        dx = float(xg) - float(x0)
        dy = float(yg) - float(y0)

        c = math.cos(float(psi))
        s = math.sin(float(psi))

        xb =  c * dx + s * dy
        yb = -s * dx + c * dy
        return xb, yb

    def get_nth_buoy_xy_by_id(self, color: str, n: int):
        if not hasattr(self, "_known_buoys"):
            return None

        lst = sorted(self._known_buoys.get(color, []), key=lambda b: b["id"])
        if n < 1 or n > len(lst):
            return None

        b = lst[n - 1]
        return float(b["x"]), float(b["y"]), int(b["id"])

    def send_controller_command(self, text: str):
        msg = String()
        msg.data = text
        self.pub_controller.publish(msg)

    def _task2_nav_channel(self):
        self.send_controller_command("HS")
        self.calculate_pose_errors(
                self.x_goal_poses[self.point],
                self.y_goal_poses[self.point]
            )
        self.psi_goal_poses[self.point] = self.pi_wrap(math.atan2(self.e_y, self.e_x))
        self.publish_desired_state(
            x=self.x_goal_poses[self.point],
            y=self.y_goal_poses[self.point],
            psi=self.psi_goal_poses[self.point],
            # Desired velocities / accelerations
            vx=1.5,   
            vy=0.0,
            w=0.0,
            ax=0.0,
            ay=0.0,
            alpha=0.0,
        )
        
        res_gr = self.get_nth_buoy_xy_by_id("green_buoy", 4)
        res_red = self.get_nth_buoy_xy_by_id("red_buoy", 4)

        if not self.stop_counting:
            chosen = None

            if res_gr:
                x, y, bid = res_gr
                xb, yb = self.ned_to_body(x, y, self.x_usv_NED, self.y_usv_NED, self.psi_usv_NED)
                if xb < -2.0:
                    chosen = ("green_buoy", xb, yb, bid)

            if chosen is None and res_red:
                x, y, bid = res_red
                xb, yb = self.ned_to_body(x, y, self.x_usv_NED, self.y_usv_NED, self.psi_usv_NED)
                if xb < -2.0:
                    chosen = ("red_buoy", xb, yb, bid)

            if chosen is not None:
                cls, xb, yb, bid = chosen
                self.get_logger().info(f"Go to PEEEEEEE (triggered by {cls} id={bid} xb={xb:.2f})")
                self.move_task_msg.data = 5
                self.move_task_pub.publish(self.move_task_msg)

    def _task3_speed(self):
        # Controller mode for this task
        self.send_controller_command("HS")

        # ----------------------------
        # One-time init (per node life)
        # ----------------------------
        if not hasattr(self, "_task3_start_time"):
            self._task3_start_time = None

        # Tunables (change here if you want)
        TASK3_TIMEOUT_S = 60.0         # how long you allow "going to yellow"
        ARRIVAL_RADIUS_M = 2.0          # consider arrived if within this distance

        # ----------------------------
        # Sound gate: if we heard sound -> go Task 1
        # ----------------------------
        if getattr(self, "heard_sound", False):
            self.get_logger().info("Sound detected -> switching to TASK 1")
            self.move_task_msg.data = 2
            self.move_task_pub.publish(self.move_task_msg)
            self.qualy = True

            # reset task3 timer and sound flag
            self._task3_start_time = None
            self.heard_sound = False
            return

        # ----------------------------
        # Need yellow buoy global NED
        # ----------------------------
        xb = getattr(self, "yellow_buoy_x_ned", None)
        yb = getattr(self, "yellow_buoy_y_ned", None)
        if xb is None or yb is None:
            # don’t run timer until we actually have a target
            self._task3_start_time = None
            return

        xb = float(xb)
        yb = float(yb)

        # Start timer the first time we have a valid target
        if self._task3_start_time is None:
            self._task3_start_time = time.time()
            self.get_logger().info("Task3: started timer -> going to yellow buoy")

        elapsed = time.time() - self._task3_start_time

        # Distance to yellow in GLOBAL NED
        dx = xb - float(self.x_usv_NED)
        dy = yb - float(self.y_usv_NED)
        dist = math.hypot(dx, dy)

        # ----------------------------
        # Arrival condition
        # ----------------------------
        if dist <= ARRIVAL_RADIUS_M:
            self.get_logger().info(f"Task3: ARRIVED at yellow (dist={dist:.2f}m, t={elapsed:.1f}s)")
            self._task3_start_time = None
            # If you want to advance state/point, do it here:
            # self.point += 1
            return

        # ----------------------------
        # Timeout condition
        # ----------------------------
        if elapsed >= TASK3_TIMEOUT_S:
            self.get_logger().info(f"Task3: TIMEOUT ({elapsed:.1f}s) -> switching to TASK 1")
            self.move_task_msg.data = 2
            self.move_task_pub.publish(self.move_task_msg)
            self.qualy = True

            self._task3_start_time = None
            return

        # ----------------------------
        # Set goal pose to yellow buoy
        # ----------------------------
        self.x_goal_poses[self.point] = xb
        self.y_goal_poses[self.point] = yb

        # Heading towards goal
        self.calculate_pose_errors(self.x_goal_poses[self.point], self.y_goal_poses[self.point])
        self.psi_goal_poses[self.point] = self.pi_wrap(math.atan2(self.e_y, self.e_x))

        # Publish desired state
        self.publish_desired_state(
            x=self.x_goal_poses[self.point],
            y=self.y_goal_poses[self.point],
            psi=self.psi_goal_poses[self.point],
            vx=0.5,
            vy=0.0,
            w=0.0,
            ax=0.0,
            ay=0.0,
            alpha=0.0,
        )

        self.get_logger().info(
            f"Task3: Going to Yellow (t={elapsed:.1f}s dist={dist:.2f}m) "
            f"x={xb:.2f} y={yb:.2f}"
        )

    def _on_sound_signal(self, msg: UInt32MultiArray):
        if len(msg.data) < 3:
            self.get_logger().warn(
                "sound_signal expects 3 ints: [signal_type, frequency_hz, assigned_task]"
            )
            return

        signal_type = int(msg.data[0])
        freq = int(msg.data[1])
        assigned_task = int(msg.data[2])

        # Keep your original behavio

        # ✅ Just raise flag
        self.heard_sound = True

        self.get_logger().info(
            f"Sound received → heard_sound = True (freq={freq})"
        )
    def _task4_delivery(self):
        self.send_controller_command("HS")
        if self.found_yellow_boat:          #for peeing boats
            self.point = 0
            self.calculate_pose_errors(
                    self.x_goal_poses[self.point],
                    self.y_goal_poses[self.point]
                )
            self.psi_goal_poses[self.point] = self.pi_wrap(math.atan2(self.e_y, self.e_x))
            self.publish_desired_state(
                    x=self.x_goal_poses[self.point],
                    y=self.y_goal_poses[self.point],
                    psi=self.psi_goal_poses[self.point],
                    vx=0.3,
                    vy=0.0,
                    w=0.0,
                    ax=0.0,
                    ay=0.0,
                    alpha=0.0,
                )

            distance = math.sqrt(self.yellow_boat_x**2 + self.yellow_boat_y**2)
            #self.get_logger().info(f"{distance}")
            if distance < 7.0 and not self.peeing:
                self.start_peeing()  
                self.get_logger().info("Trying to pee")
                self.peeing_timer_ = self.create_timer(15.0, self.stop_peeing) 
                lat, lon = self.ned_to_geodetic(self.yellow_boat_x, self.yellow_boat_y, self.lat0, self.long0)
                # Send desired state
        
        elif not self.found_yellow_boat:
            psi_boat = -1.0
            self.publish_desired_state(
                    x=0.0,
                    y=0.0,
                    psi=psi_boat,
                    vx=1.5,
                    vy=0.0,
                    w=0.0,
                    ax=0.0,
                    ay=0.0,
                    alpha=0.0,
                )
        

    def _task5_docking(self):
        return
    # ---------------- Helpers ---------------

    def calculate_pose_errors(self, x_goal: float, y_goal: float):
        self.e_x = x_goal - self.x_usv_NED
        self.e_y = y_goal - self.y_usv_NED
        self.e_xy = math.sqrt(self.e_x**2 + self.e_y**2)

    def calculate_gate_errors(self, x_goal: float, y_goal: float):
        e_x = x_goal - self.x_usv_NED
        e_y = y_goal - self.y_usv_NED
        e_xy = math.sqrt(e_x**2 + e_y**2)

        return e_xy

    def publish_desired_state(
        self,
        x: float,
        y: float,
        psi: float,
        vx: float = 0.0,
        vy: float = 0.0,
        w: float = 0.0,
        ax: float = 0.0,
        ay: float = 0.0,
        alpha: float = 0.0,
    ):
        msg = Float32MultiArray()
        msg.data = [
            float(x),
            float(y),
            float(psi),
            float(vx),
            float(vy),
            float(w),
            float(ax),
            float(ay),
            float(alpha),
        ]
        self.desired_state_pub.publish(msg)
        self._active_goal_xy = (float(x), float(y))



def main():
    rclpy.init()
    node = ASVPathPlanner()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
