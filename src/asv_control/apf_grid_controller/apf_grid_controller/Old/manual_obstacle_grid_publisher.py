#!/usr/bin/env python3

import math
import time
from typing import Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from geometry_msgs.msg import Pose2D
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray

from yolov26_msgs.msg import ZedDetection


class ZedBuoyObstacleMapPublisher(Node):
    def __init__(self):
        super().__init__("manual_obstacle_grid_publisher")

        # ------------------------------------------------------------
        # Topics
        # ------------------------------------------------------------
        self.declare_parameter("zed_detection_topic", "/zed_custom_detections")
        self.declare_parameter("pose_topic", "/asv/vehicle_pose")

        # APF controller map input
        self.declare_parameter("map_topic", "/asv/map/local_occupancy_2")

        # RViz marker output
        self.declare_parameter("marker_topic", "/asv/viz/zed_buoy_obstacles")
        self.declare_parameter("frame_id", "map")

        self.zed_detection_topic = str(
            self.get_parameter("zed_detection_topic").value
        )
        self.pose_topic = str(self.get_parameter("pose_topic").value)
        self.map_topic = str(self.get_parameter("map_topic").value)
        self.marker_topic = str(self.get_parameter("marker_topic").value)
        self.frame_id = str(self.get_parameter("frame_id").value)

        # ------------------------------------------------------------
        # Detection filtering
        # ------------------------------------------------------------
        self.declare_parameter("min_confidence", 0.35)
        self.declare_parameter("min_range_m", 0.30)
        self.declare_parameter("max_range_m", 40.0)
        self.declare_parameter("body_ahead_min_m", 0.20)

        self.min_confidence = float(self.get_parameter("min_confidence").value)
        self.min_range_m = float(self.get_parameter("min_range_m").value)
        self.max_range_m = float(self.get_parameter("max_range_m").value)
        self.body_ahead_min_m = float(self.get_parameter("body_ahead_min_m").value)

        # ------------------------------------------------------------
        # Obstacle radius
        # ------------------------------------------------------------
        # 6 inches = 0.1524 m
        self.declare_parameter("obstacle_radius_m", 6.0 * 0.0254)
        self.obstacle_radius_m = float(
            self.get_parameter("obstacle_radius_m").value
        )

        # ------------------------------------------------------------
        # Occupancy grid settings
        # ------------------------------------------------------------
        # Local window centered around the boat.
        # The map origin is still expressed in GLOBAL NED.
        self.declare_parameter("resolution", 0.05)
        self.declare_parameter("width_m", 80.0)
        self.declare_parameter("height_m", 80.0)

        self.resolution = float(self.get_parameter("resolution").value)
        self.width_m = float(self.get_parameter("width_m").value)
        self.height_m = float(self.get_parameter("height_m").value)

        self.width = int(math.ceil(self.width_m / self.resolution))
        self.height = int(math.ceil(self.height_m / self.resolution))

        # ------------------------------------------------------------
        # Track smoothing
        # ------------------------------------------------------------
        self.declare_parameter("track_merge_distance_m", 1.0)
        self.declare_parameter("track_timeout_s", 2.0)
        self.declare_parameter("track_alpha", 0.35)
        self.declare_parameter("min_hits_to_publish", 1)

        self.track_merge_distance_m = float(
            self.get_parameter("track_merge_distance_m").value
        )
        self.track_timeout_s = float(self.get_parameter("track_timeout_s").value)
        self.track_alpha = float(self.get_parameter("track_alpha").value)
        self.min_hits_to_publish = int(
            self.get_parameter("min_hits_to_publish").value
        )

        # ------------------------------------------------------------
        # Logging timers
        # ------------------------------------------------------------
        self.last_no_pose_warn_time = 0.0
        self.no_pose_warn_period = 2.0

        self.last_pub_log_time = 0.0
        self.pub_log_period = 1.0

        self.last_detection_log_time = 0.0
        self.detection_log_period = 1.0

        self.last_invalid_detection_warn_time = 0.0
        self.invalid_detection_warn_period = 1.0

        self.last_invalid_track_warn_time = 0.0
        self.invalid_track_warn_period = 1.0

        # ------------------------------------------------------------
        # Vehicle pose state
        # ------------------------------------------------------------
        self.have_pose = False

        # Global NED:
        # x = North
        # y = East
        # theta = heading/yaw in NED
        self.x_usv_ned = 0.0
        self.y_usv_ned = 0.0
        self.psi_usv_ned = 0.0

        # ------------------------------------------------------------
        # Buoy tracks
        # ------------------------------------------------------------
        self.tracks: List[Dict] = []
        self.next_track_id = 0

        # ------------------------------------------------------------
        # QoS
        # ------------------------------------------------------------
        sensor_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        pose_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        # ------------------------------------------------------------
        # Subscribers
        # ------------------------------------------------------------
        self.pose_sub = self.create_subscription(
            Pose2D,
            self.pose_topic,
            self.pose_callback,
            pose_qos,
        )

        self.zed_detection_sub = self.create_subscription(
            ZedDetection,
            self.zed_detection_topic,
            self.zed_detection_callback,
            sensor_qos,
        )

        # ------------------------------------------------------------
        # Publishers
        # ------------------------------------------------------------
        self.map_pub = self.create_publisher(
            OccupancyGrid,
            self.map_topic,
            map_qos,
        )

        self.marker_pub = self.create_publisher(
            MarkerArray,
            self.marker_topic,
            map_qos,
        )

        # Publish map at 5 Hz
        self.timer = self.create_timer(0.20, self.publish_map_and_markers)

        self.get_logger().info("ZED buoy obstacle map publisher started")
        self.get_logger().info(f"Subscribing pose:       {self.pose_topic}")
        self.get_logger().info(f"Subscribing detections: {self.zed_detection_topic}")
        self.get_logger().info(f"Publishing map:         {self.map_topic}")
        self.get_logger().info(f"Publishing markers:     {self.marker_topic}")
        self.get_logger().info(f"Frame ID:               {self.frame_id}")
        self.get_logger().info(f"Grid resolution:        {self.resolution:.3f} m/cell")
        self.get_logger().info(f"Grid size:              {self.width} x {self.height}")
        self.get_logger().info(f"Obstacle radius:        {self.obstacle_radius_m:.4f} m")
        self.get_logger().info(
            "Frame convention: ZED local NWU -> local/body NED -> global NED"
        )

    # ============================================================
    # Generic finite checks
    # ============================================================

    @staticmethod
    def is_finite(*values) -> bool:
        for v in values:
            try:
                if not math.isfinite(float(v)):
                    return False
            except Exception:
                return False
        return True

    def warn_invalid_detection_limited(self, text: str):
        now = time.time()

        if (now - self.last_invalid_detection_warn_time) >= self.invalid_detection_warn_period:
            self.last_invalid_detection_warn_time = now
            self.get_logger().warn(text)

    def warn_invalid_track_limited(self, text: str):
        now = time.time()

        if (now - self.last_invalid_track_warn_time) >= self.invalid_track_warn_period:
            self.last_invalid_track_warn_time = now
            self.get_logger().warn(text)

    # ============================================================
    # Pose callback
    # ============================================================

    def pose_callback(self, msg: Pose2D):
        """
        Vehicle pose in GLOBAL NED.

        msg.x     = North
        msg.y     = East
        msg.theta = yaw/heading in NED
        """

        x = float(msg.x)
        y = float(msg.y)
        psi = float(msg.theta)

        if not self.is_finite(x, y, psi):
            self.get_logger().warn(
                f"Received invalid vehicle pose: x={x}, y={y}, theta={psi}. Ignoring."
            )
            return

        self.x_usv_ned = x
        self.y_usv_ned = y
        self.psi_usv_ned = psi
        self.have_pose = True

    # ============================================================
    # Coordinate transforms
    # ============================================================

    @staticmethod
    def local_ned_to_global_ned(
        x_local: float,
        y_local: float,
        x0: float,
        y0: float,
        psi: float,
    ) -> Tuple[float, float]:
        """
        LOCAL/BODY NED -> GLOBAL NED.

        Global NED:
            x = North
            y = East

        Local/body NED:
            x = forward
            y = right

        Rotation:
            xg = x0 + cos(psi) * x_local - sin(psi) * y_local
            yg = y0 + sin(psi) * x_local + cos(psi) * y_local
        """

        c = math.cos(psi)
        s = math.sin(psi)

        dx = c * x_local - s * y_local
        dy = s * x_local + c * y_local

        xg = x0 + dx
        yg = y0 + dy

        return xg, yg

    # ============================================================
    # Label filtering
    # ============================================================

    @staticmethod
    def normalize_label(label: str) -> str:
        cls = str(label).strip().lower()
        cls = cls.replace(" ", "_")
        cls = cls.replace("-", "_")
        return cls

    @staticmethod
    def is_buoy_label(label: str) -> bool:
        cls = ZedBuoyObstacleMapPublisher.normalize_label(label)

        if "buoy" in cls:
            return True

        if cls in ("red", "green", "yellow", "black"):
            return True

        return False

    # ============================================================
    # ZED detection callback
    # ============================================================

    def zed_detection_callback(self, msg: ZedDetection):
        """
        ZED custom detections.

        Input position is assumed LOCAL NWU:
            x_loc = forward
            y_loc = left
            z_loc = up

        Convert to LOCAL/BODY NED:
            x_local_ned =  x_loc
            y_local_ned = -y_loc
            z_local_ned = -z_loc

        Then convert LOCAL/BODY NED -> GLOBAL NED using vehicle pose.
        """

        if not self.have_pose:
            now = time.time()

            if (now - self.last_no_pose_warn_time) >= self.no_pose_warn_period:
                self.last_no_pose_warn_time = now
                self.get_logger().warn(
                    "No vehicle pose yet. Cannot convert ZED detections to global NED."
                )

            return

        if not self.is_finite(self.x_usv_ned, self.y_usv_ned, self.psi_usv_ned):
            self.get_logger().warn(
                "Current stored vehicle pose is invalid. Clearing pose flag."
            )
            self.have_pose = False
            return

        n = min(
            len(msg.class_name),
            len(msg.confidence),
            len(msg.x_loc),
            len(msg.y_loc),
            len(msg.z_loc),
        )

        if n == 0:
            return

        now = time.time()
        accepted_count = 0

        for i in range(n):
            raw_cls = str(msg.class_name[i])
            cls = self.normalize_label(raw_cls)

            try:
                conf = float(msg.confidence[i])
                x_nwu = float(msg.x_loc[i])
                y_nwu = float(msg.y_loc[i])
                z_nwu = float(msg.z_loc[i])
            except Exception:
                self.warn_invalid_detection_limited(
                    f"Skipping detection with non-convertible values at index {i}."
                )
                continue

            if not self.is_buoy_label(cls):
                continue

            if not self.is_finite(conf, x_nwu, y_nwu, z_nwu):
                self.warn_invalid_detection_limited(
                    f"Skipping invalid ZED detection: "
                    f"class={cls}, conf={conf}, x={x_nwu}, y={y_nwu}, z={z_nwu}"
                )
                continue

            if conf < self.min_confidence:
                continue

            # --------------------------------------------------------
            # ZED LOCAL NWU
            # --------------------------------------------------------
            # x_nwu = forward
            # y_nwu = left
            # z_nwu = up

            # --------------------------------------------------------
            # LOCAL NWU -> LOCAL/BODY NED
            # --------------------------------------------------------
            # x_ned = forward
            # y_ned = right
            # z_ned = down
            x_local_ned = x_nwu
            y_local_ned = -y_nwu
            z_local_ned = -z_nwu

            if not self.is_finite(x_local_ned, y_local_ned, z_local_ned):
                self.warn_invalid_detection_limited(
                    f"Skipping invalid local NED detection: "
                    f"class={cls}, x={x_local_ned}, y={y_local_ned}, z={z_local_ned}"
                )
                continue

            rng = math.sqrt(
                x_local_ned * x_local_ned + y_local_ned * y_local_ned
            )

            if not self.is_finite(rng):
                self.warn_invalid_detection_limited(
                    f"Skipping detection with invalid range: class={cls}, range={rng}"
                )
                continue

            if rng < self.min_range_m or rng > self.max_range_m:
                continue

            # Only use detections in front of the boat.
            if x_local_ned <= self.body_ahead_min_m:
                continue

            # --------------------------------------------------------
            # LOCAL/BODY NED -> GLOBAL NED
            # --------------------------------------------------------
            xg, yg = self.local_ned_to_global_ned(
                x_local=x_local_ned,
                y_local=y_local_ned,
                x0=self.x_usv_ned,
                y0=self.y_usv_ned,
                psi=self.psi_usv_ned,
            )

            if not self.is_finite(xg, yg, z_local_ned):
                self.warn_invalid_detection_limited(
                    f"Skipping detection because global NED became invalid: "
                    f"class={cls}, N={xg}, E={yg}, z={z_local_ned}"
                )
                continue

            self.update_or_create_track(
                class_name=cls,
                xg=xg,
                yg=yg,
                zg=z_local_ned,
                confidence=conf,
                stamp=now,
            )

            accepted_count += 1

        if accepted_count > 0:
            if (now - self.last_detection_log_time) >= self.detection_log_period:
                self.last_detection_log_time = now
                self.get_logger().info(
                    f"Accepted {accepted_count} ZED buoy detections"
                )

    # ============================================================
    # Tracking helpers
    # ============================================================

    def update_or_create_track(
        self,
        class_name: str,
        xg: float,
        yg: float,
        zg: float,
        confidence: float,
        stamp: float,
    ):
        if not self.is_finite(xg, yg, zg, confidence):
            self.warn_invalid_track_limited(
                f"Rejecting invalid track update: class={class_name}, "
                f"N={xg}, E={yg}, z={zg}, conf={confidence}"
            )
            return

        best_idx: Optional[int] = None
        best_dist = float("inf")

        for idx, trk in enumerate(self.tracks):
            if str(trk["class_name"]) != str(class_name):
                continue

            if not self.is_finite(trk["x"], trk["y"]):
                continue

            d = math.hypot(xg - float(trk["x"]), yg - float(trk["y"]))

            if not self.is_finite(d):
                continue

            if d < best_dist:
                best_dist = d
                best_idx = idx

        if best_idx is not None and best_dist <= self.track_merge_distance_m:
            trk = self.tracks[best_idx]

            a = self.track_alpha

            old_x = float(trk["x"])
            old_y = float(trk["y"])
            old_z = float(trk["z"])

            new_x = (1.0 - a) * old_x + a * float(xg)
            new_y = (1.0 - a) * old_y + a * float(yg)
            new_z = (1.0 - a) * old_z + a * float(zg)

            if not self.is_finite(new_x, new_y, new_z):
                self.warn_invalid_track_limited(
                    f"Rejecting smoothed invalid track: class={class_name}, "
                    f"N={new_x}, E={new_y}, z={new_z}"
                )
                return

            trk["x"] = new_x
            trk["y"] = new_y
            trk["z"] = new_z
            trk["confidence"] = max(float(confidence), float(trk["confidence"]))
            trk["last_seen"] = stamp
            trk["hits"] = int(trk["hits"]) + 1

        else:
            self.tracks.append(
                {
                    "id": self.next_track_id,
                    "class_name": class_name,
                    "x": float(xg),
                    "y": float(yg),
                    "z": float(zg),
                    "confidence": float(confidence),
                    "last_seen": stamp,
                    "hits": 1,
                }
            )

            self.next_track_id += 1

    def remove_stale_tracks(self):
        now = time.time()
        clean_tracks = []

        for trk in self.tracks:
            try:
                age = now - float(trk["last_seen"])
            except Exception:
                continue

            if age > self.track_timeout_s:
                continue

            if not self.is_finite(trk["x"], trk["y"], trk["z"]):
                self.warn_invalid_track_limited(
                    f"Removing invalid buoy track id={trk.get('id', -1)} "
                    f"class={trk.get('class_name', 'unknown')} "
                    f"N={trk.get('x')}, E={trk.get('y')}, z={trk.get('z')}"
                )
                continue

            clean_tracks.append(trk)

        self.tracks = clean_tracks

    # ============================================================
    # Occupancy grid helpers
    # ============================================================

    def get_grid_origin(self) -> Tuple[float, float]:
        """
        Local map window centered around the boat.

        Origin is in GLOBAL NED coordinates.
        """

        if not self.is_finite(self.x_usv_ned, self.y_usv_ned):
            return 0.0, 0.0

        origin_x = self.x_usv_ned - 0.5 * self.width_m
        origin_y = self.y_usv_ned - 0.5 * self.height_m

        return origin_x, origin_y

    def world_to_cell(
        self,
        x: float,
        y: float,
        origin_x: float,
        origin_y: float,
    ) -> Optional[Tuple[int, int]]:
        if not self.is_finite(x, y, origin_x, origin_y, self.resolution):
            return None

        if self.resolution <= 0.0:
            return None

        gx_f = (float(x) - float(origin_x)) / self.resolution
        gy_f = (float(y) - float(origin_y)) / self.resolution

        if not self.is_finite(gx_f, gy_f):
            return None

        gx = int(gx_f)
        gy = int(gy_f)

        if gx < 0 or gy < 0:
            return None

        if gx >= self.width or gy >= self.height:
            return None

        return gx, gy

    def cell_center_world(
        self,
        gx: int,
        gy: int,
        origin_x: float,
        origin_y: float,
    ) -> Tuple[float, float]:
        x = origin_x + (gx + 0.5) * self.resolution
        y = origin_y + (gy + 0.5) * self.resolution

        return x, y

    def fill_circle(
        self,
        data: List[int],
        cx: float,
        cy: float,
        radius: float,
        origin_x: float,
        origin_y: float,
    ):
        if not self.is_finite(cx, cy, radius, origin_x, origin_y):
            return

        if radius <= 0.0:
            return

        center_cell = self.world_to_cell(
            x=cx,
            y=cy,
            origin_x=origin_x,
            origin_y=origin_y,
        )

        if center_cell is None:
            return

        cgx, cgy = center_cell
        r_cells = int(math.ceil(radius / self.resolution))

        gx_min = max(0, cgx - r_cells)
        gx_max = min(self.width - 1, cgx + r_cells)
        gy_min = max(0, cgy - r_cells)
        gy_max = min(self.height - 1, cgy + r_cells)

        r2 = radius * radius

        for gy in range(gy_min, gy_max + 1):
            for gx in range(gx_min, gx_max + 1):
                wx, wy = self.cell_center_world(
                    gx=gx,
                    gy=gy,
                    origin_x=origin_x,
                    origin_y=origin_y,
                )

                if not self.is_finite(wx, wy):
                    continue

                dx = wx - cx
                dy = wy - cy

                if dx * dx + dy * dy <= r2:
                    idx = gy * self.width + gx

                    if 0 <= idx < len(data):
                        data[idx] = 100

    # ============================================================
    # Message builders
    # ============================================================

    def build_grid_msg(self) -> OccupancyGrid:
        origin_x, origin_y = self.get_grid_origin()

        msg = OccupancyGrid()

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id

        msg.info.resolution = self.resolution
        msg.info.width = self.width
        msg.info.height = self.height

        msg.info.origin.position.x = float(origin_x)
        msg.info.origin.position.y = float(origin_y)
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0

        # 0   = free
        # 100 = occupied
        data = [0 for _ in range(self.width * self.height)]

        for trk in self.tracks:
            if int(trk["hits"]) < self.min_hits_to_publish:
                continue

            if not self.is_finite(trk["x"], trk["y"], trk["z"]):
                continue

            self.fill_circle(
                data=data,
                cx=float(trk["x"]),
                cy=float(trk["y"]),
                radius=self.obstacle_radius_m,
                origin_x=origin_x,
                origin_y=origin_y,
            )

        msg.data = data

        return msg

    def color_for_class(self, class_name: str) -> Tuple[float, float, float]:
        cls = self.normalize_label(class_name)

        if "red" in cls:
            return 1.0, 0.0, 0.0

        if "green" in cls:
            return 0.0, 1.0, 0.0

        if "yellow" in cls:
            return 1.0, 1.0, 0.0

        return 1.0, 1.0, 1.0

    def build_marker_msg(self) -> MarkerArray:
        marker_array = MarkerArray()
        now_msg = self.get_clock().now().to_msg()

        # Clear old RViz markers first.
        clear_marker = Marker()
        clear_marker.header.stamp = now_msg
        clear_marker.header.frame_id = self.frame_id
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)

        marker_id = 0

        for trk in self.tracks:
            if int(trk["hits"]) < self.min_hits_to_publish:
                continue

            if not self.is_finite(trk["x"], trk["y"], trk["z"]):
                continue

            r, g, b = self.color_for_class(str(trk["class_name"]))

            sphere = Marker()
            sphere.header.stamp = now_msg
            sphere.header.frame_id = self.frame_id
            sphere.ns = "zed_buoy_obstacles"
            sphere.id = marker_id
            marker_id += 1

            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD

            sphere.pose.position.x = float(trk["x"])
            sphere.pose.position.y = float(trk["y"])
            sphere.pose.position.z = 0.25
            sphere.pose.orientation.w = 1.0

            diameter = 2.0 * self.obstacle_radius_m
            sphere.scale.x = diameter
            sphere.scale.y = diameter
            sphere.scale.z = 0.35

            sphere.color.r = r
            sphere.color.g = g
            sphere.color.b = b
            sphere.color.a = 0.85

            marker_array.markers.append(sphere)

            text = Marker()
            text.header.stamp = now_msg
            text.header.frame_id = self.frame_id
            text.ns = "zed_buoy_obstacle_labels"
            text.id = marker_id
            marker_id += 1

            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD

            text.pose.position.x = float(trk["x"])
            text.pose.position.y = float(trk["y"])
            text.pose.position.z = 1.0
            text.pose.orientation.w = 1.0

            text.scale.z = 0.45

            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.color.a = 1.0

            text.text = (
                f"{trk['class_name']} id={trk['id']}\n"
                f"N={float(trk['x']):.2f}, E={float(trk['y']):.2f}\n"
                f"conf={float(trk['confidence']):.2f}, hits={int(trk['hits'])}"
            )

            marker_array.markers.append(text)

        return marker_array

    # ============================================================
    # Main publisher loop
    # ============================================================

    def publish_map_and_markers(self):
        if not self.have_pose:
            return

        if not self.is_finite(self.x_usv_ned, self.y_usv_ned, self.psi_usv_ned):
            self.get_logger().warn(
                "Stored vehicle pose became invalid. Clearing pose flag."
            )
            self.have_pose = False
            return

        self.remove_stale_tracks()

        grid_msg = self.build_grid_msg()
        marker_msg = self.build_marker_msg()

        self.map_pub.publish(grid_msg)
        self.marker_pub.publish(marker_msg)

        now = time.time()

        valid_tracks = [
            trk
            for trk in self.tracks
            if int(trk["hits"]) >= self.min_hits_to_publish
            and self.is_finite(trk["x"], trk["y"], trk["z"])
        ]

        if len(valid_tracks) > 0:
            if (now - self.last_pub_log_time) >= self.pub_log_period:
                self.last_pub_log_time = now

                track_str = " | ".join(
                    [
                        f"{trk['class_name']} id={trk['id']} "
                        f"N={float(trk['x']):.2f}, E={float(trk['y']):.2f}"
                        for trk in valid_tracks
                    ]
                )

                self.get_logger().info(
                    f"Published {len(valid_tracks)} ZED buoy obstacles in GLOBAL NED: "
                    f"{track_str}"
                )


def main(args=None):
    rclpy.init(args=args)

    node = ZedBuoyObstacleMapPublisher()

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()