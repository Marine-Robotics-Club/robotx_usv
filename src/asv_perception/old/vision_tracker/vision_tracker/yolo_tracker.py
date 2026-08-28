#!/usr/bin/env python3
"""
buoy_counter_node_bearing_dedup.py

Option 2: ODOM/YAW-AWARE de-dup (bearing-based)

What this does:
  1) Subscribes to YOLO detections
  2) Filters only target class (default: black_buoy)
  3) Tracks with simple IoU association (no prediction drift)
  4) Counts when track centroid is inside ROI for N frames
  5) Uses USV yaw (from Odometry) + camera HFOV to compute a WORLD bearing:
        bearing_world = wrap_pi(psi_usv + bearing_cam)
     When counting, it stores (bearing_world, usv_x, usv_y, time).
     If a future detection would be counted but matches a previous counted buoy
     by bearing + distance (in meters) + time, it is NOT counted again.
  6) Subscribes to YOLO annotated image and publishes overlay:
       /black_buoy_tracker/overlay
     including ROI + bbox + centroid + track id + world-bearing + total count.

Assumptions:
- bbox coordinates are PIXELS by default. Set bboxes_are_normalized:=true if needed.
- Odometry topic provides x,y and orientation quaternion with yaw in radians.
- Camera is forward-facing and aligned with USV yaw (approx). If camera is rotated,
  use 'camera_yaw_offset_deg' to compensate.

Dependencies:
- cv_bridge, opencv-python
- nav_msgs, tf_transformations (or tf_transformations via tf_transformations package)

"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set, Deque
from collections import deque
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from std_msgs.msg import Int32
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry

from cv_bridge import CvBridge
import cv2

from tf_transformations import euler_from_quaternion

# ---------------------------------------------------------------------------
# CHANGE THIS IMPORT to your actual detections message type
# ---------------------------------------------------------------------------
from yolov26_msgs.msg import YoloDetection  # <-- CHANGE ME


def iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0.0 else 0.0


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def wrap_pi(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def ang_diff(a: float, b: float) -> float:
    """Smallest difference a-b in radians, wrapped to [-pi,pi]."""
    return wrap_pi(a - b)


@dataclass
class Track:
    track_id: int
    bbox: Tuple[float, float, float, float]  # (x1,y1,x2,y2)
    cls: str
    conf: float

    hits: int = 1
    confirmed: bool = False
    time_since_update: int = 0
    last_update_time: float = field(default_factory=time.time)

    # Counting support
    roi_streak: int = 0
    counted: bool = False

    # store last centroid to estimate motion direction if desired
    prev_cx: float = 0.0
    prev_cy: float = 0.0

    def centroid(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

    def update(self, bbox: Tuple[float, float, float, float], conf: float):
        cx, cy = 0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3])
        self.prev_cx, self.prev_cy = cx, cy
        self.bbox = bbox
        self.conf = conf
        self.hits += 1
        self.time_since_update = 0
        self.last_update_time = time.time()


@dataclass
class CountedBuoy:
    t: float
    usv_x: float
    usv_y: float
    bearing_world: float  # radians, wrapped [-pi,pi]


class BuoyCounter(Node):
    def __init__(self):
        super().__init__('black_buoy_counter_bearing_dedup')

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        # ---------------- Parameters ----------------
        self.declare_parameter('detections_topic', '/yolov26/detections_black_buoy')
        self.declare_parameter('annotated_image_topic', '/yolov26/annotated_image_black_buoy')

        # odom topic you provided pattern: f'/{asv_name}/p3d_wamv_ned'
        self.declare_parameter('asv_name', 'wamv')
        self.declare_parameter('odom_topic', '')  # if empty, auto-build from asv_name

        self.declare_parameter('target_class', 'black_buoy')
        self.declare_parameter('conf_thresh', 0.40)
        self.declare_parameter('min_area_px', 200.0)

        self.declare_parameter('iou_thresh', 0.20)
        self.declare_parameter('max_age', 5)
        self.declare_parameter('min_hits', 3)

        self.declare_parameter('bboxes_are_normalized', False)

        # ROI normalized bounds (you set these)
        self.declare_parameter('roi_y_min_norm', 0.20)
        self.declare_parameter('roi_y_max_norm', 0.50)
        self.declare_parameter('roi_x_min_norm', 0.00)
        self.declare_parameter('roi_x_max_norm', 1.00)
        self.declare_parameter('roi_required_frames', 4)

        # ---- Bearing-based de-dup params (NEW) ----
        # Camera horizontal FOV in degrees (approx ok). Many webcams ~70-90 deg.
        self.declare_parameter('camera_hfov_deg', 90.0)
        # If camera is not aligned with boat forward (e.g., mounted angled),
        # set offset (+ means camera points left of boat forward, in degrees).
        self.declare_parameter('camera_yaw_offset_deg', 0.0)

        # Consider a counted buoy "the same" if:
        # - within this bearing difference (deg)
        # - and within this USV position distance (m) at time of count vs now
        self.declare_parameter('dedup_bearing_deg', 12.0)
        self.declare_parameter('dedup_pos_m', 6.0)

        # Keep counted buoy registry for this long (seconds)
        self.declare_parameter('count_memory_seconds', 300.0)  # 5 minutes

        # ---------------- State ----------------
        self.tracks: List[Track] = []
        self.next_id: int = 1
        self.total_count: int = 0

        # counted buoy registry (bearing + position + time)
        self.counted: Deque[CountedBuoy] = deque(maxlen=2000)

        # Latest annotated image buffer
        self.bridge = CvBridge()
        self.last_img_msg: Optional[Image] = None
        self.last_img_w: int = 0
        self.last_img_h: int = 0

        # USV pose/yaw from odom
        self.usv_x: float = 0.0
        self.usv_y: float = 0.0
        self.usv_yaw: float = 0.0
        self.have_odom: bool = False

        # ---------------- ROS I/O ----------------
        det_topic = str(self.get_parameter('detections_topic').value)
        img_topic = str(self.get_parameter('annotated_image_topic').value)

        asv_name = str(self.get_parameter('asv_name').value)
        odom_topic_param = str(self.get_parameter('odom_topic').value).strip()
        if odom_topic_param:
            odom_topic = odom_topic_param
        else:
            odom_topic = f'/{asv_name}/p3d_wamv_ned'  # your topic

        self.sub_det = self.create_subscription(YoloDetection, det_topic, self.on_detections, qos)
        self.sub_img = self.create_subscription(Image, img_topic, self.on_image, qos)
        self.sub_odom = self.create_subscription(Odometry, odom_topic, self.on_odom, 10)

        self.pub_total = self.create_publisher(Int32, 'black_buoys_count_total', 10)
        self.pub_current = self.create_publisher(Int32, 'black_buoys_count_current', 10)
        self.pub_overlay = self.create_publisher(Image, 'black_buoy_tracker/overlay', 10)

        self.timer = self.create_timer(0.1, self.on_timer)

        self.get_logger().info(f"Detections: {det_topic}")
        self.get_logger().info(f"Annotated image: {img_topic}")
        self.get_logger().info(f"Odom: {odom_topic}")
        self.get_logger().info("Overlay: black_buoy_tracker/overlay")

    def target_class(self) -> str:
        return str(self.get_parameter('target_class').value)

    # ---------------- ODOM ----------------
    def on_odom(self, msg: Odometry):
        self.usv_x = float(msg.pose.pose.position.x)
        self.usv_y = float(msg.pose.pose.position.y)

        q = msg.pose.pose.orientation
        q_list = [q.x, q.y, q.z, q.w]
        _, _, yaw = euler_from_quaternion(q_list)
        self.usv_yaw = float(yaw)  # radians
        self.have_odom = True

    # ---------------- IMAGE ----------------
    def on_image(self, msg: Image):
        self.last_img_msg = msg
        if msg.width > 0 and msg.height > 0:
            self.last_img_w = int(msg.width)
            self.last_img_h = int(msg.height)

    def on_timer(self):
        tgt = self.target_class()
        current = sum(1 for t in self.tracks if t.confirmed and t.time_since_update == 0 and t.cls == tgt)
        self.pub_current.publish(Int32(data=int(current)))
        self.pub_total.publish(Int32(data=int(self.total_count)))

        # prune old counted memory
        now = time.time()
        keep_s = float(self.get_parameter('count_memory_seconds').value)
        while self.counted and (now - self.counted[0].t) > keep_s:
            self.counted.popleft()

    # ---------------- Bearing math ----------------
    def pixel_to_bearing_cam(self, cx_px: float, img_w: float) -> float:
        """
        Convert pixel x to camera-relative bearing (radians).
        cx at center => 0 rad. Left negative, right positive (convention).
        """
        hfov = math.radians(float(self.get_parameter('camera_hfov_deg').value))
        x_norm = (cx_px / max(1.0, img_w))  # 0..1
        # map to [-0.5, +0.5] then multiply by HFOV
        bearing = (x_norm - 0.5) * hfov
        # camera yaw offset
        off = math.radians(float(self.get_parameter('camera_yaw_offset_deg').value))
        return bearing + off

    def bearing_world_from_pixel(self, cx_px: float, img_w: float) -> float:
        """bearing_world = usv_yaw + bearing_cam, wrapped."""
        bearing_cam = self.pixel_to_bearing_cam(cx_px, img_w)
        return wrap_pi(self.usv_yaw + bearing_cam)

    def is_duplicate_count(self, bearing_world: float, usv_x: float, usv_y: float) -> bool:
        """Check against counted registry using bearing + position gating."""
        bearing_gate = math.radians(float(self.get_parameter('dedup_bearing_deg').value))
        pos_gate = float(self.get_parameter('dedup_pos_m').value)

        for cb in self.counted:
            if abs(ang_diff(bearing_world, cb.bearing_world)) <= bearing_gate:
                dx = usv_x - cb.usv_x
                dy = usv_y - cb.usv_y
                if math.hypot(dx, dy) <= pos_gate:
                    return True
        return False

    # ---------------- DETECTIONS ----------------
    def on_detections(self, msg: YoloDetection):
        conf_thresh = float(self.get_parameter('conf_thresh').value)
        min_area = float(self.get_parameter('min_area_px').value)
        target = self.target_class()
        iou_thresh = float(self.get_parameter('iou_thresh').value)
        min_hits = int(self.get_parameter('min_hits').value)
        max_age = int(self.get_parameter('max_age').value)
        bboxes_norm = bool(self.get_parameter('bboxes_are_normalized').value)

        # Image size for scaling if needed
        W = float(self.last_img_w if self.last_img_w > 0 else 1280.0)
        H = float(self.last_img_h if self.last_img_h > 0 else 720.0)

        # 1) Filter detections
        dets: List[Tuple[Tuple[float, float, float, float], float, str]] = []
        n = len(msg.class_name)
        for i in range(n):
            cls = msg.class_name[i]
            if cls != target:
                continue

            conf = float(msg.confidence[i])
            if conf < conf_thresh:
                continue

            x1 = float(msg.x_min[i]); y1 = float(msg.y_min[i])
            x2 = float(msg.x_max[i]); y2 = float(msg.y_max[i])

            if bboxes_norm:
                x1 *= W; x2 *= W
                y1 *= H; y2 *= H

            if x2 <= x1 or y2 <= y1:
                continue

            area = (x2 - x1) * (y2 - y1)
            if area < min_area:
                continue

            dets.append(((x1, y1, x2, y2), conf, cls))

        # 2) Age tracks (NO prediction drift)
        for t in self.tracks:
            t.time_since_update += 1

        # 3) IoU association (greedy)
        matches: List[Tuple[int, int, float]] = []
        for ti, t in enumerate(self.tracks):
            if t.cls != target:
                continue
            for di, (bbox, conf, cls) in enumerate(dets):
                val = iou_xyxy(t.bbox, bbox)
                if val >= iou_thresh:
                    matches.append((ti, di, val))

        matches.sort(key=lambda x: x[2], reverse=True)
        used_tracks: Set[int] = set()
        used_dets: Set[int] = set()
        pairs: List[Tuple[int, int]] = []

        for ti, di, _ in matches:
            if ti in used_tracks or di in used_dets:
                continue
            used_tracks.add(ti)
            used_dets.add(di)
            pairs.append((ti, di))

        # 4) Update matched tracks
        for ti, di in pairs:
            bbox, conf, cls = dets[di]
            self.tracks[ti].update(bbox, conf)
            if self.tracks[ti].hits >= min_hits:
                self.tracks[ti].confirmed = True

        # 5) New tracks for unmatched detections
        for di, (bbox, conf, cls) in enumerate(dets):
            if di in used_dets:
                continue
            tr = Track(track_id=self.next_id, bbox=bbox, cls=cls, conf=conf)
            cx, cy = tr.centroid()
            tr.prev_cx, tr.prev_cy = cx, cy
            self.next_id += 1
            self.tracks.append(tr)

        # 6) Remove dead tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= max_age]

        # 7) Counting using ROI + bearing-based dedup
        self.apply_roi_counting(W=W, H=H)

        # 8) Publish overlay
        self.publish_overlay()

    # ---------------- COUNTING ----------------
    def apply_roi_counting(self, W: float, H: float):
        rx1 = float(self.get_parameter('roi_x_min_norm').value) * W
        rx2 = float(self.get_parameter('roi_x_max_norm').value) * W
        ry1 = float(self.get_parameter('roi_y_min_norm').value) * H
        ry2 = float(self.get_parameter('roi_y_max_norm').value) * H

        required = int(self.get_parameter('roi_required_frames').value)
        now = time.time()
        target = self.target_class()

        for t in self.tracks:
            if t.cls != target or not t.confirmed or t.counted:
                continue

            # only consider recently updated tracks
            if t.time_since_update > 2:
                t.roi_streak = 0
                continue

            cx, cy = t.centroid()
            in_roi = (rx1 <= cx <= rx2) and (ry1 <= cy <= ry2)

            if in_roi:
                t.roi_streak += 1
            else:
                t.roi_streak = 0

            if t.roi_streak < required:
                continue

            # Need odom for option 2
            if not self.have_odom:
                self.get_logger().warn("No odom yet; skipping bearing-based dedup count.")
                return

            bearing_world = self.bearing_world_from_pixel(cx, W)

            # Check if this is a duplicate buoy (bearing+pos gate)
            if self.is_duplicate_count(bearing_world, self.usv_x, self.usv_y):
                t.counted = True  # stop trying to count this track again
                continue

            # Count it
            self.total_count += 1
            t.counted = True
            self.counted.append(CountedBuoy(t=now, usv_x=self.usv_x, usv_y=self.usv_y, bearing_world=bearing_world))

            self.get_logger().info(
                f"COUNTED {target} #{self.total_count} "
                f"(track_id={t.track_id}) bearing_world={math.degrees(bearing_world):.1f}deg "
                f"usv=({self.usv_x:.1f},{self.usv_y:.1f})"
            )

    # ---------------- OVERLAY ----------------
    def publish_overlay(self):
        if self.last_img_msg is None:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(self.last_img_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f"cv_bridge convert failed: {e}")
            return

        H, W = frame.shape[:2]

        # Draw ROI
        rx1 = int(float(self.get_parameter('roi_x_min_norm').value) * W)
        rx2 = int(float(self.get_parameter('roi_x_max_norm').value) * W)
        ry1 = int(float(self.get_parameter('roi_y_min_norm').value) * H)
        ry2 = int(float(self.get_parameter('roi_y_max_norm').value) * H)
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)
        cv2.putText(frame, "ROI", (rx1, max(0, ry1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        target = self.target_class()

        # Draw tracks
        for t in self.tracks:
            if t.cls != target or not t.confirmed:
                continue

            x1, y1, x2, y2 = t.bbox
            x1 = int(clamp(x1, 0, W - 1)); x2 = int(clamp(x2, 0, W - 1))
            y1 = int(clamp(y1, 0, H - 1)); y2 = int(clamp(y2, 0, H - 1))

            color = (0, 255, 0) if not t.counted else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            cx, cy = t.centroid()
            cx_i = int(clamp(cx, 0, W - 1)); cy_i = int(clamp(cy, 0, H - 1))
            cv2.circle(frame, (cx_i, cy_i), 4, color, -1)

            # world bearing display if odom is available
            if self.have_odom:
                bw = self.bearing_world_from_pixel(cx, W)
                bw_deg = math.degrees(bw)
                extra = f" bw={bw_deg:.0f}deg"
            else:
                extra = " bw=?"

            label = f"id={t.track_id} conf={t.conf:.2f} hits={t.hits} age={t.time_since_update}{extra}"
            cv2.putText(frame, label, (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Count summary
        cv2.putText(frame, f"Total counted: {self.total_count}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Boat heading (deg) if available
        if self.have_odom:
            hdg = (math.degrees(self.usv_yaw) % 360.0)
            cv2.putText(frame, f"USV yaw: {hdg:.1f} deg", (15, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        out = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        out.header = self.last_img_msg.header
        self.pub_overlay.publish(out)


def main():
    rclpy.init()
    node = BuoyCounter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
