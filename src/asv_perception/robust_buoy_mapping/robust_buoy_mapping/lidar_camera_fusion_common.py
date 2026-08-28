#!/usr/bin/env python3

"""Shared utilities for camera + LiDAR buoy fusion.

Frame convention used by this package:
    Input detections from the camera and LiDAR are assumed to be sensor-frame NWU/FLU:
        x = forward from the sensor
        y = left from the sensor
        z = up from the sensor

    For compatibility with the existing mapper, the fusion feature frame is the
    GPS/body horizontal frame used by the dynamic KF mapper:
        x_body = forward from the GPS antenna
        y_body = right from the GPS antenna

    Therefore, NWU/FLU detections are converted by flipping y exactly once:
        x_body = x_sensor + sensor_x_offset
        y_body = -y_sensor + sensor_y_offset

    map x = North [m]
    map y = East  [m]

The GPS antenna is treated as the body-frame origin.  The default offsets are:
    camera: +26 in = +0.6604 m along body x
    LiDAR:  +20 in = +0.5080 m along body x
"""

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from sensor_msgs_py import point_cloud2
except Exception:  # pragma: no cover - handled at runtime inside ROS
    point_cloud2 = None

from robust_buoy_mapping.dynamic_kf_buoy_mapper_node import (
    arr_len,
    arr_val,
    clamp_angle,
    color_from_class,
    confidence_percent,
    finite,
)

INCH_TO_M = 0.0254
CAMERA_X_OFFSET_M = 26.0 * INCH_TO_M
LIDAR_X_OFFSET_M = 20.0 * INCH_TO_M

FUSION_FEATURES = [
    "color_code",
    "camera_confidence",
    "camera_x_body_m",
    "camera_y_body_m",
    "camera_range_m",
    "camera_bearing_rad",
    "lidar_has_cluster",
    "lidar_x_body_m",
    "lidar_y_body_m",
    "lidar_range_m",
    "lidar_bearing_rad",
    "lidar_point_count",
    "lidar_spread_x_m",
    "lidar_spread_y_m",
    "lidar_spread_r_m",
    "lidar_radius_m",
    "cam_lidar_dx_m",
    "cam_lidar_dy_m",
    "cam_lidar_dist_m",
    "time_delta_s",
    "vehicle_yaw_rad",
    "raw_fused_range_m",
    "raw_fused_bearing_rad",
]

CSV_COLUMNS = [
    "ros_time_s",
    "camera_msg_age_s",
    "lidar_msg_age_s",
    "lidar_source",
    "det_idx",
    "color",
    "class_name",
    "camera_confidence",
    "camera_x_body_m",
    "camera_y_body_m",
    "camera_range_m",
    "camera_bearing_rad",
    "camera_map_x_m",
    "camera_map_y_m",
    "lidar_has_cluster",
    "lidar_x_body_m",
    "lidar_y_body_m",
    "lidar_z_body_m",
    "lidar_range_m",
    "lidar_bearing_rad",
    "lidar_point_count",
    "lidar_spread_x_m",
    "lidar_spread_y_m",
    "lidar_spread_z_m",
    "lidar_spread_r_m",
    "lidar_radius_m",
    "cam_lidar_dx_m",
    "cam_lidar_dy_m",
    "cam_lidar_dist_m",
    "raw_fused_x_body_m",
    "raw_fused_y_body_m",
    "raw_fused_range_m",
    "raw_fused_bearing_rad",
    "raw_fused_map_x_m",
    "raw_fused_map_y_m",
    "vehicle_x_m",
    "vehicle_y_m",
    "vehicle_yaw_rad",
    "time_delta_s",
] + FUSION_FEATURES
CSV_COLUMNS = list(dict.fromkeys(CSV_COLUMNS))


def as_bool_param(value: Any) -> bool:
    """Parse ROS launch parameters that may arrive as bools or strings."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(value)


def color_code(color: str) -> int:
    return 1 if str(color).lower() == "green" else 0


def rotate_xy(x: float, y: float, yaw: float) -> Tuple[float, float]:
    c = math.cos(float(yaw))
    s = math.sin(float(yaw))
    return c * float(x) - s * float(y), s * float(x) + c * float(y)


def sensor_to_body(
    x_sensor: float,
    y_sensor: float,
    x_offset_m: float,
    y_offset_m: float = 0.0,
    yaw_offset_rad: float = 0.0,
    y_is_left: bool = True,
) -> Tuple[float, float]:
    """Convert a 2D sensor-frame measurement to the GPS/body mapper frame.

    Your camera and LiDAR detections are NWU/FLU in their own sensor frames:
        +x forward, +y left, +z up.

    The existing mapping code uses the GPS/body horizontal frame with +x forward
    and +y right.  With y_is_left=True, this function converts NWU/FLU to the
    mapper-compatible body frame by flipping y once before applying yaw and
    translation.  Do not set y_is_left=False unless the detection is already in
    +y-right body coordinates.
    """
    x = float(x_sensor)
    y_right = -float(y_sensor) if y_is_left else float(y_sensor)
    xr, yr = rotate_xy(x, y_right, float(yaw_offset_rad))
    return float(xr + x_offset_m), float(yr + y_offset_m)


def body_to_map(
    x_body: float,
    y_body: float,
    vehicle_x_m: float,
    vehicle_y_m: float,
    vehicle_yaw_rad: float,
) -> Tuple[float, float]:
    xb, yb = float(x_body), float(y_body)
    c = math.cos(float(vehicle_yaw_rad))
    s = math.sin(float(vehicle_yaw_rad))
    north = float(vehicle_x_m) + c * xb - s * yb
    east = float(vehicle_y_m) + s * xb + c * yb
    return float(north), float(east)


def map_to_body(
    map_x: float,
    map_y: float,
    vehicle_x_m: float,
    vehicle_y_m: float,
    vehicle_yaw_rad: float,
) -> Tuple[float, float]:
    dx = float(map_x) - float(vehicle_x_m)
    dy = float(map_y) - float(vehicle_y_m)
    c = math.cos(float(vehicle_yaw_rad))
    s = math.sin(float(vehicle_yaw_rad))
    xb = c * dx + s * dy
    yb = -s * dx + c * dy
    return float(xb), float(yb)


def safe_range_bearing(x_body: float, y_body: float) -> Tuple[float, float]:
    rng = math.hypot(float(x_body), float(y_body))
    brg = math.atan2(float(y_body), float(x_body))
    return float(rng), float(brg)


def extract_camera_detections(
    msg: Any,
    min_confidence: float,
    min_range_xy: float,
    max_range_xy: float,
    camera_x_offset_m: float = CAMERA_X_OFFSET_M,
    camera_y_offset_m: float = 0.0,
    camera_yaw_offset_rad: float = 0.061087,
    camera_y_is_left: bool = True,
) -> List[Dict[str, Any]]:
    dets: List[Dict[str, Any]] = []

    n = min(
        arr_len(msg, "class_name"),
        arr_len(msg, "confidence"),
        arr_len(msg, "x_loc"),
        arr_len(msg, "y_loc"),
        arr_len(msg, "z_loc"),
    )

    for i in range(n):
        cls = str(arr_val(msg, "class_name", i, "unknown"))
        color = color_from_class(cls)
        if color not in ("red", "green"):
            continue

        conf = confidence_percent(float(arr_val(msg, "confidence", i, 0.0)))
        if conf < float(min_confidence):
            continue

        x_loc = arr_val(msg, "x_loc", i, float("nan"))
        y_loc = arr_val(msg, "y_loc", i, float("nan"))
        z_loc = arr_val(msg, "z_loc", i, float("nan"))
        if not (finite(x_loc) and finite(y_loc) and finite(z_loc)):
            continue

        xb, yb = sensor_to_body(
            float(x_loc),
            float(y_loc),
            x_offset_m=float(camera_x_offset_m),
            y_offset_m=float(camera_y_offset_m),
            yaw_offset_rad=float(camera_yaw_offset_rad),
            y_is_left=bool(camera_y_is_left),
        )
        rng, bearing = safe_range_bearing(xb, yb)
        if rng < float(min_range_xy) or rng > float(max_range_xy):
            continue

        dets.append({
            "det_idx": int(i),
            "class_name": cls,
            "color": color,
            "confidence": float(conf),
            "x_body": float(xb),
            "y_body": float(yb),
            "z_camera": float(z_loc),
            "range_body": float(rng),
            "bearing_body": float(bearing),
        })

    dets.sort(key=lambda d: (-d["confidence"], d["range_body"]))
    return dets


def _point_get_xyz(point: Any) -> Optional[Tuple[float, float, float]]:
    try:
        if hasattr(point, "dtype") and getattr(point.dtype, "names", None):
            return float(point["x"]), float(point["y"]), float(point["z"])
        return float(point[0]), float(point[1]), float(point[2])
    except Exception:
        return None


def pointcloud2_to_body_array(
    cloud_msg: Any,
    lidar_x_offset_m: float = LIDAR_X_OFFSET_M,
    lidar_y_offset_m: float = 0.0,
    lidar_z_offset_m: float = 0.0,
    lidar_yaw_offset_rad: float = 0.0,
    lidar_y_is_left: bool = True,
    min_range_m: float = 0.3,
    max_range_m: float = 35.0,
    min_z_m: float = -2.0,
    max_z_m: float = 3.0,
    max_points: int = 70000,
) -> np.ndarray:
    """Return Nx3 LiDAR points transformed to body/GPS frame."""
    if cloud_msg is None or point_cloud2 is None:
        return np.zeros((0, 3), dtype=float)

    pts: List[Tuple[float, float, float]] = []
    count = 0

    try:
        iterator = point_cloud2.read_points(
            cloud_msg,
            field_names=("x", "y", "z"),
            skip_nans=True,
        )
    except Exception:
        return np.zeros((0, 3), dtype=float)

    for p in iterator:
        xyz = _point_get_xyz(p)
        if xyz is None:
            continue

        lx, ly, lz = xyz
        if not (finite(lx) and finite(ly) and finite(lz)):
            continue

        xb, yb = sensor_to_body(
            lx,
            ly,
            x_offset_m=float(lidar_x_offset_m),
            y_offset_m=float(lidar_y_offset_m),
            yaw_offset_rad=float(lidar_yaw_offset_rad),
            y_is_left=bool(lidar_y_is_left),
        )
        zb = float(lz) + float(lidar_z_offset_m)
        rng = math.hypot(xb, yb)

        if rng < float(min_range_m) or rng > float(max_range_m):
            continue
        if zb < float(min_z_m) or zb > float(max_z_m):
            continue

        pts.append((float(xb), float(yb), float(zb)))
        count += 1
        if count >= int(max_points):
            break

    if not pts:
        return np.zeros((0, 3), dtype=float)
    return np.asarray(pts, dtype=float)



def empty_lidar_cluster() -> Dict[str, Any]:
    return {
        "has_cluster": False,
        "x_body": 0.0,
        "y_body": 0.0,
        "z_body": 0.0,
        "range_body": 0.0,
        "bearing_body": 0.0,
        "point_count": 0,
        "spread_x": 0.0,
        "spread_y": 0.0,
        "spread_z": 0.0,
        "spread_r": 0.0,
        "radius_m": 0.0,
        "cam_lidar_dx": 0.0,
        "cam_lidar_dy": 0.0,
        "cam_lidar_dist": 999.0,
    }


def _seq_len(msg: Any, field: str) -> int:
    try:
        return len(getattr(msg, field))
    except Exception:
        return 0


def _seq_val(msg: Any, field: str, idx: int, default: Any = 0.0) -> Any:
    try:
        seq = getattr(msg, field)
        if idx < len(seq):
            return seq[idx]
    except Exception:
        pass
    return default


def _lidar_detection_to_body(
    x_sensor: float,
    y_sensor: float,
    z_sensor: float,
    radius_m: float = 0.0,
    name: str = "BUOY",
    lidar_x_offset_m: float = LIDAR_X_OFFSET_M,
    lidar_y_offset_m: float = 0.0,
    lidar_z_offset_m: float = 0.0,
    lidar_yaw_offset_rad: float = 0.0,
    lidar_y_is_left: bool = True,
) -> Optional[Dict[str, Any]]:
    if not (finite(x_sensor) and finite(y_sensor) and finite(z_sensor)):
        return None

    xb, yb = sensor_to_body(
        float(x_sensor),
        float(y_sensor),
        x_offset_m=float(lidar_x_offset_m),
        y_offset_m=float(lidar_y_offset_m),
        yaw_offset_rad=float(lidar_yaw_offset_rad),
        y_is_left=bool(lidar_y_is_left),
    )
    zb = float(z_sensor) + float(lidar_z_offset_m)
    rng, bearing = safe_range_bearing(xb, yb)
    r = float(radius_m) if finite(radius_m) else 0.0
    r = max(0.0, r)

    # Your detector publishes one centroid per buoy.  We map its radius into the
    # same feature slots used earlier by the point-cloud association version.
    return {
        "has_cluster": True,
        "name": str(name),
        "x_body": float(xb),
        "y_body": float(yb),
        "z_body": float(zb),
        "range_body": float(rng),
        "bearing_body": float(bearing),
        "point_count": 1,
        "spread_x": float(2.0 * r) if r > 0.0 else 0.0,
        "spread_y": float(2.0 * r) if r > 0.0 else 0.0,
        "spread_z": 0.0,
        "spread_r": float(r),
        "radius_m": float(r),
        "cam_lidar_dx": 0.0,
        "cam_lidar_dy": 0.0,
        "cam_lidar_dist": 999.0,
    }


def extract_lidar_detections_from_fau_objects(
    msg: Any,
    lidar_x_offset_m: float = LIDAR_X_OFFSET_M,
    lidar_y_offset_m: float = 0.0,
    lidar_z_offset_m: float = 0.0,
    lidar_yaw_offset_rad: float = 0.0,
    lidar_y_is_left: bool = True,
    min_range_m: float = 0.3,
    max_range_m: float = 35.0,
    min_z_m: float = -5.0,
    max_z_m: float = 3.0,
) -> List[Dict[str, Any]]:
    """Extract LiDAR buoy centroids from fau_msgs/msg/ObjectPosition.

    Expected fields from your detector:
        object_names[], x_object[], y_object[], z_object[], radii_object[]
    """
    n = min(
        _seq_len(msg, "object_names"),
        _seq_len(msg, "x_object"),
        _seq_len(msg, "y_object"),
        _seq_len(msg, "z_object"),
    )
    out: List[Dict[str, Any]] = []
    for i in range(n):
        det = _lidar_detection_to_body(
            _seq_val(msg, "x_object", i, float("nan")),
            _seq_val(msg, "y_object", i, float("nan")),
            _seq_val(msg, "z_object", i, float("nan")),
            radius_m=float(_seq_val(msg, "radii_object", i, 0.0)),
            name=str(_seq_val(msg, "object_names", i, f"BUOY[{i}]")),
            lidar_x_offset_m=lidar_x_offset_m,
            lidar_y_offset_m=lidar_y_offset_m,
            lidar_z_offset_m=lidar_z_offset_m,
            lidar_yaw_offset_rad=lidar_yaw_offset_rad,
            lidar_y_is_left=lidar_y_is_left,
        )
        if det is None:
            continue
        if det["range_body"] < float(min_range_m) or det["range_body"] > float(max_range_m):
            continue
        if det["z_body"] < float(min_z_m) or det["z_body"] > float(max_z_m):
            continue
        out.append(det)
    return out


def extract_lidar_detections_from_buoy_detected(
    msg: Any,
    lidar_x_offset_m: float = LIDAR_X_OFFSET_M,
    lidar_y_offset_m: float = 0.0,
    lidar_z_offset_m: float = 0.0,
    lidar_yaw_offset_rad: float = 0.0,
    lidar_y_is_left: bool = True,
    min_range_m: float = 0.3,
    max_range_m: float = 35.0,
    min_z_m: float = -5.0,
    max_z_m: float = 3.0,
) -> List[Dict[str, Any]]:
    """Extract LiDAR buoy centroids from lidar_msgs/msg/BuoyDetected.

    Expected fields from your detector:
        name[], x[], y[], z[]
    """
    n = min(
        _seq_len(msg, "name"),
        _seq_len(msg, "x"),
        _seq_len(msg, "y"),
        _seq_len(msg, "z"),
    )
    out: List[Dict[str, Any]] = []
    for i in range(n):
        det = _lidar_detection_to_body(
            _seq_val(msg, "x", i, float("nan")),
            _seq_val(msg, "y", i, float("nan")),
            _seq_val(msg, "z", i, float("nan")),
            radius_m=0.0,
            name=str(_seq_val(msg, "name", i, f"BUOY[{i}]")),
            lidar_x_offset_m=lidar_x_offset_m,
            lidar_y_offset_m=lidar_y_offset_m,
            lidar_z_offset_m=lidar_z_offset_m,
            lidar_yaw_offset_rad=lidar_yaw_offset_rad,
            lidar_y_is_left=lidar_y_is_left,
        )
        if det is None:
            continue
        if det["range_body"] < float(min_range_m) or det["range_body"] > float(max_range_m):
            continue
        if det["z_body"] < float(min_z_m) or det["z_body"] > float(max_z_m):
            continue
        out.append(det)
    return out


def _copy_lidar_match_with_camera_delta(
    lidar_det: Dict[str, Any],
    camera_x_body: float,
    camera_y_body: float,
    match_dist: Optional[float] = None,
) -> Dict[str, Any]:
    """Return a LiDAR detection copy with camera-relative association features."""
    cluster = dict(lidar_det)
    cdx = float(cluster.get("x_body", 0.0)) - float(camera_x_body)
    cdy = float(cluster.get("y_body", 0.0)) - float(camera_y_body)
    cluster["cam_lidar_dx"] = float(cdx)
    cluster["cam_lidar_dy"] = float(cdy)
    cluster["cam_lidar_dist"] = float(math.hypot(cdx, cdy) if match_dist is None else match_dist)
    return cluster


def associate_lidar_detection_near_camera(
    lidar_detections_body: Sequence[Dict[str, Any]],
    camera_x_body: float,
    camera_y_body: float,
    association_radius_m: float = 1.8,
) -> Dict[str, Any]:
    """Return the closest LiDAR buoy detection to the camera detection.

    This is kept for backward compatibility.  For close buoy fields, prefer
    associate_lidar_detections_one_to_one() because nearest-neighbor matching
    can assign the same LiDAR buoy to multiple camera detections.
    """
    if not lidar_detections_body:
        return empty_lidar_cluster()

    best: Optional[Dict[str, Any]] = None
    best_dist = float("inf")
    for det in lidar_detections_body:
        if not det.get("has_cluster", False):
            continue
        dx = float(det.get("x_body", 0.0)) - float(camera_x_body)
        dy = float(det.get("y_body", 0.0)) - float(camera_y_body)
        d = math.hypot(dx, dy)
        if d < best_dist:
            best_dist = d
            best = det

    if best is None or best_dist > float(association_radius_m):
        return empty_lidar_cluster()

    return _copy_lidar_match_with_camera_delta(best, camera_x_body, camera_y_body, best_dist)


def associate_lidar_detections_one_to_one(
    camera_detections_body: Sequence[Dict[str, Any]],
    lidar_detections_body: Sequence[Dict[str, Any]],
    association_radius_m: float = 1.2,
    prefer_same_color: bool = False,
) -> List[Dict[str, Any]]:
    """Associate camera detections to LiDAR detections with one-to-one matching.

    The previous logger associated each camera detection independently to the
    nearest LiDAR detection.  With two close buoys, that can reuse the same
    LiDAR centroid or swap buoy A/B.  This function builds all valid
    camera-LiDAR pairs inside association_radius_m, sorts by cost, and greedily
    assigns each camera and each LiDAR detection at most once.

    Returns a list of LiDAR cluster dictionaries aligned with
    camera_detections_body.  Unmatched camera detections receive
    empty_lidar_cluster().
    """
    n_cam = len(camera_detections_body)
    matches: List[Dict[str, Any]] = [empty_lidar_cluster() for _ in range(n_cam)]
    if n_cam == 0 or not lidar_detections_body:
        return matches

    candidates: List[Tuple[float, float, int, int]] = []
    gate = float(association_radius_m)

    for ci, cam in enumerate(camera_detections_body):
        cx = float(cam.get("x_body", 0.0))
        cy = float(cam.get("y_body", 0.0))
        cam_color = str(cam.get("color", "")).lower()

        for li, lidar in enumerate(lidar_detections_body):
            if not lidar.get("has_cluster", False):
                continue
            lx = float(lidar.get("x_body", 0.0))
            ly = float(lidar.get("y_body", 0.0))
            d = math.hypot(lx - cx, ly - cy)
            if d > gate:
                continue

            # Most LiDAR detections are geometry-only and have no color.  This
            # penalty only matters if a future detector adds color labels.
            lidar_color = str(lidar.get("color", "")).lower()
            color_penalty = 0.0
            if prefer_same_color and lidar_color and cam_color and lidar_color != cam_color:
                color_penalty = gate

            cost = d + color_penalty
            candidates.append((float(cost), float(d), ci, li))

    candidates.sort(key=lambda item: (item[0], item[1]))

    used_cameras = set()
    used_lidars = set()
    for _cost, dist, ci, li in candidates:
        if ci in used_cameras or li in used_lidars:
            continue
        cam = camera_detections_body[ci]
        matches[ci] = _copy_lidar_match_with_camera_delta(
            lidar_detections_body[li],
            float(cam.get("x_body", 0.0)),
            float(cam.get("y_body", 0.0)),
            float(dist),
        )
        used_cameras.add(ci)
        used_lidars.add(li)

    return matches

def lidar_cluster_near_camera(
    points_body: np.ndarray,
    camera_x_body: float,
    camera_y_body: float,
    association_radius_m: float = 1.8,
    min_cluster_points: int = 4,
    max_cluster_points: int = 250,
) -> Dict[str, Any]:
    empty = empty_lidar_cluster()

    if points_body is None or len(points_body) == 0:
        return empty

    px = points_body[:, 0]
    py = points_body[:, 1]
    dx = px - float(camera_x_body)
    dy = py - float(camera_y_body)
    dist = np.hypot(dx, dy)
    mask = dist <= float(association_radius_m)
    idx = np.nonzero(mask)[0]

    if idx.size < int(min_cluster_points):
        return empty

    # Keep the closest points to avoid including the water surface or other obstacles.
    if idx.size > int(max_cluster_points):
        order = np.argsort(dist[idx])[: int(max_cluster_points)]
        idx = idx[order]

    cluster = points_body[idx, :]
    cx = float(np.median(cluster[:, 0]))
    cy = float(np.median(cluster[:, 1]))
    cz = float(np.median(cluster[:, 2]))

    sx = float(np.percentile(cluster[:, 0], 90) - np.percentile(cluster[:, 0], 10)) if len(cluster) > 1 else 0.0
    sy = float(np.percentile(cluster[:, 1], 90) - np.percentile(cluster[:, 1], 10)) if len(cluster) > 1 else 0.0
    sz = float(np.percentile(cluster[:, 2], 90) - np.percentile(cluster[:, 2], 10)) if len(cluster) > 1 else 0.0
    sr = float(math.hypot(sx, sy))
    rng, bearing = safe_range_bearing(cx, cy)
    cdx = float(cx - float(camera_x_body))
    cdy = float(cy - float(camera_y_body))
    cdist = float(math.hypot(cdx, cdy))

    return {
        "has_cluster": True,
        "x_body": cx,
        "y_body": cy,
        "z_body": cz,
        "range_body": float(rng),
        "bearing_body": float(bearing),
        "point_count": int(len(cluster)),
        "spread_x": sx,
        "spread_y": sy,
        "spread_z": sz,
        "spread_r": sr,
        "radius_m": sr,
        "cam_lidar_dx": cdx,
        "cam_lidar_dy": cdy,
        "cam_lidar_dist": cdist,
    }


def build_fusion_candidate(
    camera_det: Dict[str, Any],
    lidar_cluster: Dict[str, Any],
    vehicle_x_m: float,
    vehicle_y_m: float,
    vehicle_yaw_rad: float,
    now_s: float,
    camera_msg_age_s: float,
    lidar_msg_age_s: float,
    lidar_blend_weight: float = 0.75,
) -> Dict[str, Any]:
    cam_x = float(camera_det["x_body"])
    cam_y = float(camera_det["y_body"])
    cam_range = float(camera_det["range_body"])
    cam_bearing = float(camera_det["bearing_body"])
    cam_map_x, cam_map_y = body_to_map(cam_x, cam_y, vehicle_x_m, vehicle_y_m, vehicle_yaw_rad)

    has_lidar = bool(lidar_cluster.get("has_cluster", False))
    if has_lidar:
        lx = float(lidar_cluster["x_body"])
        ly = float(lidar_cluster["y_body"])
        w = float(np.clip(lidar_blend_weight, 0.0, 1.0))
        raw_x = w * lx + (1.0 - w) * cam_x
        raw_y = w * ly + (1.0 - w) * cam_y
    else:
        lx = 0.0
        ly = 0.0
        raw_x = cam_x
        raw_y = cam_y

    raw_range, raw_bearing = safe_range_bearing(raw_x, raw_y)
    raw_map_x, raw_map_y = body_to_map(raw_x, raw_y, vehicle_x_m, vehicle_y_m, vehicle_yaw_rad)
    time_delta_s = abs(float(camera_msg_age_s) - float(lidar_msg_age_s))

    feature_values = {
        "color_code": float(color_code(camera_det["color"])),
        "camera_confidence": float(camera_det["confidence"]),
        "camera_x_body_m": cam_x,
        "camera_y_body_m": cam_y,
        "camera_range_m": cam_range,
        "camera_bearing_rad": cam_bearing,
        "lidar_has_cluster": 1.0 if has_lidar else 0.0,
        "lidar_x_body_m": float(lidar_cluster.get("x_body", 0.0)) if has_lidar else 0.0,
        "lidar_y_body_m": float(lidar_cluster.get("y_body", 0.0)) if has_lidar else 0.0,
        "lidar_range_m": float(lidar_cluster.get("range_body", 0.0)) if has_lidar else 0.0,
        "lidar_bearing_rad": float(lidar_cluster.get("bearing_body", 0.0)) if has_lidar else 0.0,
        "lidar_point_count": float(lidar_cluster.get("point_count", 0)),
        "lidar_spread_x_m": float(lidar_cluster.get("spread_x", 0.0)),
        "lidar_spread_y_m": float(lidar_cluster.get("spread_y", 0.0)),
        "lidar_spread_r_m": float(lidar_cluster.get("spread_r", 0.0)),
        "lidar_radius_m": float(lidar_cluster.get("radius_m", lidar_cluster.get("spread_r", 0.0))),
        "cam_lidar_dx_m": float(lidar_cluster.get("cam_lidar_dx", 0.0)) if has_lidar else 0.0,
        "cam_lidar_dy_m": float(lidar_cluster.get("cam_lidar_dy", 0.0)) if has_lidar else 0.0,
        "cam_lidar_dist_m": float(lidar_cluster.get("cam_lidar_dist", 999.0)),
        "time_delta_s": float(time_delta_s),
        "vehicle_yaw_rad": float(clamp_angle(vehicle_yaw_rad)),
        "raw_fused_range_m": float(raw_range),
        "raw_fused_bearing_rad": float(raw_bearing),
    }

    row = {
        "ros_time_s": float(now_s),
        "camera_msg_age_s": float(camera_msg_age_s),
        "lidar_msg_age_s": float(lidar_msg_age_s),
        "det_idx": int(camera_det["det_idx"]),
        "color": str(camera_det["color"]),
        "class_name": str(camera_det["class_name"]),
        "camera_confidence": float(camera_det["confidence"]),
        "camera_x_body_m": cam_x,
        "camera_y_body_m": cam_y,
        "camera_range_m": cam_range,
        "camera_bearing_rad": cam_bearing,
        "camera_map_x_m": float(cam_map_x),
        "camera_map_y_m": float(cam_map_y),
        "lidar_has_cluster": int(has_lidar),
        "lidar_x_body_m": float(lidar_cluster.get("x_body", 0.0)) if has_lidar else 0.0,
        "lidar_y_body_m": float(lidar_cluster.get("y_body", 0.0)) if has_lidar else 0.0,
        "lidar_z_body_m": float(lidar_cluster.get("z_body", 0.0)) if has_lidar else 0.0,
        "lidar_range_m": float(lidar_cluster.get("range_body", 0.0)) if has_lidar else 0.0,
        "lidar_bearing_rad": float(lidar_cluster.get("bearing_body", 0.0)) if has_lidar else 0.0,
        "lidar_point_count": int(lidar_cluster.get("point_count", 0)),
        "lidar_spread_x_m": float(lidar_cluster.get("spread_x", 0.0)),
        "lidar_spread_y_m": float(lidar_cluster.get("spread_y", 0.0)),
        "lidar_spread_z_m": float(lidar_cluster.get("spread_z", 0.0)),
        "lidar_spread_r_m": float(lidar_cluster.get("spread_r", 0.0)),
        "lidar_radius_m": float(lidar_cluster.get("radius_m", lidar_cluster.get("spread_r", 0.0))),
        "cam_lidar_dx_m": float(lidar_cluster.get("cam_lidar_dx", 0.0)) if has_lidar else 0.0,
        "cam_lidar_dy_m": float(lidar_cluster.get("cam_lidar_dy", 0.0)) if has_lidar else 0.0,
        "cam_lidar_dist_m": float(lidar_cluster.get("cam_lidar_dist", 999.0)),
        "raw_fused_x_body_m": float(raw_x),
        "raw_fused_y_body_m": float(raw_y),
        "raw_fused_range_m": float(raw_range),
        "raw_fused_bearing_rad": float(raw_bearing),
        "raw_fused_map_x_m": float(raw_map_x),
        "raw_fused_map_y_m": float(raw_map_y),
        "vehicle_x_m": float(vehicle_x_m),
        "vehicle_y_m": float(vehicle_y_m),
        "vehicle_yaw_rad": float(clamp_angle(vehicle_yaw_rad)),
        "time_delta_s": float(time_delta_s),
    }
    row.update(feature_values)

    return row


def feature_vector_from_candidate(row: Dict[str, Any]) -> np.ndarray:
    return np.asarray([[float(row.get(k, 0.0)) for k in FUSION_FEATURES]], dtype=float)


def heuristic_fusion_sigma(row: Dict[str, Any], p_reliable: float = 0.5) -> float:
    cam_range = float(row.get("camera_range_m", 0.0))
    cam_conf = max(1.0, min(100.0, float(row.get("camera_confidence", 50.0))))
    sigma = 0.40 + 0.06 * cam_range
    sigma *= math.sqrt(75.0 / cam_conf)

    if int(row.get("lidar_has_cluster", 0)) == 1:
        sigma = min(sigma, 0.30 + 0.35 * float(row.get("lidar_spread_r_m", 0.5)) + 0.10 * float(row.get("cam_lidar_dist_m", 0.0)))
    else:
        sigma += 0.8

    sigma *= 1.0 + 0.7 * (1.0 - max(0.0, min(1.0, float(p_reliable))))
    return float(np.clip(sigma, 0.25, 4.0))