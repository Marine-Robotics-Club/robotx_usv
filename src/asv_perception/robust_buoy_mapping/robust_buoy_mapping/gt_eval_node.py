#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import numpy as np
import os
from typing import Any, Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray


def finite(v: Any) -> bool:
    try:
        return math.isfinite(float(v))
    except Exception:
        return False


def dist(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    return math.hypot(float(a['x']) - float(b['x']), float(a['y']) - float(b['y']))


class GroundTruthEvaluatorNode(Node):
    def __init__(self):
        super().__init__('gt_eval_node')

        self.declare_parameter('gt_csv', '/home/highlevel/roboboat_vehicle_data/ground_truth/gt_buoys_test_day.csv')
        self.declare_parameter('semantic_buoys_topic', '/asv/map/semantic_buoys')
        self.declare_parameter('gt_marker_topic', '/asv/viz/gt_buoy_markers')
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('match_gate_m', 5.0)
        self.declare_parameter('publish_period_s', 0.5)
        self.declare_parameter('marker_diameter_m', 0.75)
        self.declare_parameter('error_csv', '/home/highlevel/roboboat_vehicle_data/logs/robust_mapping_error.csv')

        self.gt_csv = str(self.get_parameter('gt_csv').value)
        self.semantic_topic = str(self.get_parameter('semantic_buoys_topic').value)
        self.gt_marker_topic = str(self.get_parameter('gt_marker_topic').value)
        self.frame_id = str(self.get_parameter('frame_id').value)
        self.match_gate_m = float(self.get_parameter('match_gate_m').value)
        self.publish_period_s = float(self.get_parameter('publish_period_s').value)
        self.marker_diameter_m = float(self.get_parameter('marker_diameter_m').value)
        self.error_csv = str(self.get_parameter('error_csv').value)

        self.gt_buoys = self._load_gt_csv(self.gt_csv)
        self.latest_payload: Optional[Dict[str, Any]] = None
        self.last_log_time = 0.0

        os.makedirs(os.path.dirname(self.error_csv), exist_ok=True)
        if not os.path.exists(self.error_csv):
            with open(self.error_csv, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow([
                    'ros_time_s', 'source', 'gt_count', 'live_count', 'matched_count',
                    'mean_error_m', 'median_error_m', 'rmse_error_m', 'max_error_m',
                    'match_details'
                ])

        self.create_subscription(String, self.semantic_topic, self._semantic_cb, 10)
        self.marker_pub = self.create_publisher(MarkerArray, self.gt_marker_topic, 10)
        self.timer = self.create_timer(self.publish_period_s, self._timer_cb)

        self.get_logger().info('GT evaluator started')
        self.get_logger().info(f'gt_csv: {self.gt_csv}')
        self.get_logger().info(f'loaded GT buoys: {len(self.gt_buoys)}')
        self.get_logger().info(f'subscribing: {self.semantic_topic}')
        self.get_logger().info(f'error_csv: {self.error_csv}')

    def _load_gt_csv(self, path: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                color = str(row.get('color', '')).strip().lower()
                if color not in ('red', 'green'):
                    continue
                x = row.get('north_m', row.get('x', None))
                y = row.get('east_m', row.get('y', None))
                if not (finite(x) and finite(y)):
                    continue
                try:
                    bid = int(row.get('id', len(out) + 1))
                except Exception:
                    bid = len(out) + 1
                out.append({'id': bid, 'color': color, 'x': float(x), 'y': float(y)})
        if not out:
            raise RuntimeError(f'No valid GT buoys in {path}')
        return out

    def _semantic_cb(self, msg: String) -> None:
        try:
            self.latest_payload = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warn(f'Could not parse semantic JSON: {e}')

    def _extract_live(self) -> Tuple[str, List[Dict[str, Any]]]:
        if not self.latest_payload:
            return 'none', []
        source = str(self.latest_payload.get('source', 'unknown'))
        live: List[Dict[str, Any]] = []
        for b in self.latest_payload.get('buoys', []):
            color = str(b.get('color', '')).strip().lower()
            if color not in ('red', 'green'):
                cls = str(b.get('class', b.get('class_name', b.get('label', '')))).lower()
                if 'green' in cls:
                    color = 'green'
                elif 'red' in cls:
                    color = 'red'
            if color not in ('red', 'green'):
                continue
            x = b.get('x', b.get('north_m', None))
            y = b.get('y', b.get('east_m', None))
            if not (finite(x) and finite(y)):
                continue
            live.append({
                'id': b.get('id', b.get('track_id', -1)),
                'track_id': b.get('track_id', b.get('id', -1)),
                'color': color,
                'x': float(x),
                'y': float(y),
            })
        return source, live

    def _match(self, live: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[float]]:
        matches: List[Dict[str, Any]] = []
        errors: List[float] = []
        used = set()
        for gt in self.gt_buoys:
            best_i = None
            best_d = float('inf')
            for i, lb in enumerate(live):
                if i in used:
                    continue
                if lb['color'] != gt['color']:
                    continue
                d = dist(gt, lb)
                if d < best_d:
                    best_d = d
                    best_i = i
            if best_i is None or best_d > self.match_gate_m:
                continue
            used.add(best_i)
            lb = live[best_i]
            errors.append(best_d)
            matches.append({
                'gt_id': gt['id'], 'gt_color': gt['color'], 'live_id': lb['id'],
                'gt_x': gt['x'], 'gt_y': gt['y'], 'live_x': lb['x'], 'live_y': lb['y'],
                'error_m': best_d,
            })
        return matches, errors

    def _gt_markers(self, stamp_msg) -> MarkerArray:
        arr = MarkerArray()
        delete = Marker()
        delete.header.stamp = stamp_msg
        delete.header.frame_id = self.frame_id
        delete.action = Marker.DELETEALL
        arr.markers.append(delete)
        mid = 1
        for b in self.gt_buoys:
            m = Marker()
            m.header.stamp = stamp_msg
            m.header.frame_id = self.frame_id
            m.ns = 'gt_buoys'
            m.id = mid
            mid += 1
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(b['x'])
            m.pose.position.y = float(b['y'])
            m.pose.position.z = 0.55
            m.pose.orientation.w = 1.0
            m.scale.x = self.marker_diameter_m
            m.scale.y = self.marker_diameter_m
            m.scale.z = self.marker_diameter_m
            if b['color'] == 'red':
                m.color.r, m.color.g, m.color.b = 1.0, 0.0, 0.0
            else:
                m.color.r, m.color.g, m.color.b = 0.0, 1.0, 0.0
            m.color.a = 0.30
            arr.markers.append(m)

            t = Marker()
            t.header.stamp = stamp_msg
            t.header.frame_id = self.frame_id
            t.ns = 'gt_buoy_labels'
            t.id = mid
            mid += 1
            t.type = Marker.TEXT_VIEW_FACING
            t.action = Marker.ADD
            t.pose.position.x = float(b['x'])
            t.pose.position.y = float(b['y'])
            t.pose.position.z = 1.35
            t.pose.orientation.w = 1.0
            t.scale.z = 0.55
            t.color.r = t.color.g = t.color.b = t.color.a = 1.0
            t.text = f"GT {b['id']} {b['color']}"
            arr.markers.append(t)
        return arr

    def _timer_cb(self) -> None:
        stamp = self.get_clock().now().to_msg()
        self.marker_pub.publish(self._gt_markers(stamp))

        source, live = self._extract_live()
        matches, errors = self._match(live)
        now = self.get_clock().now().nanoseconds * 1e-9

        if errors:
            arr = np.asarray(errors, dtype=float)
            mean = float(np.mean(arr))
            median = float(np.median(arr))
            rmse = float(np.sqrt(np.mean(arr * arr)))
            maxe = float(np.max(arr))
        else:
            mean = median = rmse = maxe = float('nan')

        if now - self.last_log_time >= 1.0:
            self.last_log_time = now
            if errors:
                self.get_logger().info(
                    f'map_eval | source={source} | gt={len(self.gt_buoys)} live={len(live)} '
                    f'matched={len(matches)} mean={mean:.3f} m median={median:.3f} m rmse={rmse:.3f} m max={maxe:.3f} m'
                )
            else:
                self.get_logger().warn(
                    f'map_eval | source={source} | gt={len(self.gt_buoys)} live={len(live)} matched=0'
                )

        details = ';'.join([
            f"gt{m['gt_id']}:{m['gt_color']}:live{m['live_id']}:err={m['error_m']:.3f}"
            for m in matches
        ])
        with open(self.error_csv, 'a', newline='') as f:
            w = csv.writer(f)
            w.writerow([
                f'{now:.6f}', source, len(self.gt_buoys), len(live), len(matches),
                f'{mean:.6f}' if finite(mean) else '',
                f'{median:.6f}' if finite(median) else '',
                f'{rmse:.6f}' if finite(rmse) else '',
                f'{maxe:.6f}' if finite(maxe) else '',
                details,
            ])


def main(args=None):
    rclpy.init(args=args)
    node = GroundTruthEvaluatorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
