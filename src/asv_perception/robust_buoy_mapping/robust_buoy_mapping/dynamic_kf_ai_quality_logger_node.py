#!/usr/bin/env python3

import csv
import os
import math
import numpy as np
import rclpy

from robust_buoy_mapping.dynamic_kf_buoy_mapper_node import DynamicKFBuoyMapper, BuoyKFTrack


class DynamicKFAIQualityLoggerNode(DynamicKFBuoyMapper):
    def __init__(self):
        super().__init__()

        self.declare_parameter("ai_log_dir", "/home/highlevel/roboboat_vehicle_data/processed/ai_self_supervised")
        self.declare_parameter("ai_log_enabled", True)
        self.declare_parameter("ai_pair_logging_gate", 50.0)
        self.declare_parameter("snapshot_period_s", 0.5)

        self.ai_log_dir = str(self.get_parameter("ai_log_dir").value)
        self.ai_log_enabled = bool(self.get_parameter("ai_log_enabled").value)
        self.ai_pair_logging_gate = float(self.get_parameter("ai_pair_logging_gate").value)
        self.snapshot_period_s = float(self.get_parameter("snapshot_period_s").value)

        self.last_snapshot_t = 0.0

        self.pair_path = os.path.join(self.ai_log_dir, "pair_candidates.csv")
        self.birth_path = os.path.join(self.ai_log_dir, "birth_candidates.csv")
        self.snapshot_path = os.path.join(self.ai_log_dir, "track_snapshots.csv")

        self.pair_fields = [
            "ros_time_s",
            "det_idx",
            "track_id",
            "color_code",
            "det_confidence",
            "det_range_m",
            "det_sigma_m",
            "det_x",
            "det_y",
            "track_x",
            "track_y",
            "dx",
            "dy",
            "euclidean_m",
            "mahalanobis_d2",
            "track_pos_sigma_m",
            "track_hits",
            "track_confirmed",
            "track_age_s",
            "track_misses",
            "same_color_track_count",
            "assigned_by_kf",
        ]

        self.birth_fields = [
            "ros_time_s",
            "det_idx",
            "birth_track_id",
            "color_code",
            "det_confidence",
            "det_range_m",
            "det_sigma_m",
            "det_x",
            "det_y",
            "nearest_same_color_dist_m",
            "nearest_same_color_mahalanobis_d2",
            "nearest_same_color_pos_sigma_m",
            "same_color_track_count",
            "suppressed_too_close",
            "created_birth",
        ]

        self.snapshot_fields = [
            "ros_time_s",
            "track_id",
            "color_code",
            "x",
            "y",
            "vx",
            "vy",
            "speed_mps",
            "pos_sigma_m",
            "hits",
            "misses",
            "confirmed",
            "age_since_update_s",
        ]

        self.pair_file = None
        self.birth_file = None
        self.snapshot_file = None
        self.pair_writer = None
        self.birth_writer = None
        self.snapshot_writer = None

        if self.ai_log_enabled:
            os.makedirs(self.ai_log_dir, exist_ok=True)

            self.pair_file, self.pair_writer = self.open_writer(self.pair_path, self.pair_fields)
            self.birth_file, self.birth_writer = self.open_writer(self.birth_path, self.birth_fields)
            self.snapshot_file, self.snapshot_writer = self.open_writer(self.snapshot_path, self.snapshot_fields)

            self.get_logger().info("SELF-SUPERVISED AI QUALITY LOGGER ENABLED")
            self.get_logger().info(f"pair log: {self.pair_path}")
            self.get_logger().info(f"birth log: {self.birth_path}")
            self.get_logger().info(f"snapshot log: {self.snapshot_path}")

    def open_writer(self, path, fields):
        new_file = not os.path.exists(path)
        f = open(path, "a", newline="")
        w = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            w.writeheader()
            f.flush()
        return f, w

    def color_code(self, color):
        return 1 if color == "green" else 0

    def same_color_count(self, color):
        return sum(1 for t in self.tracks if t.color == color)

    def pair_features_row(self, now, det_idx, det, tr, d2, assigned):
        tx, ty = tr.pos()
        dx = float(det["x"] - tx)
        dy = float(det["y"] - ty)

        return {
            "ros_time_s": float(now),
            "det_idx": int(det_idx),
            "track_id": int(tr.id),
            "color_code": int(self.color_code(det["color"])),
            "det_confidence": float(det["confidence"]),
            "det_range_m": float(det["range_xy"]),
            "det_sigma_m": float(det["sigma"]),
            "det_x": float(det["x"]),
            "det_y": float(det["y"]),
            "track_x": float(tx),
            "track_y": float(ty),
            "dx": float(dx),
            "dy": float(dy),
            "euclidean_m": float(math.hypot(dx, dy)),
            "mahalanobis_d2": float(d2),
            "track_pos_sigma_m": float(tr.pos_sigma()),
            "track_hits": int(tr.hits),
            "track_confirmed": int(tr.confirmed),
            "track_age_s": float(tr.age_since_update(now)),
            "track_misses": int(tr.misses),
            "same_color_track_count": int(self.same_color_count(det["color"])),
            "assigned_by_kf": int(assigned),
        }

    def nearest_same_color_stats(self, det):
        best_dist = 999.0
        best_d2 = 999.0
        best_sigma = 999.0

        z = np.array([det["x"], det["y"]], dtype=float)

        for tr in self.tracks:
            if tr.color != det["color"]:
                continue

            tx, ty = tr.pos()
            dist = math.hypot(det["x"] - tx, det["y"] - ty)
            _, _, _, d2 = tr.innovation_stats(z, det["R"])

            if dist < best_dist:
                best_dist = float(dist)
                best_d2 = float(d2)
                best_sigma = float(tr.pos_sigma())

        return best_dist, best_d2, best_sigma

    def birth_row(self, now, det_idx, det, birth_track_id, suppressed_too_close, created_birth):
        nearest_dist, nearest_d2, nearest_sigma = self.nearest_same_color_stats(det)

        return {
            "ros_time_s": float(now),
            "det_idx": int(det_idx),
            "birth_track_id": int(birth_track_id),
            "color_code": int(self.color_code(det["color"])),
            "det_confidence": float(det["confidence"]),
            "det_range_m": float(det["range_xy"]),
            "det_sigma_m": float(det["sigma"]),
            "det_x": float(det["x"]),
            "det_y": float(det["y"]),
            "nearest_same_color_dist_m": float(nearest_dist),
            "nearest_same_color_mahalanobis_d2": float(nearest_d2),
            "nearest_same_color_pos_sigma_m": float(nearest_sigma),
            "same_color_track_count": int(self.same_color_count(det["color"])),
            "suppressed_too_close": int(suppressed_too_close),
            "created_birth": int(created_birth),
        }

    def write_rows(self, writer, file_handle, rows):
        if not self.ai_log_enabled or writer is None:
            return
        for r in rows:
            writer.writerow(r)
        file_handle.flush()

    def associate_and_update(self, dets, now):
        used_tracks = set()
        used_dets = set()

        assignment_candidates = []
        logging_candidates = []

        for di, d in enumerate(dets):
            z = np.array([d["x"], d["y"]], dtype=float)

            for ti, tr in enumerate(self.tracks):
                if tr.color != d["color"]:
                    continue

                gate = self.mahalanobis_gate_confirmed if tr.confirmed else self.mahalanobis_gate_tentative
                _, _, _, d2 = tr.innovation_stats(z, d["R"])

                if d2 <= self.ai_pair_logging_gate:
                    logging_candidates.append((float(d2), ti, di))

                if d2 <= gate:
                    assignment_candidates.append((float(d2), ti, di))

        assignment_candidates.sort(key=lambda x: x[0])

        assignments = []
        for d2, ti, di in assignment_candidates:
            if ti in used_tracks or di in used_dets:
                continue
            used_tracks.add(ti)
            used_dets.add(di)
            assignments.append((ti, di, d2))

        assigned_pairs = set((ti, di) for ti, di, _ in assignments)

        pair_rows = []
        for d2, ti, di in logging_candidates:
            assigned = (ti, di) in assigned_pairs
            pair_rows.append(self.pair_features_row(now, di, dets[di], self.tracks[ti], d2, assigned))

        self.write_rows(self.pair_writer, self.pair_file, pair_rows)

        for ti, di, _d2 in assignments:
            tr = self.tracks[ti]
            d = dets[di]

            tr.update(
                z_x=d["x"],
                z_y=d["y"],
                R=d["R"],
                t=now,
                confidence=d["confidence"],
                range_xy=d["range_xy"],
                max_speed_mps=self.max_speed_mps,
            )

            if tr.hits >= self.confirm_hits:
                tr.confirmed = True

        for ti, tr in enumerate(self.tracks):
            if ti not in used_tracks:
                tr.misses += 1

        birth_rows = []
        births = 0

        for di, d in enumerate(dets):
            if di in used_dets:
                continue

            if self.too_close_to_existing_same_color(d):
                birth_rows.append(self.birth_row(now, di, d, -1, True, False))
                continue

            tr = BuoyKFTrack(
                track_id=self.next_id,
                color=d["color"],
                x=d["x"],
                y=d["y"],
                t=now,
                initial_position_sigma=self.initial_position_sigma,
                initial_velocity_sigma=self.initial_velocity_sigma,
            )
            tr.last_confidence = d["confidence"]
            tr.last_range = d["range_xy"]

            self.next_id += 1
            self.tracks.append(tr)
            births += 1

            birth_rows.append(self.birth_row(now, di, d, tr.id, False, True))

        self.write_rows(self.birth_writer, self.birth_file, birth_rows)

        if births > 0:
            self.get_logger().info(f"new tentative KF buoy tracks: {births}")

    def snapshot_tracks(self, now):
        if not self.ai_log_enabled or self.snapshot_writer is None:
            return

        if now - self.last_snapshot_t < self.snapshot_period_s:
            return

        self.last_snapshot_t = now

        rows = []
        for tr in self.tracks:
            px, py = tr.pos()
            vx, vy = tr.vel()
            rows.append({
                "ros_time_s": float(now),
                "track_id": int(tr.id),
                "color_code": int(self.color_code(tr.color)),
                "x": float(px),
                "y": float(py),
                "vx": float(vx),
                "vy": float(vy),
                "speed_mps": float(math.hypot(vx, vy)),
                "pos_sigma_m": float(tr.pos_sigma()),
                "hits": int(tr.hits),
                "misses": int(tr.misses),
                "confirmed": int(tr.confirmed),
                "age_since_update_s": float(tr.age_since_update(now)),
            })

        self.write_rows(self.snapshot_writer, self.snapshot_file, rows)

    def timer_cb(self):
        super().timer_cb()
        now = self.get_clock().now().nanoseconds * 1e-9
        self.snapshot_tracks(now)


def main(args=None):
    rclpy.init(args=args)
    node = DynamicKFAIQualityLoggerNode()
    try:
        rclpy.spin(node)
    finally:
        for f in [node.pair_file, node.birth_file, node.snapshot_file]:
            if f is not None:
                f.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
