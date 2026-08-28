#!/usr/bin/env python3

import csv
import os
import math
import numpy as np
import rclpy

from robust_buoy_mapping.dynamic_kf_buoy_mapper_node import DynamicKFBuoyMapper, BuoyKFTrack


class DynamicKFAILoggerNode(DynamicKFBuoyMapper):
    def __init__(self):
        super().__init__()

        self.declare_parameter(
            "ai_log_csv",
            "/home/highlevel/roboboat_vehicle_data/processed/zed_kf_teacher_candidates.csv"
        )
        self.declare_parameter("ai_log_enabled", True)
        self.declare_parameter("ai_logging_gate", 40.0)

        self.ai_log_csv = str(self.get_parameter("ai_log_csv").value)
        self.ai_log_enabled = bool(self.get_parameter("ai_log_enabled").value)
        self.ai_logging_gate = float(self.get_parameter("ai_logging_gate").value)

        self.csv_file = None
        self.csv_writer = None

        self.fields = [
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
            "label_assigned",
        ]

        if self.ai_log_enabled:
            os.makedirs(os.path.dirname(self.ai_log_csv), exist_ok=True)
            new_file = not os.path.exists(self.ai_log_csv)
            self.csv_file = open(self.ai_log_csv, "a", newline="")
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.fields)
            if new_file:
                self.csv_writer.writeheader()
            self.get_logger().info(f"AI teacher logging enabled: {self.ai_log_csv}")

    def color_code(self, color):
        return 1 if color == "green" else 0

    def make_candidate_row(self, now, det_idx, track_idx, det, tr, d2, assigned):
        tx, ty = tr.pos()
        dx = float(det["x"] - tx)
        dy = float(det["y"] - ty)

        same_color_count = sum(1 for t in self.tracks if t.color == det["color"])

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
            "same_color_track_count": int(same_color_count),
            "label_assigned": int(assigned),
        }

    def log_candidate_rows(self, rows):
        if not self.ai_log_enabled or self.csv_writer is None:
            return

        for r in rows:
            self.csv_writer.writerow(r)

        self.csv_file.flush()

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

                if d2 <= self.ai_logging_gate:
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

        rows = []
        for d2, ti, di in logging_candidates:
            assigned = (ti, di) in assigned_pairs
            rows.append(self.make_candidate_row(now, di, ti, dets[di], self.tracks[ti], d2, assigned))

        self.log_candidate_rows(rows)

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

        births = 0
        for di, d in enumerate(dets):
            if di in used_dets:
                continue

            if self.too_close_to_existing_same_color(d):
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

        if births > 0:
            self.get_logger().info(f"new tentative KF buoy tracks: {births}")


def main(args=None):
    rclpy.init(args=args)
    node = DynamicKFAILoggerNode()
    try:
        rclpy.spin(node)
    finally:
        if node.csv_file is not None:
            node.csv_file.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
