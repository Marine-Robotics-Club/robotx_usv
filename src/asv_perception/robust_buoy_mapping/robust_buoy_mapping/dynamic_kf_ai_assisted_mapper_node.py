#!/usr/bin/env python3

import math
import os
import numpy as np
import rclpy

try:
    import joblib
except Exception:
    joblib = None

from robust_buoy_mapping.dynamic_kf_buoy_mapper_node import DynamicKFBuoyMapper, BuoyKFTrack


PAIR_FEATURES = [
    "color_code",
    "det_confidence",
    "det_range_m",
    "det_sigma_m",
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
]

BIRTH_FEATURES = [
    "color_code",
    "det_confidence",
    "det_range_m",
    "det_sigma_m",
    "nearest_same_color_dist_m",
    "nearest_same_color_mahalanobis_d2",
    "nearest_same_color_pos_sigma_m",
    "same_color_track_count",
    "suppressed_too_close",
]


class DynamicKFAIAssistedMapperNode(DynamicKFBuoyMapper):
    def __init__(self):
        super().__init__()

        self.declare_parameter("ai_enabled", True)
        self.declare_parameter("pair_model_path", "/home/highlevel/roboboat_usv/src/asv_perception/robust_buoy_mapping/models/self_supervised/pair_reliable_update_rf.joblib")
        self.declare_parameter("sigma_model_path", "/home/highlevel/roboboat_usv/src/asv_perception/robust_buoy_mapping/models/self_supervised/measurement_sigma_rf.joblib")
        self.declare_parameter("birth_model_path", "/home/highlevel/roboboat_usv/src/asv_perception/robust_buoy_mapping/models/self_supervised/birth_reliable_rf.joblib")

        self.declare_parameter("p_update_threshold", 0.55)
        self.declare_parameter("p_birth_threshold", 0.55)
        self.declare_parameter("ai_score_weight", 8.0)
        self.declare_parameter("sigma_blend_alpha", 0.75)
        self.declare_parameter("ai_sigma_min", 0.35)
        self.declare_parameter("ai_sigma_max", 3.5)

        self.ai_enabled = bool(self.get_parameter("ai_enabled").value)
        self.pair_model_path = str(self.get_parameter("pair_model_path").value)
        self.sigma_model_path = str(self.get_parameter("sigma_model_path").value)
        self.birth_model_path = str(self.get_parameter("birth_model_path").value)

        self.p_update_threshold = float(self.get_parameter("p_update_threshold").value)
        self.p_birth_threshold = float(self.get_parameter("p_birth_threshold").value)
        self.ai_score_weight = float(self.get_parameter("ai_score_weight").value)
        self.sigma_blend_alpha = float(self.get_parameter("sigma_blend_alpha").value)
        self.ai_sigma_min = float(self.get_parameter("ai_sigma_min").value)
        self.ai_sigma_max = float(self.get_parameter("ai_sigma_max").value)

        self.pair_model = None
        self.sigma_model = None
        self.birth_model = None

        self.load_models()

    def load_one(self, path, name):
        if not self.ai_enabled:
            return None
        if joblib is None:
            self.get_logger().warn("joblib not available. AI disabled.")
            self.ai_enabled = False
            return None
        if not os.path.exists(path):
            self.get_logger().warn(f"{name} model not found: {path}. Falling back where needed.")
            return None
        self.get_logger().info(f"loaded {name}: {path}")
        return joblib.load(path)

    def load_models(self):
        self.pair_model = self.load_one(self.pair_model_path, "pair reliable-update")
        self.sigma_model = self.load_one(self.sigma_model_path, "measurement sigma")
        self.birth_model = self.load_one(self.birth_model_path, "birth reliable")

        if self.pair_model is not None:
            self.get_logger().info("AI-ASSISTED DYNAMIC KF MAPPER ENABLED")
        else:
            self.get_logger().warn("Pair AI model missing. Mapper will behave close to dynamic KF baseline.")

    def color_code(self, color):
        return 1 if color == "green" else 0

    def same_color_count(self, color):
        return sum(1 for t in self.tracks if t.color == color)

    def pair_feature_vector(self, now, det, tr, d2):
        tx, ty = tr.pos()
        dx = float(det["x"] - tx)
        dy = float(det["y"] - ty)

        values = {
            "color_code": int(self.color_code(det["color"])),
            "det_confidence": float(det["confidence"]),
            "det_range_m": float(det["range_xy"]),
            "det_sigma_m": float(det["sigma"]),
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
        }

        return np.array([[values[k] for k in PAIR_FEATURES]], dtype=float), values

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

    def birth_feature_vector(self, det, suppressed_too_close):
        nearest_dist, nearest_d2, nearest_sigma = self.nearest_same_color_stats(det)

        values = {
            "color_code": int(self.color_code(det["color"])),
            "det_confidence": float(det["confidence"]),
            "det_range_m": float(det["range_xy"]),
            "det_sigma_m": float(det["sigma"]),
            "nearest_same_color_dist_m": float(nearest_dist),
            "nearest_same_color_mahalanobis_d2": float(nearest_d2),
            "nearest_same_color_pos_sigma_m": float(nearest_sigma),
            "same_color_track_count": int(self.same_color_count(det["color"])),
            "suppressed_too_close": int(suppressed_too_close),
        }

        return np.array([[values[k] for k in BIRTH_FEATURES]], dtype=float), values

    def predict_pair_probability(self, X):
        if self.pair_model is None:
            return 1.0
        try:
            return float(self.pair_model.predict_proba(X)[0, 1])
        except Exception:
            return 0.0

    def predict_birth_probability(self, X):
        if self.birth_model is None:
            return 1.0
        try:
            return float(self.birth_model.predict_proba(X)[0, 1])
        except Exception:
            return 0.0

    def predict_sigma(self, X, fallback_sigma):
        if self.sigma_model is None:
            return float(fallback_sigma)

        try:
            sigma_ai = float(self.sigma_model.predict(X)[0])
            sigma_ai = float(np.clip(sigma_ai, self.ai_sigma_min, self.ai_sigma_max))
            alpha = self.sigma_blend_alpha
            return float((1.0 - alpha) * fallback_sigma + alpha * sigma_ai)
        except Exception:
            return float(fallback_sigma)

    def associate_and_update(self, dets, now):
        used_tracks = set()
        used_dets = set()
        candidates = []

        for di, d in enumerate(dets):
            z = np.array([d["x"], d["y"]], dtype=float)

            for ti, tr in enumerate(self.tracks):
                if tr.color != d["color"]:
                    continue

                gate = self.mahalanobis_gate_confirmed if tr.confirmed else self.mahalanobis_gate_tentative
                _, _, _, d2 = tr.innovation_stats(z, d["R"])

                X, _values = self.pair_feature_vector(now, d, tr, d2)
                p_update = self.predict_pair_probability(X)

                if d2 <= gate and p_update >= self.p_update_threshold:
                    sigma = self.predict_sigma(X, d["sigma"])
                    R_ai = np.diag([sigma ** 2, sigma ** 2]).astype(float)

                    # Lower cost is better. AI probability helps choose among ambiguous candidates.
                    score = float(d2 - self.ai_score_weight * math.log(max(p_update, 1e-3)))

                    candidates.append((score, ti, di, d2, p_update, R_ai, sigma))

        candidates.sort(key=lambda x: x[0])

        assignments = []
        for score, ti, di, d2, p_update, R_ai, sigma in candidates:
            if ti in used_tracks or di in used_dets:
                continue
            used_tracks.add(ti)
            used_dets.add(di)
            assignments.append((ti, di, d2, p_update, R_ai, sigma))

        for ti, di, _d2, p_update, R_ai, sigma in assignments:
            tr = self.tracks[ti]
            d = dets[di]

            tr.update(
                z_x=d["x"],
                z_y=d["y"],
                R=R_ai,
                t=now,
                confidence=d["confidence"],
                range_xy=d["range_xy"],
                max_speed_mps=self.max_speed_mps,
            )

            tr.last_ai_p_update = float(p_update)
            tr.last_ai_sigma = float(sigma)

            if tr.hits >= self.confirm_hits:
                tr.confirmed = True

        for ti, tr in enumerate(self.tracks):
            if ti not in used_tracks:
                tr.misses += 1

        births = 0
        rejected_births = 0

        for di, d in enumerate(dets):
            if di in used_dets:
                continue

            too_close = self.too_close_to_existing_same_color(d)

            Xb, _ = self.birth_feature_vector(d, too_close)
            p_birth = self.predict_birth_probability(Xb)

            if too_close:
                rejected_births += 1
                continue

            if p_birth < self.p_birth_threshold:
                rejected_births += 1
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
            tr.last_ai_p_birth = float(p_birth)

            self.next_id += 1
            self.tracks.append(tr)
            births += 1

        if births > 0 or rejected_births > 0:
            self.get_logger().info(f"AI birth decisions | created={births} rejected={rejected_births}")


def main(args=None):
    rclpy.init(args=args)
    node = DynamicKFAIAssistedMapperNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
