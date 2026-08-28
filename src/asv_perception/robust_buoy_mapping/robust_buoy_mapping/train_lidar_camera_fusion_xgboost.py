#!/usr/bin/env python3

import argparse
import json
import os
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from robust_buoy_mapping.lidar_camera_fusion_common import FUSION_FEATURES


def load_model_classes():
    try:
        from xgboost import XGBClassifier, XGBRegressor

        clf = lambda: XGBClassifier(
            n_estimators=350,
            max_depth=5,
            learning_rate=0.04,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=2.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        )
        reg = lambda: XGBRegressor(
            n_estimators=350,
            max_depth=5,
            learning_rate=0.04,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=2.0,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        )
        return "xgboost", clf, reg
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

        clf = lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.04, max_leaf_nodes=31, random_state=42)
        reg = lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, max_leaf_nodes=31, random_state=42)
        return "sklearn_hist_gradient_boosting_fallback", clf, reg


def read_gt_csv(path: str) -> pd.DataFrame:
    gt = pd.read_csv(path)
    lower = {c.lower(): c for c in gt.columns}

    color_col = lower.get("color") or lower.get("class") or lower.get("label")
    x_col = lower.get("north_m") or lower.get("x") or lower.get("map_x_m")
    y_col = lower.get("east_m") or lower.get("y") or lower.get("map_y_m")

    if color_col is None or x_col is None or y_col is None:
        raise SystemExit(f"GT CSV must include color and north/east columns: {path}")

    out = pd.DataFrame({
        "color": gt[color_col].astype(str).str.lower().str.strip(),
        "gt_x_m": gt[x_col].astype(float),
        "gt_y_m": gt[y_col].astype(float),
    })
    out = out[out["color"].isin(["red", "green"])]
    if len(out) == 0:
        raise SystemExit("GT CSV has no red/green buoy rows.")
    return out.reset_index(drop=True)


def nearest_same_color_gt(row: pd.Series, gt: pd.DataFrame) -> Tuple[float, float, float]:
    same = gt[gt["color"] == str(row["color"]).lower().strip()]
    if len(same) == 0:
        return np.nan, np.nan, np.inf
    dx = same["gt_x_m"].values - float(row["raw_fused_map_x_m"])
    dy = same["gt_y_m"].values - float(row["raw_fused_map_y_m"])
    dist = np.hypot(dx, dy)
    i = int(np.argmin(dist))
    return float(same.iloc[i]["gt_x_m"]), float(same.iloc[i]["gt_y_m"]), float(dist[i])


def build_supervised_dataset(fusion_log: str, gt_csv: str, reliable_gate_m: float, max_negative_range_m: float) -> pd.DataFrame:
    df = pd.read_csv(fusion_log).replace([np.inf, -np.inf], np.nan)
    gt = read_gt_csv(gt_csv)

    required = set(FUSION_FEATURES + ["color", "raw_fused_map_x_m", "raw_fused_map_y_m"])
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"Fusion log is missing columns: {missing}")

    rows = []
    for _, r in df.iterrows():
        if not np.isfinite(float(r["raw_fused_map_x_m"])) or not np.isfinite(float(r["raw_fused_map_y_m"])):
            continue

        gt_x, gt_y, dist = nearest_same_color_gt(r, gt)
        if not np.isfinite(dist):
            continue

        # Keep useful hard negatives near the buoy field but avoid distant junk dominating the data.
        if dist > float(max_negative_range_m):
            continue

        out = r.to_dict()
        out["nearest_gt_x_m"] = gt_x
        out["nearest_gt_y_m"] = gt_y
        out["nearest_gt_error_m"] = dist
        out["label_reliable_fusion"] = int(dist <= float(reliable_gate_m))
        out["target_dx_m"] = float(gt_x - float(r["raw_fused_map_x_m"]))
        out["target_dy_m"] = float(gt_y - float(r["raw_fused_map_y_m"]))
        out["target_sigma_m"] = float(np.clip(0.25 + 0.55 * dist, 0.25, 4.0))
        rows.append(out)

    out_df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    out_df = out_df.dropna(subset=FUSION_FEATURES + ["label_reliable_fusion", "target_dx_m", "target_dy_m", "target_sigma_m"])
    return out_df


def clf_metrics(y_test, y_pred, y_prob) -> Dict[str, float]:
    m = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }
    if len(set(y_test)) > 1:
        m["roc_auc"] = float(roc_auc_score(y_test, y_prob))
    else:
        m["roc_auc"] = None
    return m


def predict_prob(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    pred = model.predict(X)
    return np.asarray(pred, dtype=float)


def train_classifier(df, make_clf, out_path, backend_name):
    y = df["label_reliable_fusion"].astype(int).values
    X = df[FUSION_FEATURES].astype(float).values
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    print(f"fusion reliable classifier: samples={len(y)} pos={pos} neg={neg}")

    if pos < 10 or neg < 10:
        raise SystemExit("Not enough positive/negative fusion examples. Log more data or increase max-negative-range-m.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    model = make_clf()
    model.fit(X_train, y_train)

    y_prob = predict_prob(model, X_test)
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = clf_metrics(y_test, y_pred, y_prob)
    metrics.update({"samples": int(len(y)), "positives": pos, "negatives": neg, "backend": backend_name, "features": FUSION_FEATURES})

    joblib.dump(model, out_path)
    with open(out_path.replace(".joblib", "_meta.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print("saved:", out_path)


def train_regressor(df, make_reg, target, out_path, backend_name):
    # Residual and sigma regressors should learn only from reliable fusion examples.
    dfr = df[df["label_reliable_fusion"].astype(int) == 1].copy()
    if len(dfr) < 20:
        raise SystemExit(f"Not enough reliable rows for {target}. Need more positive data.")

    X = dfr[FUSION_FEATURES].astype(float).values
    y = dfr[target].astype(float).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = make_reg()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "target": target,
        "samples": int(len(dfr)),
        "mae": float(mean_absolute_error(y_test, pred)),
        "target_mean": float(np.mean(y)),
        "target_std": float(np.std(y)),
        "backend": backend_name,
        "features": FUSION_FEATURES,
    }
    joblib.dump(model, out_path)
    with open(out_path.replace(".joblib", "_meta.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print("saved:", out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fusion-log", required=True, help="CSV created by lidar_camera_fusion_logger_node")
    parser.add_argument("--gt-csv", required=True, help="GT buoy CSV with color,north_m,east_m")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--reliable-gate-m", type=float, default=1.25)
    parser.add_argument("--max-negative-range-m", type=float, default=8.0)
    parser.add_argument("--dataset-out", default="")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    backend_name, make_clf, make_reg = load_model_classes()
    print("training backend:", backend_name)

    ds = build_supervised_dataset(
        args.fusion_log,
        args.gt_csv,
        reliable_gate_m=args.reliable_gate_m,
        max_negative_range_m=args.max_negative_range_m,
    )
    if len(ds) == 0:
        raise SystemExit("No supervised rows were created. Check fusion log and GT CSV.")

    dataset_out = args.dataset_out or os.path.join(args.out_dir, "lidar_camera_fusion_supervised_dataset.csv")
    ds.to_csv(dataset_out, index=False)
    print("dataset:", dataset_out)
    print(ds["label_reliable_fusion"].value_counts(dropna=False))

    train_classifier(ds, make_clf, os.path.join(args.out_dir, "fusion_reliable_xgb.joblib"), backend_name)
    train_regressor(ds, make_reg, "target_dx_m", os.path.join(args.out_dir, "fusion_dx_xgb.joblib"), backend_name)
    train_regressor(ds, make_reg, "target_dy_m", os.path.join(args.out_dir, "fusion_dy_xgb.joblib"), backend_name)
    train_regressor(ds, make_reg, "target_sigma_m", os.path.join(args.out_dir, "fusion_sigma_xgb.joblib"), backend_name)


if __name__ == "__main__":
    main()
