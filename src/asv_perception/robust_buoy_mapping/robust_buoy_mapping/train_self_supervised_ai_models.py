#!/usr/bin/env python3

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_absolute_error
from sklearn.model_selection import train_test_split


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


def clf_metrics(y_test, y_pred, y_prob):
    out = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }

    if len(set(y_test)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_test, y_prob))
    else:
        out["roc_auc"] = None

    return out


def train_classifier(df, features, label, out_path, name):
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=features + [label])

    y = df[label].astype(int).values
    X = df[features].values

    pos = int((y == 1).sum())
    neg = int((y == 0).sum())

    print(f"{name}: samples={len(df)} pos={pos} neg={neg}")

    if pos < 20 or neg < 20:
        print(f"WARNING: not enough balanced data for {name}. Skipping.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = clf_metrics(y_test, y_pred, y_prob)
    metrics["samples"] = int(len(df))
    metrics["positives"] = pos
    metrics["negatives"] = neg
    metrics["features"] = features

    joblib.dump(model, out_path)

    meta_path = out_path.replace(".joblib", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(name, json.dumps(metrics, indent=2))
    print("saved:", out_path)

    return metrics


def train_sigma_regressor(df, out_path):
    features = PAIR_FEATURES

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=features + ["sigma_target_m", "label_reliable_update"])

    # Train sigma only on reliable positive updates.
    df = df[df["label_reliable_update"].astype(int) == 1]

    if len(df) < 50:
        print("WARNING: not enough positive reliable updates for sigma regressor. Skipping.")
        return None

    X = df[features].values
    y = df["sigma_target_m"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
    )

    model = RandomForestRegressor(
        n_estimators=250,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        "samples": int(len(df)),
        "mae_sigma_m": float(mean_absolute_error(y_test, y_pred)),
        "target_mean_sigma_m": float(np.mean(y)),
        "features": features,
    }

    joblib.dump(model, out_path)

    meta_path = out_path.replace(".joblib", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("sigma regressor", json.dumps(metrics, indent=2))
    print("saved:", out_path)

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    pair_csv = os.path.join(args.dataset_dir, "pair_reliable_update_dataset.csv")
    birth_csv = os.path.join(args.dataset_dir, "birth_reliable_dataset.csv")

    pair = pd.read_csv(pair_csv)
    birth = pd.read_csv(birth_csv)

    train_classifier(
        pair,
        PAIR_FEATURES,
        "label_reliable_update",
        os.path.join(args.out_dir, "pair_reliable_update_rf.joblib"),
        "pair_reliable_update",
    )

    train_sigma_regressor(
        pair,
        os.path.join(args.out_dir, "measurement_sigma_rf.joblib"),
    )

    train_classifier(
        birth,
        BIRTH_FEATURES,
        "label_reliable_birth",
        os.path.join(args.out_dir, "birth_reliable_rf.joblib"),
        "birth_reliable",
    )


if __name__ == "__main__":
    main()
