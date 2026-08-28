#!/usr/bin/env python3

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split


FEATURES = [
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--test-size", type=float, default=0.25)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURES + ["label_assigned"])

    df["label_assigned"] = df["label_assigned"].astype(int)

    positives = int((df["label_assigned"] == 1).sum())
    negatives = int((df["label_assigned"] == 0).sum())

    print(f"Loaded samples: {len(df)}")
    print(f"Positive assigned: {positives}")
    print(f"Negative not assigned: {negatives}")

    if positives < 20 or negatives < 20:
        raise SystemExit("Not enough positive/negative samples. Run the logger longer.")

    X = df[FEATURES].values
    y = df["label_assigned"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "samples": int(len(df)),
        "positives": positives,
        "negatives": negatives,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "features": FEATURES,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump(model, args.out)

    meta_path = args.out.replace(".joblib", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Saved model: {args.out}")
    print(f"Saved metadata: {meta_path}")


if __name__ == "__main__":
    main()
