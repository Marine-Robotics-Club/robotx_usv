#!/usr/bin/env python3

import argparse
import os
import numpy as np
import pandas as pd


def future_snapshot(snap, track_id, t, stable_window_s):
    future = snap[
        (snap["track_id"] == track_id)
        & (snap["ros_time_s"] >= t + stable_window_s)
    ].sort_values("ros_time_s")

    if len(future) == 0:
        return None

    return future.iloc[0]


def build_pair_dataset(pair, snap, stable_window_s, max_stable_sigma, min_future_hits):
    max_time = float(snap["ros_time_s"].max())
    rows = []

    for _, r in pair.iterrows():
        t = float(r["ros_time_s"])

        # Cannot know future outcome near the end of the log.
        if t > max_time - stable_window_s:
            continue

        label = 0
        sigma_target = float(r["det_sigma_m"])

        if int(r["assigned_by_kf"]) == 1:
            fs = future_snapshot(snap, int(r["track_id"]), t, stable_window_s)

            if fs is not None:
                future_confirmed = int(fs["confirmed"]) == 1
                future_sigma_ok = float(fs["pos_sigma_m"]) <= max_stable_sigma
                future_hits_ok = int(fs["hits"]) >= int(r["track_hits"]) + min_future_hits

                if future_confirmed and future_sigma_ok and future_hits_ok:
                    label = 1

                    future_residual = float(np.hypot(
                        float(r["det_x"]) - float(fs["x"]),
                        float(r["det_y"]) - float(fs["y"])
                    ))

                    sigma_target = float(np.clip(0.35 + 0.5 * future_residual, 0.35, 3.5))

        out = r.to_dict()
        out["label_reliable_update"] = int(label)
        out["sigma_target_m"] = float(sigma_target)
        rows.append(out)

    return pd.DataFrame(rows)


def build_birth_dataset(birth, snap, stable_window_s, max_stable_sigma, min_birth_hits):
    if len(birth) == 0:
        return pd.DataFrame()

    max_time = float(snap["ros_time_s"].max())
    rows = []

    for _, r in birth.iterrows():
        t = float(r["ros_time_s"])

        if t > max_time - stable_window_s:
            continue

        label = 0

        if int(r["created_birth"]) == 1 and int(r["birth_track_id"]) >= 0:
            fs = future_snapshot(snap, int(r["birth_track_id"]), t, stable_window_s)

            if fs is not None:
                future_confirmed = int(fs["confirmed"]) == 1
                future_sigma_ok = float(fs["pos_sigma_m"]) <= max_stable_sigma
                future_hits_ok = int(fs["hits"]) >= min_birth_hits

                if future_confirmed and future_sigma_ok and future_hits_ok:
                    label = 1

        out = r.to_dict()
        out["label_reliable_birth"] = int(label)
        rows.append(out)

    return pd.DataFrame(rows)


def safe_read_csv(path, name):
    if not os.path.exists(path):
        raise SystemExit(f"Missing {name}: {path}")

    df = pd.read_csv(path)

    if len(df) == 0:
        print(f"WARNING: {name} is empty: {path}")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--stable-window-s", type=float, default=5.0)
    parser.add_argument("--max-stable-sigma", type=float, default=2.5)
    parser.add_argument("--min-future-hits", type=int, default=3)
    parser.add_argument("--min-birth-hits", type=int, default=6)
    args = parser.parse_args()

    pair_path = os.path.join(args.log_dir, "pair_candidates.csv")
    birth_path = os.path.join(args.log_dir, "birth_candidates.csv")
    snap_path = os.path.join(args.log_dir, "track_snapshots.csv")

    pair = safe_read_csv(pair_path, "pair_candidates.csv")
    snap = safe_read_csv(snap_path, "track_snapshots.csv")

    if os.path.exists(birth_path):
        birth = pd.read_csv(birth_path).replace([np.inf, -np.inf], np.nan).dropna()
    else:
        birth = pd.DataFrame()

    if len(snap) == 0:
        raise SystemExit("track_snapshots.csv has no usable rows. Run the logger longer.")

    if len(pair) == 0:
        raise SystemExit("pair_candidates.csv has no usable rows. Run the logger longer or increase ai_pair_logging_gate.")

    os.makedirs(args.out_dir, exist_ok=True)

    pair_ds = build_pair_dataset(
        pair,
        snap,
        stable_window_s=args.stable_window_s,
        max_stable_sigma=args.max_stable_sigma,
        min_future_hits=args.min_future_hits,
    )

    birth_ds = build_birth_dataset(
        birth,
        snap,
        stable_window_s=args.stable_window_s,
        max_stable_sigma=args.max_stable_sigma,
        min_birth_hits=args.min_birth_hits,
    )

    pair_out = os.path.join(args.out_dir, "pair_reliable_update_dataset.csv")
    birth_out = os.path.join(args.out_dir, "birth_reliable_dataset.csv")

    pair_ds.to_csv(pair_out, index=False)
    birth_ds.to_csv(birth_out, index=False)

    print()
    print("Pair dataset:", pair_out)
    print("Rows:", len(pair_ds))
    if len(pair_ds) > 0:
        print(pair_ds["label_reliable_update"].value_counts(dropna=False))

    print()
    print("Birth dataset:", birth_out)
    print("Rows:", len(birth_ds))
    if len(birth_ds) > 0 and "label_reliable_birth" in birth_ds.columns:
        print(birth_ds["label_reliable_birth"].value_counts(dropna=False))

    print()
    print("Done.")


if __name__ == "__main__":
    main()
