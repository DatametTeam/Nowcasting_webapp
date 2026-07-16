#!/usr/bin/env python3
"""
FSS computation for the probabilistic ensemble model.

At each grid point the ensemble predicts rain when at least `min_votes` individual
models predict rain above the threshold.  FSS is then computed between this
binary ensemble forecast and the ground-truth HDF field.

Usage:
    python fss_ensemble.py --date 30-04-2026 --time 00:10 [--config fss_config.yaml]
    python fss_ensemble.py --backfill [--config fss_config.yaml]
"""

import argparse
import glob
import os
import sys
from datetime import datetime, timedelta

import h5py
import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import uniform_filter


# ── FSS core ──────────────────────────────────────────────────────────────────


def compute_ensemble_fss(
    obs: np.ndarray,
    vote_map: np.ndarray,
    obs_threshold: float,
    min_votes: int,
    scale: int,
) -> float:
    """
    FSS where `obs` is binarized at `obs_threshold` and the forecast is
    `vote_map` binarized at `min_votes`.  This keeps both binarization
    thresholds independent, which is necessary because vote_map values are
    model counts (integers), not mm/h values.
    """
    valid = np.isfinite(obs) & np.isfinite(vote_map)
    if valid.sum() == 0:
        return np.nan

    obs_bin = np.where(valid & (obs >= obs_threshold), 1.0, 0.0)
    fct_bin = np.where(valid & (vote_map >= min_votes), 1.0, 0.0)

    obs_frac = uniform_filter(obs_bin, size=scale, mode="constant", cval=0.0)
    fct_frac = uniform_filter(fct_bin, size=scale, mode="constant", cval=0.0)

    obs_v = obs_frac[valid]
    fct_v = fct_frac[valid]

    mse = np.mean((fct_v - obs_v) ** 2)
    mse_ref = np.mean(fct_v**2) + np.mean(obs_v**2)
    if mse_ref == 0.0:
        return np.nan
    return float(1.0 - mse / mse_ref)


# ── I/O helpers ───────────────────────────────────────────────────────────────


def load_ground_truth(
    data_folder: str, dt: datetime, datetime_format: str, hdf_key: str
) -> np.ndarray | None:
    path = os.path.join(data_folder, dt.strftime(datetime_format) + ".hdf")
    if not os.path.exists(path):
        return None
    try:
        with h5py.File(path, "r") as f:
            dataset = f[hdf_key]
            assert isinstance(dataset, h5py.Dataset)
            data: np.ndarray = dataset[:]
        return data.astype(np.float32)
    except Exception as e:
        print(f"[WARNING] Failed to load GT {path}: {e}", file=sys.stderr)
        return None


def load_prediction(
    predictions_root: str, model: str, dt: datetime, datetime_format: str
) -> np.ndarray | None:
    path = os.path.join(predictions_root, model, dt.strftime(datetime_format) + ".npy")
    if not os.path.exists(path):
        return None
    try:
        return np.load(path).astype(np.float32)
    except Exception as e:
        print(f"[WARNING] Failed to load prediction {path}: {e}", file=sys.stderr)
        return None


def build_vote_map(
    predictions_root: str,
    models: list[str],
    pred_dt: datetime,
    datetime_format: str,
    channel_idx: int,
    obs_threshold: float,
    gt_shape: tuple,
) -> tuple[np.ndarray, int]:
    """
    For a given (lead_time, threshold), load predictions from all models and
    sum the binary exceedance maps.  Returns (vote_map, n_contributing_models).
    """
    vote_map = np.zeros(gt_shape, dtype=np.float32)
    n_contributing = 0

    for model in models:
        pred = load_prediction(predictions_root, model, pred_dt, datetime_format)
        if pred is None:
            continue
        if pred.ndim != 3 or pred.shape[0] <= channel_idx:
            print(
                f"[WARNING] Unexpected shape {pred.shape} for {model} "
                f"channel {channel_idx} @ {pred_dt.strftime(datetime_format)}",
                file=sys.stderr,
            )
            continue

        fct = pred[channel_idx]
        if fct.shape != gt_shape:
            if fct.T.shape == gt_shape:
                fct = fct.T
            else:
                print(
                    f"[WARNING] Shape mismatch: gt={gt_shape}, fct={fct.shape} "
                    f"for {model}",
                    file=sys.stderr,
                )
                continue

        vote_map += (fct >= obs_threshold).astype(np.float32)
        n_contributing += 1

    return vote_map, n_contributing


# ── CSV helpers ───────────────────────────────────────────────────────────────


def _thr_str(threshold: float) -> str:
    return str(int(threshold)) if threshold == int(threshold) else str(threshold)


def csv_path(fss_root: str, model: str, lead_time: int, threshold: float, dt: datetime) -> str:
    thr = _thr_str(threshold)
    dirpath = os.path.join(fss_root, model, f"lt{lead_time}", f"thr{thr}")
    os.makedirs(dirpath, exist_ok=True)
    return os.path.join(
        dirpath, f"FSS_@{lead_time}min_{thr}mm_h_{dt.strftime('%m-%Y')}.csv"
    )


def row_exists(path: str, date_str: str, time_str: str) -> bool:
    if not os.path.exists(path):
        return False
    try:
        df = pd.read_csv(path, sep=r"\s+")
        return not df[(df["date"] == date_str) & (df["time"] == time_str)].empty
    except Exception:
        return False


def lead_time_done(dt: datetime, model: str, cfg: dict, lead_time: int) -> bool:
    date_str = dt.strftime("%d-%m-%Y")
    time_str = dt.strftime("%H:%M")
    return all(
        row_exists(csv_path(cfg["fss_root"], model, lead_time, thr, dt), date_str, time_str)
        for thr in cfg["thresholds"]
    )


def timestamp_fully_done(dt: datetime, model: str, cfg: dict) -> bool:
    return all(lead_time_done(dt, model, cfg, lt) for lt in cfg["lead_times"])


def append_csv_row(
    path: str,
    date_str: str,
    time_str: str,
    n_valid: int,
    fss_values: list[float],
    scales: list[int],
) -> None:
    columns = ["date", "time", "n_valid"] + [f"sc_{s}" for s in scales]
    row: dict = {"date": date_str, "time": time_str, "n_valid": n_valid}
    for s, v in zip(scales, fss_values):
        row[f"sc_{s}"] = round(v, 4) if np.isfinite(v) else np.nan

    if os.path.exists(path):
        df = pd.read_csv(path, sep=r"\s+")
        if not df[(df["date"] == date_str) & (df["time"] == time_str)].empty:
            return
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])[columns]

    with open(path, "w") as f:
        f.write(df.to_string(index=False, float_format="%.4f") + "\n")


# ── Core processing ───────────────────────────────────────────────────────────


def process_timestamp(dt: datetime, cfg: dict) -> None:
    datetime_format = cfg["datetime_format"]
    thresholds = cfg["thresholds"]
    scales = cfg["scales"]
    lead_times = cfg["lead_times"]
    models = cfg["models"]
    min_votes = cfg.get("min_votes", 3)
    ensemble_name = cfg.get("ensemble_model_name", "Probabilistic")

    date_str = dt.strftime("%d-%m-%Y")
    time_str = dt.strftime("%H:%M")

    if timestamp_fully_done(dt, ensemble_name, cfg):
        return

    gt = load_ground_truth(cfg["data_folder"], dt, datetime_format, cfg["hdf_key"])
    if gt is None:
        print(f"[WARNING] No ground truth for {date_str} {time_str}", file=sys.stderr)
        return

    any_done = False
    for lead_time in lead_times:
        if lead_time_done(dt, ensemble_name, cfg, lead_time):
            continue

        pred_dt = dt - timedelta(minutes=lead_time)
        channel_idx = lead_time // 5 - 1

        for thr in thresholds:
            if row_exists(
                csv_path(cfg["fss_root"], ensemble_name, lead_time, thr, dt),
                date_str,
                time_str,
            ):
                continue

            vote_map, n_contributing = build_vote_map(
                cfg["predictions_root"],
                models,
                pred_dt,
                datetime_format,
                channel_idx,
                float(thr),
                gt.shape,
            )

            if n_contributing == 0:
                continue

            n_valid = int(np.sum(np.isfinite(gt) & (gt >= thr)))
            fss_values = [
                compute_ensemble_fss(gt, vote_map, float(thr), min_votes, sc)
                for sc in scales
            ]
            path = csv_path(cfg["fss_root"], ensemble_name, lead_time, thr, dt)
            append_csv_row(path, date_str, time_str, n_valid, fss_values, scales)

        any_done = True

    if any_done:
        print(f"[INFO] {ensemble_name} @ {date_str} {time_str} done")


# ── Discovery ─────────────────────────────────────────────────────────────────


def discover_gt_timestamps(data_folder: str, datetime_format: str) -> list[datetime]:
    timestamps = []
    for path in sorted(glob.glob(os.path.join(data_folder, "*.hdf"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        try:
            timestamps.append(datetime.strptime(stem, datetime_format))
        except ValueError:
            continue
    return timestamps


# ── Config ────────────────────────────────────────────────────────────────────


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    required = [
        "data_folder",
        "datetime_format",
        "hdf_key",
        "predictions_root",
        "fss_root",
        "thresholds",
        "scales",
        "lead_times",
    ]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")

    # Active models and ensemble name come from the main webapp config
    # (single source of truth, shared with nwc_webapp_v2/backend).
    main_cfg_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "nwc_webapp_v2", "cfg.yaml"
    )
    with open(main_cfg_path) as f:
        main_cfg = yaml.safe_load(f)
    cfg["models"] = main_cfg["models"]
    cfg["ensemble_model_name"] = main_cfg.get("fss", {}).get("ensemble_model_name", "Probabilistic")
    return cfg


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute FSS for the probabilistic ensemble")
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "fss_config.yaml"),
    )
    parser.add_argument("--date", help="Ground-truth date DD-MM-YYYY")
    parser.add_argument("--time", dest="time_", metavar="TIME", help="Ground-truth time HH:MM")
    parser.add_argument("--backfill", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    min_votes = cfg.get("min_votes", 3)
    ensemble_name = cfg.get("ensemble_model_name", "Probabilistic")
    n_models = len(cfg["models"])
    print(
        f"[INFO] Ensemble: '{ensemble_name}', min_votes={min_votes}/{n_models}, "
        f"models={cfg['models']}"
    )

    if args.backfill:
        timestamps = discover_gt_timestamps(cfg["data_folder"], cfg["datetime_format"])
        if not timestamps:
            print(
                f"[WARNING] No HDF files found in {cfg['data_folder']}", file=sys.stderr
            )
            sys.exit(0)
        print(f"[INFO] Backfill: {len(timestamps)} timestamps — processing {ensemble_name}")
        for dt in timestamps:
            process_timestamp(dt, cfg)
        print(f"[INFO] Backfill complete for {ensemble_name}")
    else:
        if not args.date or not args.time_:
            parser.error("--date and --time are required (or use --backfill)")
        dt_str = f"{args.date}-{args.time_.replace(':', '-')}"
        try:
            dt = datetime.strptime(dt_str, cfg["datetime_format"])
        except ValueError:
            print(
                f"[ERROR] Cannot parse '{dt_str}' with format '{cfg['datetime_format']}'",
                file=sys.stderr,
            )
            sys.exit(1)
        process_timestamp(dt, cfg)


if __name__ == "__main__":
    main()