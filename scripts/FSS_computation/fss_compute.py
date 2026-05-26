#!/usr/bin/env python3
"""
Standalone FSS (Fractions Skill Score) metrics computation.

Computes FSS between nowcasting model predictions and ground-truth HDF files,
appending results to per-model CSV databases.

Usage:
    # Crontab mode — process one specific timestamp:
    python fss_compute.py --model ConvLSTM --date 30-04-2026 --time 00:10

    # Backfill — process all available ground-truth files (initial run):
    python fss_compute.py --model ConvLSTM --backfill

Crontab example (runs every 5 minutes):
    */5 * * * * /path/to/venv/bin/python /path/to/fss_compute.py \
        --model ConvLSTM \
        --date $(date +%%d-%%m-%%Y) \
        --time $(date +%%H:%%M)
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


def compute_fss(obs: np.ndarray, fct: np.ndarray, threshold: float, scale: int) -> float:
    valid = np.isfinite(obs) & np.isfinite(fct)
    if valid.sum() == 0:
        return np.nan

    obs_bin = np.where(valid & (obs >= threshold), 1.0, 0.0)
    fct_bin = np.where(valid & (fct >= threshold), 1.0, 0.0)

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


# ── CSV helpers ───────────────────────────────────────────────────────────────


def row_exists(path: str, date_str: str, time_str: str) -> bool:
    """Return True if this date+time row is already present in the CSV."""
    if not os.path.exists(path):
        return False
    try:
        df = pd.read_csv(path, sep=r"\s+")
        return not df[(df["date"] == date_str) & (df["time"] == time_str)].empty
    except Exception:
        return False


def lead_time_done(
    dt: datetime, model: str, cfg: dict, lead_time: int
) -> bool:
    """Return True if every threshold CSV already has a row for this timestamp."""
    date_str = dt.strftime("%d-%m-%Y")
    time_str = dt.strftime("%H:%M")
    return all(
        row_exists(csv_path(cfg["fss_root"], model, lead_time, thr, dt), date_str, time_str)
        for thr in cfg["thresholds"]
    )


def timestamp_fully_done(dt: datetime, model: str, cfg: dict) -> bool:
    """Return True if every lead_time×threshold CSV already has a row for this timestamp."""
    return all(lead_time_done(dt, model, cfg, lt) for lt in cfg["lead_times"])


def _thr_str(threshold: float) -> str:
    return str(int(threshold)) if threshold == int(threshold) else str(threshold)


def csv_path(
    fss_root: str, model: str, lead_time: int, threshold: float, dt: datetime
) -> str:
    thr = _thr_str(threshold)
    dirpath = os.path.join(fss_root, model, f"lt{lead_time}", f"thr{thr}")
    os.makedirs(dirpath, exist_ok=True)
    return os.path.join(
        dirpath, f"FSS_@{lead_time}min_{thr}mm_h_{dt.strftime('%m-%Y')}.csv"
    )


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
            return  # already recorded — skip duplicate
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])[columns]

    with open(path, "w") as f:
        f.write(df.to_string(index=False, float_format="%.4f") + "\n")


# ── Core processing ───────────────────────────────────────────────────────────


def process_timestamp(dt: datetime, model: str, cfg: dict) -> None:
    """Process a single ground-truth timestamp: load GT, match predictions, write FSS rows."""
    datetime_format = cfg["datetime_format"]
    thresholds = cfg["thresholds"]
    scales = cfg["scales"]
    lead_times = cfg["lead_times"]

    date_str = dt.strftime("%d-%m-%Y")
    time_str = dt.strftime("%H:%M")

    # Skip entirely if every lead_time×threshold row is already recorded
    if timestamp_fully_done(dt, model, cfg):
        return

    gt = load_ground_truth(cfg["data_folder"], dt, datetime_format, cfg["hdf_key"])
    if gt is None:
        print(f"[WARNING] No ground truth for {date_str} {time_str}", file=sys.stderr)
        return

    any_done = False
    for lead_time in lead_times:
        # Skip this lead_time if all threshold rows are already recorded
        if lead_time_done(dt, model, cfg, lead_time):
            continue

        pred_dt = dt - timedelta(minutes=lead_time)
        pred = load_prediction(
            cfg["predictions_root"], model, pred_dt, datetime_format
        )
        if pred is None:
            continue

        if pred.ndim != 3 or pred.shape[0] < lead_time // 5:
            print(
                f"[WARNING] Unexpected prediction shape {pred.shape} for "
                f"{model} lt{lead_time} @ {pred_dt.strftime(datetime_format)}",
                file=sys.stderr,
            )
            continue

        channel_idx = lead_time // 5 - 1
        fct = pred[channel_idx]

        # Handle transposed arrays (W×H instead of H×W)
        if fct.shape != gt.shape:
            if fct.T.shape == gt.shape:
                fct = fct.T
            else:
                print(
                    f"[WARNING] Irreconcilable shape mismatch: gt={gt.shape}, "
                    f"fct={fct.shape} for {model} lt{lead_time}",
                    file=sys.stderr,
                )
                continue

        for thr in thresholds:
            n_valid = int(np.sum(np.isfinite(gt) & (gt >= thr)))
            fss_values = [compute_fss(gt, fct, thr, sc) for sc in scales]
            path = csv_path(cfg["fss_root"], model, lead_time, thr, dt)
            append_csv_row(path, date_str, time_str, n_valid, fss_values, scales)

        any_done = True

    if any_done:
        print(f"[INFO] {model} @ {date_str} {time_str} done")


# ── Discovery ─────────────────────────────────────────────────────────────────


def discover_gt_timestamps(data_folder: str, datetime_format: str) -> list[datetime]:
    """Return sorted list of all datetimes for which ground-truth HDF files exist."""
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
        "models",
    ]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")
    return cfg


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute FSS metrics for nowcasting models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model", required=True, help="Model name (e.g. ConvLSTM)")
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "fss_config.yaml"),
        help="Path to config YAML (default: fss_config.yaml next to this script)",
    )
    parser.add_argument("--date", help="Ground-truth date DD-MM-YYYY (crontab mode)")
    parser.add_argument(
        "--time", dest="time_", metavar="TIME", help="Ground-truth time HH:MM (crontab mode)"
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Process all available ground-truth files (initial run)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.model not in cfg["models"]:
        print(
            f"[ERROR] Model '{args.model}' not in config. Known models: {cfg['models']}",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.backfill:
        timestamps = discover_gt_timestamps(cfg["data_folder"], cfg["datetime_format"])
        if not timestamps:
            print(
                f"[WARNING] No HDF ground-truth files found in {cfg['data_folder']}",
                file=sys.stderr,
            )
            sys.exit(0)
        print(
            f"[INFO] Backfill: {len(timestamps)} ground-truth timestamps found — "
            f"processing {args.model}"
        )
        for dt in timestamps:
            process_timestamp(dt, args.model, cfg)
        print(f"[INFO] Backfill complete for {args.model}")

    else:
        if not args.date or not args.time_:
            parser.error("--date and --time are required in crontab mode (or use --backfill)")
        dt_str = f"{args.date}-{args.time_.replace(':', '-')}"
        try:
            dt = datetime.strptime(dt_str, cfg["datetime_format"])
        except ValueError:
            print(
                f"[ERROR] Cannot parse '{dt_str}' with format '{cfg['datetime_format']}'",
                file=sys.stderr,
            )
            sys.exit(1)
        process_timestamp(dt, args.model, cfg)


if __name__ == "__main__":
    main()