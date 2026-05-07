"""
Compute dense Lucas-Kanade optical flow from recent SRI HDF files and save
the result as a leaflet-velocity compatible JSON file.

Usage:
    python compute_lk.py --config lk_config.yaml
    python compute_lk.py --config lk_config.yaml --timestamp 2026-05-07T14:30
    python compute_lk.py --config lk_config.yaml --dry-run

If --timestamp is omitted the script uses the current time (rounded down to
the nearest 5-minute slot) as the reference and searches backwards for the
N most recent available HDF files.

Output format: two-element JSON list [U_component, V_component] compatible
with leaflet-velocity.  Each component has a 'header' dict (grid metadata)
and a flat 'data' list (m/s, row-major from northernmost row).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import h5py
import numpy as np
import yaml
from scipy.ndimage import gaussian_filter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Config helpers ────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── File discovery ────────────────────────────────────────────────────────────

def find_sri_files(cfg: dict, reference_dt: datetime) -> list[Path]:
    """
    Return the N most recent HDF files up to and including reference_dt,
    oldest first.  Searches backwards in 5-minute steps up to 3*N slots.
    """
    folder = Path(cfg["data"]["folder"])
    pattern = cfg["data"]["file_pattern"]
    n = cfg["data"]["n_frames"]
    timestep = timedelta(seconds=cfg["data"]["timestep_s"])

    found: list[tuple[datetime, Path]] = []
    dt = reference_dt
    search_limit = n * 3   # allow for some gaps in the archive

    for _ in range(search_limit):
        filename = dt.strftime(pattern)
        path = folder / filename
        if path.exists():
            found.append((dt, path))
            if len(found) == n:
                break
        dt -= timestep

    if not found:
        raise FileNotFoundError(
            f"No SRI files found in {folder} looking back from {reference_dt}"
        )

    if len(found) < 2:
        raise RuntimeError(
            f"Need at least 2 frames for optical flow, found {len(found)}"
        )

    found.sort(key=lambda x: x[0])   # oldest first
    paths = [p for _, p in found]
    logger.info("Using %d frames: %s … %s", len(paths), paths[0].name, paths[-1].name)
    return paths


# ── HDF reading ───────────────────────────────────────────────────────────────

def read_hdf(path: Path, cfg: dict) -> np.ndarray:
    """Read a 2D field from an HDF5 file and apply scale/offset."""
    key = cfg["data"]["hdf_key"]
    scale = cfg["data"].get("scale_factor", 1.0)
    offset = cfg["data"].get("offset", 0.0)

    with h5py.File(path, "r") as f:
        raw = f[key][:]

    data = raw.astype(np.float32)
    if scale != 1.0:
        data *= scale
    if offset != 0.0:
        data += offset
    return data


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(data: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Threshold → optional log transform → Gaussian smoothing.

    Log transform converts mm/h to a dBR-like scale so that LK tracks
    structure across a wide dynamic range of rain rates.
    """
    pcfg = cfg["preprocessing"]

    threshold = pcfg.get("threshold_mm_h", 0.1)
    data = np.where(data < threshold, 0.0, data)

    if pcfg.get("log_transform", True):
        eps = pcfg.get("log_epsilon", 0.01)
        # Shift so that zero-rain (= 0 + eps) maps to 0 after subtraction
        data = 10.0 * np.log10(data + eps) - 10.0 * np.log10(eps)
        data = np.maximum(data, 0.0)

    sigma = pcfg.get("smoothing_sigma", 1.0)
    if sigma > 0:
        data = gaussian_filter(data, sigma=sigma)

    return data


# ── Rain mask ────────────────────────────────────────────────────────────────

def apply_rain_mask(flow: np.ndarray, latest_raw: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Zero out flow vectors in pixels where the most recent SRI frame has no rain.

    With soft=True the mask fades linearly from 0 at `threshold_mm_h` to 1 at
    `soft_max_mm_h`, giving a smooth edge instead of a hard cut.  This avoids
    visible discontinuities in the particle animation at the rain/no-rain boundary.

    `latest_raw` must be the raw SRI in mm/h (before log transform).
    """
    mask_cfg = cfg.get("rain_mask", {})
    if not mask_cfg.get("enabled", True):
        return flow

    thr = mask_cfg.get("threshold_mm_h", 0.5)

    if mask_cfg.get("soft", True):
        soft_max = mask_cfg.get("soft_max_mm_h", 2.0)
        # Linear ramp: 0 below thr, 1 above soft_max
        weight = np.clip((latest_raw - thr) / max(soft_max - thr, 1e-6), 0.0, 1.0)
    else:
        weight = (latest_raw >= thr).astype(np.float32)

    masked = flow * weight[np.newaxis, :, :]   # broadcast over (2, H, W)
    n_masked = int(np.sum(weight == 0))
    logger.info(
        "Rain mask applied — %.1f%% of pixels zeroed (threshold %.2f mm/h, soft=%s)",
        100.0 * n_masked / weight.size,
        thr,
        mask_cfg.get("soft", True),
    )
    return masked


# ── Lucas-Kanade ──────────────────────────────────────────────────────────────

def compute_lk(frames: list[np.ndarray], cfg: dict) -> np.ndarray:
    """
    Run pysteps dense Lucas-Kanade on a sequence of preprocessed frames.

    Returns flow of shape (2, H, W):
      flow[0] = column-direction displacement (pixels / timestep, east = positive)
      flow[1] = row-direction displacement    (pixels / timestep, down  = positive)
    """
    import inspect
    from pysteps.motion.lucaskanade import dense_lucaskanade

    stack = np.stack(frames, axis=0)   # (N, H, W)
    lk = cfg.get("lk", {})

    # Only pass kwargs that this installed version of pysteps actually accepts
    valid_params = inspect.signature(dense_lucaskanade).parameters
    candidates = {
        "fd_method": lk.get("fd_method"),
        "nr_levels": lk.get("nr_levels"),
        "nr_iter": lk.get("nr_iter"),
        "nr_features": lk.get("nr_features"),
        "max_speed_kmh": lk.get("max_speed_kmh"),
    }
    kwargs = {k: v for k, v in candidates.items() if k in valid_params and v is not None}
    if len(kwargs) < len([v for v in candidates.values() if v is not None]):
        skipped = [k for k, v in candidates.items() if v is not None and k not in valid_params]
        logger.warning("pysteps ignored unsupported kwargs: %s", skipped)

    flow = dense_lucaskanade(stack, **kwargs)

    logger.info(
        "LK flow — u: [%.2f, %.2f] pix/step, v: [%.2f, %.2f] pix/step",
        float(np.nanmin(flow[0])), float(np.nanmax(flow[0])),
        float(np.nanmin(flow[1])), float(np.nanmax(flow[1])),
    )
    return flow   # (2, H, W), pixels per timestep


# ── Unit conversion ───────────────────────────────────────────────────────────

def pixels_to_ms(flow: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Convert LK flow from pixels/timestep to geographic m/s.

    Convention (matching leaflet-velocity / meteorological):
      u  =  eastward  velocity  (m/s, positive = east)
      v  =  northward velocity  (m/s, positive = north)

    The radar grid convention (from coordinates.py):
      x  = (col - xoff) * xres      →  col+ = x+ = east   → u = flow[0] * xres / dt
      y  = (row - yoff) * yres      →  row+ = y- = south  → v = flow[1] * yres / dt
    Since yres is negative, flow[1] > 0 (southward) gives v < 0 (correct sign).
    """
    proj = cfg["projection"]
    dt = cfg["data"]["timestep_s"]

    flow_ms = np.empty_like(flow)
    flow_ms[0] = flow[0] * proj["xres"] / dt    # u: east positive
    flow_ms[1] = flow[1] * proj["yres"] / dt    # v: north positive (yres < 0 flips sign)

    logger.info(
        "Flow in m/s — u: [%.2f, %.2f], v: [%.2f, %.2f]",
        float(np.nanmin(flow_ms[0])), float(np.nanmax(flow_ms[0])),
        float(np.nanmin(flow_ms[1])), float(np.nanmax(flow_ms[1])),
    )
    return flow_ms


# ── Reprojection to lat/lon grid ──────────────────────────────────────────────

def reproject_to_latlon(flow_ms: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Interpolate the (2, H, W) flow field from the radar TM pixel grid onto
    a regular lat/lon grid suitable for leaflet-velocity.

    Steps:
      1. Build the output lon/lat meshgrid.
      2. Convert each output point to TM coordinates (meters) via pyproj.
      3. Map TM coords to fractional radar pixel coords.
      4. Bilinear-interpolate u and v from the radar grid.
      5. Fill any edge NaNs with nearest-neighbour.
    """
    import pyproj
    from scipy.interpolate import RegularGridInterpolator, griddata

    proj_cfg = cfg["projection"]
    grid_cfg = cfg["output_grid"]

    xoff = proj_cfg["xoff"]
    xres = proj_cfg["xres"]
    yoff = proj_cfg["yoff"]
    yres = proj_cfg["yres"]

    H, W = flow_ms.shape[1], flow_ms.shape[2]
    u_field = flow_ms[0]
    v_field = flow_ms[1]

    # Output lat/lon grid (north → south, west → east)
    out_lons = np.linspace(grid_cfg["lo1"], grid_cfg["lo2"], grid_cfg["nx"])
    out_lats = np.linspace(grid_cfg["la1"], grid_cfg["la2"], grid_cfg["ny"])
    gx, gy = np.meshgrid(out_lons, out_lats)   # shapes: (ny, nx)

    # Transform each output (lon, lat) → TM (x_m, y_m)
    proj = pyproj.Proj(
        proj=proj_cfg["proj"],
        lat_0=proj_cfg["lat_0"],
        lon_0=proj_cfg["lon_0"],
    )
    tm_x, tm_y = proj(gx.ravel(), gy.ravel())   # meters from TM origin

    # TM coords → fractional radar pixel coords
    frac_col = tm_x / xres + xoff    # x_m = (col - xoff) * xres  → col = x/xres + xoff
    frac_row = tm_y / yres + yoff    # y_m = (row - yoff) * yres  → row = y/yres + yoff

    # Clamp to valid pixel range (points outside Italy's radar coverage land here)
    frac_col = np.clip(frac_col, 0, W - 1)
    frac_row = np.clip(frac_row, 0, H - 1)

    # Bilinear interpolation over the radar pixel grid
    rows_ax = np.arange(H, dtype=np.float64)
    cols_ax = np.arange(W, dtype=np.float64)

    interp_u = RegularGridInterpolator(
        (rows_ax, cols_ax), u_field.astype(np.float64),
        method="linear", bounds_error=False, fill_value=np.nan,
    )
    interp_v = RegularGridInterpolator(
        (rows_ax, cols_ax), v_field.astype(np.float64),
        method="linear", bounds_error=False, fill_value=np.nan,
    )

    query_pts = np.column_stack([frac_row, frac_col])
    u_flat = interp_u(query_pts)
    v_flat = interp_v(query_pts)

    u_grid = u_flat.reshape(grid_cfg["ny"], grid_cfg["nx"])
    v_grid = v_flat.reshape(grid_cfg["ny"], grid_cfg["nx"])

    # Fill boundary NaNs (points outside the valid TM extent) with nearest-neighbour
    nan_mask = np.isnan(u_grid)
    if nan_mask.any():
        valid = ~nan_mask
        pts_valid = np.column_stack([gx[valid], gy[valid]])
        pts_all   = np.column_stack([gx.ravel(), gy.ravel()])
        u_nn = griddata(pts_valid, u_grid[valid], pts_all, method="nearest")
        v_nn = griddata(pts_valid, v_grid[valid], pts_all, method="nearest")
        u_grid[nan_mask] = u_nn.reshape(u_grid.shape)[nan_mask]
        v_grid[nan_mask] = v_nn.reshape(v_grid.shape)[nan_mask]

    return u_grid, v_grid


# ── leaflet-velocity JSON serialisation ───────────────────────────────────────

def build_velocity_json(u_grid: np.ndarray, v_grid: np.ndarray,
                        cfg: dict, ref_time: str) -> list[dict]:
    """
    Build the two-element [U, V] list expected by leaflet-velocity.
    Matches the format produced by /api/wind/data (wind.py).
    """
    grid_cfg = cfg["output_grid"]
    nx, ny = grid_cfg["nx"], grid_cfg["ny"]
    dx = (grid_cfg["lo2"] - grid_cfg["lo1"]) / (nx - 1)
    dy = (grid_cfg["la1"] - grid_cfg["la2"]) / (ny - 1)   # positive step size

    header_base = {
        "parameterCategory": 2,
        "lo1": grid_cfg["lo1"],
        "la1": grid_cfg["la1"],
        "lo2": grid_cfg["lo2"],
        "la2": grid_cfg["la2"],
        "dx": round(dx, 6),
        "dy": round(dy, 6),
        "nx": nx,
        "ny": ny,
        "refTime": ref_time,
    }

    return [
        {
            "header": {**header_base, "parameterNumber": 2},   # U (eastward)
            "data": [round(float(v), 4) for v in u_grid.flatten()],
        },
        {
            "header": {**header_base, "parameterNumber": 3},   # V (northward)
            "data": [round(float(v), 4) for v in v_grid.flatten()],
        },
    ]


# ── Persistence ───────────────────────────────────────────────────────────────

def save_output(payload: list[dict], cfg: dict, ref_dt: datetime) -> Path:
    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    latest = out_dir / cfg["output"]["filename"]
    latest.write_text(json.dumps(payload, separators=(",", ":")))
    logger.info("Saved → %s", latest)

    if cfg["output"].get("keep_history", False):
        ts_name = ref_dt.strftime("%d-%m-%Y-%H-%M.json")
        hist = out_dir / ts_name
        hist.write_text(json.dumps(payload, separators=(",", ":")))
        logger.info("Saved history → %s", hist)

    return latest


# ── Backend notification ──────────────────────────────────────────────────────

def notify_backend(cfg: dict) -> None:
    import urllib.request
    url = cfg["notify"]["url"]
    timeout = cfg["notify"].get("timeout_s", 5)
    try:
        req = urllib.request.Request(url, method="POST", data=b"")
        with urllib.request.urlopen(req, timeout=timeout):
            pass
        logger.info("Notified backend: %s", url)
    except Exception as exc:
        logger.warning("Notify failed (%s): %s", url, exc)


# ── Entry point ───────────────────────────────────────────────────────────────

def target_timestamp(cfg: dict) -> datetime:
    """Return the most recent complete 5-min slot (now - 5 min, rounded down)."""
    now = datetime.now(tz=timezone.utc)
    dt = now.replace(second=0, microsecond=0)
    dt -= timedelta(minutes=dt.minute % 5)   # round down to 5-min boundary
    dt -= timedelta(seconds=cfg["data"]["timestep_s"])   # one slot back = the last complete one
    return dt


def poll_for_hdf(cfg: dict, ref_dt: datetime) -> Path:
    """
    Block until the HDF file for ref_dt exists on disk, then return its path.
    Raises TimeoutError if max_wait_s is exceeded.
    """
    import time

    folder = Path(cfg["data"]["folder"])
    pattern = cfg["data"]["file_pattern"]
    max_wait = cfg["polling"]["max_wait_s"]
    interval = cfg["polling"]["interval_s"]

    path = folder / ref_dt.strftime(pattern)
    logger.info("Polling for %s (timeout %ds)", path, max_wait)

    waited = 0
    while not path.exists():
        if waited >= max_wait:
            raise TimeoutError(f"HDF not found after {max_wait}s: {path}")
        time.sleep(interval)
        waited += interval

    logger.info("File arrived (+%ds)", waited)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to lk_config.yaml")
    parser.add_argument(
        "--timestamp",
        default=None,
        help="Override reference timestamp YYYY-MM-DDTHH:MM (UTC). Skips polling.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Compute and log stats but do not write output or notify.",
    )
    parser.add_argument("--no-notify", action="store_true", help="Skip backend notification.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.timestamp:
        ref_dt = datetime.strptime(args.timestamp, "%Y-%m-%dT%H:%M").replace(tzinfo=timezone.utc)
        logger.info("Manual timestamp: %s (skipping poll)", ref_dt.strftime("%Y-%m-%dT%H:%M"))
    else:
        ref_dt = target_timestamp(cfg)
        logger.info("Target timestamp: %s", ref_dt.strftime("%Y-%m-%dT%H:%M"))
        try:
            poll_for_hdf(cfg, ref_dt)
        except TimeoutError as exc:
            logger.warning("[SKIP] %s", exc)
            sys.exit(0)

    # 1. Find files
    try:
        paths = find_sri_files(cfg, ref_dt)
    except (FileNotFoundError, RuntimeError) as exc:
        logger.error("%s", exc)
        sys.exit(1)

    # 2. Read + preprocess (keep latest raw frame for rain masking)
    frames = []
    latest_raw = None
    for p in paths:
        raw = read_hdf(p, cfg)
        latest_raw = raw   # last iteration = most recent frame
        frames.append(preprocess(raw, cfg))
    logger.info("Frames loaded — shape: %s, value range: [%.3f, %.3f]",
                frames[0].shape, float(np.min(frames[-1])), float(np.max(frames[-1])))

    # 3. Lucas-Kanade
    flow_pix = compute_lk(frames, cfg)

    # 4. Mask flow to zero in non-precipitating pixels
    flow_pix = apply_rain_mask(flow_pix, latest_raw, cfg)

    # 5. Convert to m/s
    flow_ms = pixels_to_ms(flow_pix, cfg)

    # 5. Reproject to lat/lon grid
    u_grid, v_grid = reproject_to_latlon(flow_ms, cfg)
    logger.info(
        "Reprojected grid (%dx%d) — u: [%.2f, %.2f] m/s, v: [%.2f, %.2f] m/s",
        cfg["output_grid"]["nx"], cfg["output_grid"]["ny"],
        float(np.nanmin(u_grid)), float(np.nanmax(u_grid)),
        float(np.nanmin(v_grid)), float(np.nanmax(v_grid)),
    )

    # 7. Build JSON payload
    ref_time_str = ref_dt.strftime("%Y-%m-%d %H:%M:%S")
    payload = build_velocity_json(u_grid, v_grid, cfg, ref_time_str)

    if args.dry_run:
        logger.info("Dry run — skipping write and notify.")
        return

    # 8. Save
    save_output(payload, cfg, ref_dt)

    # 9. Notify
    if not args.no_notify:
        notify_backend(cfg)


if __name__ == "__main__":
    main()
