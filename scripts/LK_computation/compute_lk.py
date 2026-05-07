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


# ── Duplicate frame detection ─────────────────────────────────────────────────

def deduplicate_frames(raw_frames: list[np.ndarray], cfg: dict) -> list[np.ndarray]:
    """
    Remove consecutive near-identical frames caused by radars that update every
    10 minutes instead of 5.  Two frames are considered duplicates when their
    maximum absolute difference is below `duplicate_threshold`.
    """
    threshold = cfg["data"].get("duplicate_threshold", 0.05)
    unique = [raw_frames[0]]
    n_dropped = 0
    for frame in raw_frames[1:]:
        if np.max(np.abs(frame - unique[-1])) < threshold:
            n_dropped += 1
            logger.warning("Duplicate frame dropped (max diff < %.3f)", threshold)
        else:
            unique.append(frame)
    if n_dropped:
        logger.info("Deduplication: kept %d / %d frames", len(unique), len(raw_frames))
    if len(unique) < 2:
        logger.warning("Only %d unique frame(s) after dedup — using all original frames", len(unique))
        return raw_frames
    return unique


# ── Rain mask ────────────────────────────────────────────────────────────────

def apply_rain_mask(flow: np.ndarray, raw_frames: list[np.ndarray], cfg: dict) -> np.ndarray:
    """
    Zero out flow vectors in pixels with no precipitation.

    Strategy:
      1. Compute the pixel-wise maximum SRI across all frames in the sequence.
         Using the max (rather than just the last frame) avoids masking areas
         where rain existed earlier but moved out of frame by the last step.
      2. Threshold at `threshold_mm_h` to get a binary rain mask.
      3. Dilate by `buffer_km` pixels (1 pixel ≈ 1 km) so motion vectors are
         visible in a halo around each rain cell — this looks better in the
         particle animation and is meteorologically meaningful (it shows where
         the system is heading).
    """
    from scipy.ndimage import binary_dilation

    mask_cfg = cfg.get("rain_mask", {})
    if not mask_cfg.get("enabled", True):
        return flow

    thr = mask_cfg.get("threshold_mm_h", 0.2)
    buffer_px = int(mask_cfg.get("buffer_km", 35))   # 1 km ≈ 1 pixel

    # Max SRI across the whole sequence
    seq_max = np.max(np.stack(raw_frames, axis=0), axis=0)
    rain_binary = seq_max >= thr

    if buffer_px > 0:
        # Circular structuring element
        r = buffer_px
        y, x = np.ogrid[-r:r + 1, -r:r + 1]
        disk = (x ** 2 + y ** 2) <= r ** 2
        rain_mask = binary_dilation(rain_binary, structure=disk).astype(np.float32)
    else:
        rain_mask = rain_binary.astype(np.float32)

    masked = flow * rain_mask[np.newaxis, :, :]   # broadcast over (2, H, W)
    pct_kept = 100.0 * float(np.sum(rain_mask > 0)) / rain_mask.size
    logger.info(
        "Rain mask: %.1f%% of pixels kept (thr=%.2f mm/h, buffer=%dpx, seq_max=[%.2f,%.2f])",
        pct_kept, thr, buffer_px,
        float(np.min(seq_max)), float(np.max(seq_max)),
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


# ── Arrow PNG ─────────────────────────────────────────────────────────────────

# Dark navy → deep blue → purple → red.
# Deliberately different from SRI precipitation colors (light-blue → green → yellow → red)
# and from the leaflet-velocity particle colormap (blue-green rainbow).
# Near-zero speeds are near-black so they don't look like rain data.
_ARROW_COLORS = [
    ( 8/255,   0/255,  60/255),   # near-black dark navy  (≈ 0 m/s)
    (30/255,  50/255, 200/255),   # deep blue             (≈ 5 m/s)
    (120/255, 10/255, 200/255),   # purple                (≈ 12 m/s)
    (220/255,  0/255,  60/255),   # deep red              (≈ 20 m/s)
    (255/255, 30/255,   0/255),   # bright red            (≈ 25 m/s)
]


def save_arrow_png(u_grid: np.ndarray, v_grid: np.ndarray,
                   cfg: dict, ref_dt: datetime) -> None:
    """
    Render a quiver-arrow plot and save as a transparent PNG alongside the JSON.

    Design choices:
    - ALL arrows have the SAME display length (= 70% of grid-cell spacing).
      Direction shows motion; COLOR shows speed.  This avoids the problem where
      slow arrows are invisible dots and fast ones overlap neighbours.
    - Colormap goes near-black-navy → blue → purple → red, keeping it clearly
      distinct from both the SRI precipitation palette and the leaflet-velocity
      particle rainbow.
    - The image bounds exactly match output_grid so it overlays on the Leaflet
      map as a pixel-perfect ImageOverlay.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        logger.warning("matplotlib not available — skipping PNG generation")
        return

    grid = cfg["output_grid"]
    lo1, lo2 = grid["lo1"], grid["lo2"]
    la1, la2 = grid["la1"], grid["la2"]   # la1=47.75 (north), la2=35.5 (south)
    nx,  ny  = grid["nx"],  grid["ny"]    # 60, 50

    lons = np.linspace(lo1, lo2, nx)
    lats = np.linspace(la1, la2, ny)     # descending: north → south
    lons_2d, lats_2d = np.meshgrid(lons, lats)

    speed = np.sqrt(u_grid ** 2 + v_grid ** 2)

    # ── Downsample (every 2nd grid point → 30×25 = 750 arrows) ──────────────
    step = 2
    u_s  = u_grid [::step, ::step].copy().astype(np.float64)
    v_s  = v_grid [::step, ::step].copy().astype(np.float64)
    s_s  = speed  [::step, ::step]
    ln_s = lons_2d[::step, ::step]
    la_s = lats_2d[::step, ::step]

    # ── Normalize to fixed display length ────────────────────────────────────
    # Cell spacing between sampled points in degrees
    cell_lon = (lo2 - lo1) / (nx - 1) * step   # ≈ 0.508°
    cell_lat = abs(la1 - la2) / (ny - 1) * step # ≈ 0.500°

    # Arrow length = 70% of cell spacing so adjacent arrows never touch
    arrow_len_lon = cell_lon * 0.70
    arrow_len_lat = cell_lat * 0.70

    masked = s_s < 0.05   # rain-masked / no-flow pixels → skip
    speed_safe = np.where(masked, 1.0, s_s)  # avoid div-by-zero
    u_plot = np.where(masked, np.nan, (u_s / speed_safe) * arrow_len_lon)
    v_plot = np.where(masked, np.nan, (v_s / speed_safe) * arrow_len_lat)

    # ── Figure: axes fill the entire canvas, no margins ──────────────────────
    # At 150 dpi: 6 in × 150 = 900 px wide, 5 in × 150 = 750 px tall.
    dpi = 150
    fig = plt.figure(figsize=(nx / 10, ny / 10), dpi=dpi)
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(lo1, lo2)
    ax.set_ylim(la2, la1)   # south at bottom, north at top — Leaflet orientation
    ax.axis("off")
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    cmap    = mcolors.LinearSegmentedColormap.from_list("lk_arrows", _ARROW_COLORS)
    max_vel = 25.0   # m/s — upper end of the color scale

    # scale=1, scale_units='xy': the plotted vector length IS the data value
    # (in degrees), so u_plot/v_plot already encode the display length.
    # width is in fraction of axes width (default matplotlib behaviour).
    # headwidth/headlength are multiples of width.
    # Target: shaft ≈ 75% of arrow, head ≈ 25%.
    ax.quiver(
        ln_s, la_s, u_plot, v_plot, s_s,
        cmap=cmap,
        norm=mcolors.Normalize(vmin=0, vmax=max_vel),
        scale=1,
        scale_units="xy",
        angles="xy",
        width=0.0015,          # shaft ≈ 1.35 px at 900 px wide → thin clean line
        headwidth=4,           # head ≈ 5.4 px wide
        headlength=5,          # head ≈ 6.75 px long
        headaxislength=4.5,    # slightly recessed so shaft tip is visible
        pivot="tail",          # arrow base sits at the grid point
        alpha=0.92,
    )

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    png_name   = ref_dt.strftime("%d-%m-%Y-%H-%M.png")
    png_path   = out_dir / png_name
    latest_png = out_dir / "latest_flow.png"

    save_kw = dict(dpi=dpi, transparent=True, bbox_inches=None, pad_inches=0)
    fig.savefig(str(png_path),   **save_kw)
    logger.info("Saved PNG  → %s", png_path)
    fig.savefig(str(latest_png), **save_kw)

    plt.close(fig)


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

    # 2. Read all raw frames
    raw_frames = [read_hdf(p, cfg) for p in paths]
    logger.info("Frames loaded — shape: %s, SRI range: [%.3f, %.3f]",
                raw_frames[0].shape, float(np.min(raw_frames[-1])), float(np.max(raw_frames[-1])))

    # 3. Drop consecutive duplicates (10-min radars repeat the same field at 5-min slots)
    raw_frames = deduplicate_frames(raw_frames, cfg)

    # 4. Preprocess for LK
    frames = [preprocess(r, cfg) for r in raw_frames]

    # 5. Lucas-Kanade
    flow_pix = compute_lk(frames, cfg)

    # 6. Mask flow: zero outside rain areas (max of sequence + buffer)
    flow_pix = apply_rain_mask(flow_pix, raw_frames, cfg)

    # 7. Convert to m/s
    flow_ms = pixels_to_ms(flow_pix, cfg)

    # 8. Reproject to lat/lon grid
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

    # 8. Save JSON
    save_output(payload, cfg, ref_dt)

    # 9. Save PNG arrow overlay
    if cfg["output"].get("save_png", True):
        save_arrow_png(u_grid, v_grid, cfg, ref_dt)

    # 10. Notify
    if not args.no_notify:
        notify_backend(cfg)


if __name__ == "__main__":
    main()
