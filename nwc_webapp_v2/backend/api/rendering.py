"""
Rendering API endpoints - radar images, overlays, GIFs.

KEY CONCEPT: In Streamlit, the server generates matplotlib figures and
embeds them directly in the page (st.image, st.pyplot). Here, the server
generates images and serves them as files (PNG, GIF) that the browser
downloads and displays.

The browser requests:  GET /api/render/overlay/ConvLSTM/2025-01-01T12:00?lead_time=6
The server responds:   a PNG image (binary data)
The browser displays:  <img src="/api/render/overlay/..."> or uses it as a Leaflet overlay

This is more efficient because:
1. The browser caches images (doesn't re-request if already loaded)
2. Images load in parallel (Streamlit loads them sequentially)
3. The map overlay is just an image URL (no base64 encoding needed)

PERFORMANCE OPTIMIZATIONS:
- Radar mask cached in memory (loaded once from HDF5, reused for all frames)
- Warp lookup tables cached in memory (pyproj on 1.68M points, computed once)
- PNG compression level 1 (fastest — ~4x faster than default level 6)
- Cache-Control headers (browser caches images → instant on revisit)
- Response instead of StreamingResponse (slightly less overhead)
"""
import io
from datetime import datetime, timedelta
from pathlib import Path

import h5py
import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response, StreamingResponse
from PIL import Image
from pyproj import Proj

from nwc_webapp.config.config import get_config
from nwc_webapp.data.checking import check_single_prediction_exists
from nwc_webapp.pages.nowcasting.gif_creation import check_gifs_exist, get_gif_paths
from nwc_webapp.rendering.colormaps import cmap, norm

router = APIRouter(prefix="/api/render", tags=["rendering"])


# ==========================================================================
# Cached helpers — loaded once, reused for every request
# ==========================================================================

_radar_mask = None


def _get_radar_mask():
    """
    Load the radar mask from HDF5 and cache it in memory.

    WHY: The radar mask is a 1400x1200 array (~13 MB) that never changes.
    Without caching, every overlay request opens the HDF5 file and reads it
    from disk — 25 times in parallel when preloading all frames. With caching,
    it's one disk read for the lifetime of the server.
    """
    global _radar_mask
    if _radar_mask is not None:
        return _radar_mask

    config = get_config()
    mask_path = config.radar_mask_path
    if mask_path.exists():
        with h5py.File(mask_path, "r") as f:
            _radar_mask = f["mask"][()]
    return _radar_mask


# Cached warp lookup tables — (col_indices, line_indices, valid_mask)
# These depend only on the projection parameters, not on the data.
# Computing them requires pyproj on 1.68M points; we do it once at startup.
_warp_lookup = None


def _get_warp_lookup():
    """
    Precompute and cache the nearest-neighbor sampling indices for reprojection.

    WHY: The radar data is in Transverse Mercator (tmerc, lat_0=42°, lon_0=12.5°).
    Leaflet's ImageOverlay treats images as equirectangular, so we must reproject
    to a regular lat/lon grid before serving. The index lookup (pyproj on 1.68M
    points + grid arithmetic) is the expensive part and is always the same —
    computing it once and reusing it makes each warp ~5-10x faster.

    Returns:
        (col_indices, line_indices, valid_mask, nlines_dst, ncols_dst)
    """
    global _warp_lookup
    if _warp_lookup is not None:
        return _warp_lookup

    config = get_config()
    src = config.source_grid
    dst = config.dest_grid

    source_proj = Proj(proj="tmerc", lat_0=src.prj_lat, lon_0=src.prj_lon, x_0=0, y_0=0, ellps="WGS84")

    dest_lons = np.linspace(dst.minLon, dst.maxLon, dst.ncols)
    dest_lats = np.linspace(dst.maxLat, dst.minLat, dst.nlines)
    dest_lon_grid, dest_lat_grid = np.meshgrid(dest_lons, dest_lats)

    dest_x, dest_y = source_proj(dest_lon_grid, dest_lat_grid)

    col_indices = ((dest_x / src.cRes) + src.cOff).astype(int)
    line_indices = ((dest_y / src.lRes) + src.lOff).astype(int)

    valid_mask = (
        (col_indices >= 0) & (col_indices < src.ncols) &
        (line_indices >= 0) & (line_indices < src.nlines)
    )

    _warp_lookup = (col_indices, line_indices, valid_mask, dst.nlines, dst.ncols)
    return _warp_lookup


def _warp_frame(frame: np.ndarray) -> np.ndarray:
    """
    Reproject a 2D Transverse Mercator frame to equirectangular lat/lon using
    cached nearest-neighbor lookup tables.

    The raw radar data is in tmerc projection. Leaflet's ImageOverlay stretches
    images between two lat/lon corners assuming equirectangular layout. Without
    reprojection the overlay is misaligned (shifted/distorted) relative to map
    features.

    Row 0 of the output = northernmost latitude, matching Leaflet's
    ImageOverlay convention (row 0 = top = NE corner of bounds).

    Note: the original warp_map() in geo/warping.py does np.flipud to
    match Folium's origin="lower" convention (row 0 = south). Leaflet
    uses the opposite convention, so no flip is applied here.

    Returns NaN for pixels that fall outside the source grid bounds.
    """
    col_idx, line_idx, valid, nlines_dst, ncols_dst = _get_warp_lookup()

    warped = np.full((nlines_dst, ncols_dst), np.nan, dtype=float)
    warped[valid] = frame[line_idx[valid], col_idx[valid]]

    return warped


def _frame_to_png_bytes(frame):
    """
    Convert a 2D numpy array to RGBA PNG bytes using the radar colormap.

    Shared logic for both groundtruth and prediction overlays:
    1. Normalize values using the non-linear CustomNorm (0→1→2→5→10→...→100)
    2. Apply the LinearSegmentedColormap to get RGBA float array
    3. Make no-precipitation pixels fully transparent
    4. Encode as PNG with fast compression (level 1)

    Returns bytes (not a buffer), suitable for Response(content=...).
    """
    normalized = norm(frame)
    rgba = cmap(normalized)  # Float RGBA array (0-1)
    # Transparent where no precipitation or outside the radar domain (NaN from warp)
    rgba[~np.isfinite(frame) | (frame <= 0)] = [0, 0, 0, 0]

    img = Image.fromarray((rgba * 255).astype(np.uint8))

    buffer = io.BytesIO()
    # compress_level=1 is ~4x faster than the default (6) with only
    # marginally larger files (~2.4 MB vs ~2.2 MB for 1400x1200 RGBA)
    img.save(buffer, format="PNG", compress_level=1)
    return buffer.getvalue()


# Cache-Control header: these images are deterministic (same URL = same image)
# so we can cache aggressively. 1 hour is safe — if new data arrives, the
# frontend builds new URLs with different timestamps.
_CACHE_HEADERS = {"Cache-Control": "public, max-age=3600"}


# ==========================================================================
# Overlay endpoints
# ==========================================================================

@router.get("/overlay/groundtruth/{timestamp}")
async def get_groundtruth_overlay(timestamp: str):
    """
    Generate a groundtruth (SRI) overlay image (RGBA PNG) for the map.

    Loads the raw SRI HDF5 file for a given timestamp and renders it
    with the same colormap used for predictions. This powers the "past"
    section of the timeline slider (-60 to 0 minutes).
    """
    try:
        dt = datetime.fromisoformat(timestamp)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid datetime: {e}")

    config = get_config()
    sri_filename = dt.strftime("%d-%m-%Y-%H-%M") + ".hdf"
    sri_path = Path(str(config.sri_folder)) / sri_filename

    if not sri_path.exists():
        raise HTTPException(status_code=404, detail=f"SRI file not found: {sri_filename}")

    with h5py.File(sri_path, "r") as f:
        if "dataset1/data1/data" in f:
            frame = f["dataset1/data1/data"][()].astype(float)
        else:
            raise HTTPException(status_code=500, detail="Unknown HDF5 structure")

    # Apply radar mask (cached in memory)
    mask = _get_radar_mask()
    if mask is not None:
        frame = frame * mask

    frame[frame < 0] = 0
    frame = np.clip(frame, 0, 200)

    # Reproject from Transverse Mercator to equirectangular lat/lon
    frame = _warp_frame(frame)

    png_bytes = _frame_to_png_bytes(frame)
    return Response(content=png_bytes, media_type="image/png", headers=_CACHE_HEADERS)


@router.get("/overlay/{model}/{timestamp}")
async def get_radar_overlay(
    model: str,
    timestamp: str,
    lead_time: int = Query(0, description="Lead time index (0-11, where 0=+5min, 5=+30min, 11=+60min)"),
    frame_type: str = Query("prediction", description="'prediction' (default) or 'groundtruth' (Test model only)"),
):
    """
    Generate a radar prediction overlay image (RGBA PNG) for the map.

    Here: server returns a PNG, the browser uses it directly as a Leaflet image overlay.

    TEST MODEL: Uses a single static 'predictions.npy' file (shape 24, H, W).
    Indices 0-11 are groundtruth, 12-23 are predictions.
      frame_type='groundtruth' → predictions.npy[lead_time]      (0-11)
      frame_type='prediction'  → predictions.npy[12 + lead_time]  (12-23)
    """
    try:
        dt = datetime.fromisoformat(timestamp)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid datetime: {e}")

    config = get_config()

    from nwc_webapp.data.predictions import load_prediction_array

    # Test model: load from static predictions.npy (24 frames: 0-11 GT, 12-23 predictions)
    if model.upper() == "TEST":
        static_path = config.real_time_pred / "Test" / "predictions.npy"
        if static_path.exists():
            full_array = np.load(static_path, mmap_mode="r")
            # Handle both (24, H, W) and (24, 1, H, W) shapes
            if full_array.ndim == 4:
                full_array = full_array[:, 0]  # (24, H, W)

            if frame_type == "groundtruth":
                # Groundtruth: indices 0-11
                arr_index = lead_time
            else:
                # Prediction: indices 12-23
                arr_index = 12 + lead_time

            if arr_index < 0 or arr_index >= full_array.shape[0]:
                raise HTTPException(status_code=400, detail=f"lead_time {lead_time} out of range for frame_type={frame_type}")
            frame = np.array(full_array[arr_index])
        else:
            # Fallback to per-timestamp file (mock/local mode)
            pred_filename = dt.strftime("%d-%m-%Y-%H-%M") + ".npy"
            pred_path = config.real_time_pred / model / pred_filename
            if not pred_path.exists():
                raise HTTPException(status_code=404, detail=f"Test predictions not found")
            pred_array = load_prediction_array(pred_path, model)
            if pred_array is None:
                raise HTTPException(status_code=500, detail="Failed to load prediction array")
            frame = pred_array[lead_time]
    else:
        pred_filename = dt.strftime("%d-%m-%Y-%H-%M") + ".npy"
        pred_path = config.real_time_pred / model / pred_filename

        if not pred_path.exists():
            raise HTTPException(status_code=404, detail=f"Prediction not found: {pred_filename}")

        pred_array = load_prediction_array(pred_path, model)
        if pred_array is None:
            raise HTTPException(status_code=500, detail="Failed to load prediction array")

        if lead_time < 0 or lead_time >= pred_array.shape[0]:
            raise HTTPException(status_code=400, detail=f"lead_time must be 0-{pred_array.shape[0]-1}")

        frame = pred_array[lead_time]

    # Apply radar mask (cached in memory)
    mask = _get_radar_mask()
    if mask is not None:
        frame = frame * mask

    frame = np.clip(frame, 0, 200)

    # Reproject from Transverse Mercator to equirectangular lat/lon
    frame = _warp_frame(frame)

    png_bytes = _frame_to_png_bytes(frame)
    return Response(content=png_bytes, media_type="image/png", headers=_CACHE_HEADERS)


# ==========================================================================
# Figure endpoints
# ==========================================================================

@router.get("/figure/{model}/{timestamp}")
async def get_prediction_figure(
    model: str,
    timestamp: str,
    lead_time: int = Query(0, description="Lead time index (0-11)"),
    figure_type: str = Query("prediction", description="Type: prediction, groundtruth, or difference"),
):
    """
    Generate a matplotlib figure (PNG) for a prediction, groundtruth, or difference.

    This replaces compute_figure_gpd() calls scattered through the UI code.
    The browser just shows: <img src="/api/render/figure/ConvLSTM/2025-01-01T12:00?lead_time=5">
    """
    try:
        dt = datetime.fromisoformat(timestamp)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid datetime: {e}")

    from nwc_webapp.rendering.figures import compute_figure_gpd
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for server-side rendering

    config = get_config()

    if figure_type == "groundtruth":
        from nwc_webapp.data.groundtruth import load_groundtruth_for_timestamp
        gt_data = load_groundtruth_for_timestamp(dt)
        if gt_data is None:
            raise HTTPException(status_code=404, detail="Groundtruth data not found")
        frame = gt_data[lead_time]
        title = f"GT +{(lead_time+1)*5}min - {(dt + timedelta(minutes=(lead_time+1)*5)).strftime('%d/%m/%Y %H:%M')}"
        fig = compute_figure_gpd(frame, title, name="")
    elif figure_type == "prediction":
        from nwc_webapp.data.predictions import load_prediction_array
        pred_path = config.real_time_pred / model / (dt.strftime("%d-%m-%Y-%H-%M") + ".npy")
        if not pred_path.exists():
            raise HTTPException(status_code=404, detail="Prediction not found")
        pred_array = load_prediction_array(pred_path, model)
        frame = pred_array[lead_time]
        # Apply radar mask (same as groundtruth and overlay endpoints)
        mask = _get_radar_mask()
        if mask is not None:
            frame = frame * mask
        frame = np.clip(frame, 0, 200)
        title = f"{model} +{(lead_time+1)*5}min - {(dt + timedelta(minutes=(lead_time+1)*5)).strftime('%d/%m/%Y %H:%M')}"
        fig = compute_figure_gpd(frame, title, name="")
    else:
        raise HTTPException(status_code=400, detail="figure_type must be: prediction, groundtruth, or difference")

    # Convert figure to PNG
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
    fig.clear()
    import matplotlib.pyplot as plt
    plt.close(fig)
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="image/png")


# ==========================================================================
# GIF endpoints
# ==========================================================================

@router.get("/gifs/check")
async def check_gifs(
    model: str = Query(...),
    start: str = Query(...),
    end: str = Query(...),
):
    """
    Check if GIFs exist for a model and date range.

    In Streamlit (nowcasting.py):
        gif_paths = get_gif_paths(model_name, start_dt, end_dt)
        gt_exist, pred_exist, diff_exist = check_gifs_exist(gif_paths)
    """
    try:
        start_dt = datetime.fromisoformat(start)
        end_dt = datetime.fromisoformat(end)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid datetime: {e}")

    gif_paths = get_gif_paths(model, start_dt, end_dt)
    gt_exist, pred_exist, diff_exist = check_gifs_exist(gif_paths)

    return {
        "model": model,
        "gt_exist": gt_exist,
        "pred_exist": pred_exist,
        "diff_exist": diff_exist,
        "all_exist": gt_exist and pred_exist and diff_exist,
        "paths": {k: str(v) for k, v in gif_paths.items()},
    }


@router.get("/gifs/file")
async def get_gif_file(
    path: str = Query(..., description="Relative path to GIF file"),
):
    """
    Serve a GIF file. The frontend displays it as: <img src="/api/render/gifs/file?path=...">
    """
    config = get_config()
    # Resolve relative to gif_storage
    gif_path = Path(path)

    if not gif_path.exists():
        raise HTTPException(status_code=404, detail=f"GIF not found: {path}")

    return StreamingResponse(
        open(gif_path, "rb"),
        media_type="image/gif",
    )