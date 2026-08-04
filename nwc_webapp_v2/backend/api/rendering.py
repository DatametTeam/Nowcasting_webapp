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
import asyncio
import io
import logging
import threading
import zipfile
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

import h5py
import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response, StreamingResponse
from PIL import Image
from pydantic import BaseModel
from pyproj import Proj

from nwc_webapp.config.config import get_config
from nwc_webapp.data.checking import check_single_prediction_exists
from nwc_webapp.rendering.colormaps import cmap, norm

router = APIRouter(prefix="/api/render", tags=["rendering"])


# ==========================================================================
# WR10 polar → Cartesian → Web Mercator rendering
# ==========================================================================

# WR10 radar parameters (from HDF5 where/what attributes)
_WR10_RADAR_LAT = 39.18960189819336
_WR10_RADAR_LON = 9.15820026397705
_WR10_N_RAYS = 360
_WR10_N_BINS = 480
_WR10_RSCALE = 150.0   # metres per range bin
_WR10_MAX_RANGE_M = _WR10_N_BINS * _WR10_RSCALE   # 72 000 m
_WR10_IMG_SIZE = 600   # pixels in the output PNG (each side)

_wr10_warp_lookup = None  # (ray_idx, bin_idx, valid_mask) — computed once


def _get_wr10_warp_lookup():
    """
    Precompute and cache the polar→Cartesian→WebMercator lookup table.

    For each pixel (row, col) of the _WR10_IMG_SIZE × _WR10_IMG_SIZE output PNG:
      1. Convert pixel position → Web Mercator metres
      2. Web Mercator → lat/lon
      3. lat/lon → metres (dx, dy) in azimuthal-equidistant projection centred on the radar
      4. (dx, dy) → (azimuth, range) → polar (ray_idx, bin_idx)

    Returns (ray_idx, bin_idx, valid_mask), all shape (_WR10_IMG_SIZE, _WR10_IMG_SIZE).
    """
    global _wr10_warp_lookup
    if _wr10_warp_lookup is not None:
        return _wr10_warp_lookup

    from pyproj import Proj

    R = 6378137.0  # WGS84 equatorial radius (Web Mercator)

    # Coverage bounding box in Web Mercator, centred on the radar
    proj_aeqd = Proj(proj="aeqd", lat_0=_WR10_RADAR_LAT, lon_0=_WR10_RADAR_LON, ellps="WGS84")
    # SW and NE corners of the ±max_range square in lat/lon
    lon_sw, lat_sw = proj_aeqd(-_WR10_MAX_RANGE_M, -_WR10_MAX_RANGE_M, inverse=True)
    lon_ne, lat_ne = proj_aeqd(+_WR10_MAX_RANGE_M, +_WR10_MAX_RANGE_M, inverse=True)

    # Web Mercator bounds
    x_sw = R * np.radians(lon_sw)
    y_sw = R * np.log(np.tan(np.pi / 4 + np.radians(lat_sw) / 2))
    x_ne = R * np.radians(lon_ne)
    y_ne = R * np.log(np.tan(np.pi / 4 + np.radians(lat_ne) / 2))

    # Destination grid: uniform in Web Mercator (north→south rows, west→east cols)
    dest_x = np.linspace(x_sw, x_ne, _WR10_IMG_SIZE)
    dest_y = np.linspace(y_ne, y_sw, _WR10_IMG_SIZE)   # y_ne > y_sw → top = north
    dest_xm, dest_ym = np.meshgrid(dest_x, dest_y)

    # Web Mercator → lat/lon
    dest_lon = np.degrees(dest_xm / R)
    dest_lat = np.degrees(2 * np.arctan(np.exp(dest_ym / R)) - np.pi / 2)

    # lat/lon → aeqd metres (dx=East, dy=North) relative to radar centre
    dx, dy = proj_aeqd(dest_lon, dest_lat)

    # Polar coordinates: azimuth (0° = North, clockwise), range in metres
    range_m = np.sqrt(dx ** 2 + dy ** 2)
    azimuth_deg = np.degrees(np.arctan2(dx, dy)) % 360.0  # arctan2(East, North)

    # Map to integer polar indices
    ray_idx = (azimuth_deg * _WR10_N_RAYS / 360.0).astype(int) % _WR10_N_RAYS
    bin_idx = (range_m / _WR10_RSCALE).astype(int)

    # Valid only within the radar's measurement range
    valid = (range_m < _WR10_MAX_RANGE_M) & (bin_idx >= 0) & (bin_idx < _WR10_N_BINS)

    _wr10_warp_lookup = (ray_idx, bin_idx, valid)
    return _wr10_warp_lookup


def _render_wr10_frame(file_path: Path, legend_name: str) -> bytes:
    """
    Load a WR10 HDF5 file and return a Web-Mercator PNG overlay.

    Steps:
      1. Read uint8 polar data (360 × 480)
      2. Read gain/offset from HDF5 attributes and convert to physical values
      3. Mark no-data pixels (raw == nodata) as NaN
      4. Reproject via cached lookup table → _WR10_IMG_SIZE × _WR10_IMG_SIZE
      5. Apply colormap and encode as PNG
    """
    import h5py

    with h5py.File(file_path, "r") as f:
        raw = f["dataset1/data1/data"][()].astype(float)
        what = f["dataset1/data1/what"]
        gain = float(what.attrs.get("gain", 0.5))
        offset = float(what.attrs.get("offset", -32.0))
        nodata = float(what.attrs.get("nodata", 0.0))
        undetect = float(what.attrs.get("undetect", 0.0))

    # Convert to physical values; mark no-data/undetect as NaN
    no_signal = (raw == nodata) | (raw == undetect)
    physical = offset + gain * raw
    physical[no_signal] = np.nan

    # Polar → Web Mercator (nearest-neighbour lookup)
    ray_idx, bin_idx, valid = _get_wr10_warp_lookup()
    warped = np.full((_WR10_IMG_SIZE, _WR10_IMG_SIZE), np.nan, dtype=float)
    warped[valid] = physical[ray_idx[valid], bin_idx[valid]]

    frame_cmap, frame_norm = _get_product_cmap_norm(legend_name)
    return _frame_to_png_bytes(warped, frame_cmap, frame_norm, transparent_zero=False)


@router.get("/overlay/wr10/{timestamp}")
async def get_wr10_overlay(
    timestamp: str,
    product:    str = Query("SRI",  description="WR10 product: SRI, VMI, or PPI"),
    elevation:  str = Query("0015", description="PPI elevation code (e.g. 0015 = 1.5°) — ignored for SRI/VMI"),
    correction: str = Query("C",    description="PPI correction: C (corrected) or U (uncorrected) — ignored for SRI/VMI"),
):
    """
    Generate a WR10 polar radar overlay (RGBA PNG) for the map.

    For SRI/VMI: reads the product folder as before.
    For PPI: reads from the PPI folder, filtered by elevation code and correction (C/U).
    All products use the same polar→Web-Mercator reprojection.
    """
    from api.wr10 import find_wr10_file, find_ppi_file, _wr10_cfg

    try:
        dt = datetime.fromisoformat(timestamp)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid datetime: {e}")

    if product == "PPI":
        file_path = find_ppi_file(dt, elevation, correction)
        if file_path is None:
            raise HTTPException(
                status_code=404,
                detail=f"PPI file not found: elev={elevation} corr={correction} @ {timestamp}",
            )
        legend_name = "CZ"
    else:
        file_path = find_wr10_file(product, dt)
        if file_path is None:
            raise HTTPException(status_code=404, detail=f"WR10 file not found: {product} @ {timestamp}")
        cfg = _wr10_cfg()
        legend_name = cfg.get("products", {}).get(product, {}).get("legend", "CZ")

    try:
        _get_wr10_warp_lookup()
        png_bytes = _render_wr10_frame(file_path, legend_name)
    except OSError as e:
        raise HTTPException(
            status_code=404,
            detail=f"File not yet readable (possibly still being written): {e}",
        )

    return Response(content=png_bytes, media_type="image/png", headers=_CACHE_HEADERS)


# ==========================================================================
# Cagliari X-band radar — Cartesian rendering
#
# The Cagliari data is already in Cartesian format (960×960 float32) in
# azimuthal equidistant projection centred on the radar.  No polar→Cartesian
# warp is needed; we just downsample to the output size and apply the
# colormap.
# ==========================================================================

def _render_cagliari_frame(file_path: Path, legend_name: str) -> bytes:
    """Load a Cagliari HDF5 file and return a PNG overlay at full 960×960 resolution."""
    import h5py

    with h5py.File(file_path, "r") as f:
        data = f["dataset1/data1/data"][()].astype(float)
        what = f["dataset1/data1/what"]
        nodata   = what.attrs.get("nodata",   float("nan"))
        undetect = what.attrs.get("undetect", float("nan"))

    # Values are already physical (gain=1.0, offset=0.0); mark fill values as NaN
    try:
        nd = float(nodata)
        if np.isfinite(nd):
            data[data == nd] = np.nan
    except (TypeError, ValueError):
        pass
    try:
        ud = float(undetect)
        if np.isfinite(ud):
            data[data == ud] = np.nan
    except (TypeError, ValueError):
        pass

    frame_cmap, frame_norm = _get_product_cmap_norm(legend_name)
    return _frame_to_png_bytes(data, frame_cmap, frame_norm, transparent_zero=False)


@router.get("/overlay/cagliari/{timestamp}")
async def get_cagliari_overlay(
    timestamp: str,
    product: str = Query("RR", description="Cagliari product: RR, CZ, OZ, or PPI"),
    idx: str = Query(None, description="PPI elevation index (e.g. 801). Required when product=PPI"),
):
    """Generate a Cagliari radar overlay (RGBA PNG) for the map."""
    from api.cagliari import find_cagliari_file, _cagliari_cfg

    try:
        dt = datetime.fromisoformat(timestamp)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid datetime: {e}")

    file_path = find_cagliari_file(product, dt, idx)
    if file_path is None:
        raise HTTPException(status_code=404, detail=f"Cagliari file not found: {product} @ {timestamp}")

    cfg = _cagliari_cfg()
    legend_name = cfg.get("products", {}).get(product, {}).get("legend", "CZ")

    try:
        png_bytes = _render_cagliari_frame(file_path, legend_name)
    except OSError as e:
        raise HTTPException(status_code=404, detail=f"File not yet readable: {e}")

    return Response(content=png_bytes, media_type="image/png", headers=_CACHE_HEADERS)


# ==========================================================================
# Torchiarolo (Puglia) composite — Transverse Mercator rendering
#
# Unlike Cagliari, this data is an ODIM composite on a tmerc grid, so it needs
# the same tmerc → Web Mercator reprojection as the national mosaic (see
# _get_warp_lookup below for why the destination grid is uniform in EPSG:3857
# metres). It also stores uint8 counts with a gain/offset that must be applied.
# ==========================================================================

_torchiarolo_warp_lookup = None


def _get_torchiarolo_warp_lookup():
    """Precompute and cache nearest-neighbour indices for the Torchiarolo reprojection.

    Same approach as _get_warp_lookup(), but the source grid parameters come
    from the HDF5 files rather than cfg.yaml, and the destination bounds are
    derived from the grid itself.

    Returns:
        (col_indices, line_indices, valid_mask, nlines_dst, ncols_dst)
    """
    global _torchiarolo_warp_lookup
    if _torchiarolo_warp_lookup is not None:
        return _torchiarolo_warp_lookup

    from api.torchiarolo import _grid_geometry, get_overlay_bounds, get_grid_extent_m

    geom = _grid_geometry()
    (min_lat, min_lon), (max_lat, max_lon) = get_overlay_bounds()
    half_x, half_y = get_grid_extent_m(geom)

    source_proj = Proj(geom["projdef"])

    # Output resolution: keep it square and close to the native pixel count.
    n_dst = int(max(geom["ncols"], geom["nlines"]))

    # EPSG:3857 uses the WGS84 equatorial radius
    R = 6378137.0

    x_min = R * np.radians(min_lon)
    x_max = R * np.radians(max_lon)
    y_min = R * np.log(np.tan(np.pi / 4 + np.radians(min_lat) / 2))
    y_max = R * np.log(np.tan(np.pi / 4 + np.radians(max_lat) / 2))

    dest_x_merc = np.linspace(x_min, x_max, n_dst)
    dest_y_merc = np.linspace(y_max, y_min, n_dst)   # north (top) to south (bottom)
    dest_xm_grid, dest_ym_grid = np.meshgrid(dest_x_merc, dest_y_merc)

    # EPSG:3857 → EPSG:4326
    dest_lon = np.degrees(dest_xm_grid / R)
    dest_lat = np.degrees(2 * np.arctan(np.exp(dest_ym_grid / R)) - np.pi / 2)

    # EPSG:4326 → source tmerc → source grid indices.
    # Row 0 is the northernmost row, so the line index decreases as y increases.
    dest_x_tmerc, dest_y_tmerc = source_proj(dest_lon, dest_lat)
    col_indices = ((dest_x_tmerc + half_x) / geom["xscale"]).astype(int)
    line_indices = ((half_y - dest_y_tmerc) / geom["yscale"]).astype(int)

    valid_mask = (
        (col_indices >= 0) & (col_indices < geom["ncols"]) &
        (line_indices >= 0) & (line_indices < geom["nlines"])
    )

    _torchiarolo_warp_lookup = (col_indices, line_indices, valid_mask, n_dst, n_dst)
    return _torchiarolo_warp_lookup


def _render_torchiarolo_frame(file_path: Path, legend_name: str) -> bytes:
    """Load a Torchiarolo HDF5 file, decode, reproject and return a PNG overlay."""
    import h5py

    from api.torchiarolo import fill_values, scaling

    with h5py.File(file_path, "r") as f:
        raw = f["dataset1/data1/data"][()]
        what = dict(f["dataset1/data1/what"].attrs)

    data = raw.astype(float)

    # ODIM: nodata/undetect are raw counts, so mask them before applying gain.
    for fv in fill_values(what):
        if np.isfinite(fv):
            data[raw == fv] = np.nan

    gain, offset = scaling(what)
    if gain != 1.0 or offset != 0.0:
        data = data * gain + offset

    col_idx, line_idx, valid, nlines_dst, ncols_dst = _get_torchiarolo_warp_lookup()
    warped = np.full((nlines_dst, ncols_dst), np.nan, dtype=float)
    warped[valid] = data[line_idx[valid], col_idx[valid]]

    # transparent_zero: SRI and VIL encode "nothing" as a real 0.0 that would
    # otherwise paint the whole coverage disc; VMI's sub-zero dBZ are
    # noise-level returns. All four should be see-through below zero.
    frame_cmap, frame_norm = _get_product_cmap_norm(legend_name)
    return _frame_to_png_bytes(warped, frame_cmap, frame_norm, transparent_zero=True)


@router.get("/overlay/torchiarolo/{timestamp}")
async def get_torchiarolo_overlay(
    timestamp: str,
    product: str = Query("SRI", description="Torchiarolo product: SRI, VMI, VIL or ETM"),
):
    """Generate a Torchiarolo radar overlay (RGBA PNG) for the map."""
    from api.torchiarolo import find_torchiarolo_file, _torchiarolo_cfg

    try:
        dt = datetime.fromisoformat(timestamp)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid datetime: {e}")

    file_path = find_torchiarolo_file(product, dt)
    if file_path is None:
        raise HTTPException(status_code=404, detail=f"Torchiarolo file not found: {product} @ {timestamp}")

    cfg = _torchiarolo_cfg()
    legend_name = cfg.get("products", {}).get(product, {}).get("legend", "CZ")

    try:
        png_bytes = _render_torchiarolo_frame(file_path, legend_name)
    except OSError as e:
        raise HTTPException(status_code=404, detail=f"File not yet readable: {e}")

    return Response(content=png_bytes, media_type="image/png", headers=_CACHE_HEADERS)


# ==========================================================================
# Cached helpers — loaded once, reused for every request
# ==========================================================================

# Per-product colormap/norm cache — keyed by product legend name (e.g. 'R', 'CZ')
_product_cmaps: dict = {}

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

    WHY — Web Mercator vs equirectangular:
    Leaflet internally uses EPSG:3857 (Web Mercator) and stretches ImageOverlay
    images linearly in meters between the SW and NE corners. If we create a grid
    that is uniform in lat/lon degrees (equirectangular), the vertical pixel
    spacing in Web Mercator increases toward the poles, so northern features end
    up displayed a few km too far north.

    FIX: create the destination grid uniform in EPSG:3857 meters instead.
    Then Leaflet's linear stretch is exact and the overlay aligns perfectly.

    The bounds passed to Leaflet stay in EPSG:4326 lat/lon — Leaflet converts
    them to EPSG:3857 internally, which is consistent with our grid.

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

    # EPSG:3857 uses the WGS84 equatorial radius
    R = 6378137.0

    # Convert lat/lon bounds to Web Mercator (EPSG:3857) in meters
    x_min = R * np.radians(dst.minLon)
    x_max = R * np.radians(dst.maxLon)
    y_min = R * np.log(np.tan(np.pi / 4 + np.radians(dst.minLat) / 2))
    y_max = R * np.log(np.tan(np.pi / 4 + np.radians(dst.maxLat) / 2))

    # Destination grid: uniform spacing in EPSG:3857 meters
    dest_x_merc = np.linspace(x_min, x_max, dst.ncols)
    dest_y_merc = np.linspace(y_max, y_min, dst.nlines)   # north (top) to south (bottom)
    dest_xm_grid, dest_ym_grid = np.meshgrid(dest_x_merc, dest_y_merc)

    # EPSG:3857 → EPSG:4326 (lat/lon in degrees)
    dest_lon = np.degrees(dest_xm_grid / R)
    dest_lat = np.degrees(2 * np.arctan(np.exp(dest_ym_grid / R)) - np.pi / 2)

    # EPSG:4326 → source tmerc → source grid indices
    dest_x_tmerc, dest_y_tmerc = source_proj(dest_lon, dest_lat)

    col_indices = ((dest_x_tmerc / src.cRes) + src.cOff).astype(int)
    line_indices = ((dest_y_tmerc / src.lRes) + src.lOff).astype(int)

    valid_mask = (
        (col_indices >= 0) & (col_indices < src.ncols) &
        (line_indices >= 0) & (line_indices < src.nlines)
    )

    _warp_lookup = (col_indices, line_indices, valid_mask, dst.nlines, dst.ncols)
    return _warp_lookup


def _warp_frame(frame: np.ndarray) -> np.ndarray:
    """
    Reproject a 2D Transverse Mercator frame to Web Mercator (EPSG:3857) using
    cached nearest-neighbor lookup tables.

    The raw radar data is in tmerc. The output is a Web Mercator image: pixels
    are uniform in EPSG:3857 meters, so Leaflet's linear stretch between the
    SW/NE bounds is geometrically exact and the overlay aligns with the basemap.

    Row 0 = northernmost (top of image = NE corner of Leaflet bounds).

    Note: the original warp_map() in geo/warping.py does np.flipud to
    match Folium's origin="lower" convention (row 0 = south). Leaflet
    uses the opposite convention, so no flip is applied here.

    Returns NaN for pixels that fall outside the source grid bounds.
    """
    col_idx, line_idx, valid, nlines_dst, ncols_dst = _get_warp_lookup()

    warped = np.full((nlines_dst, ncols_dst), np.nan, dtype=float)
    warped[valid] = frame[line_idx[valid], col_idx[valid]]

    return warped


def _get_product_cmap_norm(legend_name: str):
    """
    Return (cmap, norm) for a given legend name, cached per legend.
    Falls back to the default SRI 'R' colormap if the legend is not found.
    """
    global _product_cmaps
    if legend_name in _product_cmaps:
        return _product_cmaps[legend_name]

    from nwc_webapp.rendering.colormaps import configure_colorbar
    result = configure_colorbar(legend_name, min_val=0, max_val=100)
    product_cmap, product_norm = result[0], result[1]
    # configure_colorbar returns 'jet' string when legend file missing — fall back to default
    if isinstance(product_cmap, str):
        product_cmap, product_norm = cmap, norm
    _product_cmaps[legend_name] = (product_cmap, product_norm)
    return product_cmap, product_norm


def _frame_to_png_bytes(frame, frame_cmap=None, frame_norm=None, transparent_zero=True):
    """
    Convert a 2D numpy array to RGBA PNG bytes using the radar colormap.

    Shared logic for both groundtruth and prediction overlays:
    1. Normalize values using the non-linear CustomNorm (0→1→2→5→10→...→100)
    2. Apply the LinearSegmentedColormap to get RGBA float array
    3. Make no-precipitation pixels fully transparent
    4. Encode as PNG with fast compression (level 1)

    frame_cmap / frame_norm: optional colormap/norm overrides.
    If None, uses the default SRI 'R' colormap.

    transparent_zero: if True (default, SRI/radar), also make frame <= 0 transparent.
    Set to False for products like IR_108 where 0 and negative values are valid data
    (cold cloud tops have negative temperatures). The legend's own alpha channel
    (e.g. alpha=0 for warm/clear areas) handles product-specific transparency.

    Returns bytes (not a buffer), suitable for Response(content=...).
    """
    if frame_cmap is None:
        frame_cmap = cmap
    if frame_norm is None:
        frame_norm = norm

    normalized = frame_norm(frame)
    rgba = frame_cmap(normalized)  # Float RGBA array (0-1)
    if transparent_zero:
        # Transparent where no precipitation or outside the radar domain (NaN from warp)
        rgba[~np.isfinite(frame) | (frame <= 0)] = [0, 0, 0, 0]
    else:
        # Transparent only for pixels outside the domain (NaN from warp).
        # The colormap's own alpha channel handles product-specific transparency.
        rgba[~np.isfinite(frame)] = [0, 0, 0, 0]

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
# Server-side PNG cache — holds up to 12 h of rendered frames
# ==========================================================================
# Key: (timestamp_iso, product) → PNG bytes
# Eviction: when the number of distinct timestamps exceeds _PNG_CACHE_MAX_TIMESTAMPS,
# the oldest timestamp and all its products are dropped together so the cache
# stays time-aligned (all 5 products for a slot are always present or absent).
_PNG_CACHE_MAX_TIMESTAMPS = 144  # 12 h × 12 timestamps/h
_png_cache: OrderedDict[tuple, bytes] = OrderedDict()
_png_cache_timestamps: OrderedDict[str, None] = OrderedDict()  # insertion-ordered set
_png_cache_lock = threading.Lock()


def _cache_get(timestamp_iso: str, product: str) -> bytes | None:
    with _png_cache_lock:
        return _png_cache.get((timestamp_iso, product))


def _cache_put(timestamp_iso: str, product: str, data: bytes) -> None:
    with _png_cache_lock:
        _png_cache[(timestamp_iso, product)] = data
        if timestamp_iso not in _png_cache_timestamps:
            _png_cache_timestamps[timestamp_iso] = None
            while len(_png_cache_timestamps) > _PNG_CACHE_MAX_TIMESTAMPS:
                oldest_ts, _ = _png_cache_timestamps.popitem(last=False)
                for key in [k for k in _png_cache if k[0] == oldest_ts]:
                    del _png_cache[key]


# ==========================================================================
# Overlay endpoints
# ==========================================================================

def _render_groundtruth_frame(dt: datetime, product: str) -> bytes:
    """
    Render one radar product frame to PNG bytes (CPU-bound, runs in a thread).

    Raises:
        ValueError  — unknown product
        FileNotFoundError — data file missing
        OSError     — file mid-write / unreadable
        RuntimeError — unexpected HDF5 structure
    """
    config = get_config()
    products = config.radar_products
    if product not in products:
        raise ValueError(f"Unknown product: {product}")

    product_cfg = products[product]
    product_folder = config.get_product_folder(product)
    if not product_folder and not config.data_archive_folder:
        raise FileNotFoundError(f"No data folder configured for {product}")

    legend_name = product_cfg.get("legend", "R")
    p_cmap, p_norm = _get_product_cmap_norm(legend_name)
    file_format = product_cfg.get("file_format", "hdf")

    if file_format == "tiff":
        stem = dt.strftime("%d-%m-%Y-%H-%M")
        file_path = config.find_product_file(product, dt, stem + ".tif")
        if file_path is None:
            file_path = config.find_product_file(product, dt, stem + ".tiff")
        if file_path is None:
            raise FileNotFoundError(f"File not found: {stem}.tif")
        try:
            pil_img = Image.open(file_path)
            frame = np.array(pil_img, dtype=float)
        except (Image.UnidentifiedImageError, OSError) as e:
            raise OSError(f"File not yet readable: {file_path.name} ({e})")
        clip_min = p_norm.thresh[0] if hasattr(p_norm, "thresh") else -100
        clip_max = p_norm.thresh[-1] if hasattr(p_norm, "thresh") else 100
        frame = np.clip(frame, clip_min, clip_max)
        frame = _warp_frame(frame)
        return _frame_to_png_bytes(frame, p_cmap, p_norm, transparent_zero=False)
    else:
        filename = dt.strftime("%d-%m-%Y-%H-%M") + ".hdf"
        file_path = config.find_product_file(product, dt, filename)
        if file_path is None:
            raise FileNotFoundError(f"File not found: {filename}")
        try:
            with h5py.File(file_path, "r") as f:
                if "dataset1/data1/data" in f:
                    frame = f["dataset1/data1/data"][()].astype(float)
                else:
                    raise RuntimeError("Unknown HDF5 structure")
        except OSError as e:
            raise OSError(f"File not yet readable: {file_path.name} ({e})")
        mask = _get_radar_mask()
        if mask is not None:
            frame = frame * mask
        frame[frame < 0] = 0
        clip_max = p_norm.thresh[-1] if hasattr(p_norm, "thresh") else 200
        frame = np.clip(frame, 0, clip_max)
        frame = _warp_frame(frame)
        return _frame_to_png_bytes(frame, p_cmap, p_norm)


def prerender_recent_frames(hours: int = 12) -> None:
    """
    Render the last `hours` of radar frames for all products and populate the
    server-side PNG cache. Called once at startup in a background thread so the
    cache is warm before the first user connects.
    """
    config = get_config()
    products = list(config.radar_products.keys())

    now = datetime.now()
    start = now - timedelta(hours=hours)
    dt = start.replace(second=0, microsecond=0)
    dt = dt.replace(minute=(dt.minute // 5) * 5)
    timestamps = []
    while dt <= now:
        timestamps.append(dt)
        dt += timedelta(minutes=5)
    timestamps.reverse()  # newest first — most recent frames ready before first user connects

    total = len(timestamps) * len(products)
    cached_count = 0
    cached_lock = threading.Lock()

    print(f"  PNG cache: pre-rendering {len(timestamps)} timestamps × {len(products)} products ({total} frames)...")

    def _render_one(dt: datetime, product: str) -> None:
        nonlocal cached_count
        ts_iso = dt.isoformat()
        if _cache_get(ts_iso, product) is not None:
            with cached_lock:
                cached_count += 1
            return
        try:
            data = _render_groundtruth_frame(dt, product)
            _cache_put(ts_iso, product, data)
            with cached_lock:
                cached_count += 1
        except Exception:
            pass  # file missing or unreadable — skip silently

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda args: _render_one(*args), [(dt, p) for dt in timestamps for p in products]))

    print(f"  PNG cache: pre-render complete — {cached_count}/{total} frames cached")


def prerender_timestamp(timestamp_iso: str) -> None:
    """
    Pre-render all products for a single timestamp and add them to the cache.
    Called by RealtimeService when new radar data arrives.
    """
    try:
        dt = datetime.fromisoformat(timestamp_iso)
    except ValueError:
        return
    config = get_config()
    for product in config.radar_products:
        if _cache_get(timestamp_iso, product) is not None:
            continue
        try:
            data = _render_groundtruth_frame(dt, product)
            _cache_put(timestamp_iso, product, data)
        except Exception:
            pass


@router.get("/overlay/groundtruth/{timestamp}")
async def get_groundtruth_overlay(
    timestamp: str,
    product: str = Query("SRI_adj", description="Radar product (SRI_adj, VMI, ETM, VIL)"),
):
    """
    Generate a radar product overlay image (RGBA PNG) for the map.

    Checks the server-side PNG cache first — cache is pre-populated at startup
    and updated when new radar data arrives, so most requests return immediately.
    """
    try:
        dt = datetime.fromisoformat(timestamp)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid datetime: {e}")

    cached = _cache_get(timestamp, product)
    if cached is not None:
        return Response(content=cached, media_type="image/png", headers=_CACHE_HEADERS)

    loop = asyncio.get_event_loop()
    try:
        png_bytes = await loop.run_in_executor(None, _render_groundtruth_frame, dt, product)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except OSError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    _cache_put(timestamp, product, png_bytes)
    return Response(content=png_bytes, media_type="image/png", headers=_CACHE_HEADERS)


# ==========================================================================
# Ensemble probability overlay
# IMPORTANT: must be registered BEFORE /overlay/{model}/{timestamp} so that
# "ensemble" is not captured as the {model} path parameter.
# ==========================================================================

@router.get("/overlay/ensemble/{timestamp}")
async def get_ensemble_overlay(
    timestamp: str,
    lead_time: int = Query(0, description="Lead time index (0-11)"),
    threshold: float = Query(2.0, description="Rainfall threshold in mm/h"),
    models: str = Query(..., description="Comma-separated model names"),
    contours: bool = Query(False, description="Overlay dark probability contour lines"),
):
    """
    Generate a probabilistic ensemble overlay: P(rain > threshold) per pixel.

    Loads real-time predictions from each requested model, stacks them, and
    computes the fraction that exceed the threshold at every pixel.
    Renders as a Blues colormap PNG (transparent at 0%, dark blue at 100%).
    Missing model predictions are silently skipped — not an error.
    """
    try:
        dt = datetime.fromisoformat(timestamp)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid datetime: {e}")

    config = get_config()
    model_list = [m.strip() for m in models.split(',') if m.strip()]
    if not model_list:
        raise HTTPException(status_code=400, detail="At least one model must be specified")

    from nwc_webapp.data.predictions import load_prediction_array
    from matplotlib.cm import Oranges as prob_cmap

    frames = []
    for model in model_list:
        pred_filename = dt.strftime("%d-%m-%Y-%H-%M") + ".npy"
        pred_path = config.real_time_pred / model / pred_filename
        if not pred_path.exists():
            continue
        pred_array = load_prediction_array(pred_path, model)
        if pred_array is None or lead_time >= pred_array.shape[0]:
            continue
        frames.append(pred_array[lead_time].astype(float))

    if not frames:
        raise HTTPException(
            status_code=404,
            detail=f"No predictions found for any of: {model_list} at {timestamp}",
        )

    # Apply radar mask to each frame before computing probability
    mask = _get_radar_mask()
    if mask is not None:
        frames = [f * mask for f in frames]

    # Stack → (N_models, H, W), compute fraction exceeding threshold
    stack = np.stack(frames)
    prob_map = (stack > threshold).mean(axis=0)   # values 0.0–1.0

    # Reproject to Web Mercator (same lookup table as all other overlays)
    warped = _warp_frame(prob_map)

    # Oranges colormap: 0.0 → near-white, 1.0 → deep orange/brown.
    # Stays legible on dark, OSM, and satellite basemaps (blue/green) alike.
    warped_safe = np.where(np.isfinite(warped), np.clip(warped, 0.0, 1.0), 0.0)
    rgba = prob_cmap(warped_safe)   # (H, W, 4) float 0–1

    # Transparent where probability == 0 (no model predicts rain) or outside domain
    rgba[~np.isfinite(warped) | (warped <= 0.0)] = [0.0, 0.0, 0.0, 0.0]

    # Optional dark probability contour lines for extra legibility.
    # Compute pixel-level boundaries where `warped > level`: any pixel that is
    # >level but has a 4-neighbor that is <=level is on the boundary. We OR
    # boundaries across several levels and stamp them dark grey on the RGBA.
    if contours:
        contour_levels = (0.25, 0.5, 0.75)
        valid = np.isfinite(warped)
        edges = np.zeros(warped.shape, dtype=bool)
        for lvl in contour_levels:
            above = (warped > lvl) & valid
            e = np.zeros_like(above)
            e[1:, :]  |= above[1:, :]  & ~above[:-1, :]
            e[:-1, :] |= above[:-1, :] & ~above[1:, :]
            e[:, 1:]  |= above[:, 1:]  & ~above[:, :-1]
            e[:, :-1] |= above[:, :-1] & ~above[:, 1:]
            edges |= e
        rgba[edges] = [0.15, 0.15, 0.15, 1.0]

    img = Image.fromarray((rgba * 255).astype(np.uint8))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG", compress_level=1)

    return Response(content=buffer.getvalue(), media_type="image/png", headers=_CACHE_HEADERS)


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
# Batch overlay endpoint
# ==========================================================================

def _render_single_frame(idx: int, ts_str: str, product_key: str):
    """
    Render one radar frame as PNG bytes. Designed to run inside a ThreadPoolExecutor.

    Returns (idx, png_bytes) where png_bytes is None if the file is missing or
    any error occurs. Reuses all cached helpers (mask, warp lookup, colormaps).
    """
    try:
        config = get_config()
        dt = datetime.fromisoformat(ts_str)
        products = config.radar_products
        product_cfg = products[product_key]
        legend_name = product_cfg.get("legend", "R")
        file_format = product_cfg.get("file_format", "hdf")
        p_cmap, p_norm = _get_product_cmap_norm(legend_name)

        if file_format == "tiff":
            stem = dt.strftime("%d-%m-%Y-%H-%M")
            file_path = config.find_product_file(product_key, dt, stem + ".tif")
            if file_path is None:
                file_path = config.find_product_file(product_key, dt, stem + ".tiff")
            if file_path is None:
                return idx, None
            pil_img = Image.open(file_path)
            frame = np.array(pil_img, dtype=float)
            clip_min = p_norm.thresh[0] if hasattr(p_norm, "thresh") else -100
            clip_max = p_norm.thresh[-1] if hasattr(p_norm, "thresh") else 100
            frame = np.clip(frame, clip_min, clip_max)
            frame = _warp_frame(frame)
            return idx, _frame_to_png_bytes(frame, p_cmap, p_norm, transparent_zero=False)
        else:
            filename = dt.strftime("%d-%m-%Y-%H-%M") + ".hdf"
            file_path = config.find_product_file(product_key, dt, filename)
            if file_path is None:
                return idx, None
            with h5py.File(file_path, "r") as f:
                if "dataset1/data1/data" not in f:
                    return idx, None
                frame = f["dataset1/data1/data"][()].astype(float)
            mask = _get_radar_mask()
            if mask is not None:
                frame = frame * mask
            frame[frame < 0] = 0
            clip_max = p_norm.thresh[-1] if hasattr(p_norm, "thresh") else 200
            frame = np.clip(frame, 0, clip_max)
            frame = _warp_frame(frame)
            return idx, _frame_to_png_bytes(frame, p_cmap, p_norm)
    except Exception:
        return idx, None


class BatchOverlayRequest(BaseModel):
    product: str
    timestamps: List[str]  # ISO format strings, only the ones that exist


@router.post("/overlay/batch")
async def get_batch_overlay(request: BatchOverlayRequest):
    """
    Render all frames for one product in parallel and return as a ZIP file.

    WHY: The Data Explorer loads up to 144 frames × 4 products = 576 images.
    Individual requests are throttled to ~6 concurrent by the browser's
    per-domain connection limit, so large ranges trickle in slowly.
    This endpoint collapses 144 round trips into 1 per product.

    ZIP contents: NNNN.png files (zero-padded index matching request.timestamps).
    Missing files are absent from the ZIP — the frontend maps by index.

    Caches are warmed before spawning threads so threads don't race to
    initialise the radar mask or warp lookup on first use.
    """
    config = get_config()
    if request.product not in config.radar_products:
        raise HTTPException(status_code=400, detail=f"Unknown product: {request.product}")

    # Warm up all caches before threads start (avoids redundant work / races)
    _get_radar_mask()
    _get_warp_lookup()
    legend_name = config.radar_products[request.product].get("legend", "R")
    _get_product_cmap_norm(legend_name)

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            loop.run_in_executor(executor, _render_single_frame, idx, ts, request.product)
            for idx, ts in enumerate(request.timestamps)
        ]
        rendered = await asyncio.gather(*futures)

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_STORED) as zf:
        for idx, png_bytes in rendered:
            if png_bytes is not None:
                zf.writestr(f"{idx:04d}.png", png_bytes)

    return Response(
        content=zip_buffer.getvalue(),
        media_type="application/zip",
    )


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
