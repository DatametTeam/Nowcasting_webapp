"""
WR10 small X-band radar API endpoints.

Provides:
- GET /api/wr10/config   — radar parameters and overlay bounds for the frontend map
- GET /api/wr10/timestamps — list available files in a lookback window
- WS  /api/wr10/ws       — push notification when a new file arrives
"""

import asyncio
import logging
import os
import re
from datetime import datetime, timedelta
from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from pyproj import Proj

from nwc_webapp.config.environment import is_server

router = APIRouter(prefix="/api/wr10", tags=["wr10"])
logger = logging.getLogger(__name__)

# ==========================================================================
# WR10 WebSocket connection manager (separate from the realtime WS)
# ==========================================================================

class _WR10WsManager:
    def __init__(self):
        self._connections: list[WebSocket] = []
        self._loop: asyncio.AbstractEventLoop | None = None

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self._connections.append(ws)
        logger.debug("WR10 WS client connected (%d total)", len(self._connections))

    def disconnect(self, ws: WebSocket):
        if ws in self._connections:
            self._connections.remove(ws)
        logger.debug("WR10 WS client disconnected (%d remaining)", len(self._connections))

    async def _broadcast(self, message: dict):
        dead = []
        for ws in list(self._connections):
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    def broadcast_sync(self, message: dict):
        """Thread-safe broadcast from the WR10Service background thread."""
        if not self._connections:
            return
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        asyncio.run_coroutine_threadsafe(self._broadcast(message), loop)


wr10_ws_manager = _WR10WsManager()

# ==========================================================================
# Config helpers (read cfg.yaml once)
# ==========================================================================

_cfg_cache: dict | None = None


def _wr10_cfg() -> dict:
    """Load the wr10 section of cfg.yaml (cached)."""
    global _cfg_cache
    if _cfg_cache is not None:
        return _cfg_cache
    cfg_path = Path(__file__).parent.parent.parent / "cfg.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    _cfg_cache = cfg.get("wr10", {})
    return _cfg_cache


def _get_data_root() -> Path:
    cfg = _wr10_cfg()
    if is_server():
        return Path(cfg.get("server_data_root", "/data/wr10"))
    return Path(cfg.get("local_data_root", "data/wr10"))


def _get_product_folder(product: str) -> Path:
    return _get_data_root() / product


# ==========================================================================
# Filename parsing
# HDF-SRI-A00-202605041605-B-0720-0150-0010-0000-C.z
# timestamp group: YYYYMMDDHHSS (12 digits)
# ==========================================================================

_FNAME_RE = re.compile(r"^HDF-(\w+)-A00-(\d{12})-.*\.z$")


def parse_wr10_filename(filename: str):
    """Return (product, datetime) or (None, None) for a WR10 filename."""
    m = _FNAME_RE.match(filename)
    if not m:
        return None, None
    product = m.group(1)
    ts_str = m.group(2)
    try:
        return product, datetime.strptime(ts_str, "%Y%m%d%H%M")
    except ValueError:
        return None, None


def find_wr10_file(product: str, dt: datetime) -> Path | None:
    """Return the path of the WR10 file for a given product and timestamp, or None."""
    folder = _get_product_folder(product)
    if not folder.exists():
        return None
    ts_str = dt.strftime("%Y%m%d%H%M")
    prefix = f"HDF-{product}-A00-{ts_str}-"
    for fname in os.listdir(folder):
        if fname.startswith(prefix) and fname.endswith(".z"):
            return folder / fname
    return None


# ==========================================================================
# Overlay bounds (computed once and cached)
# ==========================================================================

_overlay_bounds: list | None = None


def get_overlay_bounds() -> list:
    """Return [[lat_sw, lon_sw], [lat_ne, lon_ne]] for the WR10 coverage area."""
    global _overlay_bounds
    if _overlay_bounds is not None:
        return _overlay_bounds
    cfg = _wr10_cfg()
    lat = cfg.get("radar_lat", 41.84239959716797)
    lon = cfg.get("radar_lon", 12.646699905395508)
    n_bins = cfg.get("n_bins", 480)
    rscale = cfg.get("rscale", 150.0)
    max_m = n_bins * rscale
    proj = Proj(proj="aeqd", lat_0=lat, lon_0=lon, ellps="WGS84")
    lon_sw, lat_sw = proj(-max_m, -max_m, inverse=True)
    lon_ne, lat_ne = proj(+max_m, +max_m, inverse=True)
    _overlay_bounds = [[lat_sw, lon_sw], [lat_ne, lon_ne]]
    return _overlay_bounds


# ==========================================================================
# Pixel sampling helper
# ==========================================================================

def _sample_wr10_pixel(file_path: Path, ray_idx: int, bin_idx: int) -> float | None:
    """Read one polar pixel from a WR10 HDF5 file and return the physical value."""
    import h5py
    try:
        with h5py.File(file_path, "r") as f:
            raw = float(f["dataset1/data1/data"][ray_idx, bin_idx])
            what = f["dataset1/data1/what"]
            gain   = float(what.attrs.get("gain",    0.5))
            offset = float(what.attrs.get("offset", -32.0))
            nodata   = float(what.attrs.get("nodata",   0.0))
            undetect = float(what.attrs.get("undetect", 0.0))
        if raw == nodata or raw == undetect:
            return None
        return offset + gain * raw
    except Exception:
        return None


# ==========================================================================
# API endpoints
# ==========================================================================

@router.get("/config")
async def get_wr10_config():
    """Return WR10 radar parameters and map configuration (including colorbar data)."""
    from nwc_webapp.rendering.colormaps import get_legend_data, build_legend_file_path

    cfg = _wr10_cfg()
    bounds = get_overlay_bounds()
    lat = cfg.get("radar_lat", 41.84239959716797)
    lon = cfg.get("radar_lon", 12.646699905395508)
    products = {}
    for name, meta in cfg.get("products", {}).items():
        legend_name = meta.get("legend", "CZ")
        thresholds: list = []
        colors: list = []
        try:
            legend_path = build_legend_file_path(legend_name)
            if legend_path.exists():
                legend_data = get_legend_data(legend_path)
                thresholds = legend_data["Thresh"]
                colors = [
                    f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
                    for r, g, b, *_ in legend_data["rgb"]
                ]
        except Exception:
            pass
        products[name] = {
            "legend": legend_name,
            "label": meta.get("label", name),
            "unit": meta.get("unit", "dBZ"),
            "thresholds": thresholds,
            "colors": colors,
        }
    return {
        "radar_lat": lat,
        "radar_lon": lon,
        "overlay_bounds": bounds,
        "center": [lat, lon],
        "zoom": 10,
        "products": products,
    }


@router.get("/timestamps")
async def get_wr10_timestamps(
    product: str = Query("SRI", description="Product name: SRI or VMI"),
    lookback_minutes: int = Query(60, description="Lookback window in minutes (default 60)"),
):
    """
    List available WR10 timestamps for a product within the lookback window.
    Returns ISO-format timestamps (UTC) sorted oldest-first.
    """
    cfg = _wr10_cfg()
    valid_products = list(cfg.get("products", {}).keys()) or ["SRI", "VMI"]
    if product not in valid_products:
        raise HTTPException(status_code=400, detail=f"Unknown product: {product}. Available: {valid_products}")

    folder = _get_product_folder(product)
    if not folder.exists():
        return {"product": product, "timestamps": [], "total": 0}

    now = datetime.utcnow()
    cutoff = now - timedelta(minutes=lookback_minutes)

    found: list[tuple[datetime, str]] = []
    try:
        for fname in os.listdir(folder):
            p, dt = parse_wr10_filename(fname)
            if p == product and dt is not None and dt >= cutoff:
                found.append((dt, fname))
    except OSError as e:
        logger.warning("Cannot list WR10 folder %s: %s", folder, e)
        return {"product": product, "timestamps": [], "total": 0}

    found.sort(key=lambda x: x[0])
    return {
        "product": product,
        "timestamps": [dt.isoformat() for dt, _ in found],
        "total": len(found),
    }


@router.get("/sample")
async def sample_wr10_pixel(
    lat: float = Query(..., description="Latitude (WGS84)"),
    lon: float = Query(..., description="Longitude (WGS84)"),
    timestamp: str = Query(..., description="Datetime ISO (YYYY-MM-DDTHH:MM)"),
    products: str = Query("SRI", description="Comma-separated: SRI, VMI"),
):
    """
    Sample WR10 polar radar values at a clicked (lat, lon) for the current frame.

    Converts lat/lon to azimuthal-equidistant metres relative to the radar
    centre, then to (ray_idx, bin_idx) in the 360×480 polar grid.
    Returns physical values (gain/offset applied) or null for no-data pixels.
    """
    import math

    try:
        dt = datetime.fromisoformat(timestamp)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid datetime: {e}")

    cfg = _wr10_cfg()
    radar_lat = cfg.get("radar_lat", 41.84239959716797)
    radar_lon = cfg.get("radar_lon", 12.646699905395508)
    n_rays    = cfg.get("n_rays",  360)
    n_bins    = cfg.get("n_bins",  480)
    rscale    = cfg.get("rscale",  150.0)
    max_range_m = n_bins * rscale

    # lat/lon → metres (dx=East, dy=North) relative to radar centre
    proj = Proj(proj="aeqd", lat_0=radar_lat, lon_0=radar_lon, ellps="WGS84")
    dx, dy = proj(lon, lat)

    range_m     = math.sqrt(dx ** 2 + dy ** 2)
    azimuth_deg = math.degrees(math.atan2(dx, dy)) % 360.0
    ray_idx     = int(azimuth_deg * n_rays / 360.0) % n_rays
    bin_idx     = int(range_m / rscale)
    in_bounds   = range_m < max_range_m

    result: dict = {
        "lat": lat,
        "lon": lon,
        "ray": ray_idx,
        "bin": bin_idx,
        "range_km": round(range_m / 1000.0, 2),
        "azimuth_deg": round(azimuth_deg, 1),
        "in_bounds": in_bounds,
        "timestamp": timestamp,
        "values": {},
    }

    if not in_bounds:
        return result

    product_list  = [p.strip() for p in products.split(",") if p.strip()]
    valid_products = list(cfg.get("products", {}).keys()) or ["SRI", "VMI"]

    for product in product_list:
        if product not in valid_products:
            result["values"][product] = None
            continue
        file_path = find_wr10_file(product, dt)
        if file_path is None:
            result["values"][product] = None
            continue
        result["values"][product] = _sample_wr10_pixel(file_path, ray_idx, bin_idx)

    return result


@router.websocket("/ws")
async def wr10_websocket(websocket: WebSocket):
    """WebSocket endpoint — server pushes {type: 'wr10_update', data: {...}} on new data."""
    await wr10_ws_manager.connect(websocket)
    try:
        while True:
            # We only push from the server; just drain any keep-alive pings.
            await websocket.receive_text()
    except WebSocketDisconnect:
        wr10_ws_manager.disconnect(websocket)
