"""
Cagliari X-band radar API endpoints.

Provides:
- GET /api/cagliari/config     — radar parameters and overlay bounds for the frontend map
- GET /api/cagliari/timestamps — list available files in a lookback window
- GET /api/cagliari/sample     — pixel sampling for click-to-inspect popup
- WS  /api/cagliari/ws         — push notification when a new file arrives

Filename format: {PP}W{YY}{DOY}{HHMM}{S}{SITE}.{IDX}.h5
  e.g. RRW2613210550L.001.h5
  PP   = product prefix (RR=RainRate, CZ=CorrectedZ/VMI, OZ=OriginalZ/CAPPI, PZ=PPI)
  YY   = 2-digit year  (26 → 2026)
  DOY  = 3-digit day-of-year  (132 → May 12)
  HHMM = hour+minute  (1055 → 10:55)
  S    = scan number (fixed 0 for this radar)
  SITE = site identifier char (fixed L for this radar)
  IDX  = 3-digit product/elevation index
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

router = APIRouter(prefix="/api/cagliari", tags=["cagliari"])
logger = logging.getLogger(__name__)

# ==========================================================================
# Cagliari WebSocket connection manager
# ==========================================================================

class _CagliariWsManager:
    def __init__(self):
        self._connections: list[WebSocket] = []
        self._loop: asyncio.AbstractEventLoop | None = None

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self._connections.append(ws)
        logger.debug("Cagliari WS client connected (%d total)", len(self._connections))

    def disconnect(self, ws: WebSocket):
        if ws in self._connections:
            self._connections.remove(ws)
        logger.debug("Cagliari WS client disconnected (%d remaining)", len(self._connections))

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
        """Thread-safe broadcast from the CagliariService background thread."""
        if not self._connections:
            return
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        asyncio.run_coroutine_threadsafe(self._broadcast(message), loop)


cagliari_ws_manager = _CagliariWsManager()

# ==========================================================================
# Config helpers (read cfg.yaml once)
# ==========================================================================

_cfg_cache: dict | None = None


def _cagliari_cfg() -> dict:
    """Load the cagliari section of cfg.yaml (cached)."""
    global _cfg_cache
    if _cfg_cache is not None:
        return _cfg_cache
    cfg_path = Path(__file__).parent.parent.parent / "cfg.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    _cfg_cache = cfg.get("cagliari", {})
    return _cfg_cache


def _get_data_root() -> Path:
    cfg = _cagliari_cfg()
    if is_server():
        return Path(cfg.get("server_data_root", "/data/cagliari_xband"))
    return Path(cfg.get("local_data_root", "data/cagliari_xband"))


# Maps logical product names to the 2-char prefix used in filenames
_PRODUCT_TO_FILE_PREFIX = {
    "RR": "RR",
    "CZ": "CZ",
    "OZ": "OZ",
    "PPI": "PZ",
}


def _get_product_folder(product: str, idx: str | None = None) -> Path:
    root = _get_data_root()
    if product == "PPI" and idx:
        return root / "PPI" / idx
    return root / product


# ==========================================================================
# Filename parsing
# Format: {PP}W{YY}{DOY}{HHMM}{S}{SITE}.{IDX}.h5
# e.g. RRW2613210550L.001.h5
# ==========================================================================

_FNAME_RE = re.compile(
    r"^(?P<prefix>[A-Z]{2})W(?P<ts>\d{10})[A-Z]\.(?P<idx>\d{3})\.h5$"
)


def parse_cagliari_filename(filename: str):
    """Return (prefix, idx, datetime) or (None, None, None)."""
    m = _FNAME_RE.match(filename)
    if not m:
        return None, None, None
    prefix = m.group("prefix")
    idx = m.group("idx")
    ts_str = m.group("ts")
    try:
        yy  = int(ts_str[0:2])
        doy = int(ts_str[2:5])
        hh  = int(ts_str[5:7])
        mm  = int(ts_str[7:9])
        base = datetime(2000 + yy, 1, 1) + timedelta(days=doy - 1)
        return prefix, idx, base.replace(hour=hh, minute=mm)
    except (ValueError, IndexError):
        return None, None, None


def find_cagliari_file(product: str, dt: datetime, idx: str | None = None) -> Path | None:
    """Return the path of the Cagliari file for a given product and timestamp, or None.

    For PPI files, pass idx (e.g. '801') to select the elevation subfolder.
    """
    folder = _get_product_folder(product, idx)
    if not folder.exists():
        return None
    file_prefix = _PRODUCT_TO_FILE_PREFIX.get(product, product)
    yy  = dt.year - 2000
    doy = dt.timetuple().tm_yday
    ts_prefix = f"{yy:02d}{doy:03d}{dt.hour:02d}{dt.minute:02d}"  # 9 chars
    try:
        for fname in os.listdir(folder):
            m = _FNAME_RE.match(fname)
            if m and m.group("prefix") == file_prefix and m.group("ts")[:9] == ts_prefix:
                return folder / fname
    except OSError:
        pass
    return None


# ==========================================================================
# Overlay bounds (computed once and cached)
# ==========================================================================

_overlay_bounds: list | None = None


def get_overlay_bounds() -> list:
    """Return [[lat_sw, lon_sw], [lat_ne, lon_ne]] for the Cagliari coverage area."""
    global _overlay_bounds
    if _overlay_bounds is not None:
        return _overlay_bounds
    cfg = _cagliari_cfg()
    lat     = cfg.get("radar_lat", 39.271488189697266)
    lon     = cfg.get("radar_lon", 9.122883796691895)
    n_px    = cfg.get("n_pixels", 960)
    xscale  = cfg.get("xscale", 125.0)  # metres per pixel
    max_m   = (n_px / 2) * xscale       # half-extent in metres
    proj = Proj(proj="aeqd", lat_0=lat, lon_0=lon, ellps="WGS84")
    lon_sw, lat_sw = proj(-max_m, -max_m, inverse=True)
    lon_ne, lat_ne = proj(+max_m, +max_m, inverse=True)
    _overlay_bounds = [[lat_sw, lon_sw], [lat_ne, lon_ne]]
    return _overlay_bounds


# ==========================================================================
# Pixel sampling helper
# ==========================================================================

def _sample_cagliari_pixel(file_path: Path, row: int, col: int) -> float | None:
    """Read one Cartesian pixel from a Cagliari HDF5 file and return the physical value."""
    import h5py
    try:
        with h5py.File(file_path, "r") as f:
            val = float(f["dataset1/data1/data"][row, col])
            what = f["dataset1/data1/what"]
            nodata   = what.attrs.get("nodata",   float("nan"))
            undetect = what.attrs.get("undetect", float("nan"))
        import math
        if math.isnan(val) or val == nodata or val == undetect:
            return None
        return val
    except Exception:
        return None


# ==========================================================================
# API endpoints
# ==========================================================================

@router.get("/config")
async def get_cagliari_config():
    """Return Cagliari radar parameters and map configuration (including colorbar data)."""
    from nwc_webapp.rendering.colormaps import get_legend_data, build_legend_file_path

    cfg = _cagliari_cfg()
    bounds = get_overlay_bounds()
    lat = cfg.get("radar_lat", 39.271488189697266)
    lon = cfg.get("radar_lon", 9.122883796691895)
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
async def get_cagliari_timestamps(
    product: str = Query("RR", description="Product name: RR, CZ, OZ, or PPI"),
    lookback_minutes: int = Query(60, description="Lookback window in minutes"),
    idx: str = Query(None, description="PPI elevation index (801-805). Required when product=PPI"),
):
    """List available Cagliari timestamps for a product within the lookback window."""
    cfg = _cagliari_cfg()
    valid_products = list(cfg.get("products", {}).keys()) or ["RR", "CZ"]
    if product not in valid_products:
        raise HTTPException(status_code=400, detail=f"Unknown product: {product}. Available: {valid_products}")

    folder = _get_product_folder(product, idx)
    if not folder.exists():
        return {"product": product, "timestamps": [], "total": 0}

    file_prefix = _PRODUCT_TO_FILE_PREFIX.get(product, product)
    cutoff = datetime.utcnow() - timedelta(minutes=lookback_minutes + 6)

    found: list[tuple[datetime, str]] = []
    try:
        for fname in os.listdir(folder):
            file_prefix_parsed, _, dt = parse_cagliari_filename(fname)
            if file_prefix_parsed == file_prefix and dt is not None and dt >= cutoff:
                found.append((dt, fname))
    except OSError as e:
        logger.warning("Cannot list Cagliari folder %s: %s", folder, e)
        return {"product": product, "timestamps": [], "total": 0}

    found.sort(key=lambda x: x[0])
    return {
        "product": product,
        "timestamps": [dt.isoformat() for dt, _ in found],
        "total": len(found),
    }


@router.get("/sample")
async def sample_cagliari_pixel(
    lat: float = Query(..., description="Latitude (WGS84)"),
    lon: float = Query(..., description="Longitude (WGS84)"),
    timestamp: str = Query(..., description="Datetime ISO (YYYY-MM-DDTHH:MM)"),
    products: str = Query("RR", description="Comma-separated: RR, CZ, OZ, PPI"),
    ppi_idx: str = Query(None, description="PPI elevation index (e.g. 801). Required when products includes PPI"),
):
    """
    Sample Cagliari Cartesian radar values at a clicked (lat, lon).

    Converts lat/lon to azimuthal-equidistant metres relative to the radar
    centre, then to (row, col) in the 960×960 Cartesian grid.
    """
    import math

    try:
        dt = datetime.fromisoformat(timestamp)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid datetime: {e}")

    cfg = _cagliari_cfg()
    radar_lat = cfg.get("radar_lat", 39.271488189697266)
    radar_lon = cfg.get("radar_lon", 9.122883796691895)
    n_pixels  = cfg.get("n_pixels", 960)
    xscale    = cfg.get("xscale", 125.0)  # metres per pixel
    max_m     = (n_pixels / 2) * xscale

    proj = Proj(proj="aeqd", lat_0=radar_lat, lon_0=radar_lon, ellps="WGS84")
    dx, dy = proj(lon, lat)  # East, North in metres

    range_m     = math.sqrt(dx ** 2 + dy ** 2)
    azimuth_deg = math.degrees(math.atan2(dx, dy)) % 360.0
    in_bounds   = range_m < max_m

    # Grid origin is at (-max_m, +max_m) in (West, North); row 0 = north
    col = int((dx + max_m) / xscale)
    row = int((max_m - dy) / xscale)
    col = max(0, min(n_pixels - 1, col))
    row = max(0, min(n_pixels - 1, row))

    result: dict = {
        "lat": lat,
        "lon": lon,
        "row": row,
        "col": col,
        "range_km": round(range_m / 1000.0, 2),
        "azimuth_deg": round(azimuth_deg, 1),
        "in_bounds": in_bounds,
        "timestamp": timestamp,
        "values": {},
    }

    if not in_bounds:
        return result

    product_list   = [p.strip() for p in products.split(",") if p.strip()]
    valid_products = list(cfg.get("products", {}).keys()) or ["RR", "CZ"]

    for product in product_list:
        if product not in valid_products:
            result["values"][product] = None
            continue
        file_path = find_cagliari_file(product, dt, ppi_idx if product == "PPI" else None)
        if file_path is None:
            result["values"][product] = None
            continue
        result["values"][product] = _sample_cagliari_pixel(file_path, row, col)

    return result


@router.websocket("/ws")
async def cagliari_websocket(websocket: WebSocket):
    """WebSocket endpoint — server pushes {type: 'cagliari_update', data: {...}} on new data."""
    await cagliari_ws_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        cagliari_ws_manager.disconnect(websocket)
