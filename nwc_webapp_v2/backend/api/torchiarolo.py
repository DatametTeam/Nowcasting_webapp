"""
Torchiarolo (Puglia) radar composite API endpoints.

Provides:
- GET /api/torchiarolo/config     — grid geometry and overlay bounds for the frontend map
- GET /api/torchiarolo/timestamps — list available files in a lookback window
- GET /api/torchiarolo/sample     — pixel sampling for click-to-inspect popup
- WS  /api/torchiarolo/ws         — push notification when a new file arrives

Filename format: DD-MM-YYYY-HH-MM.hdf (UTC), e.g. 04-08-2026-12-40.hdf

Unlike the Cagliari X-band radar (azimuthal-equidistant, centred on the site),
this is an ODIM composite on a Transverse Mercator grid — structurally the same
as the national mosaic. The grid geometry is read from the HDF5 files
themselves rather than hardcoded; see _grid_geometry().
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

router = APIRouter(prefix="/api/torchiarolo", tags=["torchiarolo"])
logger = logging.getLogger(__name__)

# ==========================================================================
# Torchiarolo WebSocket connection manager
# ==========================================================================


class _TorchiaroloWsManager:
    def __init__(self):
        self._connections: list[WebSocket] = []
        self._loop: asyncio.AbstractEventLoop | None = None

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self._connections.append(ws)
        logger.debug("Torchiarolo WS client connected (%d total)", len(self._connections))

    def disconnect(self, ws: WebSocket):
        if ws in self._connections:
            self._connections.remove(ws)
        logger.debug("Torchiarolo WS client disconnected (%d remaining)", len(self._connections))

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
        """Thread-safe broadcast from the TorchiaroloService background thread."""
        if not self._connections:
            return
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        asyncio.run_coroutine_threadsafe(self._broadcast(message), loop)


torchiarolo_ws_manager = _TorchiaroloWsManager()

# ==========================================================================
# Config helpers (read cfg.yaml once)
# ==========================================================================

_cfg_cache: dict | None = None


def _torchiarolo_cfg() -> dict:
    """Load the torchiarolo section of cfg.yaml (cached)."""
    global _cfg_cache
    if _cfg_cache is not None:
        return _cfg_cache
    cfg_path = Path(__file__).parent.parent.parent / "cfg.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    _cfg_cache = cfg.get("torchiarolo", {})
    return _cfg_cache


def _get_data_root() -> Path:
    cfg = _torchiarolo_cfg()
    if is_server():
        return Path(cfg.get("server_data_root", "/data/torchiarolo"))
    return Path(cfg.get("local_data_root", "data/torchiarolo"))


def _get_product_folder(product: str) -> Path:
    return _get_data_root() / product


# ==========================================================================
# Filename parsing — DD-MM-YYYY-HH-MM.hdf (UTC)
# ==========================================================================

_FNAME_RE = re.compile(r"^(\d{2})-(\d{2})-(\d{4})-(\d{2})-(\d{2})\.hdf$")


def parse_torchiarolo_filename(filename: str) -> datetime | None:
    """Return the datetime encoded in the filename, or None if it doesn't match."""
    m = _FNAME_RE.match(filename)
    if not m:
        return None
    day, month, year, hour, minute = (int(g) for g in m.groups())
    try:
        return datetime(year, month, day, hour, minute)
    except ValueError:
        return None


def find_torchiarolo_file(product: str, dt: datetime) -> Path | None:
    """Return the path of the file for a given product and timestamp, or None."""
    path = _get_product_folder(product) / dt.strftime("%d-%m-%Y-%H-%M.hdf")
    return path if path.exists() else None


# ==========================================================================
# Grid geometry — read from the data, not hardcoded
#
# Every Torchiarolo HDF5 carries its own navigation in where/: projdef (a
# proj4 string), xscale/yscale (metres per pixel) and xsize/ysize. Reading it
# once at startup means the overlay stays correct if the provider ever
# re-centres or re-scales the grid, instead of silently drifting against a
# stale constant in cfg.yaml.
# ==========================================================================

_geometry: dict | None = None

_LAT0_RE = re.compile(r"\+lat_0=([-\d.]+)")
_LON0_RE = re.compile(r"\+lon_0=([-\d.]+)")


def _newest_file() -> Path | None:
    """Return the newest HDF5 file across all product folders, or None."""
    newest: tuple[datetime, Path] | None = None
    cfg = _torchiarolo_cfg()
    for product in cfg.get("products", {}):
        folder = _get_product_folder(product)
        if not folder.exists():
            continue
        try:
            for fname in os.listdir(folder):
                dt = parse_torchiarolo_filename(fname)
                if dt is not None and (newest is None or dt > newest[0]):
                    newest = (dt, folder / fname)
        except OSError:
            continue
    return newest[1] if newest else None


def _fallback_geometry() -> dict:
    """Geometry from cfg.yaml, used only when no data file is available yet."""
    fb = _torchiarolo_cfg().get("fallback_grid", {})
    lat = fb.get("prj_lat", 40.5064)
    lon = fb.get("prj_lon", 18.0598)
    n = int(fb.get("n_pixels", 400))
    scale = float(fb.get("xscale", 1000.0))
    return {
        "projdef": f"+proj=tmerc +lat_0={lat} +lon_0={lon} +ellps=WGS84",
        "prj_lat": lat,
        "prj_lon": lon,
        "ncols": n,
        "nlines": n,
        "xscale": scale,
        "yscale": scale,
        "from_file": False,
    }


def _grid_geometry() -> dict:
    """Read and cache the grid geometry from the newest available HDF5 file.

    Falls back to cfg.yaml constants when no file exists (fresh install before
    the first download). The fallback result is NOT cached, so the real
    geometry is picked up as soon as data lands.
    """
    global _geometry
    if _geometry is not None:
        return _geometry

    path = _newest_file()
    if path is None:
        logger.warning("Torchiarolo: no data file found, using fallback grid from cfg.yaml")
        return _fallback_geometry()

    try:
        import h5py

        with h5py.File(path, "r") as f:
            where = f["where"].attrs
            projdef = where["projdef"]
            if isinstance(projdef, bytes):
                projdef = projdef.decode()
            geom = {
                "projdef": projdef,
                "ncols": int(where["xsize"]),
                "nlines": int(where["ysize"]),
                "xscale": float(where["xscale"]),
                "yscale": float(where["yscale"]),
                "from_file": True,
            }
    except (OSError, KeyError) as e:
        logger.warning("Torchiarolo: cannot read geometry from %s (%s), using fallback", path, e)
        return _fallback_geometry()

    lat_m = _LAT0_RE.search(geom["projdef"])
    lon_m = _LON0_RE.search(geom["projdef"])
    geom["prj_lat"] = float(lat_m.group(1)) if lat_m else 40.5064
    geom["prj_lon"] = float(lon_m.group(1)) if lon_m else 18.0598

    logger.info(
        "Torchiarolo grid: %dx%d @ %.0f m, centre (%.4f, %.4f)",
        geom["ncols"], geom["nlines"], geom["xscale"], geom["prj_lat"], geom["prj_lon"],
    )
    _geometry = geom
    return geom


def get_grid_extent_m(geom: dict) -> tuple[float, float]:
    """Return (half_width_m, half_height_m) of the grid, centred on the projection origin."""
    return (geom["ncols"] / 2) * geom["xscale"], (geom["nlines"] / 2) * geom["yscale"]


def get_overlay_bounds() -> list:
    """Return [[lat_sw, lon_sw], [lat_ne, lon_ne]] enclosing the Torchiarolo grid.

    The grid is a square in Transverse Mercator, so its lat/lon envelope is
    slightly larger than any single pair of corners; we take the extremes over
    all four so the warped image is fully contained.
    """
    geom = _grid_geometry()
    proj = Proj(geom["projdef"])
    half_x, half_y = get_grid_extent_m(geom)

    lats, lons = [], []
    for x in (-half_x, half_x):
        for y in (-half_y, half_y):
            lon, lat = proj(x, y, inverse=True)
            lats.append(lat)
            lons.append(lon)
    return [[min(lats), min(lons)], [max(lats), max(lons)]]


# ==========================================================================
# Value decoding — ODIM gain/offset
# ==========================================================================


def _attr(what_attrs: dict, key: str) -> float | None:
    """Return an ODIM what/ attribute as a float, or None if absent or unparseable."""
    value = what_attrs.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fill_values(what_attrs: dict) -> list[float]:
    """Return the raw counts that represent no data (nodata and undetect)."""
    return [
        v for v in (_attr(what_attrs, "nodata"), _attr(what_attrs, "undetect"))
        if v is not None
    ]


def scaling(what_attrs: dict) -> tuple[float, float]:
    """Return the (gain, offset) that convert raw counts to physical units.

    This reproduces the provider's own convention, as implemented in datamet
    (sou_py/dpg/calibration.py): the value scale may start at a raw count other
    than zero — datamet calls it 'bottom' and applies it by rolling the lookup
    table. When the declared offset is 0 and raw 0 is the no-data marker, the
    physical zero sits at raw 1, so a naive raw*gain + offset would shift every
    value up by one gain step.

    For these products: VIL (gain 0.5, offset 0, nodata 0) takes the shift, so
    raw 1 decodes to 0.0 kg/m²; VMI (offset -31) and ETM (offset 2000) do not,
    so for them undetect = 1 remains a fill value.
    """
    gain = float(what_attrs.get("gain", 1.0))
    offset = float(what_attrs.get("offset", 0.0))

    bottom = 0
    if offset == 0.0:
        nodata = _attr(what_attrs, "nodata")
        undetect = _attr(what_attrs, "undetect")
        if nodata == 0.0:
            bottom = 1
        if undetect is not None and undetect >= 0:
            bottom = int(undetect) + 1

    return gain, offset - gain * bottom


def decode_physical(raw, what_attrs):
    """Convert a raw ODIM count to physical units, returning None for fill values.

    Unlike the Cagliari products (already physical), VMI/VIL/ETM here are uint8
    with a gain/offset. Per ODIM, nodata and undetect are expressed in RAW
    counts, so they must be tested before applying the scaling.
    """
    import math

    gain, offset = scaling(what_attrs)

    value = float(raw)
    if any(value == fill for fill in fill_values(what_attrs)):
        return None
    physical = value * gain + offset
    return None if math.isnan(physical) else physical


# ==========================================================================
# Pixel sampling helper
# ==========================================================================


def _sample_torchiarolo_pixel(file_path: Path, row: int, col: int) -> float | None:
    """Read one Cartesian pixel from a Torchiarolo HDF5 file and return the physical value."""
    import h5py

    try:
        with h5py.File(file_path, "r") as f:
            raw = f["dataset1/data1/data"][row, col]
            what = dict(f["dataset1/data1/what"].attrs)
        return decode_physical(raw, what)
    except Exception:
        return None


# ==========================================================================
# API endpoints
# ==========================================================================


@router.get("/config")
async def get_torchiarolo_config():
    """Return Torchiarolo grid parameters and map configuration (including colorbar data)."""
    from nwc_webapp.rendering.colormaps import get_legend_data, build_legend_file_path

    cfg = _torchiarolo_cfg()
    geom = _grid_geometry()
    bounds = get_overlay_bounds()
    lat, lon = geom["prj_lat"], geom["prj_lon"]

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

    half_x, _ = get_grid_extent_m(geom)
    return {
        "radar_lat": lat,
        "radar_lon": lon,
        "overlay_bounds": bounds,
        "center": [lat, lon],
        "zoom": cfg.get("zoom", 8),
        "grid": {
            "ncols": geom["ncols"],
            "nlines": geom["nlines"],
            "xscale": geom["xscale"],
            "range_km": round(half_x / 1000.0),
            "from_file": geom["from_file"],
        },
        "products": products,
    }


@router.get("/timestamps")
async def get_torchiarolo_timestamps(
    product: str = Query("SRI", description="Product name: SRI, VMI, VIL or ETM"),
    lookback_minutes: int = Query(60, description="Lookback window in minutes"),
):
    """List available Torchiarolo timestamps for a product within the lookback window."""
    cfg = _torchiarolo_cfg()
    valid_products = list(cfg.get("products", {}).keys()) or ["SRI", "VMI", "VIL", "ETM"]
    if product not in valid_products:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown product: {product}. Available: {valid_products}",
        )

    folder = _get_product_folder(product)
    if not folder.exists():
        return {"product": product, "timestamps": [], "total": 0}

    cutoff = datetime.utcnow() - timedelta(minutes=lookback_minutes + 6)

    found: list[datetime] = []
    try:
        for fname in os.listdir(folder):
            dt = parse_torchiarolo_filename(fname)
            if dt is not None and dt >= cutoff:
                found.append(dt)
    except OSError as e:
        logger.warning("Cannot list Torchiarolo folder %s: %s", folder, e)
        return {"product": product, "timestamps": [], "total": 0}

    found.sort()
    return {
        "product": product,
        "timestamps": [dt.isoformat() for dt in found],
        "total": len(found),
    }


@router.get("/sample")
async def sample_torchiarolo_pixel(
    lat: float = Query(..., description="Latitude (WGS84)"),
    lon: float = Query(..., description="Longitude (WGS84)"),
    timestamp: str = Query(..., description="Datetime ISO (YYYY-MM-DDTHH:MM)"),
    products: str = Query("SRI", description="Comma-separated: SRI, VMI, VIL, ETM"),
):
    """
    Sample Torchiarolo values at a clicked (lat, lon).

    Converts lat/lon to Transverse Mercator metres on the file's own grid, then
    to (row, col). Row 0 is the northernmost row (ODIM convention).
    """
    import math

    try:
        dt = datetime.fromisoformat(timestamp)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid datetime: {e}")

    cfg = _torchiarolo_cfg()
    geom = _grid_geometry()
    proj = Proj(geom["projdef"])
    half_x, half_y = get_grid_extent_m(geom)

    dx, dy = proj(lon, lat)  # East, North in metres relative to the grid centre

    range_m = math.sqrt(dx ** 2 + dy ** 2)
    azimuth_deg = math.degrees(math.atan2(dx, dy)) % 360.0
    in_bounds = abs(dx) < half_x and abs(dy) < half_y

    col = int((dx + half_x) / geom["xscale"])
    row = int((half_y - dy) / geom["yscale"])
    col = max(0, min(geom["ncols"] - 1, col))
    row = max(0, min(geom["nlines"] - 1, row))

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

    product_list = [p.strip() for p in products.split(",") if p.strip()]
    valid_products = list(cfg.get("products", {}).keys()) or ["SRI", "VMI", "VIL", "ETM"]

    for product in product_list:
        if product not in valid_products:
            result["values"][product] = None
            continue
        file_path = find_torchiarolo_file(product, dt)
        if file_path is None:
            result["values"][product] = None
            continue
        result["values"][product] = _sample_torchiarolo_pixel(file_path, row, col)

    return result


@router.websocket("/ws")
async def torchiarolo_websocket(websocket: WebSocket):
    """WebSocket endpoint — server pushes {type: 'torchiarolo_update', data: {...}} on new data."""
    await torchiarolo_ws_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        torchiarolo_ws_manager.disconnect(websocket)