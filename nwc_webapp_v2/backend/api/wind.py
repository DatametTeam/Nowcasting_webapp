"""
Wind / AMV (Atmospheric Motion Vector) API endpoints.

AMV data arrives every ~20 minutes as shapefiles (one per timestamp).
Each file contains 500-600 scattered observation points with:
  - AMV:  wind speed in knots
  - Dir:  meteorological direction in degrees (FROM which direction)
  - geometry: Point (lon, lat) in WGS84

These endpoints:
  GET /api/wind/timestamps  → list of ISO timestamps with available AMV data
  GET /api/wind/data        → leaflet-velocity JSON for a specific timestamp

The leaflet-velocity format is a 2-element list [U-component, V-component],
each with a 'header' (grid metadata) and a flat 'data' array (m/s, row-major
from top-left / highest latitude to bottom-right / lowest latitude).
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from ws.manager import ConnectionManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/wind", tags=["wind"])
wind_ws_manager = ConnectionManager()

# In-memory cache: timestamp string → velocity JSON list.
# Populated on first request; lives for the lifetime of the server process.
# Eliminates repeated shapefile reads and scipy interpolation for the same file.
_velocity_cache: dict = {}

# Shapefile filename pattern: DD-MM-YYYY-HH-MM.shp
_FILENAME_RE = re.compile(
    r"^(\d{2})-(\d{2})-(\d{4})-(\d{2})-(\d{2})\.shp$"
)

# Output grid covering Italy and surroundings (WGS84 lon/lat)
_LO1 = 4.5
_LA1 = 47.75   # top (highest latitude — leaflet-velocity scans north→south)
_LO2 = 19.5
_LA2 = 35.5    # bottom
_NX  = 60
_NY  = 50


def _amv_folder() -> Optional[Path]:
    from nwc_webapp.config.config import get_config
    return get_config().amv_folder


def _parse_filename(name: str) -> Optional[datetime]:
    """Convert 'DD-MM-YYYY-HH-MM.shp' → UTC datetime, or None if no match."""
    m = _FILENAME_RE.match(name)
    if not m:
        return None
    dd, mo, yyyy, hh, mm = (int(x) for x in m.groups())
    try:
        return datetime(yyyy, mo, dd, hh, mm, tzinfo=timezone.utc)
    except ValueError:
        return None


def _shapefile_to_velocity(path: Path, ref_time: str) -> list[dict]:
    """
    Read a shapefile and return a leaflet-velocity compatible payload.

    Conversion steps:
      1. Parse speed (knots → m/s) and meteorological direction → u/v.
      2. Bilinear interpolation of scattered points onto a regular lon/lat grid.
      3. Fill boundary NaNs with nearest-neighbour.
      4. Pack into the two-element [U, V] leaflet-velocity JSON structure.
    """
    import geopandas as gpd
    from scipy.interpolate import griddata

    gdf = gpd.read_file(path)

    lons  = gdf.geometry.x.values
    lats  = gdf.geometry.y.values
    # AMV pipeline (readLastAMV with as_is=False) converts knots → km/h (* 1.852)
    # and FROM direction → TO direction (+ 180°).  So the shapefile already
    # contains km/h velocities and direction-of-motion (not meteorological FROM).
    speed = gdf["AMV"].values / 3.6              # km/h → m/s
    direc = np.radians(gdf["Dir"].values)        # direction of motion, clockwise from North

    # Standard vector decomposition for a TO-direction:
    # u (eastward)  = speed · sin(dir)
    # v (northward) = speed · cos(dir)
    u = speed * np.sin(direc)
    v = speed * np.cos(direc)

    # Regular grid: top (LA1) → bottom (LA2), left (LO1) → right (LO2)
    grid_lons = np.linspace(_LO1, _LO2, _NX)
    grid_lats = np.linspace(_LA1, _LA2, _NY)   # descending so row 0 = northernmost
    gx, gy = np.meshgrid(grid_lons, grid_lats)

    points = np.column_stack([lons, lats])
    u_grid = griddata(points, u, (gx, gy), method="linear")
    v_grid = griddata(points, v, (gx, gy), method="linear")

    # Fill boundary NaNs (outside the convex hull of obs points) with nearest neighbour
    mask = np.isnan(u_grid)
    if mask.any():
        u_nn   = griddata(points, u, (gx, gy), method="nearest")
        v_nn   = griddata(points, v, (gx, gy), method="nearest")
        u_grid[mask] = u_nn[mask]
        v_grid[mask] = v_nn[mask]

    dx = (_LO2 - _LO1) / (_NX - 1)
    dy = (_LA1 - _LA2) / (_NY - 1)   # positive step size (abs value of lat step)

    header_base = {
        "parameterCategory": 2,
        "lo1": _LO1, "la1": _LA1,
        "lo2": _LO2, "la2": _LA2,
        "dx":  round(dx, 6),
        "dy":  round(dy, 6),
        "nx":  _NX,
        "ny":  _NY,
        "refTime": ref_time,
    }

    return [
        {
            "header": {**header_base, "parameterNumber": 2},   # U
            "data":   [round(float(x), 4) for x in u_grid.flatten()],
        },
        {
            "header": {**header_base, "parameterNumber": 3},   # V
            "data":   [round(float(x), 4) for x in v_grid.flatten()],
        },
    ]


@router.get("/timestamps")
async def get_wind_timestamps(
    lookback_hours: int = Query(24, ge=1, le=720, description="Only return timestamps within this many hours"),
):
    """
    Return a sorted list of ISO timestamps for which AMV shapefiles exist,
    limited to the last `lookback_hours` hours.
    """
    folder = _amv_folder()
    if not folder or not folder.is_dir():
        return {"timestamps": []}

    cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=lookback_hours)
    result = []
    for shp in folder.glob("*.shp"):
        dt = _parse_filename(shp.name)
        if dt is not None and dt >= cutoff:
            result.append(dt.strftime("%Y-%m-%dT%H:%M"))

    result.sort()
    return {"timestamps": result}


@router.get("/data")
async def get_wind_data(timestamp: str):
    """
    Return leaflet-velocity JSON for the given timestamp.
    `timestamp` must be in "YYYY-MM-DDTHH:MM" format (UTC).
    """
    folder = _amv_folder()
    if not folder or not folder.is_dir():
        raise HTTPException(status_code=503, detail="AMV data folder not configured or missing.")

    # Convert ISO timestamp → filename pattern DD-MM-YYYY-HH-MM.shp
    try:
        dt = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M")
    except ValueError:
        raise HTTPException(status_code=400, detail="timestamp must be YYYY-MM-DDTHH:MM")

    filename = dt.strftime("%d-%m-%Y-%H-%M.shp")
    path = folder / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"No AMV data for {timestamp}.")

    if timestamp not in _velocity_cache:
        try:
            ref_time = dt.strftime("%Y-%m-%d %H:%M:%S")
            _velocity_cache[timestamp] = _shapefile_to_velocity(path, ref_time)
        except Exception as exc:
            logger.error("Failed to process AMV file %s: %s", path, exc, exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to process AMV file: {exc}")

    return JSONResponse(content=_velocity_cache[timestamp])


@router.post("/notify")
async def wind_notify():
    """
    Called by a cron/pipeline script after a new AMV shapefile is saved.
    Broadcasts amv_ready to all connected WS clients.
    """
    ts = datetime.utcnow().isoformat()
    _velocity_cache.clear()   # invalidate so next request reads fresh shapefile
    await wind_ws_manager.broadcast({"type": "amv_ready", "ts": ts})
    logger.info("AMV notify broadcast sent at %s", ts)
    return {"ok": True, "ts": ts}


@router.websocket("/ws")
async def wind_websocket(websocket: WebSocket):
    await wind_ws_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        wind_ws_manager.disconnect(websocket)
