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

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/api/wind", tags=["wind"])

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
    speed = gdf["AMV"].values * 0.51444          # knots → m/s
    direc = np.radians(gdf["Dir"].values)        # meteorological degrees → radians

    # Meteorological convention: direction is FROM which compass point the wind blows.
    # u (eastward)  = −speed · sin(dir)
    # v (northward) = −speed · cos(dir)
    u = -speed * np.sin(direc)
    v = -speed * np.cos(direc)

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
async def get_wind_timestamps():
    """
    Return a sorted list of ISO timestamps for which AMV shapefiles exist.
    Format: "YYYY-MM-DDTHH:MM" (UTC, no seconds — matches radar timestamp format).
    """
    folder = _amv_folder()
    if not folder or not folder.is_dir():
        return {"timestamps": []}

    result = []
    for shp in folder.glob("*.shp"):
        dt = _parse_filename(shp.name)
        if dt is not None:
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

    try:
        ref_time = dt.strftime("%Y-%m-%d %H:%M:%S")
        payload = _shapefile_to_velocity(path, ref_time)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to process AMV file: {exc}")

    return JSONResponse(content=payload)
