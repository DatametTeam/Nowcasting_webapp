"""
Lucas-Kanade optical flow API endpoints.

LK flow is pre-computed by the cron script (scripts/LK_computation/compute_lk.py)
and saved as leaflet-velocity compatible JSON files named DD-MM-YYYY-HH-MM.json.

  GET  /api/lk/timestamps  → sorted list of ISO timestamps with available LK data
  GET  /api/lk/data        → leaflet-velocity JSON for a specific timestamp
  POST /api/lk/notify      → cron trigger: broadcasts lk_updated to WS clients
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from ws.manager import ConnectionManager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/lk", tags=["lk"])

lk_ws_manager = ConnectionManager()

# In-memory cache: timestamp string → parsed JSON list.
_data_cache: dict = {}

# Filename pattern: DD-MM-YYYY-HH-MM.json
_FILENAME_RE = re.compile(r"^(\d{2})-(\d{2})-(\d{4})-(\d{2})-(\d{2})\.json$")


def _lk_folder() -> Optional[Path]:
    from nwc_webapp.config.config import get_config
    return get_config().lk_folder


def _parse_filename(name: str) -> Optional[datetime]:
    """Convert 'DD-MM-YYYY-HH-MM.json' → UTC datetime, or None if no match."""
    m = _FILENAME_RE.match(name)
    if not m:
        return None
    dd, mo, yyyy, hh, mm = (int(x) for x in m.groups())
    try:
        return datetime(yyyy, mo, dd, hh, mm, tzinfo=timezone.utc)
    except ValueError:
        return None


@router.get("/timestamps")
async def get_lk_timestamps():
    """Return a sorted list of ISO timestamps for which LK JSON files exist."""
    folder = _lk_folder()
    if not folder or not folder.is_dir():
        return {"timestamps": []}

    result = []
    for f in folder.glob("*.json"):
        if f.name == "latest_flow.json":
            continue
        dt = _parse_filename(f.name)
        if dt is not None:
            result.append(dt.strftime("%Y-%m-%dT%H:%M"))

    result.sort()
    return {"timestamps": result}


@router.get("/data")
async def get_lk_data(timestamp: str):
    """
    Return leaflet-velocity JSON for the given timestamp.
    `timestamp` must be in "YYYY-MM-DDTHH:MM" format (UTC).
    """
    folder = _lk_folder()
    if not folder or not folder.is_dir():
        raise HTTPException(status_code=503, detail="LK data folder not configured or missing.")

    try:
        dt = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M")
    except ValueError:
        raise HTTPException(status_code=400, detail="timestamp must be YYYY-MM-DDTHH:MM")

    filename = dt.strftime("%d-%m-%Y-%H-%M.json")
    path = folder / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"No LK data for {timestamp}.")

    if timestamp not in _data_cache:
        try:
            _data_cache[timestamp] = json.loads(path.read_text())
        except Exception as exc:
            logger.error("Failed to read LK file %s: %s", path, exc)
            raise HTTPException(status_code=500, detail=f"Failed to read LK file: {exc}")

    return JSONResponse(content=_data_cache[timestamp])


@router.get("/image")
async def get_lk_image(timestamp: str):
    """
    Return the pre-rendered quiver-arrow PNG for the given timestamp.
    `timestamp` must be in "YYYY-MM-DDTHH:MM" format (UTC).
    """
    folder = _lk_folder()
    if not folder or not folder.is_dir():
        raise HTTPException(status_code=503, detail="LK data folder not configured or missing.")

    try:
        dt = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M")
    except ValueError:
        raise HTTPException(status_code=400, detail="timestamp must be YYYY-MM-DDTHH:MM")

    filename = dt.strftime("%d-%m-%Y-%H-%M.png")
    path = folder / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"No LK arrow image for {timestamp}.")

    return FileResponse(str(path), media_type="image/png")


@router.post("/notify")
async def lk_notify():
    """
    Called by the cron script after a new LK JSON is saved.
    Broadcasts lk_updated to all connected WS clients.
    """
    ts = datetime.utcnow().isoformat()
    # Invalidate cache for the latest entry so next request reads fresh data
    _data_cache.clear()
    await lk_ws_manager.broadcast({"type": "lk_updated", "ts": ts})
    logger.info("LK notify broadcast sent at %s", ts)
    return {"ok": True, "ts": ts}


@router.websocket("/ws")
async def lk_websocket(websocket: WebSocket):
    await lk_ws_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        lk_ws_manager.disconnect(websocket)
