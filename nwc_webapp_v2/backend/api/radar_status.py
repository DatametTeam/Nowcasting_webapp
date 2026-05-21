"""
Radar availability status API.

Files named DD-MM-YYYY-HH-MM.txt are downloaded every 5 min from FTP.
Each file lists the active radar sites at that timestamp.

GET  /api/radar-status/timestamps?lookback_hours=24   → sorted ISO timestamps with available data
GET  /api/radar-status/range?start=...&end=...        → { statuses: {ts: [site, ...], ...} }
POST /api/radar-status/notify                         → cron trigger: broadcast WS event
WS   /api/radar-status/ws                            → push notifications
"""

import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect

from ws.manager import ConnectionManager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/radar-status", tags=["radar-status"])

radar_status_ws_manager = ConnectionManager()

# Filename pattern: DD-MM-YYYY-HH-MM.txt
_FILENAME_RE = re.compile(r"^(\d{2})-(\d{2})-(\d{4})-(\d{2})-(\d{2})\.txt$")


def _status_folder() -> Optional[Path]:
    from nwc_webapp.config.config import get_config
    return get_config().radar_status_folder


def _parse_filename(name: str) -> Optional[datetime]:
    """Convert 'DD-MM-YYYY-HH-MM.txt' → UTC datetime, or None if no match."""
    m = _FILENAME_RE.match(name)
    if not m:
        return None
    dd, mo, yyyy, hh, mm = (int(x) for x in m.groups())
    try:
        return datetime(yyyy, mo, dd, hh, mm, tzinfo=timezone.utc)
    except ValueError:
        return None


def _parse_status_file(path: Path) -> list[str]:
    """Parse a radar status file and return list of active site names."""
    sites = []
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if line.startswith("site"):
                parts = line.split("=", 1)
                if len(parts) == 2:
                    site = parts[1].strip()
                    if site:
                        sites.append(site)
    except Exception as exc:
        logger.warning("Failed to parse %s: %s", path, exc)
    return sites


@router.get("/timestamps")
async def get_radar_status_timestamps(
    lookback_hours: int = Query(24, ge=1, le=720, description="Only return timestamps within this many hours"),
):
    """Return sorted list of ISO timestamps for which status files exist."""
    folder = _status_folder()
    if not folder or not folder.is_dir():
        return {"timestamps": []}

    cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=lookback_hours)
    result = []
    for f in folder.glob("*.txt"):
        dt = _parse_filename(f.name)
        if dt is not None and dt >= cutoff:
            result.append(dt.strftime("%Y-%m-%dT%H:%M"))

    result.sort()
    return {"timestamps": result}


@router.get("/range")
async def get_radar_status_range(
    start: str = Query(..., description="Start timestamp YYYY-MM-DDTHH:MM (UTC)"),
    end: str = Query(..., description="End timestamp YYYY-MM-DDTHH:MM (UTC)"),
):
    """
    Return active radar sites for every available timestamp in [start, end].
    Response: { "statuses": { "2026-05-11T13:00": ["BRIC", "ARMIDDA", ...], ... } }
    """
    folder = _status_folder()
    if not folder or not folder.is_dir():
        return {"statuses": {}}

    try:
        dt_start = datetime.strptime(start, "%Y-%m-%dT%H:%M").replace(tzinfo=timezone.utc)
        dt_end   = datetime.strptime(end,   "%Y-%m-%dT%H:%M").replace(tzinfo=timezone.utc)
    except ValueError:
        raise HTTPException(status_code=400, detail="start/end must be YYYY-MM-DDTHH:MM")

    statuses = {}
    for f in folder.glob("*.txt"):
        dt = _parse_filename(f.name)
        if dt is None or dt < dt_start or dt > dt_end:
            continue
        ts_key = dt.strftime("%Y-%m-%dT%H:%M")
        statuses[ts_key] = _parse_status_file(f)

    return {"statuses": statuses}


@router.post("/notify")
async def radar_status_notify():
    """Called by the cron script after a new status file is downloaded.
    Broadcasts radar_status_updated to all connected WS clients."""
    ts = datetime.utcnow().isoformat()
    await radar_status_ws_manager.broadcast({"type": "radar_status_updated", "ts": ts})
    logger.info("Radar status notify broadcast sent at %s", ts)
    return {"ok": True, "ts": ts}


@router.websocket("/ws")
async def radar_status_websocket(websocket: WebSocket):
    """WebSocket — pushes {type: 'radar_status_updated', ts: '...'} when a new file arrives."""
    await radar_status_ws_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        radar_status_ws_manager.disconnect(websocket)