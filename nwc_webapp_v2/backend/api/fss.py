"""
FSS real-time assessment API.

GET  /api/fss/recent?scale=5&hours=24  → last N hours of 5-min FSS time series
GET  /api/fss/daily?scale=5&days=90    → last N days of daily-mean FSS
POST /api/fss/notify                   → cron trigger: broadcast fss_updated to WS clients
WS   /api/fss/ws                       → push notifications
"""
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from nwc_webapp.config.environment import is_server
from ws.manager import ConnectionManager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/fss", tags=["fss"])

fss_ws_manager = ConnectionManager()

_cfg_cache = None


def _fss_cfg() -> dict:
    global _cfg_cache
    if _cfg_cache is not None:
        return _cfg_cache
    cfg_path = Path(__file__).parent.parent.parent / "cfg.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    _cfg_cache = cfg.get("fss", {})
    return _cfg_cache


def _fss_root() -> Path:
    cfg = _fss_cfg()
    if is_server():
        return Path(cfg.get("server_fss_root", "/data/FSS_metrics"))
    return Path(cfg.get("local_fss_root", "data/FSS_metrics"))


def _thr_str(thr: float) -> str:
    return str(int(thr)) if thr == int(thr) else str(thr)


def _discover_models() -> list[str]:
    root = _fss_root()
    if not root.exists():
        return []
    return [d.name for d in sorted(root.iterdir()) if d.is_dir()]


def _load_series(
    model: str,
    lead_time: int,
    threshold: float,
    scale: int,
    n_months: int = 2,
) -> pd.DataFrame:
    """Return DataFrame with [sc_{scale}, n_valid] indexed by datetime."""
    thr = _thr_str(threshold)
    dirpath = _fss_root() / model / f"lt{lead_time}" / f"thr{thr}"
    col = f"sc_{scale}"

    now = pd.Timestamp.now()
    frames = []
    seen: set = set()
    for i in range(n_months):
        ts = now - pd.DateOffset(months=i)
        key = (ts.year, ts.month)
        if key in seen:
            continue
        seen.add(key)
        mm = f"{ts.month:02d}"
        yyyy = str(ts.year)
        path = dirpath / f"FSS_@{lead_time}min_{thr}mm_h_{mm}-{yyyy}.csv"
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, sep=r"\s+")
            df["datetime"] = pd.to_datetime(
                df["date"] + " " + df["time"], format="%d-%m-%Y %H:%M"
            )
            df = df.set_index("datetime").sort_index()
            if col not in df.columns:
                continue
            frames.append(df[[col, "n_valid"]])
        except Exception:
            continue

    if not frames:
        return pd.DataFrame(
            {col: pd.Series(dtype=float), "n_valid": pd.Series(dtype=float)}
        )

    combined = pd.concat(frames).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    combined[col] = combined[col].astype(float)
    return combined


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/recent")
async def get_recent_fss(
    scale: int = Query(5, description="Spatial scale km — 1, 5, or 20"),
    hours: int = Query(24, description="Lookback window in hours"),
):
    """Return last `hours` hours of 5-min FSS time series for all available models."""
    models = _discover_models()
    lead_times = [15, 30, 45, 60]
    thresholds = [5.0, 10.0, 25.0]
    col = f"sc_{scale}"

    now = pd.Timestamp.now()
    cutoff = now - pd.Timedelta(hours=hours)

    series_out: dict = {}
    means_out: dict = {}
    last_updated = None

    for model in models:
        series_out[model] = {}
        means_out[model] = {}
        for lt in lead_times:
            lt_key = f"lt{lt}"
            series_out[model][lt_key] = {}
            means_out[model][lt_key] = {}
            for thr in thresholds:
                thr_key = f"thr{int(thr)}"
                df = _load_series(model, lt, thr, scale)
                if not df.empty and col in df.columns:
                    s = df.loc[df.index >= cutoff, col]
                    valid = s.dropna()
                    if not valid.empty:
                        ts_last = valid.index[-1]
                        if last_updated is None or ts_last > last_updated:
                            last_updated = ts_last
                    points = [
                        {
                            "t": idx.isoformat(),
                            "v": round(float(v), 4) if not np.isnan(v) else None,
                        }
                        for idx, v in s.items()
                    ]
                    mean_val = float(valid.mean()) if not valid.empty else None
                else:
                    points, mean_val = [], None

                series_out[model][lt_key][thr_key] = points
                means_out[model][lt_key][thr_key] = (
                    round(mean_val, 4) if mean_val is not None else None
                )

    return {
        "scale": scale,
        "hours": hours,
        "models": models,
        "lead_times": lead_times,
        "thresholds": thresholds,
        "series": series_out,
        "means": means_out,
        "last_updated": last_updated.isoformat() if last_updated else None,
    }


@router.get("/daily")
async def get_daily_fss(
    scale: int = Query(5, description="Spatial scale km"),
    days: int = Query(90, description="Lookback window in days"),
    min_valid: str = Query(
        "5:1000,10:500,25:100",
        description="Min n_valid per threshold: '5:1000,10:500,25:100'",
    ),
):
    """Return last `days` days of daily-mean FSS for all available models."""
    models = _discover_models()
    lead_times = [15, 30, 45, 60]
    thresholds = [5.0, 10.0, 25.0]
    col = f"sc_{scale}"

    min_valid_map: dict[float, int] = {}
    try:
        for part in min_valid.split(","):
            t, v = part.split(":")
            min_valid_map[float(t)] = int(v)
    except Exception:
        min_valid_map = {5.0: 1000, 10.0: 500, 25.0: 100}

    now = pd.Timestamp.now().floor("D")
    cutoff = now - pd.Timedelta(days=days)
    full_dates = pd.date_range(start=cutoff, end=now, freq="D")

    series_out: dict = {}
    means_out: dict = {}

    for model in models:
        series_out[model] = {}
        means_out[model] = {}
        for lt in lead_times:
            lt_key = f"lt{lt}"
            series_out[model][lt_key] = {}
            means_out[model][lt_key] = {}
            for thr in thresholds:
                thr_key = f"thr{int(thr)}"
                df = _load_series(model, lt, thr, scale, n_months=4)
                if not df.empty and col in df.columns:
                    df = df.loc[df.index >= cutoff].copy()
                    mv = min_valid_map.get(thr, 0)
                    if mv > 0:
                        df.loc[df["n_valid"] < mv, col] = np.nan
                    s = df[col].resample("D").mean().reindex(full_dates)
                else:
                    s = pd.Series(dtype=float, index=full_dates)

                valid_s = s.dropna()
                points = [
                    {
                        "t": idx.strftime("%Y-%m-%d"),
                        "v": round(float(v), 4) if not np.isnan(v) else None,
                    }
                    for idx, v in s.items()
                ]
                mean_val = float(valid_s.mean()) if not valid_s.empty else None
                series_out[model][lt_key][thr_key] = points
                means_out[model][lt_key][thr_key] = (
                    round(mean_val, 4) if mean_val is not None else None
                )

    return {
        "scale": scale,
        "days": days,
        "models": models,
        "lead_times": lead_times,
        "thresholds": thresholds,
        "series": series_out,
        "means": means_out,
    }


@router.post("/notify")
async def fss_notify():
    """
    Called by the cron script after FSS CSVs are updated.
    Broadcasts fss_updated to all connected WS clients so the frontend
    refetches immediately instead of waiting for the 5-min poll.
    """
    ts = datetime.utcnow().isoformat()
    await fss_ws_manager.broadcast({"type": "fss_updated", "ts": ts})
    logger.info("FSS notify broadcast sent at %s", ts)
    return {"ok": True, "ts": ts}


@router.websocket("/ws")
async def fss_websocket(websocket: WebSocket):
    """WebSocket — pushes {type: 'fss_updated', ts: '...'} when CSVs change."""
    await fss_ws_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()  # keep-alive; client sends nothing
    except WebSocketDisconnect:
        fss_ws_manager.disconnect(websocket)