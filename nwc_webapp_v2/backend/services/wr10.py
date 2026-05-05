"""
WR10Service — background thread that polls /data/wr10/SRI and /data/wr10/VMI
for new files and broadcasts via WebSocket when one arrives.

Mirrors the pattern of services/realtime.py but is much simpler: no job
submission, just folder-watching and WS push.
"""

import logging
import os
import threading
from datetime import datetime
from pathlib import Path

import yaml

from nwc_webapp.config.environment import is_server
from api.wr10 import wr10_ws_manager, parse_wr10_filename, _get_product_folder, _wr10_cfg

logger = logging.getLogger(__name__)


class WR10Service:
    """
    Singleton service that watches WR10 data folders for new files and
    broadcasts a 'wr10_update' WebSocket message when new data arrives.
    """

    _instance = None
    _instance_lock = threading.Lock()

    def __new__(cls):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._active = False
        self._last_seen: dict[str, str] = {}  # product → latest filename

    def start(self):
        if self._active:
            return
        self._active = True
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._poll_loop,
            daemon=True,
            name="wr10-watcher",
        )
        self._thread.start()
        logger.info("WR10Service started")

    def stop(self):
        self._active = False
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        logger.info("WR10Service stopped")

    # ------------------------------------------------------------------
    # Background poll loop
    # ------------------------------------------------------------------

    def _poll_loop(self):
        """Poll each WR10 product folder every second for new files."""
        cfg = _wr10_cfg()
        products = list(cfg.get("products", {}).keys()) or ["SRI", "VMI"]

        POLL_INTERVAL = 1  # seconds

        # Initialise last_seen to the current latest file so we don't
        # immediately blast a notification for files that were already there.
        for product in products:
            latest = self._find_latest(product)
            if latest:
                self._last_seen[product] = latest
                logger.info("WR10 initial latest %s: %s", product, latest)

        while not self._stop_event.is_set():
            for product in products:
                latest = self._find_latest(product)
                if latest and latest != self._last_seen.get(product):
                    self._last_seen[product] = latest
                    _, dt = parse_wr10_filename(latest)
                    ts_iso = dt.isoformat() if dt else None
                    logger.info("New WR10 file detected: %s / %s", product, latest)
                    wr10_ws_manager.broadcast_sync({
                        "type": "wr10_update",
                        "data": {
                            "product": product,
                            "filename": latest,
                            "timestamp": ts_iso,
                        },
                    })

            if self._stop_event.wait(timeout=POLL_INTERVAL):
                break

    @staticmethod
    def _find_latest(product: str) -> str | None:
        """Return the filename of the newest WR10 file for the given product."""
        from api.wr10 import _FNAME_RE
        folder = _get_product_folder(product)
        if not folder.exists():
            return None
        files = []
        try:
            for fname in os.listdir(folder):
                m = _FNAME_RE.match(fname)
                if m and m.group(1) == product:
                    try:
                        dt = datetime.strptime(m.group(2), "%Y%m%d%H%M")
                        files.append((dt, fname))
                    except ValueError:
                        pass
        except OSError:
            return None
        if not files:
            return None
        files.sort(key=lambda x: x[0], reverse=True)
        return files[0][1]
