"""
CagliariService — background thread that polls /data/cagliari_xband/RR and /data/cagliari_xband/CZ
for new files and broadcasts via WebSocket when one arrives.

Mirrors the pattern of services/wr10.py.
"""

import logging
import os
import threading
from datetime import datetime
from pathlib import Path

from nwc_webapp.config.environment import is_server
from api.cagliari import cagliari_ws_manager, parse_cagliari_filename, _get_product_folder, _cagliari_cfg

logger = logging.getLogger(__name__)


class CagliariService:
    """
    Singleton service that watches Cagliari data folders for new files and
    broadcasts a 'cagliari_update' WebSocket message when new data arrives.
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
            name="cagliari-watcher",
        )
        self._thread.start()
        logger.info("CagliariService started")

    def stop(self):
        self._active = False
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        logger.info("CagliariService stopped")

    # ------------------------------------------------------------------
    # Background poll loop
    # ------------------------------------------------------------------

    def _poll_loop(self):
        """Poll each Cagliari product folder every second for new files."""
        cfg = _cagliari_cfg()
        products = list(cfg.get("products", {}).keys()) or ["RR", "CZ"]

        POLL_INTERVAL = 1  # seconds

        for product in products:
            latest = self._find_latest(product)
            if latest:
                self._last_seen[product] = latest
                logger.info("Cagliari initial latest %s: %s", product, latest)

        while not self._stop_event.is_set():
            for product in products:
                latest = self._find_latest(product)
                if latest and latest != self._last_seen.get(product):
                    self._last_seen[product] = latest
                    _, _, dt = parse_cagliari_filename(latest)
                    ts_iso = dt.isoformat() if dt else None
                    logger.info("New Cagliari file detected: %s / %s", product, latest)
                    cagliari_ws_manager.broadcast_sync({
                        "type": "cagliari_update",
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
        """Return the filename of the newest Cagliari file for the given product."""
        folder = _get_product_folder(product)
        if not folder.exists():
            return None
        files = []
        try:
            for fname in os.listdir(folder):
                prefix, idx, dt = parse_cagliari_filename(fname)
                if prefix == product and dt is not None:
                    files.append((dt, fname))
        except OSError:
            return None
        if not files:
            return None
        files.sort(key=lambda x: x[0], reverse=True)
        return files[0][1]
