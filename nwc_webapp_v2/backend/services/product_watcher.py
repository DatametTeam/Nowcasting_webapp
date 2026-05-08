"""
ProductWatcherService — watches each radar product folder and broadcasts
a `product_ready` WebSocket event whenever a new file lands on disk.

This replaces the frontend polling loop: instead of the browser asking
"is VMI/ETM/VIL/IR_108 ready yet?" every 3 s, the backend pushes
{ type: "product_ready", product: "VMI", timestamp: "2026-05-08T10:15:00" }
the instant the file appears.

One file per product per 5-min slot: DD-MM-YYYY-HH-MM.hdf (or .tif/.tiff).
"""

import logging
import os
import threading
from datetime import datetime
from pathlib import Path

from ws.manager import ws_manager
from api.wind import wind_ws_manager

logger = logging.getLogger(__name__)


class ProductWatcherService:
    """
    Singleton that watches all configured radar product folders for new files.
    Broadcasts `product_ready` events via the shared WebSocket manager.
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
        self._thread = None
        self._last_seen: dict[str, str | None] = {}  # product → filename stem

    # ------------------------------------------------------------------

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._watch_loop, daemon=True, name="product-watcher"
        )
        self._thread.start()
        logger.info("ProductWatcherService started")

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3)
        logger.info("ProductWatcherService stopped")

    # ------------------------------------------------------------------

    def _watch_loop(self):
        from nwc_webapp.config.config import get_config

        config = get_config()
        products = config.radar_products

        # Seed last-seen with the current newest file so we don't fire
        # events for files that already existed before startup.
        for product, cfg in products.items():
            folder = config.get_product_folder(product)
            ext = ".tif" if cfg.get("file_format", "hdf") == "tiff" else ".hdf"
            self._last_seen[product] = self._find_latest(folder, ext)

        # Also seed AMV (downloaded shapefiles — no notify script, watcher is the trigger).
        amv_folder = config.amv_folder
        self._last_seen["__amv__"] = self._find_latest(amv_folder, ".shp")

        logger.info(
            "ProductWatcher seeded: %s",
            {p: v for p, v in self._last_seen.items() if v},
        )

        while not self._stop_event.is_set():
            # ---- Radar products → realtime WS ----
            for product, cfg in products.items():
                folder = config.get_product_folder(product)
                if not folder:
                    continue
                ext = ".tif" if cfg.get("file_format", "hdf") == "tiff" else ".hdf"
                latest = self._find_latest(folder, ext)
                if latest and latest != self._last_seen.get(product):
                    self._last_seen[product] = latest
                    ts = self._parse_ts(latest)
                    logger.info("ProductWatcher: new %s → %s", product, latest)
                    ws_manager.broadcast_sync({
                        "type": "product_ready",
                        "product": product,
                        "timestamp": ts.isoformat() if ts else None,
                    })

            # ---- AMV shapefiles → wind WS ----
            if amv_folder:
                latest_amv = self._find_latest(amv_folder, ".shp")
                if latest_amv and latest_amv != self._last_seen.get("__amv__"):
                    self._last_seen["__amv__"] = latest_amv
                    ts = self._parse_ts(latest_amv)
                    logger.info("ProductWatcher: new AMV → %s", latest_amv)
                    wind_ws_manager.broadcast_sync({
                        "type": "amv_ready",
                        "timestamp": ts.isoformat() if ts else None,
                    })

            self._stop_event.wait(timeout=1)

    # ------------------------------------------------------------------

    @staticmethod
    def _find_latest(folder: Path | str | None, ext: str) -> str | None:
        """Return the stem of the newest DD-MM-YYYY-HH-MM file, or None."""
        if not folder:
            return None
        p = Path(folder)
        if not p.exists():
            return None
        try:
            files = [f for f in os.listdir(p) if f.endswith(ext)]
        except OSError:
            return None
        if not files:
            return None
        # Sort by parsed datetime (filename encodes the timestamp)
        def _key(name):
            try:
                return datetime.strptime(name.split(".")[0], "%d-%m-%Y-%H-%M")
            except ValueError:
                return datetime.min
        return max(files, key=_key).split(".")[0]  # return stem only

    @staticmethod
    def _parse_ts(stem: str) -> datetime | None:
        try:
            return datetime.strptime(stem, "%d-%m-%Y-%H-%M")
        except ValueError:
            return None
