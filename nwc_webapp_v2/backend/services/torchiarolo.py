"""
TorchiaroloService — background thread that polls the Torchiarolo product folders
for new files and broadcasts via WebSocket when one arrives.

Mirrors the pattern of services/cagliari.py.
"""

import logging
import os
import threading

from api.torchiarolo import (
    torchiarolo_ws_manager,
    parse_torchiarolo_filename,
    _get_product_folder,
    _torchiarolo_cfg,
)

logger = logging.getLogger(__name__)


class TorchiaroloService:
    """
    Singleton service that watches the Torchiarolo data folders for new files and
    broadcasts a 'torchiarolo_update' WebSocket message when new data arrives.
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
            name="torchiarolo-watcher",
        )
        self._thread.start()
        logger.info("TorchiaroloService started")

    def stop(self):
        self._active = False
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        logger.info("TorchiaroloService stopped")

    # ------------------------------------------------------------------
    # Background poll loop
    # ------------------------------------------------------------------

    def _poll_loop(self):
        """Poll each Torchiarolo product folder every second for new files."""
        cfg = _torchiarolo_cfg()
        products = list(cfg.get("products", {}).keys()) or ["SRI", "VMI", "VIL", "ETM"]

        POLL_INTERVAL = 1  # seconds

        for product in products:
            latest = self._find_latest(product)
            if latest:
                self._last_seen[product] = latest
                logger.info("Torchiarolo initial latest %s: %s", product, latest)

        while not self._stop_event.is_set():
            for product in products:
                latest = self._find_latest(product)
                if latest and latest != self._last_seen.get(product):
                    self._last_seen[product] = latest
                    dt = parse_torchiarolo_filename(latest)
                    ts_iso = dt.isoformat() if dt else None
                    logger.info("New Torchiarolo file detected: %s / %s", product, latest)
                    torchiarolo_ws_manager.broadcast_sync({
                        "type": "torchiarolo_update",
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
        """Return the filename of the newest Torchiarolo file for the given product."""
        folder = _get_product_folder(product)
        if not folder.exists():
            return None
        files = []
        try:
            for fname in os.listdir(folder):
                dt = parse_torchiarolo_filename(fname)
                if dt is not None:
                    files.append((dt, fname))
        except OSError:
            return None
        if not files:
            return None
        files.sort(key=lambda x: x[0], reverse=True)
        return files[0][1]