"""
RealtimeService — Backend singleton that drives the real-time prediction cycle.

HOW THIS WORKS:
===============
This is a thread-safe singleton that manages a background daemon thread.
When started, it continuously:
  - HPC mode: polls SRI folder for new radar files → submits PBS jobs → monitors them
  - Local mode: generates mock data → simulates queued/computing/ready transitions

Any number of browser tabs can poll GET /api/realtime/status and see the same state.
The cycle survives browser closes — only an explicit POST /api/realtime/stop kills it.

KEY PATTERNS:
- threading.Event.wait(timeout) instead of time.sleep() → instant clean shutdown
- threading.Lock protects all state reads/writes
- copy.deepcopy on get_state() prevents mutation during JSON serialization
- Daemon thread so it dies automatically when uvicorn exits
"""

import copy
import logging
import os
import random
import threading
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from nwc_webapp.config.config import get_config
from nwc_webapp.config.environment import is_hpc

logger = logging.getLogger(__name__)


class RealtimeService:
    """
    Singleton service that manages the real-time prediction background loop.

    Usage:
        service = RealtimeService()   # always returns the same instance
        service.start()               # spawns background thread
        state = service.get_state()   # read current state (thread-safe)
        service.stop()                # signals thread to exit cleanly
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

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None

        # State that get_state() returns
        self._active = False
        self._latest_sri = None
        self._latest_sri_timestamp = None
        self._notification = ""
        self._models = {}  # { model_name: {"status": "idle", "job_id": None} }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        """
        Start the real-time prediction loop.
        Returns dict with 'ok' and optionally 'reason'.
        """
        with self._lock:
            if self._active:
                return {"ok": False, "reason": "already_running"}

            config = get_config()
            for model in config.models:
                self._models[model] = {"status": "idle", "job_id": None}

            self._active = True
            self._stop_event.clear()
            self._notification = ""

        # Spawn the appropriate loop as a daemon thread
        target = self._hpc_loop if is_hpc() else self._local_loop
        self._thread = threading.Thread(target=target, daemon=True, name="realtime-loop")
        self._thread.start()
        mode = "hpc" if is_hpc() else "local"
        logger.info("RealtimeService started (%s mode)", mode)

        return {"ok": True, "mode": mode}

    def stop(self):
        """
        Signal the background thread to stop and wait for it to exit.
        Returns dict with 'ok'.
        """
        with self._lock:
            if not self._active:
                return {"ok": False, "reason": "not_running"}
            self._active = False

        # Signal the thread to wake up and exit
        self._stop_event.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

        # Reset state
        with self._lock:
            self._latest_sri = None
            self._latest_sri_timestamp = None
            self._notification = ""
            for model in self._models:
                self._models[model] = {"status": "idle", "job_id": None}

        logger.info("RealtimeService stopped")
        return {"ok": True}

    def get_state(self):
        """
        Return a deep copy of the current state (thread-safe).
        Called by GET /api/realtime/status.
        """
        with self._lock:
            return copy.deepcopy({
                "active": self._active,
                "mode": "hpc" if is_hpc() else "local",
                "latest_sri": self._latest_sri,
                "latest_sri_timestamp": self._latest_sri_timestamp,
                "notification": self._notification,
                "models": self._models,
            })

    # ------------------------------------------------------------------
    # HPC loop — real PBS jobs
    # ------------------------------------------------------------------

    def _hpc_loop(self):
        """
        Real HPC loop:
        1. Poll SRI folder every 30s for a new .hdf file
        2. On new file → submit PBS jobs for all models
        3. Poll job statuses every 5s until all resolve
        4. Loop back to step 1
        """
        from nwc_webapp.hpc.pbs import start_prediction_job

        config = get_config()
        sri_folder = Path(str(config.sri_folder))
        last_seen_file = self._find_latest_sri(sri_folder)

        logger.info("HPC loop started. Last known SRI: %s", last_seen_file)

        while not self._stop_event.is_set():
            # Step 1: Poll for new SRI file
            latest = self._find_latest_sri(sri_folder)

            if latest and latest != last_seen_file:
                last_seen_file = latest
                logger.info("New SRI detected: %s", latest)

                # Update state
                sri_dt = self._parse_sri_datetime(latest)
                with self._lock:
                    self._latest_sri = latest
                    self._latest_sri_timestamp = sri_dt.isoformat() if sri_dt else None
                    self._notification = f"New data found! {self._format_display(latest)}"

                # Step 2: Submit PBS jobs for all models
                for model in config.models:
                    with self._lock:
                        self._models[model] = {"status": "queued", "job_id": None}
                    try:
                        job_id = start_prediction_job(model, latest)
                        with self._lock:
                            self._models[model]["job_id"] = job_id
                        if job_id is None:
                            with self._lock:
                                self._models[model]["status"] = "failed"
                            logger.error("Failed to submit job for %s", model)
                    except Exception as e:
                        with self._lock:
                            self._models[model]["status"] = "failed"
                        logger.error("Exception submitting job for %s: %s", model, e)

                # Step 3: Monitor all jobs until resolved
                self._monitor_hpc_jobs(config.models)

            # Wait 30s before checking for next new file (or exit instantly on stop)
            if self._stop_event.wait(timeout=30):
                break

    def _monitor_hpc_jobs(self, models):
        """
        Poll job statuses every 5s until all models are resolved (ready/failed)
        or 30 minutes have passed.

        Uses get_job_status(job_id) directly instead of get_model_job_status(model),
        because the latter tries to read streamlit.session_state which doesn't
        exist in the FastAPI context.
        """
        from nwc_webapp.hpc.pbs import get_job_status

        config = get_config()
        start_time = datetime.now()
        timeout = timedelta(minutes=30)

        while not self._stop_event.is_set():
            all_resolved = True

            for model in models:
                with self._lock:
                    current = self._models[model]["status"]
                if current in ("ready", "failed"):
                    continue

                all_resolved = False
                job_id = None
                with self._lock:
                    job_id = self._models[model].get("job_id")

                if not job_id:
                    continue

                try:
                    status = get_job_status(job_id)
                except Exception:
                    status = "ended"

                with self._lock:
                    if status == "Q":
                        self._models[model]["status"] = "queued"
                    elif status == "R":
                        self._models[model]["status"] = "computing"
                    elif status == "ended":
                        # Job left the queue — check if prediction file exists
                        pred_folder = config.real_time_pred / model
                        sri_name = self._latest_sri.replace(".hdf", "") if self._latest_sri else ""
                        pred_file = pred_folder / f"{sri_name}.npy"
                        if pred_file.exists():
                            self._models[model]["status"] = "ready"
                        else:
                            self._models[model]["status"] = "failed"

            if all_resolved:
                logger.info("All HPC jobs resolved")
                break

            # Timeout check
            if datetime.now() - start_time > timeout:
                logger.warning("HPC job monitoring timed out after 30 minutes")
                with self._lock:
                    for model in models:
                        if self._models[model]["status"] not in ("ready", "failed"):
                            self._models[model]["status"] = "failed"
                break

            # Wait 5s between status checks
            if self._stop_event.wait(timeout=5):
                break

    # ------------------------------------------------------------------
    # Local loop — mock simulation
    # ------------------------------------------------------------------

    def _local_loop(self):
        """
        Mock loop that simulates real-time prediction cycles locally.
        Timing mirrors what the frontend simulation used to do:
          0s  → generate mock data, all models "queued"
          5s  → all models "computing"
         15s  → each model → "ready" (80%) or "failed" (20%)
         45s  → next cycle
        """
        from nwc_webapp.mock.generator import (
            create_mock_hdf_file,
            generate_temporal_sequence,
        )

        config = get_config()

        while not self._stop_event.is_set():
            # --- Generate mock data ---
            sri_folder = Path(str(config.sri_folder))
            sri_folder.mkdir(parents=True, exist_ok=True)

            next_dt = self._compute_next_mock_timestamp(sri_folder)

            # Create SRI file for current timestamp
            sri_filename = next_dt.strftime("%d-%m-%Y-%H-%M") + ".hdf"
            create_mock_hdf_file(sri_folder / sri_filename, next_dt)
            logger.info("Mock SRI created: %s", sri_filename)

            # Also create past SRI files so the groundtruth timeline has data.
            # The map shows -60 to 0 min (13 frames at 5-min intervals).
            for offset_min in range(5, 65, 5):  # -5, -10, ... -60
                past_dt = next_dt - timedelta(minutes=offset_min)
                past_filename = past_dt.strftime("%d-%m-%Y-%H-%M") + ".hdf"
                past_path = sri_folder / past_filename
                if not past_path.exists():
                    create_mock_hdf_file(past_path, past_dt)

            # Create prediction files for each model
            for model_name in config.models:
                pred_folder = config.real_time_pred / model_name
                pred_folder.mkdir(parents=True, exist_ok=True)

                pred_filename = next_dt.strftime("%d-%m-%Y-%H-%M") + ".npy"
                pred_path = pred_folder / pred_filename

                if not pred_path.exists():
                    prediction = generate_temporal_sequence(
                        num_timesteps=12,
                        shape=(1400, 1200),
                        base_seed=int(next_dt.timestamp()),
                    )
                    if model_name == "ED_ConvLSTM":
                        prediction = np.expand_dims(prediction, axis=0)
                    np.save(pred_path, prediction)

            # Update state: new data found
            with self._lock:
                self._latest_sri = sri_filename
                self._latest_sri_timestamp = next_dt.isoformat()
                self._notification = f"New data found! {next_dt.strftime('%d/%m/%Y %H:%M')}"
                for model in config.models:
                    self._models[model] = {"status": "queued", "job_id": None}

            # --- 5s: all models → "computing" ---
            if self._stop_event.wait(timeout=5):
                break
            with self._lock:
                for model in config.models:
                    if self._models[model]["status"] == "queued":
                        self._models[model]["status"] = "computing"

            # --- 10s later (15s total): each model → "ready" or "failed" ---
            if self._stop_event.wait(timeout=10):
                break
            with self._lock:
                for model in config.models:
                    if self._models[model]["status"] == "computing":
                        self._models[model]["status"] = (
                            "ready" if random.random() < 0.8 else "failed"
                        )
                # Clear the notification now that results are in
                self._notification = ""

            # --- 30s later (45s total): next cycle ---
            if self._stop_event.wait(timeout=30):
                break

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_latest_sri(sri_folder: Path) -> str | None:
        """Find the latest .hdf file in the SRI folder by timestamp in filename."""
        if not sri_folder.exists():
            return None
        hdf_files = [f for f in os.listdir(sri_folder) if f.endswith(".hdf")]
        if not hdf_files:
            return None
        hdf_files.sort(
            key=lambda x: datetime.strptime(x.split(".")[0], "%d-%m-%Y-%H-%M"),
            reverse=True,
        )
        return hdf_files[0]

    @staticmethod
    def _parse_sri_datetime(filename: str) -> datetime | None:
        """Parse a SRI filename like '12-02-2026-15-00.hdf' into a datetime."""
        try:
            name = filename.replace(".hdf", "")
            return datetime.strptime(name, "%d-%m-%Y-%H-%M")
        except ValueError:
            return None

    @staticmethod
    def _format_display(filename: str) -> str:
        """Format SRI filename for display: '12-02-2026-15-00.hdf' → '12/02/2026 15:00'."""
        name = filename.replace(".hdf", "")
        parts = name.split("-")
        if len(parts) != 5:
            return filename
        return f"{parts[0]}/{parts[1]}/{parts[2]} {parts[3]}:{parts[4]}"

    @staticmethod
    def _compute_next_mock_timestamp(sri_folder: Path) -> datetime:
        """Determine the next mock timestamp (latest + 5 min, or now rounded to 5 min)."""
        hdf_files = [f for f in os.listdir(sri_folder) if f.endswith(".hdf")]
        if hdf_files:
            hdf_files.sort(
                key=lambda x: datetime.strptime(x.split(".")[0], "%d-%m-%Y-%H-%M"),
                reverse=True,
            )
            latest_name = hdf_files[0].replace(".hdf", "")
            latest_dt = datetime.strptime(latest_name, "%d-%m-%Y-%H-%M")
            return latest_dt + timedelta(minutes=5)
        else:
            now = datetime.now()
            return now.replace(minute=(now.minute // 5) * 5, second=0, microsecond=0)
