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
from zoneinfo import ZoneInfo

import numpy as np

from nwc_webapp.config.config import get_config
from nwc_webapp.config.environment import is_hpc, is_server

logger = logging.getLogger(__name__)

ROME_TZ = ZoneInfo("Europe/Rome")
UTC_TZ = ZoneInfo("UTC")


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

        Pre-checks prediction availability for the latest SRI so model
        statuses are accurate from the very first status poll.
        """
        with self._lock:
            if self._active:
                return {"ok": False, "reason": "already_running"}

            config = get_config()
            sri_folder = Path(str(config.sri_folder))
            latest_sri = self._find_latest_sri(sri_folder)

            # Set initial SRI info so the frontend sees it immediately
            if latest_sri:
                sri_dt = self._parse_sri_datetime(latest_sri)
                self._latest_sri = latest_sri
                self._latest_sri_timestamp = sri_dt.isoformat() if sri_dt else None

            # Pre-check which models already have predictions for the latest SRI
            for model in config.models:
                if model.upper() == "TEST":
                    self._models[model] = {"status": "ready", "job_id": None}
                elif latest_sri:
                    sri_stem = latest_sri.replace(".hdf", "")
                    pred_file = config.real_time_pred / model / f"{sri_stem}.npy"
                    if pred_file.exists():
                        self._models[model] = {"status": "ready", "job_id": None}
                    else:
                        self._models[model] = {"status": "queued", "job_id": None}
                else:
                    self._models[model] = {"status": "idle", "job_id": None}

            self._active = True
            self._stop_event.clear()
            self._notification = ""

        # Spawn the appropriate loop as a daemon thread
        target = self._hpc_loop if (is_hpc() or is_server()) else self._local_loop
        self._thread = threading.Thread(target=target, daemon=True, name="realtime-loop")
        self._thread.start()
        mode = "hpc" if is_hpc() else ("server" if is_server() else "local")
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
                "mode": "hpc" if is_hpc() else ("server" if is_server() else "local"),
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
        Real prediction loop — polls the SRI folder every 1 s continuously.
        Used for both HPC (PBS jobs) and server mode (direct subprocess).

        SRI files arrive every 5 minutes with a ~6-min server delay, so the
        file timestamped HH:25 typically lands around HH:31:30.  1-second
        polling is cheap (a single listdir) and gets us the file within 1 s
        of arrival.

        If the previous boundary's file still hasn't appeared 3 minutes into
        the current boundary, a warning notification is set (once per boundary).

        On new data:
          1. Check each model — skip if prediction .npy already exists
          2. Submit jobs (PBS on HPC, subprocess on server)
          3. Monitor job statuses until all resolve
          4. Resume polling for the next arrival
        """
        if is_server():
            from nwc_webapp.hpc.jobs import start_realtime_prediction_server as _submit
        else:
            from nwc_webapp.hpc.pbs import start_prediction_job as _submit

        config = get_config()
        sri_folder = Path(str(config.sri_folder))

        # Initialize last_seen_file to None so the first iteration processes
        # the existing latest SRI (checks predictions, submits jobs if needed).
        # start() already set self._latest_sri for display.
        last_seen_file = None
        # Boundary we've already warned about, so we warn at most once per cycle.
        last_warning_boundary = None

        logger.info("HPC loop started. Will process existing SRI on first iteration.")

        POLL_INTERVAL = 1    # always poll every 1s — a listdir is cheap
        DATA_TIMEOUT  = 180  # seconds past boundary before warning (3 min)

        while not self._stop_event.is_set():
            now = datetime.now()
            seconds_past_boundary = (now.minute % 5) * 60 + now.second
            current_boundary = now.replace(
                minute=(now.minute // 5) * 5, second=0, microsecond=0
            )

            # --- Check for new SRI file ---
            latest = self._find_latest_sri(sri_folder)

            if latest and latest != last_seen_file:
                last_seen_file = latest
                logger.info("New SRI detected: %s", latest)

                sri_dt = self._parse_sri_datetime(latest)
                with self._lock:
                    self._latest_sri = latest
                    self._latest_sri_timestamp = sri_dt.isoformat() if sri_dt else None
                    self._notification = f"New data found! {self._format_display(latest)}"

                # --- Submit PBS jobs for models that need predictions ---
                sri_stem = latest.replace(".hdf", "")
                for model in config.models:
                    # Skip Test model — it uses pre-existing static data, always ready
                    if model.upper() == "TEST":
                        with self._lock:
                            self._models[model] = {"status": "ready", "job_id": None}
                        continue

                    pred_file = config.real_time_pred / model / f"{sri_stem}.npy"
                    if pred_file.exists():
                        logger.info("Prediction already exists for %s/%s, skipping", model, sri_stem)
                        with self._lock:
                            self._models[model] = {"status": "ready", "job_id": None}
                        continue

                    with self._lock:
                        self._models[model] = {"status": "queued", "job_id": None}
                    try:
                        job_id = _submit(model, latest)
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

                # Monitor all jobs until resolved
                self._monitor_hpc_jobs(config.models)

                # Re-evaluate timing from the top of the loop after monitoring
                continue

            # --- No new data this tick: warn if the previous boundary's file
            # hasn't arrived 3 minutes into the current boundary.  With a
            # ~6-min server delay the HH:25 file lands at ~HH:31:30, so it
            # should always be there well before HH:33.  Only warn if we
            # truly haven't seen it yet (once per boundary).
            if seconds_past_boundary >= DATA_TIMEOUT:
                previous_boundary = current_boundary - timedelta(minutes=5)
                last_seen_dt = (
                    self._parse_sri_datetime(last_seen_file) if last_seen_file else None
                )
                missing_previous = last_seen_dt is None or last_seen_dt < previous_boundary
                if missing_previous and last_warning_boundary != current_boundary:
                    last_warning_boundary = current_boundary
                    with self._lock:
                        self._notification = (
                            f"Warning: no new data since "
                            f"{previous_boundary.strftime('%H:%M')} "
                            f"({seconds_past_boundary}s overdue)"
                        )
                    logger.warning(
                        "No SRI file newer than %s by %ds past %s boundary",
                        previous_boundary.strftime("%H:%M"),
                        seconds_past_boundary,
                        current_boundary.strftime("%H:%M"),
                    )

            if self._stop_event.wait(timeout=POLL_INTERVAL):
                break

    def _monitor_hpc_jobs(self, models):
        """
        Poll job statuses every 5s until all models are resolved (ready/failed)
        or 30 minutes have passed.

        Handles both PBS job IDs (HPC) and server_{pid} job IDs (server mode).
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
                    if job_id and job_id.startswith("server_"):
                        from nwc_webapp.hpc.jobs import get_server_process_status
                        status = get_server_process_status(job_id)
                    else:
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

            # Wait 1s between status checks
            if self._stop_event.wait(timeout=1):
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

        SAFETY: This loop writes mock data to disk. It must NEVER run
        if the SRI folder points to real HPC data.
        """
        from nwc_webapp.mock.generator import (
            create_mock_hdf_file,
            generate_temporal_sequence,
        )

        config = get_config()

        # SAFETY CHECK: refuse to write mock data to HPC production paths
        sri_folder = Path(str(config.sri_folder))
        if str(sri_folder).startswith("/davinci"):
            logger.error(
                "SAFETY: _local_loop refused to run — SRI folder points to "
                "HPC production path: %s", sri_folder
            )
            with self._lock:
                self._active = False
            return

        while not self._stop_event.is_set():
            # --- Generate mock data ---
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

                # Test model uses a static predictions.npy (24 frames: 0-11 GT, 12-23 preds)
                if model_name.upper() == "TEST":
                    static_path = pred_folder / "predictions.npy"
                    if not static_path.exists():
                        test_data = generate_temporal_sequence(
                            num_timesteps=24,
                            shape=(1400, 1200),
                            base_seed=42,
                        )
                        np.save(static_path, test_data)
                        logger.info("Created static predictions.npy for Test model")
                    continue

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
            # Test model is always "ready" (static data), others start "queued"
            with self._lock:
                self._latest_sri = sri_filename
                self._latest_sri_timestamp = next_dt.isoformat()
                rome_dt = next_dt.replace(tzinfo=UTC_TZ).astimezone(ROME_TZ)
                self._notification = f"New data found! {rome_dt.strftime('%d/%m/%Y %H:%M')}"
                for model in config.models:
                    if model.upper() == "TEST":
                        self._models[model] = {"status": "ready", "job_id": None}
                    else:
                        self._models[model] = {"status": "queued", "job_id": None}

            # --- 5s: all models → "computing" (skip Test) ---
            if self._stop_event.wait(timeout=5):
                break
            with self._lock:
                for model in config.models:
                    if model.upper() == "TEST":
                        continue
                    if self._models[model]["status"] == "queued":
                        self._models[model]["status"] = "computing"

            # --- 10s later (15s total): each model → "ready" or "failed" (skip Test) ---
            if self._stop_event.wait(timeout=10):
                break
            with self._lock:
                for model in config.models:
                    if model.upper() == "TEST":
                        continue
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
        """Format SRI filename for display, converting UTC → Europe/Rome."""
        name = filename.replace(".hdf", "")
        try:
            utc_dt = datetime.strptime(name, "%d-%m-%Y-%H-%M").replace(tzinfo=UTC_TZ)
            rome_dt = utc_dt.astimezone(ROME_TZ)
            return rome_dt.strftime("%d/%m/%Y %H:%M")
        except ValueError:
            return filename

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
