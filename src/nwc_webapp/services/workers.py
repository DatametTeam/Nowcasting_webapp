"""
Background worker threads for real-time prediction monitoring.
Handles file monitoring and prediction job submission.
"""
import os
import time
import threading
from pathlib import Path
from datetime import datetime, timedelta

from streamlit.runtime import get_instance
from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx

from nwc_webapp.config.config import get_config
from nwc_webapp.config.environment import is_hpc
from nwc_webapp.logging_config import setup_logger

# Set up logger
logger = setup_logger(__name__)

# Import environment-aware job submission
if is_hpc():
    from nwc_webapp.services.pbs import start_prediction_job
else:
    from nwc_webapp.services.mock.mock import start_prediction_job


def get_latest_file(folder_path, terminate_event):
    """
    Background thread that monitors for new radar data files.
    Runs continuously and triggers app refresh when new files arrive.

    Args:
        folder_path: Path to SRI folder to monitor
        terminate_event: Threading event to signal termination
    """
    logger.info("AUTO SCAN --> " + str(folder_path))
    ctx = get_script_run_ctx()
    runtime = get_instance()
    thread_id = threading.get_ident()
    ctx.session_state["thread_ID_get_latest_file"] = thread_id
    logger.debug("thread_ID in thread --> " + str(thread_id))

    latest_file = None

    while not terminate_event.is_set():
        logger.debug("Scanning for new SRI files...")
        files = [f for f in os.listdir(folder_path) if f.endswith(".hdf")]
        if not files:
            return None

        files.sort(key=lambda x: datetime.strptime(x.split(".")[0], "%d-%m-%Y-%H-%M"), reverse=True)

        logger.debug(f"Latest SRI file: {files[0]}")

        if files[0] != latest_file:
            ctx.session_state['latest_thread'] = files[0]
            ctx.session_state['new_update'] = True
            latest_file = files[0]

        now = datetime.now()

        # Calculate the next 5-minute interval
        next_minute = (now.minute // 5 + 1) * 5
        if next_minute == 60:  # Handle the hour rollover case
            next_interval = now.replace(hour=(now.hour + 1) % 24, minute=0, second=0, microsecond=0)
        else:
            next_interval = now.replace(minute=next_minute, second=0, microsecond=0)

        wait_time = (next_interval - now).total_seconds()
        logger.debug(f"Waiting {wait_time:.0f}s until next check interval")
        time.sleep(wait_time)

        # Polling cycle for the research of a new input file after the current 5 minutes slot is terminated
        while True:
            files = [f for f in os.listdir(folder_path) if f.endswith(".hdf")]
            if not files:
                time.sleep(1)
                continue

            files.sort(key=lambda x: datetime.strptime(x.split(".")[0], "%d-%m-%Y-%H-%M"), reverse=True)
            new_file = files[0]

            if new_file != latest_file:
                logger.info(f"New file detected: {new_file}")
                latest_file = new_file
                break

            time.sleep(1)

        # Restart the application to force the refresh of the main loop
        logger.info("Rerun main")
        session_info = runtime._session_mgr.get_active_session_info(ctx.session_id)
        time.sleep(0.2)
        session_info.session.request_rerun(None)
    logger.warning("TERMINATE event is_set().")


def worker_thread(event, latest_file, model, ctx):
    """
    Worker thread that submits prediction jobs and waits for completion.
    Uses config-based paths instead of hardcoded paths.

    Args:
        event: Threading event to signal completion
        latest_file: Name of the latest input file
        model: Model name to compute predictions for
        ctx: Streamlit script run context for session state access
    """
    config = get_config()
    output_dir = config.real_time_pred

    thread_id = threading.get_ident()
    logger.info(f"Worker thread (ID: {thread_id}) starting prediction for model: {model}")

    jobs_ids = []

    # Special handling for TEST model - always uses predictions.npy (case-insensitive)
    if model.upper() == "TEST":
        model_output = output_dir / "Test" / "predictions.npy"
        logger.info(f"[{model}] TEST model - checking for {model_output}")

        # TEST model: Check if file exists and exit immediately (no waiting)
        if model_output.exists():
            logger.info(f"[{model}] TEST predictions.npy exists - ready!")
        else:
            logger.warning(f"[{model}] TEST predictions.npy not found - marking as failed")
            if "failed_models" not in ctx.session_state:
                ctx.session_state["failed_models"] = set()
            ctx.session_state["failed_models"].add(model)

        # Remove from computing set and exit immediately
        if model in ctx.session_state["computing_models"]:
            ctx.session_state["computing_models"].remove(model)
        event.set()
        return

    # For non-TEST models: use date-based filename
    new_file = Path(latest_file).stem + '.npy'
    logger.info(f"Looking for prediction file: {new_file}")
    model_output = output_dir / model / new_file

    # STEP 1: Check if prediction already exists
    if model_output.exists():
        logger.info(f"[{model}] Prediction already exists at {model_output}")
        # Remove from computing set immediately
        if model in ctx.session_state["computing_models"]:
            ctx.session_state["computing_models"].remove(model)
        event.set()
        return

    # STEP 2: Submit job
    logger.info(f"[{model}] File does not exist - submitting PBS job")
    job_id = start_prediction_job(model, latest_file)
    if job_id:
        logger.info(f"[{model}] Job submitted: {job_id}")
        ctx.session_state["submitted_models"].add(model)
        # Store job_id for fast status checking
        ctx.session_state[f"job_id_{model}"] = job_id
    else:
        logger.error(f"[{model}] Job submission failed!")

    # STEP 3: Monitor job status and wait for file
    from nwc_webapp.services.pbs import get_model_job_status, is_pbs_available
    last_status = None
    check_count = 0
    max_wait_iterations = 1800  # 1 hour max (1800 * 2 seconds)

    logger.debug(f"[{model}] Monitoring for output file: {model_output.name}")

    while not model_output.exists() and check_count < max_wait_iterations:
        check_count += 1

        # Check job status every 3 iterations (6 seconds) for non-TEST models
        if model.upper() != "TEST" and is_pbs_available() and check_count % 3 == 0:
            current_status = get_model_job_status(model)

            # Status changed - no need to trigger rerun, fragment handles UI updates
            if current_status != last_status:
                logger.info(f"[{model}] Job status changed: {last_status} -> {current_status}")
                last_status = current_status
                # Fragment updates independently every 2s - no manual rerun needed!

            # Job disappeared from queue - check if prediction exists
            if last_status and not current_status:
                logger.info(f"[{model}] Job no longer in queue - polling for output file...")

                # Poll for up to 10 seconds to detect failures faster
                max_polls = 5  # 5 polls * 2 seconds = 10 seconds
                for poll_attempt in range(max_polls):
                    time.sleep(2)

                    if model_output.exists():
                        logger.info(f"[{model}] SUCCESS - Prediction file found after job completion!")
                        # Fragment will pick up the change automatically - no rerun needed!
                        # Exit the wait loop - file is ready
                        break

                    logger.debug(f"[{model}] Poll {poll_attempt + 1}/{max_polls}: File not found yet...")

                # Final check after polling
                if not model_output.exists():
                    logger.warning(f"[{model}] Job ended but no prediction file after 10s - marking as failed")
                    # Add to failed models set
                    if "failed_models" not in ctx.session_state:
                        ctx.session_state["failed_models"] = set()
                    ctx.session_state["failed_models"].add(model)

                # Exit the main wait loop since job is done
                break

        # Check if job never appeared in queue after 30 seconds - likely failed at submission
        if check_count > 15 and last_status is None and not model_output.exists():
            logger.warning(f"[{model}] Job never appeared in queue after 30s - marking as failed")
            if "failed_models" not in ctx.session_state:
                ctx.session_state["failed_models"] = set()
            ctx.session_state["failed_models"].add(model)
            # Fragment will show failed status automatically - no rerun needed!
            break

        # Log progress every 60 seconds (reduced from 30s to reduce noise)
        if check_count % 30 == 0:
            logger.info(f"[{model}] Still waiting for output... ({check_count * 2}s elapsed)")

        time.sleep(2)

    # STEP 4: Final check and cleanup
    if model_output.exists():
        logger.info(f"[{model}] SUCCESS - Prediction file ready!")
    else:
        logger.error(f"[{model}] TIMEOUT or FAILED - No prediction after {check_count * 2}s")

    # Remove model from computing set
    if model in ctx.session_state["computing_models"]:
        ctx.session_state["computing_models"].remove(model)
        logger.info(f"[{model}] Removed from computing_models set")

    event.set()  # Signal that the worker thread is done


def worker_thread_test(event):
    """
    Test worker thread that simulates prediction time.

    Args:
        event: Threading event to signal completion
    """
    thread_id = threading.get_ident()
    logger.info(f"Worker thread (ID: {thread_id}) is starting prediction...")

    time.sleep(10)

    logger.info(f"Worker thread (ID: {thread_id}) has finished!")
    event.set()


def launch_thread_execution(st, latest_file, columns, model_list):
    """
    Launch worker threads for prediction execution for all models.

    Args:
        st: Streamlit module
        latest_file: Latest input file name
        columns: Streamlit columns for UI display
        model_list: List of models to compute predictions for
    """
    ctx = get_script_run_ctx()
    runtime = get_instance()

    ctx.session_state.latest_file = latest_file
    logger.info(f"New SRI file available! {latest_file}")
    logger.info(f"Launching predictions for models: {model_list}")

    # Add all models to computing set
    for model in model_list:
        ctx.session_state["computing_models"].add(model)

    # Create events and threads for each model
    threads = []
    events = []

    for model in model_list:
        event = threading.Event()
        events.append(event)

        # Start worker thread for this model
        thread = threading.Thread(target=worker_thread, args=(event, latest_file, model, ctx), daemon=True)
        from streamlit.runtime.scriptrunner_utils.script_run_context import add_script_run_ctx
        add_script_run_ctx(thread, ctx)
        threads.append(thread)
        thread.start()
        logger.debug(f"Started worker thread for model: {model}")

    st.session_state.thread_started = True

    # Wait for all threads to complete
    logger.info(f"Monitoring {len(threads)} prediction jobs...")
    for thread, model in zip(threads, model_list):
        thread.join()
        logger.debug(f"Worker thread completed for: {model}")

    # Reset
    ctx.session_state["launch_prediction_thread"] = None

    # State update
    ctx.session_state.latest_file = latest_file
    ctx.session_state.selection = None
    ctx.session_state["new_prediction"] = True
    logger.info("All prediction threads TERMINATED..")

    session_info = runtime._session_mgr.get_active_session_info(ctx.session_id)
    session_info.session.request_rerun(None)


def load_prediction_thread(st, time_options, latest_file):
    """
    Load prediction data in a background thread.

    Args:
        st: Streamlit module
        time_options: List of time options
        latest_file: Latest input file name
    """
    from nwc_webapp.data.loaders import load_prediction_data

    ctx = get_script_run_ctx()
    runtime = get_instance()

    rgba_img = load_prediction_data(st, time_options, latest_file)

    ctx.session_state['prediction_data_thread'] = rgba_img
    ctx.session_state['new_prediction'] = False
    ctx.session_state['load_prediction_thread'] = False
    ctx.session_state['display_prediction'] = True

    logger.info("load prediction TERMINATED..")

    session_info = runtime._session_mgr.get_active_session_info(ctx.session_id)
    session_info.session.request_rerun(None)