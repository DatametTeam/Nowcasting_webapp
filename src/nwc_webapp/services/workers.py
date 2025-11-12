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
        logger.info("start global cycle thread searching file")
        files = [f for f in os.listdir(folder_path) if f.endswith(".hdf")]
        if not files:
            return None

        files.sort(key=lambda x: datetime.strptime(x.split(".")[0], "%d-%m-%Y-%H-%M"), reverse=True)

        print(f"Input file found: {files[0]}")

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
        logger.info(f"{datetime.now()}: Waiting for {wait_time:.2f} seconds until the next interval...")
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


def worker_thread(event, latest_file, models_list=None):
    """
    Worker thread that submits prediction jobs and waits for completion.
    Uses config-based paths instead of hardcoded paths.

    Args:
        event: Threading event to signal completion
        latest_file: Name of the latest input file
        models_list: Optional list of models (currently unused, defaults to ED_ConvLSTM)
    """
    config = get_config()
    output_dir = config.prediction_output / "real_time_pred"

    thread_id = threading.get_ident()
    logger.info(f"Worker thread (ID: {thread_id}) is starting prediction...")

    model = 'ED_ConvLSTM'

    jobs_ids = []
    new_file = Path(latest_file).stem + '.npy'
    logger.info(f"Looking for prediction file: {new_file}")

    model_output = output_dir / model / new_file

    if not model_output.exists():
        logger.warning(f"File {model_output} does not exist. Starting prediction")
        job_id = start_prediction_job(model, latest_file)
        jobs_ids.append(job_id)
    else:
        logger.warning(f"Prediction already computed! {model_output} exists for {model}.")

    logger.info(f"Waiting for {model_output}")
    while not model_output.exists():
        logger.info("Prediction still going")
        time.sleep(2)

    logger.info(f"Worker thread (ID: {thread_id}) has finished!")
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


def launch_thread_execution(st, latest_file, columns):
    """
    Launch a worker thread for prediction execution.

    Args:
        st: Streamlit module
        latest_file: Latest input file name
        columns: Streamlit columns for UI display
    """
    ctx = get_script_run_ctx()
    runtime = get_instance()

    ctx.session_state.latest_file = latest_file
    logger.info(f"New SRI file available! {latest_file}")

    with columns[1]:
        event = threading.Event()
        logger.info(f"Thread started status: {st.session_state.thread_started}")
        logger.info("Starting thread")

        # Start the worker thread
        thread = threading.Thread(target=worker_thread, args=(event, latest_file))

        st.session_state.thread_started = True
        thread.start()

        status_placeholder = st.empty()
        i = 1
        time_prediction = time.time()
        while not event.is_set():
            i += 1
            time.sleep(1)
        thread.join()

        # Reset
        ctx.session_state["launch_prediction_thread"] = None
        ctx.session_state["computing_model"] = None

        # State update
        ctx.session_state.latest_file = latest_file
        ctx.session_state.selection = None
        ctx.session_state["new_prediction"] = True
        logger.info("launch prediction TERMINATED..")

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