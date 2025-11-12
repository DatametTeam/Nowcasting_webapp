"""
Background thread workers for async operations.
"""
import time
import threading
from datetime import datetime
import streamlit as st
from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx, add_script_run_ctx

from nwc_webapp.utils import load_prediction_thread
from nwc_webapp.logging_config import setup_logger

# Set up logger
logger = setup_logger(__name__)

def thread_for_position():
    """
    Background thread to monitor and update map position.
    """
    thread_id = threading.get_ident()
    logger.info(f"Worker thread (ID: {thread_id}) is starting...")
    while True:
        if "st_map" in st.session_state:
            st_map = st.session_state["st_map"]
            if 'center' in st_map.keys() and 'zoom' in st_map.keys():
                # print("THREAD - " + str(st_map['center']) + " -- " + str(st_map['zoom']))
                st.session_state["center"] = st_map['center']
                st.session_state["zoom"] = st_map['zoom']
            else:
                # print("THREAD - center / zoom not available..")
                pass
        else:
            # print("THREAD - st_map not available..")
            pass
        time.sleep(0.4)


def load_prediction(time_options, latest_file, prediction_num):
    """
    Start a background thread to load prediction data.

    Args:
        time_options: List of available time options
        latest_file: Path to latest file
        prediction_num: Prediction number identifier
    """
    ctx = get_script_run_ctx()
    load_pred_thread = threading.Thread(target=load_prediction_thread,
                                        args=(st, time_options, latest_file), daemon=True)
    add_script_run_ctx(load_pred_thread, ctx)
    logger.info("LOAD PREDICTION -- " + str(prediction_num) + " --..")
    st.session_state['load_prediction_thread'] = True
    load_pred_thread.start()


def monitor_time():
    """
    Monitor time and trigger app rerun at specific intervals.
    Checks if current minute is a multiple of 5 to force new predictions.
    """
    logger.debug("Starting Monitor time thread")
    while True:
        now = datetime.now()
        # Check if the current minute is a multiple of 5 and seconds are close to 0
        if now.minute % 5 == 0 and now.second < 5:
            logger.info(f"Time is {now}! Restarting app to force new prediction")
            time.sleep(5)
            st.rerun()

        time.sleep(2)  # Check every second