"""
Main entry point for the weather nowcasting Streamlit application.
"""
import streamlit as st
from datetime import datetime

from layouts import configure_sidebar, show_metrics_page
from nwc_webapp.config import get_config
from nwc_webapp.page_modules.nowcasting import main_page
from nwc_webapp.page_modules.prediction_by_date import show_prediction_page
from nwc_webapp.page_modules.home import show_home_page
from nwc_webapp.page_modules.real_time import show_real_time_prediction
from nwc_webapp.threading.workers import monitor_time

# Configure Streamlit page
st.set_page_config(page_title="Weather prediction", page_icon=":flag-eu:", layout="wide")


def main(app_config):
    """
    Main application entry point.

    Args:
        app_config: Application configuration object
    """
    model_list = app_config.models
    sidebar_args = configure_sidebar(model_list)

    if sidebar_args['submitted'] and 'prediction_result' in st.session_state:
        st.session_state.prediction_result = {}

    # Create tabs using st.tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Real Time Prediction", "Nowcasting", "Prediction by Date & Time", "Metrics"])

    with tab1:
        st.session_state["sync_end"] = 1
        show_real_time_prediction(model_list, SRI_FOLDER_DIR, COUNT)

    with tab2:
        main_page(sidebar_args, SRI_FOLDER_DIR)

    with tab3:
        show_prediction_page(model_list)

    with tab4:
        show_metrics_page(model_list)


# Initial auto-refresh interval (in seconds)
COUNT = None

# Initialize the thread only once using session state
if "autorefresh_thread_started" not in st.session_state:
    st.session_state["autorefresh_thread_started"] = False

if not st.session_state["autorefresh_thread_started"]:
    # thread = threading.Thread(target=monitor_time, daemon=True)
    # thread.start()
    st.session_state["autorefresh_thread_started"] = True

# Get SRI folder from config (environment-aware: HPC or local)
app_config = get_config()
SRI_FOLDER_DIR = str(app_config.sri_folder)
model_list = app_config.models

# Start mock realtime service if running locally
from nwc_webapp.environment import is_local
if is_local() and "mock_service_started" not in st.session_state:
    from nwc_webapp.services.mock_realtime_service import start_mock_service
    import logging
    logging.basicConfig(level=logging.INFO)

    # Start mock service with 60-second check intervals (files created at 5-min intervals)
    start_mock_service(interval_seconds=60, generate_history=True)
    st.session_state.mock_service_started = True
    print("🎭 Mock realtime service started (local mode)")

if __name__ == "__main__":
    print(f"***NEWRUN @ {datetime.now()}***")
    # print(st.session_state)

    main(app_config)