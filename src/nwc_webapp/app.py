"""
Main entry point for the weather nowcasting Streamlit application.
"""

from datetime import datetime

import streamlit as st

from nwc_webapp.core.workers import monitor_time
from nwc_webapp.config.config import get_config
from nwc_webapp.logging_config import setup_logger
from nwc_webapp.pages.csi_analysis import show_csi_analysis_page
from nwc_webapp.pages.home import show_home_page
from nwc_webapp.pages.model_comparison import show_model_comparison_page
from nwc_webapp.pages.nowcasting import main_page
from nwc_webapp.pages.prediction_by_date import show_prediction_page
from nwc_webapp.pages.real_time import show_real_time_prediction
from nwc_webapp.ui.components import configure_sidebar

# Set up logger
logger = setup_logger(__name__)

# Configure Streamlit page
st.set_page_config(page_title="Weather prediction", page_icon=":flag-eu:", layout="wide")


def main(app_config, sri_folder, count_value):
    """
    Main application entry point.

    Args:
        app_config: Application configuration object
        sri_folder: Path to SRI folder
        count_value: Auto-refresh count value
    """
    # Initialize autorefresh thread (moved from module level)
    if "autorefresh_thread_started" not in st.session_state:
        st.session_state["autorefresh_thread_started"] = False

    if not st.session_state["autorefresh_thread_started"]:
        st.session_state["autorefresh_thread_started"] = True

    # Start mock realtime service if running locally (moved from module level)
    from nwc_webapp.config.environment import is_local

    if is_local() and "mock_service_started" not in st.session_state:
        import logging

        from nwc_webapp.mock.realtime import start_mock_service

        logging.basicConfig(level=logging.INFO)

        # Start mock service with 60-second check intervals
        start_mock_service(interval_seconds=60, generate_history=True)
        st.session_state.mock_service_started = True
        logger.info("🎭 Mock realtime service started (local mode)")

    model_list = app_config.models

    # Filter out "Test" model for all tabs except Real-time Prediction
    model_list_no_test = [m for m in model_list if m.upper() != "TEST"]

    sidebar_args = configure_sidebar(model_list_no_test)

    if sidebar_args["submitted"] and "prediction_result" in st.session_state:
        st.session_state.prediction_result = {}

    # Create tabs using st.tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Real Time Prediction",
        "Nowcasting",
        "Prediction by Date & Time",
        "Model Comparison",
        "Metrics Analysis"
    ])

    with tab1:
        st.session_state["sync_end"] = 1
        # Real-time tab uses full model list (includes Test)
        show_real_time_prediction(model_list, sri_folder, count_value)

    with tab2:
        # Nowcasting uses sidebar_args which already has filtered models
        main_page(sidebar_args, sri_folder)

    with tab3:
        # Prediction by Date uses filtered list (no Test)
        show_prediction_page(model_list_no_test)

    with tab4:
        # Model Comparison uses filtered list (no Test)
        show_model_comparison_page(model_list_no_test)

    with tab5:
        # Metrics Analysis uses filtered list (no Test) - also filters internally
        show_csi_analysis_page(model_list_no_test)


# Get configuration (safe to call at module level - no session state access)
app_config = get_config()
SRI_FOLDER_DIR = str(app_config.sri_folder)
COUNT = None  # Auto-refresh interval

# Call main function (all session state access happens inside function)
main(app_config, SRI_FOLDER_DIR, COUNT)
