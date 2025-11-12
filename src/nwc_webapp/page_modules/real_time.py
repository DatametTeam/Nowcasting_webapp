"""
Real-time prediction page with live updates.
"""
import os
import time
import threading
import streamlit as st
from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx, add_script_run_ctx

from nwc_webapp.ui.state import initial_state_management
from nwc_webapp.ui.maps import create_only_map, create_animated_map_html
from nwc_webapp.ui.spinners import (
    background_checker_spinner,
    background_prediction_calculator_spinner,
    background_prediction_loader_spinner
)
from nwc_webapp.background.workers import load_prediction
from nwc_webapp.utils import get_latest_file, launch_thread_execution
from nwc_webapp.data.loaders import load_all_predictions


def show_real_time_prediction(model_list, sri_folder_dir, COUNT=None):
    """
    Display real-time prediction page with live map updates.

    Args:
        model_list: List of available models
        sri_folder_dir: Path to SRI folder directory
        COUNT: Optional count value for auto-refresh
    """
    columns = st.columns([0.5, 0.5])
    st.session_state["sync_end"] = 1

    # Initial state management
    initial_state_management(COUNT)

    # Clear cached predictions on first load to prevent showing stale data
    if "page_initialized" not in st.session_state:
        st.session_state["page_initialized"] = True
        if "all_predictions_data" in st.session_state:
            del st.session_state["all_predictions_data"]
        st.session_state["new_prediction"] = False

    model_options = model_list
    # Include ground truth times (-30 to 0) and prediction times (+5 to +60)
    time_options = ["-30min", "-25min", "-20min", "-15min", "-10min", "-5min", "0min",
                    "+5min", "+10min", "+15min", "+20min", "+25min",
                    "+30min", "+35min", "+40min", "+45min", "+50min",
                    "+55min", "+60min"]

    with (columns[0]):
        # Select model only (time selection removed - animation shows all times)
        st.selectbox(
            "Select a model",
            options=model_options,
            key="selected_model"
        )

        # THREAD per l'ottenimento automatico di nuovi file di input
        print("Sto entrando in get_latest_file_thread")
        st.session_state["run_get_latest_file"] = True
        ctx = get_script_run_ctx()

        if "prev_thread_ID_get_latest_file" in st.session_state:
            thread_ID = st.session_state["prev_thread_ID_get_latest_file"]
            print(f"NEWRUN --> main process {os.getpid()}")
            print(f"NEWRUN --> KILLING {thread_ID} thread")
            terminate_event = st.session_state["terminate_event"]
            terminate_event.set()
            time.sleep(0.5)
            del (st.session_state["prev_thread_ID_get_latest_file"])
            del (st.session_state["terminate_event"])

        # lanciato una volta sola questo thread gira autonomamente
        terminate_event = threading.Event()
        st.session_state["terminate_event"] = terminate_event
        obtain_input_th = threading.Thread(target=get_latest_file, args=(sri_folder_dir, terminate_event), daemon=True)
        add_script_run_ctx(obtain_input_th, ctx)
        obtain_input_th.start()

        # per dare tempo al thread di settare in sessione un nuovo file se esiste
        time.sleep(0.4)
        if "thread_ID_get_latest_file" in st.session_state:
            thread_ID = st.session_state["thread_ID_get_latest_file"]
            print("thread_ID in main --> " + str(thread_ID))
            del (st.session_state["thread_ID_get_latest_file"])
            st.session_state["prev_thread_ID_get_latest_file"] = thread_ID

        st.markdown("<div style='text-align: center; font-size: 18px;'>"
                    f"<b>Current Date: {st.session_state.latest_file}</b>"
                    "</div>",
                    unsafe_allow_html=True)

        latest_file = st.session_state["latest_thread"]
        if latest_file != st.session_state.latest_file:
            # calcolo della previsione in background
            if st.session_state["launch_prediction_thread"] is None:
                print("LAUNCH PREDICTION..")

                st.session_state["launch_prediction_thread"] = True

                ctx = get_script_run_ctx()
                launch_thread = threading.Thread(target=launch_thread_execution, args=(st, latest_file, columns),
                                                 daemon=True)
                add_script_run_ctx(launch_thread, ctx)
                launch_thread.start()
        else:
            print(f"Current SRI == Latest file processed! {latest_file}. Skipped prediction")

        # Load and display animated predictions
        if st.session_state.selected_model:
            # Check if we need to load new predictions (first time or model changed)
            if ("new_prediction" in st.session_state and st.session_state["new_prediction"]) or \
               ("previous_model" in st.session_state and st.session_state['previous_model'] != st.session_state['selected_model']) or \
               "all_predictions_data" not in st.session_state:

                st.session_state.previous_model = st.session_state.selected_model

                # Load all 12 predictions at once
                with st.spinner("Loading all prediction timesteps...", show_time=True):
                    rgba_images_dict = load_all_predictions(st, time_options, latest_file)

                if rgba_images_dict is not None:
                    st.session_state['all_predictions_data'] = rgba_images_dict
                    st.session_state['display_prediction'] = True
                    st.session_state['new_prediction'] = False
                    create_animated_map_html(rgba_images_dict)
                else:
                    # No predictions available yet
                    create_only_map(None)
            else:
                # Display existing predictions
                if "all_predictions_data" in st.session_state and st.session_state['all_predictions_data'] is not None:
                    create_animated_map_html(st.session_state['all_predictions_data'])
                else:
                    create_only_map(None)
        else:
            create_only_map(None)

    # Spinner section
    # --------------------------------------------------------------------------------------------
    if "load_prediction_thread" in st.session_state and st.session_state["load_prediction_thread"]:
        background_prediction_loader_spinner(columns)
    # --------------------------------------------------------------------------------------------

    # ---------------------------------------------
    if st.session_state["launch_prediction_thread"]:
        background_prediction_calculator_spinner(columns)
    # ---------------------------------------------

    # ----------------------------------------
    if st.session_state["run_get_latest_file"]:
        background_checker_spinner(columns)
    # ----------------------------------------