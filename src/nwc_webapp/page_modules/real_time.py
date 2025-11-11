"""
Real-time prediction page with live updates.
"""
import os
import time
import threading
import streamlit as st
from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx, add_script_run_ctx

from nwc_webapp.ui.state import initial_state_management
from nwc_webapp.ui.maps import create_only_map
from nwc_webapp.ui.spinners import (
    background_checker_spinner,
    background_prediction_calculator_spinner,
    background_prediction_loader_spinner
)
from nwc_webapp.threading.workers import load_prediction
from nwc_webapp.utils import get_latest_file, launch_thread_execution


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

    model_options = model_list
    time_options = ["+5min", "+10min", "+15min", "+20min", "+25min",
                    "+30min", "+35min", "+40min", "+45min", "+50min",
                    "+55min", "+60min"]

    with (columns[0]):
        internal_columns = st.columns([0.3, 0.1, 0.3])
        with internal_columns[0]:
            # Select model, bound to session state
            st.selectbox(
                "Select a model",
                options=model_options,
                key="selected_model"
            )

        with internal_columns[2]:
            # Select time, bound to session state
            st.selectbox(
                "Select a prediction time",
                options=time_options,
                key="selected_time",
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

        if st.session_state.selected_model and st.session_state.selected_time:
            # carico di default l'ultima previsione disponibile solo la prima volta
            if "first_prediction_visualization" not in st.session_state:
                st.session_state["first_prediction_visualization"] = True
                if latest_file is not None:
                    pass
                load_prediction(time_options, latest_file, 3)

            # se st.session_state["new_prediction"] == True allora posso fare il caricamente di una nuova previsione
            if "new_prediction" in st.session_state and st.session_state["new_prediction"] or \
                    (st.session_state['previous_time'] != st.session_state['selected_time'] or
                     st.session_state['previous_model'] != st.session_state['selected_model']):
                st.session_state.previous_time = st.session_state.selected_time
                st.session_state.previous_model = st.session_state.selected_model

                if "prediction_data_thread" not in st.session_state:
                    st.session_state["prediction_data_thread"] = None

                if "load_prediction_thread" in st.session_state:
                    if st.session_state["load_prediction_thread"] is False:
                        load_prediction(time_options, latest_file, 1)
                else:
                    print("LATEST FILE")
                    print(str(latest_file))
                    load_prediction(time_options, latest_file, 2)

                if "prediction_data_thread" in st.session_state:
                    rgba_img = st.session_state["prediction_data_thread"]
                    if rgba_img is not None:
                        st.session_state['display_prediction'] = True
                        with st.spinner("Loading **DATA**..", show_time=True):
                            create_only_map(rgba_img, prediction=True)
                    else:
                        create_only_map(None)
                else:
                    create_only_map(None)
            else:
                # se st.session_state["new_prediction"] == False allora posso semplicemente applicare la predizione
                # alla mappa
                if "prediction_data_thread" in st.session_state:
                    rgba_img = st.session_state["prediction_data_thread"]
                    if rgba_img is not None:
                        with st.spinner("Loading **DATA**..", show_time=True):
                            create_only_map(rgba_img, prediction=True)
                    else:
                        create_only_map(None)
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