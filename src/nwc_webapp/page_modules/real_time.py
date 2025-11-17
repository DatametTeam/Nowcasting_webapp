"""
Real-time prediction page with live updates.
"""
import os
import time
import threading
import streamlit as st
from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx, add_script_run_ctx

from pathlib import Path
from nwc_webapp.ui.state import initial_state_management
from nwc_webapp.ui.maps import create_only_map, create_animated_map_html
from nwc_webapp.utils import get_latest_file, launch_thread_execution
from nwc_webapp.data.loaders import load_all_predictions
from nwc_webapp.logging_config import setup_logger

# Set up logger
logger = setup_logger(__name__)


def show_real_time_prediction(model_list, sri_folder_dir, COUNT=None):
    """
    Display real-time prediction page with live map updates.

    Args:
        model_list: List of available models
        sri_folder_dir: Path to SRI folder directory
        COUNT: Optional count value for auto-refresh
    """
    columns = st.columns([0.7, 0.3])  # 70% map, 30% status panel
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
        logger.debug("Entering get_latest_file_thread")
        st.session_state["run_get_latest_file"] = True
        ctx = get_script_run_ctx()

        if "prev_thread_ID_get_latest_file" in st.session_state:
            thread_ID = st.session_state["prev_thread_ID_get_latest_file"]
            logger.debug(f"NEWRUN --> main process {os.getpid()}")
            logger.debug(f"NEWRUN --> KILLING {thread_ID} thread")
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
            logger.debug(f"thread_ID in main --> {thread_ID}")
            del (st.session_state["thread_ID_get_latest_file"])
            st.session_state["prev_thread_ID_get_latest_file"] = thread_ID

        # Format current date nicely (from "DD-MM-YYYY-HH-MM.hdf" to "HH:MM DD-MM-YYYY")
        latest_file_display = st.session_state.latest_file
        if latest_file_display and latest_file_display != "N/A":
            try:
                from datetime import datetime
                # Remove .hdf extension and parse
                filename_clean = Path(latest_file_display).stem
                dt = datetime.strptime(filename_clean, "%d-%m-%Y-%H-%M")
                latest_file_display = dt.strftime("%H:%M %d-%m-%Y")
            except:
                # Fallback to original if parsing fails
                latest_file_display = latest_file_display.replace('.hdf', '')

        st.markdown("<div style='text-align: center; font-size: 18px;'>"
                    f"<b>Current Date: {latest_file_display}</b>"
                    "</div>",
                    unsafe_allow_html=True)

        latest_file = st.session_state["latest_thread"]
        if latest_file != st.session_state.latest_file:
            # calcolo della previsione in background
            if st.session_state["launch_prediction_thread"] is None:
                logger.info(f"Launching predictions for all models for {latest_file}")

                # Mark that prediction threads are launching
                st.session_state["launch_prediction_thread"] = "ALL_MODELS"

                ctx = get_script_run_ctx()
                launch_thread = threading.Thread(target=launch_thread_execution,
                                                 args=(st, latest_file, columns, model_list),
                                                 daemon=True)
                add_script_run_ctx(launch_thread, ctx)
                launch_thread.start()
        else:
            logger.debug(f"Current SRI == Latest file processed! {latest_file}. Skipped prediction")

        # Load and display animated predictions
        if st.session_state.selected_model:
            # Check if we need to load new predictions (first time or model changed)
            if ("new_prediction" in st.session_state and st.session_state["new_prediction"]) or \
               ("previous_model" in st.session_state and st.session_state['previous_model'] != st.session_state['selected_model']) or \
               "all_predictions_data" not in st.session_state:

                st.session_state.previous_model = st.session_state.selected_model

                # Load all 12 predictions at once
                with st.spinner("Loading all prediction timesteps...", show_time=True):
                    rgba_images_dict, status_info = load_all_predictions(st, time_options, latest_file)

                # Store status information for display in status panel
                st.session_state['data_load_status'] = status_info

                if rgba_images_dict is not None:
                    st.session_state['all_predictions_data'] = rgba_images_dict
                    st.session_state['display_prediction'] = True
                    st.session_state['new_prediction'] = False
                    create_animated_map_html(rgba_images_dict, st.session_state.latest_file)
                else:
                    # No predictions available yet
                    create_only_map(None)
            else:
                # Display existing predictions
                if "all_predictions_data" in st.session_state and st.session_state['all_predictions_data'] is not None:
                    create_animated_map_html(st.session_state['all_predictions_data'], st.session_state.latest_file)
                else:
                    create_only_map(None)
        else:
            create_only_map(None)

    # Status Panel - Right Column
    with columns[1]:
        # Add CSS for animated dots on Computing status
        st.markdown("""
        <style>
        .computing-text::after {
            content: '';
            animation: dots 1.5s steps(4, end) infinite;
        }
        @keyframes dots {
            0%, 24% { content: ''; }
            25%, 49% { content: '.'; }
            50%, 74% { content: '..'; }
            75%, 100% { content: '...'; }
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown("### System Status")

        # Latest data timestamp
        latest_file = st.session_state.get("latest_file", "N/A")
        st.markdown(f"**Last Data Found:**  \n`{latest_file}`")

        # Check if new data is available
        latest_thread = st.session_state.get("latest_thread", None)
        if latest_thread and latest_thread != latest_file:
            st.success("🆕 New data available!")

        st.markdown("---")

        # Model Prediction Status
        st.markdown("**Model Predictions:**")

        from nwc_webapp.config.config import get_config
        config = get_config()

        for model in model_options:
            # Check if prediction exists for this model
            latest_npy = Path(latest_file).stem + '.npy' if latest_file != "N/A" else None

            if latest_npy:
                # All models now use the same path structure in real_time_pred
                pred_path = config.real_time_pred / model / latest_npy

                if pred_path.exists():
                    st.markdown(f"- ✅ **{model}**: Ready")
                else:
                    # Check PBS job status (only on HPC)
                    job_status = None
                    is_computing = model in st.session_state.get("computing_models", set())
                    was_submitted = model in st.session_state.get("submitted_models", set())

                    try:
                        from nwc_webapp.services.pbs import get_model_job_status, is_pbs_available
                        if is_pbs_available():
                            job_status = get_model_job_status(model)
                    except:
                        pass

                    # Determine display status
                    if job_status == 'Q':
                        st.markdown(f"- 📋 **{model}**: Queue")
                    elif job_status == 'R':
                        st.markdown(f"- ⚙️ **{model}**: <span class='computing-text'>Computing</span>", unsafe_allow_html=True)
                    elif is_computing:
                        # Worker thread is still polling for output file
                        st.markdown(f"- 🔄 **{model}**: Finalizing...")
                    elif was_submitted and not job_status:
                        # Job was submitted but is no longer in queue and no output file
                        # This means the job finished but the prediction file was not created
                        st.markdown(f"- ❌ **{model}**: Failed")
                    else:
                        st.markdown(f"- ⏹️ **{model}**: Not computed")
            else:
                st.markdown(f"- ⏹️ **{model}**: Waiting for data")

        st.markdown("---")

        # System Info
        st.markdown("**System Info:**")
        checking_status = "🔄 Active" if st.session_state.get("run_get_latest_file") else "⏸️ Paused"
        st.markdown(f"- Data Monitor: {checking_status}")

        if "all_predictions_data" in st.session_state and st.session_state["all_predictions_data"]:
            num_frames = len(st.session_state["all_predictions_data"])
            st.markdown(f"- Loaded Frames: {num_frames}")

        # Auto-refresh indicator
        st.markdown(f"- Auto-refresh: Every 5 min")

        # Data Load Status / Errors
        if "data_load_status" in st.session_state:
            status = st.session_state["data_load_status"]

            # Show ground truth status
            if not status.get('ground_truth_available', False):
                st.markdown("---")
                st.warning(f"⚠️ **Ground Truth Data Missing**")
                if status.get('error'):
                    st.error(status['error'])
                st.markdown("_Ground truth radar data is required for predictions. Please ensure SRI files are available._")
            elif status.get('ground_truth_count', 0) < 7:
                st.markdown("---")
                st.warning(f"⚠️ **Partial Ground Truth**  \nOnly {status.get('ground_truth_count', 0)}/7 frames loaded")