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
from nwc_webapp.config.config import get_config

# Set up logger
logger = setup_logger(__name__)


def _get_model_status(model, latest_file, config):
    """
    Fast helper function to determine model status.
    Returns status text immediately without blocking.

    Args:
        model: Model name
        latest_file: Latest SRI file
        config: Config instance

    Returns:
        str: Formatted status text for the model
    """
    if latest_file == "N/A":
        return f"- ⏹️ **{model}**: Waiting for data"

    # Check if prediction exists for this model
    # TEST model always uses predictions.npy (case-insensitive)
    if model.upper() == "TEST":
        pred_path = config.real_time_pred / "Test" / "predictions.npy"
    else:
        # Other models use date-based filename
        latest_npy = Path(latest_file).stem + '.npy'
        pred_path = config.real_time_pred / model / latest_npy

    if pred_path.exists():
        return f"- ✅ **{model}**: Ready"

    # File doesn't exist - check job status
    # Get status from session state (fast, non-blocking)
    # Safely access session state - handle case where context is not available
    try:
        is_computing = model in st.session_state.get("computing_models", set())
        was_submitted = model in st.session_state.get("submitted_models", set())
        has_failed = model in st.session_state.get("failed_models", set())
    except RuntimeError:
        # Script run context not available - return default status
        return f"- ⏹️ **{model}**: Initializing..."

    # Try to get PBS job status (quick check, non-blocking with try/except)
    job_status = None
    try:
        from nwc_webapp.services.pbs import get_model_job_status, is_pbs_available
        if is_pbs_available():
            job_status = get_model_job_status(model)
    except Exception:
        pass  # Silently fail - don't block on errors

    # Determine display status (fast logic)
    if has_failed:
        return f"- ❌ **{model}**: Failed prediction!"
    elif job_status == 'Q':
        return f"- 📋 **{model}**: <span class='queue-text'>Queue</span>"
    elif job_status == 'R':
        return f"- ⚙️ **{model}**: <span class='computing-text'>Computing</span>"
    elif is_computing:
        return f"- 🔄 **{model}**: Finalizing..."
    elif was_submitted and not job_status:
        return f"- ❌ **{model}**: Failed"
    else:
        return f"- ⏹️ **{model}**: Not computed"


@st.fragment(run_every=5)
def update_model_predictions(model_list):
    """
    Update all model predictions together every 5 seconds.
    Pre-computes all statuses first, then renders all at once for simultaneous display.
    """
    config = get_config()
    latest_file = st.session_state.get("latest_file", "N/A")

    # STEP 1: Pre-compute all model statuses BEFORE rendering
    # This loop can be sequential - it's hidden from the user
    model_statuses = {}
    for model in model_list:
        try:
            model_statuses[model] = _get_model_status(model, latest_file, config)
        except Exception as e:
            model_statuses[model] = f"- ⚠️ **{model}**: Error"
            logger.error(f"Error getting status for {model}: {e}")

    # STEP 2: Render all models at once with pre-computed statuses
    # This is fast - just outputting strings, no PBS checks or file I/O
    for model in model_list:
        st.markdown(model_statuses[model], unsafe_allow_html=True)


@st.fragment
def render_status_panel(model_list):
    """
    Status panel with static structure and smart model status updates.
    Only Model Predictions section updates (every 2s), everything else is static.
    """
    # Add CSS for animated dots (only once)
    st.markdown("""
    <style>
    .queue-text::after, .computing-text::after {
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

    # Static header
    st.markdown("### System Status")

    # Static label
    st.markdown("**Last Data Found:**")
    # Dynamic content (updates every 2s)
    latest_file = st.session_state.get("latest_file", "N/A")
    st.code(latest_file, language=None)

    # Check if new data is available but not yet displayed on map
    latest_thread = st.session_state.get("latest_thread", None)
    displayed_file = st.session_state.get("displayed_file", None)

    if latest_thread and displayed_file and latest_thread != displayed_file:
        st.success("🆕 **New data available!**  \n_Map will update when prediction loads_")

    st.markdown("---")

    # Model Prediction Status - updates all models together every 5 seconds
    st.markdown("**Model Predictions:**")

    # Initialize on first render, then fragment takes over
    if "model_predictions_initialized" not in st.session_state:
        st.session_state["model_predictions_initialized"] = True
        # Render initial statuses immediately
        config = get_config()
        latest_file = st.session_state.get("latest_file", "N/A")
        for model in model_list:
            try:
                status_text = _get_model_status(model, latest_file, config)
                st.markdown(status_text, unsafe_allow_html=True)
            except Exception as e:
                st.markdown(f"- ⚠️ **{model}**: Error", unsafe_allow_html=True)
    else:
        # Fragment pre-computes all statuses then renders all at once
        update_model_predictions(model_list)

    st.markdown("---")

    # System Info - static section
    st.markdown("**System Info:**")
    checking_status = "🔄 Active" if st.session_state.get("run_get_latest_file") else "⏸️ Paused"
    st.markdown(f"- Data Monitor: {checking_status}")

    if "all_predictions_data" in st.session_state and st.session_state["all_predictions_data"]:
        num_frames = len(st.session_state["all_predictions_data"])
        st.markdown(f"- Loaded Frames: {num_frames}")

    st.markdown(f"- Auto-refresh: Every 5 min")

    st.markdown("---")

    # Display missing groundtruth data warnings
    if "data_load_status" in st.session_state:
        status = st.session_state["data_load_status"]
        missing_gt = status.get('missing_groundtruth', [])

        if missing_gt:
            st.markdown("---")
            st.warning(f"⚠️ **Missing Groundtruth Data**")
            for timestamp in missing_gt:
                st.markdown(f"- Groundtruth data **{timestamp}** is missing")

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

        # Display date for the data that's ACTUALLY shown on the map (not latest available)
        # Use displayed_file to track what's currently visible, not latest_file
        displayed_file = st.session_state.get("displayed_file", st.session_state.latest_file)
        latest_file_display = displayed_file

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

                # Update latest_file immediately so we don't relaunch for the same file
                st.session_state.latest_file = latest_file

                # Clear all model state from previous run so they can start fresh with new data
                if "failed_models" in st.session_state:
                    st.session_state["failed_models"].clear()
                    logger.info("Cleared failed_models set for new data processing")

                if "submitted_models" in st.session_state:
                    st.session_state["submitted_models"].clear()
                    logger.info("Cleared submitted_models set for new data processing")

                if "computing_models" in st.session_state:
                    st.session_state["computing_models"].clear()
                    logger.info("Cleared computing_models set for new data processing")

                ctx = get_script_run_ctx()
                launch_thread = threading.Thread(target=launch_thread_execution,
                                                 args=(st, latest_file, columns, model_list),
                                                 daemon=True)
                add_script_run_ctx(launch_thread, ctx)
                launch_thread.start()

                logger.info(f"Updated latest_file to {latest_file} - ready for next update")
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
                    # Update displayed_file to match what's actually shown on the map
                    st.session_state['displayed_file'] = latest_file
                    logger.info(f"Map data loaded - updated displayed_file to {latest_file}")
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

    # Status Panel - Right Column (updates independently every 2 seconds)
    with columns[1]:
        render_status_panel(model_list)