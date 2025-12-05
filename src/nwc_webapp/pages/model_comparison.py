import matplotlib.pyplot as plt
from io import BytesIO
"""
Model Comparison page - compare multiple models side-by-side at a single timestamp.
"""

import base64
import os
import time
from datetime import datetime, timedelta
from datetime import time as dt_time
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from nwc_webapp.config.config import get_config
from nwc_webapp.config.environment import is_hpc
from nwc_webapp.evaluation.metrics import CSI
from nwc_webapp.logging_config import setup_logger
from nwc_webapp.rendering.figures import compute_figure_gpd
from nwc_webapp.data.checking import check_single_prediction_exists
from nwc_webapp.data.predictions import load_prediction_array
from nwc_webapp.hpc.jobs import submit_date_range_prediction_job
from nwc_webapp.hpc.pbs import get_model_job_status, is_pbs_available
from nwc_webapp.pages.common import show_training_date_warning, is_training_date
from nwc_webapp.data.groundtruth import load_groundtruth_for_timestamp

# Set up logger
logger = setup_logger(__name__)


def load_prediction_for_timestamp(model_name: str, timestamp: datetime) -> Optional[np.ndarray]:
    """
    Load prediction for a given model and timestamp.

    Args:
        model_name: Model name
        timestamp: Base timestamp

    Returns:
        np.ndarray of shape (12, H, W) or None if not found
    """
    config = get_config()

    pred_filename = timestamp.strftime('%d-%m-%Y-%H-%M') + '.npy'
    pred_path = config.real_time_pred / model_name / pred_filename

    if not pred_path.exists():
        return None

    try:
        # Use helper to handle model-specific shapes
        pred_array = load_prediction_array(pred_path, model_name)

        if pred_array is None:
            return None

        # Make a writable copy to avoid read-only array issues
        pred_array = np.array(pred_array, copy=True)

        # Load mask and apply
        mask_path = Path(__file__).resolve().parent.parent / "resources/mask/radar_mask.hdf"
        with h5py.File(mask_path, "r") as f:
            radar_mask = f["mask"][()]

        # Apply mask to all frames
        for i in range(pred_array.shape[0]):
            pred_array[i] = pred_array[i] * radar_mask
            pred_array[i] = np.clip(pred_array[i], 0, 200)

        return pred_array

    except Exception as e:
        logger.error(f"Error loading prediction for {model_name} at {timestamp}: {e}")
        return None


def compute_csi_for_leadtime(
    gt_frame: np.ndarray,
    pred_frames: Dict[str, np.ndarray],
    thresholds: List[float]
) -> pd.DataFrame:
    """
    Compute CSI for all models at a specific lead time.

    Args:
        gt_frame: Ground truth frame (H, W)
        pred_frames: Dictionary {model_name: prediction_frame}
        thresholds: List of precipitation thresholds in mm/h

    Returns:
        DataFrame with models as rows and thresholds as columns
    """
    results = []

    for model_name, pred_frame in pred_frames.items():
        row = {"Model": model_name}

        for threshold in thresholds:
            csi = CSI(gt_frame, pred_frame, threshold)
            row[f"{threshold} mm/h"] = csi

        # Add average CSI
        csi_values = [row[f"{t} mm/h"] for t in thresholds]
        row["Average"] = np.mean(csi_values)

        results.append(row)

    # Create DataFrame
    df = pd.DataFrame(results)
    return df


def show_model_comparison_page(model_list: List[str]):
    """
    Main model comparison page.

    Allows users to:
    - Select a single timestamp
    - Add/remove model columns
    - View ground truth and predictions side-by-side for all lead times
    - See CSI metrics for each lead time

    Args:
        model_list: List of available models
    """
    st.title("Model Comparison")
    st.markdown("Compare multiple model predictions side-by-side at a single timestamp")

    # Initialize session state
    if "comparison_timestamp" not in st.session_state:
        st.session_state["comparison_timestamp"] = None
    if "comparison_models" not in st.session_state:
        st.session_state["comparison_models"] = []  # List of selected model names
    if "comparison_predictions" not in st.session_state:
        st.session_state["comparison_predictions"] = {}  # {model_name: np.array}
    if "comparison_gt" not in st.session_state:
        st.session_state["comparison_gt"] = None
    if "comparison_show_results" not in st.session_state:
        st.session_state["comparison_show_results"] = False
    if "comparison_add_model_dialog" not in st.session_state:
        st.session_state["comparison_add_model_dialog"] = None
    if "comparison_model_action" not in st.session_state:
        st.session_state["comparison_model_action"] = None  # "display", "compute", or None
    if "comparison_pending_model" not in st.session_state:
        st.session_state["comparison_pending_model"] = None
    if "comparison_load_gt_action" not in st.session_state:
        st.session_state["comparison_load_gt_action"] = None  # None, "show_warning", "load"

    # Timestamp selection
    st.subheader("📅 Select Timestamp")
    col1, col2 = st.columns(2)

    with col1:
        selected_date = st.date_input(
            "Date",
            min_value=datetime(2020, 1, 1).date(),
            max_value=datetime.today().date(),
            format="DD/MM/YYYY",
            value=datetime.now().date(),
            key="comparison_date"
        )

    with col2:
        selected_time = st.time_input(
            "Time",
            value=dt_time(datetime.now().hour, 0),
            step=300,  # 5 minutes
            key="comparison_time"
        )

    selected_datetime = datetime.combine(selected_date, selected_time)

    # Check if timestamp changed
    if st.session_state["comparison_timestamp"] != selected_datetime:
        logger.info(f"Timestamp changed: {selected_datetime}")
        st.session_state["comparison_timestamp"] = selected_datetime
        st.session_state["comparison_gt"] = None
        st.session_state["comparison_predictions"] = {}
        st.session_state["comparison_show_results"] = False
        st.session_state["comparison_load_gt_action"] = None
        st.session_state["training_warning_accepted_comparison"] = False

    # Load ground truth workflow
    if st.session_state["comparison_gt"] is None:
        # Step 1: Show "Load Ground Truth" button
        if st.session_state["comparison_load_gt_action"] is None:
            if st.button("📊 Load Ground Truth", type="primary", width='stretch'):
                st.session_state["comparison_load_gt_action"] = "check_training"
                st.rerun()
            return

        # Step 2: Check if training date warning needed
        if st.session_state["comparison_load_gt_action"] == "check_training":
            if is_training_date(selected_datetime):
                if not st.session_state.get("training_warning_accepted_comparison", False):
                    # Show warning dialog
                    if not show_training_date_warning("comparison"):
                        # User hasn't clicked YES or NO yet - keep showing warning
                        return
                    # User clicked NO
                    if not st.session_state.get("training_warning_accepted_comparison", False):
                        st.session_state["comparison_load_gt_action"] = None
                        st.rerun()
                        return

            # Training date accepted or not a training date - proceed to load
            st.session_state["comparison_load_gt_action"] = "load"
            st.rerun()

        # Step 3: Load ground truth
        if st.session_state["comparison_load_gt_action"] == "load":
            with st.spinner("Loading ground truth..."):
                gt_data = load_groundtruth_for_timestamp(selected_datetime)

            if gt_data is not None:
                st.session_state["comparison_gt"] = gt_data
                st.session_state["comparison_load_gt_action"] = None
                st.success(f"✅ Ground truth loaded for {selected_datetime.strftime('%d/%m/%Y %H:%M')}")
                st.rerun()
            else:
                st.error("❌ Failed to load ground truth data")
                st.session_state["comparison_load_gt_action"] = None
            return

        return

    # Ground truth loaded - show model management
    st.success(f"✅ Ground truth loaded for {selected_datetime.strftime('%d/%m/%Y %H:%M')} - 12 timesteps ready!")

    st.markdown("---")
    st.subheader("🔬 Models")

    # Show current models or empty state
    if not st.session_state["comparison_models"]:
        st.info("📌 No models added yet. Click 'Add Model Column' to start comparing!")

    # Add model button
    if st.button("➕ Add Model Column", key="add_model_btn"):
        st.session_state["comparison_add_model_dialog"] = "select"
        st.rerun()

    # Show model selection dialog
    if st.session_state["comparison_add_model_dialog"] == "select":
        # Filter out already selected models
        available_models = [m for m in model_list if m not in st.session_state["comparison_models"]]

        if not available_models:
            st.info("ℹ️ All models are already added")
            if st.button("Close", key="close_no_models"):
                st.session_state["comparison_add_model_dialog"] = None
                st.rerun()
        else:
            selected_model = st.selectbox(
                "Select model to add",
                available_models,
                key="model_selector"
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Add", key="confirm_add_model", width='stretch'):
                    st.session_state["comparison_pending_model"] = selected_model
                    st.session_state["comparison_add_model_dialog"] = "check"
                    st.rerun()

            with col2:
                if st.button("❌ Cancel", key="cancel_add_model", width='stretch'):
                    st.session_state["comparison_add_model_dialog"] = None
                    st.rerun()

    # Check prediction exists for pending model
    if st.session_state["comparison_add_model_dialog"] == "check":
        pending_model = st.session_state["comparison_pending_model"]

        with st.spinner(f"Checking if prediction exists for {pending_model}..."):
            pred_exists = check_single_prediction_exists(pending_model, selected_datetime)

        if pred_exists:
            st.session_state["comparison_add_model_dialog"] = "exists"
        else:
            st.session_state["comparison_add_model_dialog"] = "compute"
        st.rerun()

    # Show dialog for existing prediction
    if st.session_state["comparison_add_model_dialog"] == "exists":
        pending_model = st.session_state["comparison_pending_model"]

        st.success(f"✅ Prediction exists for **{pending_model}** at {selected_datetime.strftime('%d/%m/%Y %H:%M')}")
        st.info("Do you want to display the existing prediction or recompute it?")

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📺 Display", key="display_existing_pred", width='stretch', type="primary"):
                st.session_state["comparison_model_action"] = "display"
                st.session_state["comparison_add_model_dialog"] = None
                st.rerun()

        with col2:
            if st.button("🔄 Recompute", key="recompute_pred", width='stretch'):
                st.session_state["comparison_model_action"] = "compute"
                st.session_state["comparison_add_model_dialog"] = None
                st.rerun()

        with col3:
            if st.button("❌ Cancel", key="cancel_pred_dialog", width='stretch'):
                st.session_state["comparison_add_model_dialog"] = None
                st.session_state["comparison_pending_model"] = None
                st.rerun()
        return

    # Compute prediction dialog
    if st.session_state["comparison_add_model_dialog"] == "compute":
        pending_model = st.session_state["comparison_pending_model"]

        st.warning(f"⚠️ Prediction not found for **{pending_model}** at {selected_datetime.strftime('%d/%m/%Y %H:%M')}")
        st.info("Do you want to compute it now?")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Compute", key="confirm_compute_pred", width='stretch', type="primary"):
                st.session_state["comparison_model_action"] = "compute"
                st.session_state["comparison_add_model_dialog"] = None
                st.rerun()

        with col2:
            if st.button("❌ Cancel", key="cancel_compute_dialog", width='stretch'):
                st.session_state["comparison_add_model_dialog"] = None
                st.session_state["comparison_pending_model"] = None
                st.rerun()
        return

    # Handle display action
    if st.session_state["comparison_model_action"] == "display":
        pending_model = st.session_state["comparison_pending_model"]

        with st.spinner(f"Loading prediction for {pending_model}..."):
            pred_data = load_prediction_for_timestamp(pending_model, selected_datetime)

        if pred_data is not None:
            # Add to models and predictions
            st.session_state["comparison_models"].append(pending_model)
            st.session_state["comparison_predictions"][pending_model] = pred_data
            st.session_state["comparison_show_results"] = True

            # Clear action state
            st.session_state["comparison_model_action"] = None
            st.session_state["comparison_pending_model"] = None

            st.success(f"✅ {pending_model} prediction loaded!")
            st.rerun()
        else:
            st.error(f"❌ Failed to load prediction for {pending_model}")
            st.session_state["comparison_model_action"] = None
            st.session_state["comparison_pending_model"] = None

    # Handle compute action
    if st.session_state["comparison_model_action"] == "compute":
        pending_model = st.session_state["comparison_pending_model"]

        st.info(f"🚀 Submitting PBS job for {pending_model}...")

        # Submit job for single timestamp (start_dt == end_dt)
        with st.spinner(f"Submitting job for {pending_model}..."):
            job_id = submit_date_range_prediction_job(pending_model, selected_datetime, selected_datetime)

        if job_id:
            st.success(f"✅ Job submitted! Job ID: {job_id}")
            st.write(f"**Model**: {pending_model}")
            st.write(f"**Timestamp**: {selected_datetime.strftime('%d/%m/%Y %H:%M')}")

            st.markdown("---")

            # Check if this is a mock job (local mode)
            if not is_hpc() or job_id.startswith("mock_"):
                logger.info("🖥️  Local mode detected - skipping job monitoring")
                st.success("✅ Mock prediction created instantly!")

                # Load prediction
                pred_data = load_prediction_for_timestamp(pending_model, selected_datetime)

                if pred_data is not None:
                    st.session_state["comparison_models"].append(pending_model)
                    st.session_state["comparison_predictions"][pending_model] = pred_data
                    st.session_state["comparison_show_results"] = True

                    st.session_state["comparison_model_action"] = None
                    st.session_state["comparison_pending_model"] = None

                    st.success(f"✅ {pending_model} prediction loaded!")
                    st.rerun()
                else:
                    st.error("❌ Failed to load mock prediction")
                    st.session_state["comparison_model_action"] = None
                    st.session_state["comparison_pending_model"] = None
                return

            # HPC mode: Monitor job
            st.subheader("📊 Job Progress")

            # Add CSS for animated dots
            st.markdown("""
                <style>
                .queue-text::after, .running-text::after {
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

            status_placeholder = st.empty()
            progress_placeholder = st.empty()

            # Monitor job
            max_iterations = 1800  # 1 hour max
            iteration = 0
            last_status = None
            job_completed = False
            consecutive_none_count = 0

            while iteration < max_iterations and not job_completed:
                current_status = None
                if is_pbs_available():
                    try:
                        current_status = get_model_job_status(pending_model)
                        logger.debug(f"Job status: {current_status}")
                    except Exception as e:
                        logger.error(f"Error checking job status: {e}")

                # Update status display
                if current_status != last_status and current_status is not None:
                    if current_status == "Q":
                        status_placeholder.markdown("⏳ **Job in <span class='queue-text'>queue</span>**", unsafe_allow_html=True)
                        logger.info(f"Job {job_id} is in queue")
                    elif current_status == "R":
                        status_placeholder.markdown("⚙️ **Job <span class='running-text'>running</span>**", unsafe_allow_html=True)
                        logger.info(f"Job {job_id} is running")
                    consecutive_none_count = 0

                # Check if job disappeared (completed)
                if last_status is not None and current_status is None:
                    consecutive_none_count += 1
                    logger.debug(f"Job status None (consecutive: {consecutive_none_count}/3)")

                    if consecutive_none_count >= 3:
                        logger.info(f"Job {job_id} disappeared from queue - verifying")
                        job_completed = True

                        status_placeholder.info("⏳ **Job finished - verifying results...**")
                        time.sleep(5)

                        # Load prediction
                        pred_data = load_prediction_for_timestamp(pending_model, selected_datetime)

                        if pred_data is not None:
                            status_placeholder.success("✅ **Prediction completed!**")
                            st.success(f"🎉 {pending_model} prediction completed successfully!")

                            # Add to models and predictions
                            st.session_state["comparison_models"].append(pending_model)
                            st.session_state["comparison_predictions"][pending_model] = pred_data
                            st.session_state["comparison_show_results"] = True

                            st.session_state["comparison_model_action"] = None
                            st.session_state["comparison_pending_model"] = None

                            st.rerun()
                        else:
                            status_placeholder.error("❌ **Error: Prediction not found!**")
                            st.error(f"❌ Failed to load prediction for {pending_model}")

                            # Try to read PBS log
                            home_dir = os.path.expanduser("~")
                            log_file = os.path.join(home_dir, f"nwc_{pending_model}_range.o{job_id}")

                            if os.path.exists(log_file):
                                try:
                                    with open(log_file, "r") as f:
                                        log_content = f.read()
                                    st.markdown("---")
                                    st.error("### ❌ PBS Job Error Log")
                                    st.markdown(f"**Log file**: `{log_file}`")
                                    st.code(log_content, language="text")
                                except Exception as e:
                                    st.warning(f"Could not read log file: {e}")
                            else:
                                st.warning(f"Could not find PBS log file at: `{log_file}`")

                        break
                else:
                    consecutive_none_count = 0

                if current_status is not None:
                    last_status = current_status

                # Check if prediction exists
                pred_exists = check_single_prediction_exists(pending_model, selected_datetime)
                if pred_exists:
                    progress_placeholder.progress(1.0, text="Prediction: 1/1")
                else:
                    progress_placeholder.progress(0.0, text="Prediction: 0/1")

                time.sleep(2)
                iteration += 1

            if iteration >= max_iterations:
                st.warning("⚠️ Monitoring timeout reached (1 hour). Check job status manually.")
        else:
            st.error(f"❌ Failed to submit job for {pending_model}")
            st.session_state["comparison_model_action"] = None
            st.session_state["comparison_pending_model"] = None

        return

    # Display results if we have models
    if st.session_state["comparison_show_results"] and st.session_state["comparison_models"]:
        display_comparison_results(
            selected_datetime,
            st.session_state["comparison_gt"],
            st.session_state["comparison_models"],
            st.session_state["comparison_predictions"]
        )


def display_comparison_results(
    timestamp: datetime,
    gt_data: np.ndarray,
    model_names: List[str],
    predictions: Dict[str, np.ndarray]
):
    """
    Display comparison results in a grid layout.

    Layout:
    Row 1 (+5min):  [GT] [Model1] [Model2] ... [CSI Table]
    Row 2 (+10min): [GT] [Model1] [Model2] ... [CSI Table]
    ...
    Row 12 (+60min): [GT] [Model1] [Model2] ... [CSI Table]

    Args:
        timestamp: Base timestamp
        gt_data: Ground truth array (12, H, W)
        model_names: List of model names
        predictions: Dictionary {model_name: prediction_array}
    """

    config = get_config()

    st.markdown("---")
    st.subheader("📊 Comparison Results")

    # Get CSI thresholds from config
    thresholds = config.csi_threshold

    # Show current models with remove buttons
    st.write("**Current Models:**")
    cols = st.columns(len(model_names) + 1)

    for idx, model in enumerate(model_names):
        with cols[idx]:
            if st.button(f"🗑️ {model}", key=f"remove_{model}", width='stretch'):
                # Remove model
                st.session_state["comparison_models"].remove(model)
                st.session_state["comparison_predictions"].pop(model, None)

                # If no models left, hide results
                if not st.session_state["comparison_models"]:
                    st.session_state["comparison_show_results"] = False

                st.rerun()

    with cols[-1]:
        if st.button("➕ Add Model", key="add_another_model", width='stretch'):
            st.session_state["comparison_add_model_dialog"] = "select"
            st.rerun()

    st.markdown("---")
    st.info("🔍 **Synchronized Zoom**: Mouse wheel to zoom, click & drag to pan, double-click to reset")

    # Create grid for each lead time
    lead_times = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]

    for lead_idx, lead_time in enumerate(lead_times):
        st.markdown(f"### ⏱️ +{lead_time} minutes")

        # Prepare data for all models at this lead time
        gt_frame = gt_data[lead_idx]
        pred_frames = {}

        # Generate matplotlib figures and convert to base64
        images_b64 = []

        # Ground Truth
        gt_time = timestamp + timedelta(minutes=lead_time)
        timestamp_str = f"GT +{lead_time}min - {gt_time.strftime('%d/%m/%Y %H:%M')}"
        fig = compute_figure_gpd(gt_frame, timestamp_str, name="")
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        images_b64.append(base64.b64encode(buf.read()).decode())
        plt.close(fig)

        # Models
        for model_name in model_names:
            pred_data = predictions[model_name]
            pred_frame = pred_data[lead_idx]
            pred_frames[model_name] = pred_frame

            pred_time = timestamp + timedelta(minutes=lead_time)
            timestamp_str = f"{model_name} +{lead_time}min - {pred_time.strftime('%d/%m/%Y %H:%M')}"
            fig = compute_figure_gpd(pred_frame, timestamp_str, name="")
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            images_b64.append(base64.b64encode(buf.read()).decode())
            plt.close(fig)

        # Create HTML with JavaScript synchronized zoom
        html_template = """
        <div class="zoom-container" data-row="{row_id}">
            {image_divs}
        </div>

        <style>
        .zoom-container {{
            display: flex;
            gap: 10px;
            margin: 10px 0 20px 0;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 8px;
            overflow: hidden;
        }}

        .zoom-wrapper {{
            flex: 1;
            position: relative;
            overflow: hidden;
            background: white;
            border: 2px solid #dee2e6;
            border-radius: 4px;
            cursor: grab;
        }}

        .zoom-wrapper:active {{
            cursor: grabbing;
        }}

        .zoom-image {{
            width: 100%;
            height: auto;
            display: block;
            transform-origin: center center;
            transition: transform 0.05s ease-out;
        }}
        </style>

        <script>
        (function() {{
            const container = document.querySelector('[data-row="{row_id}"]');
            const images = container.querySelectorAll('.zoom-image');
            const wrappers = container.querySelectorAll('.zoom-wrapper');

            let scale = 1;
            let translateX = 0;
            let translateY = 0;
            let isDragging = false;
            let startX = 0;
            let startY = 0;

            function updateTransform() {{
                const transform = `translate(${{translateX}}px, ${{translateY}}px) scale(${{scale}})`;
                images.forEach(img => {{
                    img.style.transform = transform;
                }});
            }}

            wrappers.forEach(wrapper => {{
                // Mouse wheel for zoom
                wrapper.addEventListener('wheel', (e) => {{
                    e.preventDefault();
                    const delta = e.deltaY > 0 ? 0.9 : 1.1;
                    const newScale = scale * delta;
                    if (newScale >= 0.5 && newScale <= 10) {{
                        scale = newScale;
                        updateTransform();
                    }}
                }}, {{ passive: false }});

                // Mouse events for pan
                wrapper.addEventListener('mousedown', (e) => {{
                    isDragging = true;
                    startX = e.clientX - translateX;
                    startY = e.clientY - translateY;
                }});

                wrapper.addEventListener('mousemove', (e) => {{
                    if (!isDragging) return;
                    translateX = e.clientX - startX;
                    translateY = e.clientY - startY;
                    updateTransform();
                }});

                wrapper.addEventListener('mouseup', () => {{
                    isDragging = false;
                }});

                wrapper.addEventListener('mouseleave', () => {{
                    isDragging = false;
                }});

                // Double click to reset
                wrapper.addEventListener('dblclick', () => {{
                    scale = 1;
                    translateX = 0;
                    translateY = 0;
                    updateTransform();
                }});
            }});
        }})();
        </script>
        """

        # Build image divs
        image_divs = ""
        for img_b64 in images_b64:
            image_divs += f'<div class="zoom-wrapper"><img src="data:image/png;base64,{img_b64}" class="zoom-image" draggable="false"></div>\n'

        html_code = html_template.format(row_id=f"row_{lead_idx}", image_divs=image_divs)

        # Display synchronized images and CSI table side-by-side
        cols = st.columns([4, 1])

        with cols[0]:
            components.html(html_code, height=500, scrolling=False)

        with cols[1]:
            st.markdown("**CSI Metrics**")

            # Compute CSI for all models at this lead time
            csi_df = compute_csi_for_leadtime(gt_frame, pred_frames, thresholds)

            # Transpose: Thresholds as rows, Models as columns
            csi_df = csi_df.set_index("Model").T
            csi_df.index.name = "Threshold"

            # Format CSI values to 3 decimal places
            csi_df = csi_df.applymap(lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else x)

            st.dataframe(csi_df, use_container_width=True, height=400)

        st.markdown("---")