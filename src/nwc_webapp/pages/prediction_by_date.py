"""
Prediction by date and time page.
"""

import os
import time
from datetime import datetime, timedelta
from datetime import time as dt_time

import streamlit as st

from nwc_webapp.config.config import get_config
from nwc_webapp.config.environment import is_hpc
from nwc_webapp.logging_config import setup_logger
from nwc_webapp.pages.nowcasting_utils import (
    check_single_prediction_exists,
    is_training_date,
    load_single_prediction_data,
    submit_date_range_prediction_job,
)
from nwc_webapp.ui.components import init_second_tab_layout, precompute_images

# Set up logger
logger = setup_logger(__name__)


def show_training_date_warning() -> bool:
    """
    Show warning dialog for dates prior to Jan 1, 2025.

    Returns:
        True if user wants to proceed, False otherwise
    """
    st.warning(
        "⚠️ **Training Data Warning**\n\n"
        "Dates prior to **1st January 2025** were used for model training. "
        "The prediction results will not be accountable and may not reflect real-world performance.\n\n"
        "**Are you sure you want to proceed?**"
    )

    col1, col2, _ = st.columns([1, 1, 3])

    with col1:
        if st.button("✅ YES, Proceed", key="training_yes_third_tab", width='stretch'):
            st.session_state.training_warning_accepted_third_tab = True
            return True

    with col2:
        if st.button("❌ NO, Cancel", key="training_no_third_tab", width='stretch'):
            st.session_state.training_warning_accepted_third_tab = False
            st.info("Operation cancelled. Please select a different date.")
            return False

    return False


def show_prediction_page(model_list):
    """
    Display prediction page with date/time selection.

    Workflow:
    1. User selects date/time and model
    2. Check if prediction exists in real_time_pred folder
    3. If exists: load and display data
    4. If doesn't exist: submit job (start_date == end_date for single prediction)
    5. Monitor job status (queued → running → completed)
    6. When done: load and display OR show error

    Args:
        model_list: List of available models
    """
    st.title("Prediction by Date and Time")

    # Initialize session state for workflow tracking
    if "third_tab_checking" not in st.session_state:
        st.session_state["third_tab_checking"] = False
    if "third_tab_action" not in st.session_state:
        st.session_state["third_tab_action"] = None  # Can be: None, "display", "recompute"

    # Date and time selection
    selected_date = st.date_input(
        "Select Date",
        min_value=datetime(2020, 1, 1).date(),
        max_value=datetime.today().date(),
        format="DD/MM/YYYY",
        value=datetime.now().date(),
    )
    selected_time = st.time_input(
        "Select Time", value=dt_time(datetime.now().hour, 0), step=300
    )  # 300 seconds = 5 minutes
    selected_model = st.selectbox("Select model", model_list)

    # Combine selected date and time
    selected_datetime = datetime.combine(selected_date, selected_time)

    # Check if selection changed - clear workflow state
    current_selection = (selected_date, selected_time, selected_model)
    if "third_tab_last_selection" not in st.session_state:
        st.session_state["third_tab_last_selection"] = current_selection
    elif st.session_state["third_tab_last_selection"] != current_selection:
        st.session_state["third_tab_last_selection"] = current_selection
        st.session_state["third_tab_checking"] = False
        st.session_state["third_tab_action"] = None
        st.session_state["show_prediction_results"] = False
        st.session_state["prediction_data_cache"] = None

    # Check if we have cached results to display
    if st.session_state.get("show_prediction_results", False) and st.session_state.get("prediction_data_cache"):
        cached = st.session_state["prediction_data_cache"]

        # Check if cached data matches current selection
        cached_dt = datetime.combine(cached["date"], cached["time"])
        current_dt = datetime.combine(selected_date, selected_time)

        if (cached["model"] == selected_model and cached_dt == current_dt):
            # Display only the images - no other UI
            try:
                init_second_tab_layout(
                    cached["groundtruth_images"],
                    cached["target_dict"],
                    cached["pred_dict"]
                )
            except Exception as e:
                logger.error(f"Error displaying cached results: {e}")
                st.error(f"❌ Error displaying results: {e}")
                # Clear cache on error
                st.session_state["show_prediction_results"] = False
                st.session_state["prediction_data_cache"] = None
            return  # Stop here - don't show the button again
        else:
            # Selection changed - clear cache
            st.session_state["show_prediction_results"] = False
            st.session_state["prediction_data_cache"] = None

    if st.button("Check/Compute Prediction", type="primary", width='stretch'):
        st.session_state["third_tab_checking"] = True
        st.session_state["third_tab_action"] = None
        st.rerun()

    # Execute workflow if checking flag is set
    if st.session_state["third_tab_checking"]:
        # Combine selected date and time
        selected_datetime = datetime.combine(selected_date, selected_time)
        logger.info(f"Checking prediction for {selected_model} at {selected_datetime.strftime('%d/%m/%Y %H:%M')}")

        # Check if date is in training period (before Jan 1, 2025)
        if is_training_date(selected_datetime):
            # Check if user has already accepted the warning in this session
            if not st.session_state.get("training_warning_accepted_third_tab", False):
                # Show warning and wait for user decision
                if not show_training_date_warning():
                    return  # User cancelled, stop here

        # Check if prediction exists
        pred_exists = check_single_prediction_exists(selected_model, selected_datetime)

        if pred_exists:
            # Prediction exists - ask if user wants to display or recompute
            st.success(f"✅ Prediction exists for {selected_model} at {selected_datetime.strftime('%d/%m/%Y %H:%M')}")
            st.info("Do you want to display the existing prediction or recompute it?")

            col1, col2, _ = st.columns([1, 1, 2])

            with col1:
                if st.button("📺 Display", key="display_existing_pred"):
                    # Load and display existing prediction
                    with st.spinner("Loading data..."):
                        try:
                            gt_dict, target_dict, pred_dict = load_single_prediction_data(selected_model, selected_datetime)

                            # Check if we have enough data
                            if not gt_dict or not target_dict or not pred_dict:
                                st.error(
                                    "❌ Failed to load complete data. Some groundtruth or prediction files may be missing."
                                )
                                logger.error(
                                    f"Incomplete data: {len(gt_dict)} GT, {len(target_dict)} target, {len(pred_dict)} pred"
                                )
                                return

                            # Precompute groundtruth images
                            groundtruth_images = precompute_images(gt_dict)

                            # Cache the data and trigger display-only mode
                            st.session_state["prediction_data_cache"] = {
                                "date": selected_date,
                                "time": selected_time,
                                "model": selected_model,
                                "groundtruth_images": groundtruth_images,
                                "target_dict": target_dict,
                                "pred_dict": pred_dict,
                            }
                            st.session_state["show_prediction_results"] = True
                            logger.info(f"Caching and displaying prediction for {selected_model}")
                            st.rerun()

                        except Exception as e:
                            st.error(f"❌ Error loading data: {e}")
                            logger.error(f"Error loading data: {e}")
                            import traceback

                            logger.error(traceback.format_exc())

            with col2:
                if st.button("🔄 Recompute", key="recompute_pred"):
                    # Delete existing prediction and recompute
                    st.info("🗑️ Deleting old prediction...")
                    from nwc_webapp.pages.nowcasting_utils import delete_predictions_in_range

                    deleted = delete_predictions_in_range(selected_model, selected_datetime, selected_datetime)
                    if deleted > 0:
                        logger.info(f"Deleted {deleted} prediction(s) for {selected_model}")

                    # Set flag to trigger job submission and rerun
                    pred_exists = False  # Set to False to trigger job submission
                    st.rerun()

            # If no action taken yet, wait for user decision
            return

        if not pred_exists:
            # Prediction doesn't exist - check target availability first
            st.warning(f"⚠️ Prediction not found for {selected_model} at {selected_datetime.strftime('%d/%m/%Y %H:%M')}")

            # Check if target data exists (needed for difference images)
            from nwc_webapp.pages.nowcasting_utils import check_target_data_exists

            targets_exist, found_count, total_count = check_target_data_exists(selected_datetime)

            if not targets_exist:
                # Target data is missing - warn user
                st.warning(
                    f"⚠️ **Target Data Incomplete**\n\n"
                    f"Only {found_count}/{total_count} target frames are available for this time.\n\n"
                    f"Target data (groundtruth from {(selected_datetime + timedelta(minutes=5)).strftime('%H:%M')} to {(selected_datetime + timedelta(minutes=60)).strftime('%H:%M')}) is needed to compute difference images. "
                    f"Without it, only prediction images will be displayed.\n\n"
                    f"**Do you want to compute the prediction anyway?**"
                )

                col1, col2, _ = st.columns([1, 1, 2])

                with col1:
                    if st.button("✅ YES, Proceed", key="target_warning_yes", width='stretch'):
                        st.session_state.target_warning_accepted = True
                        st.rerun()

                with col2:
                    if st.button("❌ NO, Cancel", key="target_warning_no", width='stretch'):
                        st.session_state.target_warning_accepted = False
                        st.info("Operation cancelled. Please select a different time.")
                        return

                # If user hasn't decided yet, stop here
                if not st.session_state.get("target_warning_accepted", False):
                    return

            # Clear warning acceptance for next run
            st.session_state.target_warning_accepted = False

            st.info("📝 Submitting job to compute prediction...")

            # Submit job with start_date == end_date (single prediction)
            with st.spinner(f"Submitting job for {selected_model}..."):
                job_id = submit_date_range_prediction_job(selected_model, selected_datetime, selected_datetime)

            if not job_id:
                st.error("❌ Failed to submit prediction job. Check logs for details.")
                return

            st.success(f"✅ Job submitted successfully! Job ID: {job_id}")
            st.write(f"**Model**: {selected_model}")
            st.write(f"**DateTime**: {selected_datetime.strftime('%d/%m/%Y %H:%M')}")

            st.markdown("---")

            # Check if this is a mock job (local mode)
            if not is_hpc() or job_id.startswith("mock_"):
                # Local mode: predictions already created, skip monitoring
                logger.info("🖥️  Local mode detected - skipping job monitoring")
                st.success("✅ Mock prediction created instantly!")

                # Wait a moment for filesystem sync
                time.sleep(1)

                # Load and display the data
                with st.spinner("Loading data..."):
                    try:
                        gt_dict, target_dict, pred_dict = load_single_prediction_data(selected_model, selected_datetime)

                        if not gt_dict or not target_dict or not pred_dict:
                            st.error("❌ Failed to load data after prediction creation.")
                            return

                        # Precompute groundtruth images
                        groundtruth_images = precompute_images(gt_dict)

                        # Cache the data and trigger display-only mode
                        st.session_state["prediction_data_cache"] = {
                            "date": selected_date,
                            "time": selected_time,
                            "model": selected_model,
                            "groundtruth_images": groundtruth_images,
                            "target_dict": target_dict,
                            "pred_dict": pred_dict,
                        }
                        st.session_state["show_prediction_results"] = True
                        logger.info(f"Caching and displaying prediction for {selected_model}")
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ Error loading data: {e}")
                        logger.error(f"Error loading data: {e}")

                return

            # HPC mode: Monitor job progress
            st.subheader("📊 Job Progress")

            # Add CSS for animated dots
            st.markdown(
                """
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
            """,
                unsafe_allow_html=True,
            )

            # Create placeholders for dynamic updates
            status_placeholder = st.empty()

            # Get output folder path
            config = get_config()
            out_folder_path = config.real_time_pred / selected_model

            # Monitor job status
            from nwc_webapp.hpc.pbs import get_model_job_status, is_pbs_available

            max_iterations = 1800  # 1 hour max (1800 * 2 second checks)
            iteration = 0
            last_status = None
            job_completed = False
            consecutive_none_count = 0

            while iteration < max_iterations and not job_completed:
                # Check job status (only if PBS is available)
                current_status = None
                if is_pbs_available():
                    try:
                        current_status = get_model_job_status(selected_model)
                        logger.debug(f"Job status check: {current_status}")
                    except Exception as e:
                        logger.error(f"Error checking job status: {e}")
                        current_status = None

                # Update status display when status changes
                if current_status != last_status and current_status is not None:
                    if current_status == "Q":
                        status_placeholder.markdown(
                            "⏳ **Job in <span class='queue-text'>queue</span>**", unsafe_allow_html=True
                        )
                        logger.info(f"Job {job_id} is in queue")
                    elif current_status == "R":
                        status_placeholder.markdown(
                            f"⚙️ **Job <span class='running-text'>running</span>**<br>Results will be saved in `{out_folder_path}`",
                            unsafe_allow_html=True,
                        )
                        logger.info(f"Job {job_id} is running")
                    consecutive_none_count = 0

                # Check if job disappeared from queue (completed or failed)
                if last_status is not None and current_status is None:
                    consecutive_none_count += 1
                    logger.debug(f"Job status None (consecutive: {consecutive_none_count}/3)")

                    if consecutive_none_count >= 3:
                        logger.info(f"Job {job_id} disappeared from queue - doing final verification")
                        job_completed = True

                        # Wait for filesystem sync
                        status_placeholder.info("⏳ **Job finished - verifying results...**")
                        time.sleep(5)

                        # Check if prediction was created
                        pred_exists = check_single_prediction_exists(selected_model, selected_datetime)

                        if not pred_exists:
                            # Error occurred - prediction not created
                            status_placeholder.error("❌ **Error doing prediction!**")

                            # Try to read PBS output log
                            home_dir = os.path.expanduser("~")
                            log_file = os.path.join(home_dir, f"nwc_{selected_model}_range.o{job_id}")

                            logger.error(f"Job failed - prediction file not created")
                            logger.info(f"Looking for log file: {log_file}")

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
                                st.info("Check PBS logs manually in your home directory")
                        else:
                            # Success - prediction created
                            status_placeholder.success("✅ **Prediction completed!**")

                            # Wait a moment to show success message
                            time.sleep(1)

                            # Load and display the data
                            try:
                                gt_dict, target_dict, pred_dict = load_single_prediction_data(
                                    selected_model, selected_datetime
                                )

                                if not gt_dict or not target_dict or not pred_dict:
                                    st.error("❌ Failed to load data after prediction creation.")
                                    return

                                # Precompute groundtruth images
                                groundtruth_images = precompute_images(gt_dict)

                                # Cache the data and trigger display-only mode
                                st.session_state["prediction_data_cache"] = {
                                    "date": selected_date,
                                    "time": selected_time,
                                    "model": selected_model,
                                    "groundtruth_images": groundtruth_images,
                                    "target_dict": target_dict,
                                    "pred_dict": pred_dict,
                                }
                                st.session_state["show_prediction_results"] = True
                                logger.info(f"Caching and displaying prediction for {selected_model}")
                                st.rerun()

                            except Exception as e:
                                st.error(f"❌ Error loading data: {e}")
                                logger.error(f"Error loading data: {e}")

                        break
                else:
                    consecutive_none_count = 0

                # Update last_status for next iteration
                if current_status is not None:
                    last_status = current_status

                # Wait before next check
                time.sleep(2)
                iteration += 1

            # Check if we hit the timeout
            if iteration >= max_iterations and not job_completed:
                status_placeholder.error("⏰ **Timeout!** Job took too long to complete.")
                st.error("The job has been running for over 1 hour. Please check manually.")
                logger.error(f"Job {job_id} timeout after {iteration} iterations")