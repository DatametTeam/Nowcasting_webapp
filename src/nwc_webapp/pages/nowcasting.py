"""
Nowcasting page - main prediction interface with date range support.
"""

from datetime import datetime, timedelta
from pathlib import Path

import streamlit as st

from nwc_webapp.logging_config import setup_logger
from nwc_webapp.pages.nowcasting_utils import (
    check_gifs_exist,
    check_missing_predictions,
    check_target_data_for_range,
    create_gifs_from_prediction_range,
    delete_predictions_in_range,
    get_gif_paths,
    get_missing_range,
    is_training_date,
    submit_date_range_prediction_job,
)
from nwc_webapp.rendering.visualization import (
    compute_prediction_results,
    create_gifs_from_realtime_data,
    display_results,
    update_prediction_visualization,
)
from nwc_webapp.data.gifs import load_gif_as_bytesio

# Set up logger
logger = setup_logger(__name__)


def load_and_display_gifs(gif_paths, model_name=None, start_dt=None, end_dt=None):
    """
    Load GIFs from paths and display them using the visualization layout.
    Triggers clear page mode to show only GIFs.

    Args:
        gif_paths: Dictionary with GIF paths from get_gif_paths()
        model_name: Model name (for caching parameters)
        start_dt: Start datetime (for caching parameters)
        end_dt: End datetime (for caching parameters)
    """
    # Cache the GIF paths and trigger "show only GIFs" mode
    st.session_state["gif_paths_cache"] = gif_paths
    st.session_state["show_gifs_only"] = True

    # Store cache parameters if provided (for persistence across tab switches)
    if model_name and start_dt and end_dt:
        st.session_state["gif_cache_params"] = {
            "model_name": model_name,
            "start_dt": start_dt,
            "end_dt": end_dt,
        }

    # Trigger rerun to display only GIFs
    st.rerun()


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
        if st.button("✅ YES, Proceed", key="training_yes", width='stretch'):
            st.session_state.training_warning_accepted = True
            return True

    with col2:
        if st.button("❌ NO, Cancel", key="training_no", width='stretch'):
            st.session_state.training_warning_accepted = False
            st.info("Operation cancelled. Please select a different date.")
            return False

    return False


def show_missing_target_data_warning(start_dt: datetime, end_dt: datetime, missing_count: int, total_count: int, key_suffix: str = "") -> bool:
    """
    Show warning dialog when target data is missing.

    Args:
        start_dt: Start datetime
        end_dt: End datetime
        missing_count: Number of timestamps with missing target data
        total_count: Total number of timestamps
        key_suffix: Suffix for button keys to avoid duplicates

    Returns:
        True if user wants to proceed, False otherwise
    """
    first_missing = start_dt + timedelta(minutes=30)  # Target data starts at +30min
    last_missing = end_dt + timedelta(minutes=60)  # Target data ends at +60min

    st.warning(
        f"⚠️ **Missing Target Data**\n\n"
        f"Target data for **{missing_count}/{total_count}** timestamps is not available.\n\n"
        f"**Range affected**: {first_missing.strftime('%d/%m/%Y %H:%M')} to {last_missing.strftime('%d/%m/%Y %H:%M')}\n\n"
        f"If you proceed:\n"
        f"- Target GIFs will show **empty frames** for missing data\n"
        f"- Difference GIFs will show **empty frames** for missing data\n"
        f"- Prediction GIFs will be created normally\n\n"
        f"**Do you want to proceed?**"
    )

    col1, col2, _ = st.columns([1, 1, 3])

    with col1:
        if st.button("✅ YES, Continue", key=f"target_yes_{key_suffix}", width='stretch'):
            st.session_state[f"target_warning_accepted_{key_suffix}"] = True
            return True

    with col2:
        if st.button("❌ NO, Cancel", key=f"target_no_{key_suffix}", width='stretch'):
            st.session_state[f"target_warning_accepted_{key_suffix}"] = False
            st.info("Operation cancelled.")
            return False

    return False


def check_and_warn_missing_target_data(start_dt: datetime, end_dt: datetime, key_suffix: str = "") -> bool:
    """
    Check if target data exists and show warning if missing.

    Args:
        start_dt: Start datetime
        end_dt: End datetime
        key_suffix: Suffix for session state keys to avoid duplicates

    Returns:
        True if should proceed with GIF creation, False otherwise
    """
    # Check if target data exists
    all_target_exist, missing_target_timestamps, existing_target_timestamps = check_target_data_for_range(start_dt, end_dt)

    if all_target_exist:
        # All target data exists - proceed
        logger.info("✅ All target data available")
        return True

    # Some target data is missing - show warning
    total_count = len(missing_target_timestamps) + len(existing_target_timestamps)
    missing_count = len(missing_target_timestamps)

    logger.warning(f"⚠️  Missing target data for {missing_count}/{total_count} timestamps")

    # Check if warning was already shown and accepted
    session_key = f"target_warning_accepted_{key_suffix}"
    if session_key not in st.session_state:
        st.session_state[session_key] = None

    # Show warning if not yet decided
    if st.session_state[session_key] is None:
        # Show info message about missing data
        st.info(
            f"ℹ️ **Target data not available for {missing_count}/{total_count} timestamps**\n\n"
            f"Missing range: {(start_dt + timedelta(minutes=30)).strftime('%d/%m/%Y %H:%M')} to "
            f"{(end_dt + timedelta(minutes=60)).strftime('%d/%m/%Y %H:%M')}"
        )
        user_decision = show_missing_target_data_warning(start_dt, end_dt, missing_count, total_count, key_suffix)
        return user_decision
    elif st.session_state[session_key]:
        # User previously accepted - proceed
        logger.info(f"User accepted missing target data for range {start_dt} to {end_dt}")
        return True
    else:
        # User previously declined
        st.info("Operation cancelled. Target data is required for complete GIF creation.")
        return False


def main_page(sidebar_args, sri_folder_dir) -> None:
    """
    Main nowcasting page with date range prediction support.

    Workflow:
    1. Check if date range is before Jan 1, 2025 → show warning
    2. Check which predictions exist in the range
    3. Show smart dialog: "Predictions from X to Y exist. Missing from A to B. Recompute? YES/NO"
    4. Submit job for date range if needed
    5. Create sliding window GIFs from all predictions

    Args:
        sidebar_args: Dictionary with sidebar configuration (includes start/end dates)
        sri_folder_dir: Path to SRI folder directory
    """
    # Check if we should display GIFs only (clear page mode)
    if st.session_state.get("show_gifs_only", False) and st.session_state.get("gif_paths_cache"):
        # Display only GIFs, skip all other UI
        gif_paths = st.session_state["gif_paths_cache"]

        try:
            # Load GIFs from disk into BytesIO objects
            gt_gifs = load_gif_as_bytesio(
                [
                    gif_paths["gt_t0"],  # Groundtruth
                    gif_paths["gt_t6"],  # Target +30
                    gif_paths["gt_t12"],  # Target +60
                ]
            )

            pred_gifs = load_gif_as_bytesio(
                [
                    gif_paths["pred_t6"],  # Prediction +30
                    gif_paths["pred_t12"],  # Prediction +60
                ]
            )

            diff_gifs = load_gif_as_bytesio(
                [
                    gif_paths["diff_t6"],  # Difference +30
                    gif_paths["diff_t12"],  # Difference +60
                ]
            )

            # Display only the GIFs
            display_results(gt_gifs, pred_gifs, diff_gifs)

            # Add a button to go back
            if st.button("🔙 Back to Prediction Setup", width='stretch'):
                st.session_state["show_gifs_only"] = False
                st.rerun()

            return  # Don't show any other UI

        except Exception as e:
            logger.error(f"Error displaying GIFs: {e}")
            st.error(f"❌ Error displaying GIFs: {e}")
            st.session_state["show_gifs_only"] = False
            st.rerun()
            return

    # Check if form was submitted or if we have cached params from previous submission
    if sidebar_args.get("submitted", False):
        # Store params in session state for persistence across reruns
        st.session_state["nowcasting_params"] = {
            "model_name": sidebar_args["model_name"],
            "start_date": sidebar_args["start_date"],
            "start_time": sidebar_args["start_time"],
            "end_date": sidebar_args["end_date"],
            "end_time": sidebar_args["end_time"],
        }

    # Use cached params if available
    if "nowcasting_params" not in st.session_state:
        st.info("👈 Please select a model, start/end date and time from the sidebar, then click 'Submit'")
        return

    # Extract parameters from session state
    params = st.session_state["nowcasting_params"]
    model_name = params["model_name"]
    start_date = params["start_date"]
    start_time = params["start_time"]
    end_date = params["end_date"]
    end_time = params["end_time"]

    # Combine date and time
    start_dt = datetime.combine(start_date, start_time)
    end_dt = datetime.combine(end_date, end_time)

    # Check if we have cached GIFs that match current selection
    # This allows results to persist across tab switches
    if "gif_paths_cache" in st.session_state and st.session_state.get("gif_cache_params"):
        cached_params = st.session_state["gif_cache_params"]
        if (cached_params["model_name"] == model_name and
            cached_params["start_dt"] == start_dt and
            cached_params["end_dt"] == end_dt):
            # We have cached GIFs for this exact selection - offer to display them
            st.success(f"✅ GIFs available for {model_name}: {start_dt.strftime('%d/%m/%Y %H:%M')} to {end_dt.strftime('%d/%m/%Y %H:%M')}")

            col1, col2, _ = st.columns([1, 1, 2])
            with col1:
                if st.button("📺 Display Cached GIFs", key="display_cached_gifs", width='stretch', type="primary"):
                    # Show the cached GIFs
                    gif_paths = st.session_state["gif_paths_cache"]
                    load_and_display_gifs(gif_paths, model_name, start_dt, end_dt)
                    return

            with col2:
                if st.button("🔄 Recompute", key="recompute_from_cache", width='stretch'):
                    # Clear cache and continue to recompute
                    st.session_state.pop("gif_paths_cache", None)
                    st.session_state.pop("gif_cache_params", None)
                    st.rerun()

            return  # Don't show the rest of the UI

    # Validate date range
    if end_dt < start_dt:
        st.error("❌ End date/time must be after start date/time!")
        return

    logger.info(f"Nowcasting request: {model_name} from {start_dt} to {end_dt}")

    # Step 1: Date validation for training data
    if is_training_date(start_dt) or is_training_date(end_dt):
        # Check if warning was already shown and accepted in this session
        if "training_warning_accepted" not in st.session_state:
            st.session_state.training_warning_accepted = None

        # Show warning and wait for user decision
        if st.session_state.training_warning_accepted is None:
            user_decision = show_training_date_warning()
            if not user_decision:
                return  # Stop if user cancels
        elif not st.session_state.training_warning_accepted:
            # User previously declined
            st.info("Operation cancelled. Please select dates after 1st January 2025.")
            return

        # If we reach here, user has accepted the warning
        logger.info(f"User accepted training date warning for range {start_dt} to {end_dt}")

    # Step 2: Check which predictions exist in the range
    missing_timestamps, existing_timestamps = check_missing_predictions(model_name, start_dt, end_dt)

    # Display status to user
    total_count = len(missing_timestamps) + len(existing_timestamps)

    # Track if user wants to recompute (delete old predictions)
    should_delete_old = False

    if not missing_timestamps:
        # All predictions exist
        st.success(f"✅ All {total_count} predictions exist for {model_name}")
        st.info(f"**Range**: {start_dt.strftime('%d/%m/%Y %H:%M')} to {end_dt.strftime('%d/%m/%Y %H:%M')}")

        # Initialize session state for GIF action tracking
        if "gif_action" not in st.session_state:
            st.session_state["gif_action"] = None

        # Check if we're showing the recompute/display dialog
        if st.session_state["gif_action"] == "show_dialog":
            # GIFs already exist - show dialog
            gif_paths = get_gif_paths(model_name, start_dt, end_dt)

            st.warning("⚠️ **GIFs already exist!** Do you want to recompute them or display the existing ones?")

            # Custom CSS for taller, more readable buttons
            st.markdown(
                """
                <style>
                div[data-testid="column"] .stButton > button {
                    height: 80px;
                    font-weight: bold;
                    font-size: 18px;
                    white-space: normal;
                    word-wrap: break-word;
                }
                </style>
            """,
                unsafe_allow_html=True,
            )

            col_recompute, col_display, col_cancel = st.columns([1, 1, 1])

            with col_recompute:
                if st.button("🔄 Recompute\nGIFs", key="recompute_gifs_yes", width='stretch', type="primary"):
                    st.session_state["gif_action"] = "recompute"
                    st.rerun()

            with col_display:
                if st.button("📺 Display\nExisting", key="recompute_gifs_no", width='stretch', type="primary"):
                    st.session_state["gif_action"] = "display"
                    st.rerun()

            with col_cancel:
                if st.button("❌ Cancel", key="recompute_gifs_cancel", width='stretch'):
                    st.session_state["gif_action"] = None
                    st.rerun()

            return

        # Handle recompute action
        if st.session_state["gif_action"] == "recompute":
            st.info("🗑️ Deleting old GIFs and creating new ones...")
            gif_paths = get_gif_paths(model_name, start_dt, end_dt)

            # Delete old GIFs
            for path in [
                gif_paths["gt_t0"],
                gif_paths["gt_t6"],
                gif_paths["gt_t12"],
                gif_paths["pred_t6"],
                gif_paths["pred_t12"],
                gif_paths["diff_t6"],
                gif_paths["diff_t12"],
            ]:
                if path.exists():
                    path.unlink()
                    logger.info(f"Deleted old GIF: {path}")

            with st.spinner("Creating GIFs..."):
                gif_paths = create_gifs_from_prediction_range(model_name, start_dt, end_dt, sri_folder_dir)

            # Reset action state
            st.session_state["gif_action"] = None

            if gif_paths:
                st.success("✅ GIFs created successfully!")
                # Display the GIFs
                load_and_display_gifs(gif_paths, model_name, start_dt, end_dt)
            else:
                st.error("❌ Failed to create GIFs. Check logs for details.")
            return

        # Handle display existing action
        if st.session_state["gif_action"] == "display":
            st.success("📺 Loading existing GIFs...")
            gif_paths = get_gif_paths(model_name, start_dt, end_dt)

            # Reset action state
            st.session_state["gif_action"] = None

            # Display the GIFs
            load_and_display_gifs(gif_paths, model_name, start_dt, end_dt)
            return

        # Show initial buttons
        col1, col2, _ = st.columns([1, 1, 3])

        with col1:
            if st.button("📊 Create GIFs", key="create_gifs", width='stretch'):
                # Check if GIFs already exist
                gif_paths = get_gif_paths(model_name, start_dt, end_dt)
                gt_exist, pred_exist, diff_exist = check_gifs_exist(gif_paths)

                if gt_exist or pred_exist or diff_exist:
                    # GIFs already exist - show dialog next time
                    st.session_state["gif_action"] = "show_dialog"
                    st.rerun()
                else:
                    # GIFs don't exist - create them
                    st.info("🎬 Creating sliding window GIFs from predictions...")

                    with st.spinner("Creating GIFs..."):
                        gif_paths = create_gifs_from_prediction_range(model_name, start_dt, end_dt, sri_folder_dir)

                    if gif_paths:
                        st.success("✅ GIFs created successfully!")
                        # Display the GIFs
                        load_and_display_gifs(gif_paths, model_name, start_dt, end_dt)
                    else:
                        st.error("❌ Failed to create GIFs. Check logs for details.")
                    return

        with col2:
            if st.button("🔄 Recompute All", key="recompute_all", width='stretch'):
                # Continue to recompute - will delete old predictions
                should_delete_old = True
            else:
                return  # Wait for user decision

    elif not existing_timestamps:
        # No predictions exist
        st.warning(f"⚠️ No predictions found for {model_name} in this range")
        st.info(
            f"**Missing**: {total_count} predictions from {start_dt.strftime('%d/%m/%Y %H:%M')} to {end_dt.strftime('%d/%m/%Y %H:%M')}"
        )

        # Center the button in a half-width column
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            # Custom CSS for taller button
            st.markdown(
                """
                <style>
                .stButton > button {
                    height: 60px;
                    font-weight: bold;
                    font-size: 18px;
                }
                </style>
            """,
                unsafe_allow_html=True,
            )

            compute_clicked = st.button(
                "▶️ Compute Predictions", key="compute_missing", width='stretch', type="primary"
            )

        if not compute_clicked:
            return  # Wait for user decision

    else:
        # Some predictions exist, some are missing
        first_missing, last_missing = get_missing_range(missing_timestamps)

        st.info(f"📊 **Prediction Status for {model_name}**")
        st.write(f"✅ **Existing**: {len(existing_timestamps)}/{total_count} predictions")
        st.write(f"❌ **Missing**: {len(missing_timestamps)}/{total_count} predictions")
        st.write(f"   └─ From {first_missing.strftime('%d/%m/%Y %H:%M')} to {last_missing.strftime('%d/%m/%Y %H:%M')}")

        # Custom CSS for taller buttons
        st.markdown(
            """
            <style>
            div[data-testid="column"] .stButton > button {
                height: 60px;
                font-weight: bold;
                font-size: 18px;
            }
            </style>
        """,
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            compute_missing = st.button(
                "▶️ Compute Missing", key="compute_partial", width='stretch', type="primary"
            )

        with col2:
            recompute_all = st.button(
                "🔄 Recompute All", key="recompute_all_partial", width='stretch', type="primary"
            )

        if not (compute_missing or recompute_all):
            return  # Wait for user decision

        # If recompute all is clicked, set flag to delete old predictions
        if recompute_all:
            should_delete_old = True

    # Step 3: Delete old predictions if recompute all was clicked
    if should_delete_old:
        st.info(f"🗑️ Deleting old predictions for {model_name}...")
        deleted_count = delete_predictions_in_range(model_name, start_dt, end_dt)
        if deleted_count > 0:
            st.success(f"✅ Deleted {deleted_count} old prediction(s)")
            logger.info(f"Deleted {deleted_count} old predictions for {model_name}")

        # Wait for filesystem sync
        import time

        time.sleep(2)

        # Reset the count since we deleted everything
        missing_timestamps, existing_timestamps = check_missing_predictions(model_name, start_dt, end_dt)
        total_count = len(missing_timestamps) + len(existing_timestamps)
        logger.info(f"After deletion: {len(existing_timestamps)} existing, {len(missing_timestamps)} missing")

    # Step 4: Submit PBS job for the range
    st.info(f"🚀 Submitting PBS job for {model_name}...")

    with st.spinner(f"Submitting job for {model_name} (range: {start_dt} to {end_dt})..."):
        job_id = submit_date_range_prediction_job(model_name, start_dt, end_dt)

    if job_id:
        st.success(f"✅ Job submitted successfully! Job ID: {job_id}")
        st.write(f"**Model**: {model_name}")
        st.write(f"**Range**: {start_dt.strftime('%d/%m/%Y %H:%M')} to {end_dt.strftime('%d/%m/%Y %H:%M')}")
        st.write(f"**Total predictions**: {total_count}")

        st.markdown("---")

        # Check if this is a mock job (local mode)
        from nwc_webapp.config.environment import is_hpc

        if not is_hpc() or job_id.startswith("mock_"):
            # Local mode: predictions already created, skip monitoring and create GIFs immediately
            logger.info("🖥️  Local mode detected - skipping job monitoring")
            st.success("✅ Mock predictions created instantly!")

            # Automatically start GIF creation
            st.info("🎬 Creating GIFs from predictions...")
            logger.info(f"Auto-starting GIF creation for {model_name}")

            with st.spinner("Creating GIFs..."):
                gif_paths = create_gifs_from_prediction_range(model_name, start_dt, end_dt, sri_folder_dir)

            if gif_paths:
                st.success("✅ GIFs created successfully!")
                logger.info(f"GIF creation completed for {model_name}")
                # Display the GIFs
                load_and_display_gifs(gif_paths, model_name, start_dt, end_dt)
            else:
                st.error("❌ Failed to create GIFs. Check logs for details.")
                logger.error(f"GIF creation failed for {model_name}")

            # Exit early - no need to monitor
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
        progress_placeholder = st.empty()
        details_placeholder = st.empty()

        # Get output folder path
        from nwc_webapp.config.config import get_config

        config = get_config()
        out_folder_path = config.real_time_pred / model_name

        # Monitor job status and progress
        import os
        import time

        from nwc_webapp.hpc.pbs import get_model_job_status, is_pbs_available

        max_iterations = 1800  # 1 hour max (1800 * 2 second checks)
        iteration = 0
        last_status = None
        job_completed = False
        consecutive_none_count = 0  # Track consecutive None statuses to avoid false positives

        while iteration < max_iterations and not job_completed:
            # Check job status (only if PBS is available)
            current_status = None
            if is_pbs_available():
                try:
                    current_status = get_model_job_status(model_name)
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
                # Reset consecutive None counter when we get a valid status
                consecutive_none_count = 0

            # Check if job disappeared from queue (was running, now gone)
            # Only trigger after 3 consecutive None checks to avoid false positives from transient errors
            if last_status is not None and current_status is None:
                consecutive_none_count += 1
                logger.debug(f"Job status None (consecutive: {consecutive_none_count}/3)")

                if consecutive_none_count >= 3:
                    logger.info(f"Job {job_id} disappeared from queue (was: {last_status}) - doing final verification")
                    job_completed = True

                    # Wait 5 seconds for filesystem sync
                    status_placeholder.info("⏳ **Job finished - verifying results...**")
                    time.sleep(5)

                    # Final check of predictions
                    missing_timestamps, existing_timestamps = check_missing_predictions(model_name, start_dt, end_dt)
                    completed = len(existing_timestamps)

                    if len(missing_timestamps) > 0:
                        # Error occurred - not all predictions were created
                        status_placeholder.error("❌ **Error doing prediction!**")

                        # Try to read PBS output log
                        home_dir = os.path.expanduser("~")
                        log_file = os.path.join(home_dir, f"nwc_{model_name}_range.o{job_id}")

                        logger.error(f"Job failed - missing {len(missing_timestamps)} predictions")
                        logger.info(f"Looking for log file: {log_file}")

                        if os.path.exists(log_file):
                            try:
                                with open(log_file, "r") as f:
                                    log_content = f.read()
                                # Display error log with proper formatting
                                st.markdown("---")
                                st.error("### ❌ PBS Job Error Log")
                                st.markdown(f"**Log file**: `{log_file}`")
                                # Use code block for proper formatting and readability
                                st.code(log_content, language="text")
                            except Exception as e:
                                st.warning(f"Could not read log file: {e}")
                        else:
                            st.warning(f"Could not find PBS log file at: `{log_file}`")
                            st.info(f"Check PBS logs manually in your home directory")
                    else:
                        # Success - all predictions created
                        status_placeholder.success("✅ **All predictions completed!**")
                        progress_placeholder.progress(1.0, text=f"Predictions: {total_count}/{total_count}")
                        st.success("🎉 Prediction job completed successfully!")

                        # Automatically start GIF creation
                        st.info("🎬 Creating GIFs from predictions...")
                        logger.info(f"Auto-starting GIF creation for {model_name}")

                        with st.spinner("Creating GIFs..."):
                            gif_paths = create_gifs_from_prediction_range(model_name, start_dt, end_dt, sri_folder_dir)

                        if gif_paths:
                            st.success("✅ GIFs created successfully!")
                            logger.info(f"GIF creation completed for {model_name}")
                            # Display the GIFs
                            load_and_display_gifs(gif_paths, model_name, start_dt, end_dt)
                        else:
                            st.error("❌ Failed to create GIFs. Check logs for details.")
                            logger.error(f"GIF creation failed for {model_name}")

                    break
            else:
                consecutive_none_count = 0

            # Update last_status for next iteration
            if current_status is not None:
                last_status = current_status

            # Check prediction progress (count existing files) every 2 seconds - don't log during monitoring
            missing_timestamps, existing_timestamps = check_missing_predictions(
                model_name, start_dt, end_dt, verbose=False
            )
            completed = len(existing_timestamps)
            progress = completed / total_count if total_count > 0 else 0

            # Update progress bar
            progress_placeholder.progress(progress, text=f"Predictions: {completed}/{total_count}")

            # Show details only when job is running
            if last_status == "R" and completed > 0:
                details_placeholder.write(
                    f"✅ {completed} prediction(s) completed, {len(missing_timestamps)} remaining"
                )

            time.sleep(2)  # Check every 2 seconds
            iteration += 1

        if iteration >= max_iterations:
            st.warning("⚠️ Monitoring timeout reached (1 hour). Check job status manually.")

    else:
        st.error(f"❌ Failed to submit job for {model_name}. Check logs for details.")
        logger.error(f"Job submission failed for {model_name}")
