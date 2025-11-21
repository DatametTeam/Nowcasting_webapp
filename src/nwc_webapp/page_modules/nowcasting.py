"""
Nowcasting page - main prediction interface with date range support.
"""
from datetime import datetime
from pathlib import Path
import streamlit as st

from nwc_webapp.page_modules.nowcasting_utils import (
    is_training_date,
    get_gif_paths,
    check_gifs_exist,
    check_realtime_prediction_exists,
    load_realtime_prediction,
    check_missing_predictions,
    get_missing_range,
    submit_date_range_prediction_job,
    create_gifs_from_prediction_range,
)
from nwc_webapp.prediction.visualization import (
    compute_prediction_results,
    display_results,
    update_prediction_visualization,
    create_gifs_from_realtime_data,
)
from nwc_webapp.utils import load_gif_as_bytesio
from nwc_webapp.logging_config import setup_logger

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
        if st.button("✅ YES, Proceed", key="training_yes", use_container_width=True):
            st.session_state.training_warning_accepted = True
            return True

    with col2:
        if st.button("❌ NO, Cancel", key="training_no", use_container_width=True):
            st.session_state.training_warning_accepted = False
            st.info("Operation cancelled. Please select a different date.")
            return False

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
    # Check if form was submitted or if we have cached params from previous submission
    if sidebar_args.get('submitted', False):
        # Store params in session state for persistence across reruns
        st.session_state['nowcasting_params'] = {
            'model_name': sidebar_args['model_name'],
            'start_date': sidebar_args['start_date'],
            'start_time': sidebar_args['start_time'],
            'end_date': sidebar_args['end_date'],
            'end_time': sidebar_args['end_time'],
        }

    # Use cached params if available
    if 'nowcasting_params' not in st.session_state:
        st.info("👈 Please select a model, start/end date and time from the sidebar, then click 'Submit'")
        return

    # Extract parameters from session state
    params = st.session_state['nowcasting_params']
    model_name = params['model_name']
    start_date = params['start_date']
    start_time = params['start_time']
    end_date = params['end_date']
    end_time = params['end_time']

    # Combine date and time
    start_dt = datetime.combine(start_date, start_time)
    end_dt = datetime.combine(end_date, end_time)

    # Validate date range
    if end_dt < start_dt:
        st.error("❌ End date/time must be after start date/time!")
        return

    logger.info(f"Nowcasting request: {model_name} from {start_dt} to {end_dt}")

    # Step 1: Date validation for training data
    if is_training_date(start_dt) or is_training_date(end_dt):
        # Check if warning was already shown and accepted in this session
        if 'training_warning_accepted' not in st.session_state:
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

    if not missing_timestamps:
        # All predictions exist
        st.success(f"✅ All {total_count} predictions exist for {model_name}")
        st.info(f"**Range**: {start_dt.strftime('%d/%m/%Y %H:%M')} to {end_dt.strftime('%d/%m/%Y %H:%M')}")

        # Ask if they want to recompute
        col1, col2, _ = st.columns([1, 1, 3])

        with col1:
            if st.button("📊 Create GIFs", key="create_gifs", use_container_width=True):
                st.info("🎬 Creating sliding window GIFs from predictions...")

                with st.spinner("Creating GIFs..."):
                    success = create_gifs_from_prediction_range(model_name, start_dt, end_dt, sri_folder_dir)

                if success:
                    st.success("✅ GIFs created successfully!")
                else:
                    st.error("❌ Failed to create GIFs. Check logs for details.")
                return

        with col2:
            if st.button("🔄 Recompute All", key="recompute_all", use_container_width=True):
                # Continue to recompute
                pass
            else:
                return  # Wait for user decision

    elif not existing_timestamps:
        # No predictions exist
        st.warning(f"⚠️ No predictions found for {model_name} in this range")
        st.info(f"**Missing**: {total_count} predictions from {start_dt.strftime('%d/%m/%Y %H:%M')} to {end_dt.strftime('%d/%m/%Y %H:%M')}")

        # Center the button in a half-width column
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            # Custom CSS for taller button
            st.markdown("""
                <style>
                .stButton > button {
                    height: 60px;
                    font-weight: bold;
                    font-size: 18px;
                }
                </style>
            """, unsafe_allow_html=True)

            compute_clicked = st.button("▶️ Compute Predictions", key="compute_missing",
                                       use_container_width=True, type="primary")

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
        st.markdown("""
            <style>
            div[data-testid="column"] .stButton > button {
                height: 60px;
                font-weight: bold;
                font-size: 18px;
            }
            </style>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            compute_missing = st.button("▶️ Compute Missing", key="compute_partial",
                                       use_container_width=True, type="primary")

        with col2:
            recompute_all = st.button("🔄 Recompute All", key="recompute_all_partial",
                                     use_container_width=True, type="primary")

        if not (compute_missing or recompute_all):
            return  # Wait for user decision

    # Step 3: Submit PBS job for the range
    st.info(f"🚀 Submitting PBS job for {model_name}...")

    with st.spinner(f"Submitting job for {model_name} (range: {start_dt} to {end_dt})..."):
        job_id = submit_date_range_prediction_job(model_name, start_dt, end_dt)

    if job_id:
        st.success(f"✅ Job submitted successfully! Job ID: {job_id}")
        st.write(f"**Model**: {model_name}")
        st.write(f"**Range**: {start_dt.strftime('%d/%m/%Y %H:%M')} to {end_dt.strftime('%d/%m/%Y %H:%M')}")
        st.write(f"**Total predictions**: {total_count}")

        st.markdown("---")

        # Monitor job progress
        st.subheader("📊 Job Progress")

        # Create placeholders for dynamic updates
        status_placeholder = st.empty()
        progress_placeholder = st.empty()
        details_placeholder = st.empty()

        # Get output folder path
        from nwc_webapp.config.config import get_config
        config = get_config()
        out_folder_path = config.real_time_pred / model_name

        # Monitor job status and progress
        import time
        import os
        from nwc_webapp.services.pbs import get_model_job_status, is_pbs_available

        max_iterations = 1800  # 1 hour max (1800 * 2 second checks)
        iteration = 0
        last_status = None

        while iteration < max_iterations:
            # Check job status
            if is_pbs_available():
                current_status = get_model_job_status(model_name)

                if current_status != last_status:
                    if current_status == 'Q':
                        status_placeholder.info("⏳ **Job in queue!**")
                    elif current_status == 'R':
                        status_placeholder.success(f"⚙️ **Job running!** Results will be saved in `{out_folder_path}`")

                    last_status = current_status

                # If job disappeared from queue, it's done
                if last_status and not current_status:
                    logger.info(f"Job {job_id} disappeared from queue - doing final check")

                    # Wait 5 seconds for filesystem sync
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
                            st.error(f"**PBS Job Log** (`{log_file}`):")
                            try:
                                with open(log_file, 'r') as f:
                                    log_content = f.read()
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
                        st.info("You can now create GIFs from the predictions by clicking Submit again.")

                    break

            # Check prediction progress (count existing files) every 2 seconds
            missing_timestamps, existing_timestamps = check_missing_predictions(model_name, start_dt, end_dt)
            completed = len(existing_timestamps)
            progress = completed / total_count if total_count > 0 else 0

            # Update progress bar
            progress_placeholder.progress(progress, text=f"Predictions: {completed}/{total_count}")

            # Show details only when job is running
            if last_status == 'R' and completed > 0:
                details_placeholder.write(f"✅ {completed} prediction(s) completed, {len(missing_timestamps)} remaining")

            time.sleep(2)  # Check every 2 seconds
            iteration += 1

        if iteration >= max_iterations:
            st.warning("⚠️ Monitoring timeout reached (1 hour). Check job status manually.")

    else:
        st.error(f"❌ Failed to submit job for {model_name}. Check logs for details.")
        logger.error(f"Job submission failed for {model_name}")