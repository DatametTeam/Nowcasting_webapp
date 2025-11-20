"""
Nowcasting page - main prediction interface.
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
    Main nowcasting page with prediction submission and visualization.

    New workflow:
    1. Check if date is before Jan 1, 2025 → show warning
    2. Check if GIFs already exist → load and display
    3. Check if real-time predictions exist → create GIFs from them
    4. Submit prediction job → create GIFs from results

    Args:
        sidebar_args: Dictionary with sidebar configuration
        sri_folder_dir: Path to SRI folder directory
    """
    # Check if form was submitted
    if not sidebar_args.get('submitted', False):
        st.info("👈 Please select a model, date, and time from the sidebar, then click 'Submit'")
        return

    # Extract parameters
    model_name = sidebar_args['model_name']
    start_date = sidebar_args['start_date']
    start_time = sidebar_args['start_time']

    # Combine date and time
    selected_datetime = datetime.combine(start_date, start_time)
    date_str = selected_datetime.strftime('%d%m%Y')  # DDMMYYYY format
    time_str = selected_datetime.strftime('%H%M')

    # Step 1: Date validation for training data
    if is_training_date(selected_datetime):
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
            st.info("Operation cancelled. Please select a date after 1st January 2025.")
            return

        # If we reach here, user has accepted the warning
        logger.info(f"User accepted training date warning for {date_str}_{time_str}")

    # Step 2: Check if GIFs already exist
    gif_paths = get_gif_paths(model_name, date_str, time_str)
    gt_exist, pred_exist, diff_exist = check_gifs_exist(gif_paths)

    if gt_exist and pred_exist and diff_exist:
        logger.info(f"Found existing GIFs for {model_name} at {date_str}_{time_str}")

        # Ask user if they want to recompute
        st.success("✅ Prediction GIFs found!")
        st.info("Do you want to load existing GIFs or recompute?")

        col1, col2, _ = st.columns([1, 1, 3])

        with col1:
            if st.button("📂 Load Existing", key="load_existing", use_container_width=True):
                # Load and display existing GIFs
                gif_paths_list = [
                    gif_paths['gt_t0'],
                    gif_paths['gt_t6'],
                    gif_paths['gt_t12'],
                ]
                pred_paths_list = [
                    gif_paths['pred_t6'],
                    gif_paths['pred_t12'],
                ]
                diff_paths_list = [
                    gif_paths['diff_t6'],
                    gif_paths['diff_t12'],
                ]

                gt_gifs = load_gif_as_bytesio(gif_paths_list)
                pred_gifs = load_gif_as_bytesio(pred_paths_list)
                diff_gifs = load_gif_as_bytesio(diff_paths_list)

                display_results(gt_gifs, pred_gifs, diff_gifs)
                return

        with col2:
            if st.button("🔄 Recompute", key="recompute", use_container_width=True):
                # Continue to recompute
                pass
            else:
                return  # Wait for user decision

    # Step 3: Check if real-time predictions exist
    if check_realtime_prediction_exists(model_name, date_str, time_str):
        logger.info(f"Found real-time predictions for {model_name} at {date_str}_{time_str}")
        st.info("🔍 Found real-time prediction data. Creating GIFs...")

        # Load real-time prediction data
        pred_data = load_realtime_prediction(model_name, date_str, time_str)

        if pred_data is not None:
            # Create GIFs from real-time data
            create_gifs_from_realtime_data(
                pred_data=pred_data,
                sidebar_args=sidebar_args,
                sri_folder_dir=sri_folder_dir,
                gif_paths=gif_paths,
            )
            return
        else:
            st.error("Failed to load real-time prediction data. Will submit new prediction job.")

    # Step 4: No GIFs or predictions found → submit job
    logger.info(f"No existing data found for {model_name} at {date_str}_{time_str}. Submitting job...")
    st.info("📝 No existing predictions found. Submitting new prediction job...")

    compute_prediction_results(sidebar_args, sri_folder_dir)