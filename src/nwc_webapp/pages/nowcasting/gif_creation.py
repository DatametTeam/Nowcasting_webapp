import h5py
import time
import traceback
"""
GIF creation workflow - specific to nowcasting page.
Handles GIF path management, checking, and creation for prediction visualization.
"""

import io
from datetime import datetime, timedelta
from multiprocessing import Manager, Process
from pathlib import Path
from typing import Tuple

import imageio
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image

from nwc_webapp.config.config import get_config
from nwc_webapp.config.environment import is_hpc
from nwc_webapp.data.checking import generate_timestamp_range, check_target_data_for_range
from nwc_webapp.data.predictions import load_prediction_array
from nwc_webapp.logging_config import setup_logger
from nwc_webapp.rendering.figures import compute_figure_gpd

# Set up logger
logger = setup_logger(__name__)


def get_gif_paths(model_name: str, start_dt: datetime, end_dt: datetime) -> dict:
    """
    Get the paths where GIFs should be stored/loaded.

    Uses the naming convention: {start}_{end}.gif for full sequence,
    {start}_{end}_+30m.gif and {start}_{end}_+60m.gif for time offsets.

    Directory structure:
    - gif_storage/
      - groundtruth/
        - {start}_{end}.gif (full sequence: 0-55 min)
        - {start+30m}_{end+30m}.gif (from +30: 30-85 min)
        - {start+60m}_{end+60m}.gif (from +60: 60-115 min)
      - prediction/
        - {model_name}/
          - {start}_{end}_+30m.gif
          - {start}_{end}_+60m.gif
      - difference/
        - {model_name}/
          - {start}_{end}_+30m.gif
          - {start}_{end}_+60m.gif

    Args:
        model_name: Name of the prediction model
        start_dt: Start datetime
        end_dt: End datetime

    Returns:
        Dictionary with paths for all GIF types and directories
    """
    config = get_config()
    gif_base = config.gif_storage

    # Create subdirectories
    gt_dir = gif_base / "groundtruth"
    pred_dir = gif_base / "prediction" / model_name
    diff_dir = gif_base / "difference" / model_name

    # Base filename: {start}_{end} in format DD-MM-YYYY-HH-MM
    base_name = f"{start_dt.strftime('%d-%m-%Y-%H-%M')}_{end_dt.strftime('%d-%m-%Y-%H-%M')}"
    base_name_30m = f"{(start_dt + timedelta(minutes=30)).strftime('%d-%m-%Y-%H-%M')}_{(end_dt + timedelta(minutes=30)).strftime('%d-%m-%Y-%H-%M')}"
    base_name_60m = f"{(start_dt + timedelta(minutes=60)).strftime('%d-%m-%Y-%H-%M')}_{(end_dt + timedelta(minutes=60)).strftime('%d-%m-%Y-%H-%M')}"

    return {
        "gt_t0": gt_dir / f"{base_name}.gif",  # Full groundtruth sequence
        "gt_t6": gt_dir / f"{base_name_30m}.gif",  # Target +30min sequence
        "gt_t12": gt_dir / f"{base_name_60m}.gif",  # Target +60min sequence
        "pred_t6": pred_dir / f"{base_name}_+30.gif",
        "pred_t12": pred_dir / f"{base_name}_+60.gif",
        "diff_t6": diff_dir / f"{base_name}_+30.gif",
        "diff_t12": diff_dir / f"{base_name}_+60.gif",
        # Also return the directories for easy access
        "gt_dir": gt_dir,
        "pred_dir": pred_dir,
        "diff_dir": diff_dir,
    }


def check_gifs_exist(gif_paths: dict) -> Tuple[bool, bool, bool]:
    """
    Check if GIFs exist at the specified paths.

    Args:
        gif_paths: Dictionary of GIF paths from get_gif_paths()

    Returns:
        Tuple of (gt_exist, pred_exist, diff_exist) booleans
    """
    gt_exist = gif_paths["gt_t0"].exists() and gif_paths["gt_t6"].exists() and gif_paths["gt_t12"].exists()

    pred_exist = gif_paths["pred_t6"].exists() and gif_paths["pred_t12"].exists()

    diff_exist = gif_paths["diff_t6"].exists() and gif_paths["diff_t12"].exists()

    return gt_exist, pred_exist, diff_exist


def create_empty_frame(timestamp: datetime, message: str = "No Data Available"):
    """
    Create an empty/blank frame with a message.

    Args:
        timestamp: Timestamp for the frame
        message: Message to display on the empty frame

    Returns:
        Matplotlib figure with empty frame
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Add timestamp at the top
    ax.text(0.5, 0.7, timestamp.strftime("%d/%m/%Y %H:%M"),
            ha='center', va='center', fontsize=16, fontweight='bold')

    # Add "No Data Available" message
    ax.text(0.5, 0.5, message,
            ha='center', va='center', fontsize=20, color='gray', style='italic')

    # Add a light gray background
    ax.add_patch(plt.Rectangle((0.1, 0.3), 0.8, 0.5,
                                facecolor='lightgray', alpha=0.3, zorder=-1))

    return fig


def create_groundtruth_figures(all_timestamps, gt_raw_data, gt_figures, gt_found_count, gt_missing_count):
    """
    Load groundtruth data and create figures.

    If groundtruth data is missing, creates empty/blank frames with a "No Data Available" message.
    """

    config = get_config()

    for idx, timestamp in enumerate(all_timestamps, 1):
        logger.info(f"Loading groundtruth {idx}/{len(all_timestamps)}: {timestamp.strftime('%d/%m/%Y %H:%M')}")
        gt_filename = timestamp.strftime("%d-%m-%Y-%H-%M") + ".hdf"

        # Determine paths based on environment
        gt_path = None

        if is_hpc():
            # HPC: Try data1 first (recent data, faster), then data (archived)
            gt_path_data1 = Path("/davinci-1/work/protezionecivile/data1/SRI_adj") / gt_filename
            year = timestamp.strftime("%Y")
            month = timestamp.strftime("%m")
            day = timestamp.strftime("%d")
            gt_path_data = Path(f"/davinci-1/work/protezionecivile/data/{year}/{month}/{day}/SRI_adj") / gt_filename

            if gt_path_data1.exists():
                gt_path = gt_path_data1
            elif gt_path_data.exists():
                gt_path = gt_path_data
        else:
            # Local: Use config sri_folder
            gt_path_local = config.sri_folder / gt_filename
            if gt_path_local.exists():
                gt_path = gt_path_local

        if gt_path:
            try:
                with h5py.File(gt_path, "r") as hdf:
                    gt_data = hdf["/dataset1/data1/data"][:]
                    gt_data[gt_data < 0] = 0
                    # Store raw data for difference calculation
                    gt_raw_data[timestamp] = gt_data
                    # Create figure for ground truth (target)
                    fig = compute_figure_gpd(gt_data, timestamp.strftime("%d/%m/%Y %H:%M"))
                    gt_figures[timestamp] = fig
                    gt_found_count += 1
            except Exception as e:
                logger.warning(f"Error loading GT at {gt_path}: {e}")
                # Create empty frame with error message
                fig = create_empty_frame(timestamp, "Error Loading Data")
                gt_figures[timestamp] = fig
                gt_missing_count += 1
        else:
            gt_missing_count += 1
            logger.debug(f"GT not found for {timestamp}: {gt_filename}")
            # Create empty frame with "No Data" message
            fig = create_empty_frame(timestamp, "No Data Available")
            gt_figures[timestamp] = fig
            # Store empty raw data (zeros) for difference calculation
            gt_raw_data[timestamp] = np.zeros((1400, 1200))

    return gt_figures, gt_raw_data, gt_found_count, gt_missing_count


def create_gifs_from_prediction_range(
        model_name: str, start_dt: datetime, end_dt: datetime, sri_folder_dir: str
) -> bool:
    """
    Create 7 GIFs from prediction range: groundtruth, target+30, target+60, pred+30, pred+60, diff+30, diff+60.

    This function:
    1. Loads all predictions in the range
    2. Loads corresponding ground truth data (base, +30min, +60min)
    3. Separates predictions into +30 and +60 dictionaries
    4. Computes difference arrays
    5. Creates 7 GIFs and saves them to gif_storage location

    If target data is missing, empty frames with "No Data Available" message will be shown.

    Args:
        model_name: Model name
        start_dt: Start datetime
        end_dt: End datetime
        sri_folder_dir: Path to SRI folder (unused, kept for compatibility)

    Returns:
        Dictionary with GIF paths if successful, False otherwise
    """
    config = get_config()

    logger.info(f"Creating GIFs for {model_name} from {start_dt} to {end_dt}")

    # Check for missing target data and inform user
    all_target_exist, missing_target_timestamps, existing_target_timestamps = check_target_data_for_range(start_dt, end_dt)

    if not all_target_exist:
        total_count = len(missing_target_timestamps) + len(existing_target_timestamps)
        missing_count = len(missing_target_timestamps)

        first_missing = (start_dt + timedelta(minutes=30))
        last_missing = (end_dt + timedelta(minutes=60))

        logger.warning(f"⚠️  Missing target data for {missing_count}/{total_count} timestamps")
        st.info(
            f"ℹ️ **Target data not available for {missing_count}/{total_count} timestamps**\n\n"
            f"**Missing range**: {first_missing.strftime('%d/%m/%Y %H:%M')} to {last_missing.strftime('%d/%m/%Y %H:%M')}\n\n"
            f"Empty frames will be shown for missing data in Target and Difference GIFs."
        )

    try:
        # ========== STEP 1: Load all groundtruth data ==========
        all_timestamps = generate_timestamp_range(start_dt, end_dt)

        # Dictionaries to store figures
        gt_figures = {}
        target30_figures = {}
        target60_figures = {}
        pred_30_figures = {}
        pred_60_figures = {}

        # Store raw data for difference calculation
        gt_raw_data = {}
        target30_raw_data = {}
        target60_raw_data = {}
        pred_30_raw_data = {}
        pred_60_raw_data = {}

        # Track file status
        gt_found_count = 0
        gt_missing_count = 0

        # Load groundtruth (base interval)
        logger.info(f"📊 Loading {len(all_timestamps)} groundtruth frames...")
        gt_figures, gt_raw_data, gt_found_count, gt_missing_count = create_groundtruth_figures(
            all_timestamps, gt_raw_data, gt_figures, gt_found_count, gt_missing_count
        )
        logger.info(f"✅ Groundtruth: {len(gt_figures)} frames loaded")

        # Load target +30 (shifted by +30 minutes)
        all_target_timestamps_30 = generate_timestamp_range(
            start_dt + timedelta(minutes=30), end_dt + timedelta(minutes=30)
        )
        logger.info(f"📊 Loading {len(all_target_timestamps_30)} target +30min frames...")
        target30_figures, target30_raw_data, _, _ = create_groundtruth_figures(
            all_target_timestamps_30, target30_raw_data, target30_figures, 0, 0
        )
        logger.info(f"✅ Target +30: {len(target30_figures)} frames loaded")

        # Load target +60 (shifted by +60 minutes)
        all_target_timestamps_60 = generate_timestamp_range(
            start_dt + timedelta(minutes=60), end_dt + timedelta(minutes=60)
        )
        logger.info(f"📊 Loading {len(all_target_timestamps_60)} target +60min frames...")
        target60_figures, target60_raw_data, _, _ = create_groundtruth_figures(
            all_target_timestamps_60, target60_raw_data, target60_figures, 0, 0
        )
        logger.info(f"✅ Target +60: {len(target60_figures)} frames loaded")

        # ========== STEP 2: Load predictions and separate into +30 and +60 ==========
        logger.info(f"📊 Loading {len(all_timestamps)} prediction files...")

        for idx, timestamp in enumerate(all_timestamps, 1):
            filename = timestamp.strftime("%d-%m-%Y-%H-%M") + ".npy"
            pred_path = config.real_time_pred / model_name / filename

            if not pred_path.exists():
                logger.warning(f"Prediction not found: {pred_path}")
                continue

            # Use helper function to handle model-specific shapes
            pred_data = load_prediction_array(pred_path, model_name)

            if pred_data is None:
                logger.error(f"Failed to load prediction: {pred_path}")
                continue

            # Extract pred[5] for +30min and pred[11] for +60min
            pred_30_time = timestamp + timedelta(minutes=30)
            pred_60_time = timestamp + timedelta(minutes=60)

            # Prediction +30 (index 5)
            pred_30_array = pred_data[5]
            pred_30_raw_data[pred_30_time] = pred_30_array
            fig_30 = compute_figure_gpd(pred_30_array, pred_30_time.strftime("%d/%m/%Y %H:%M"))
            pred_30_figures[pred_30_time] = fig_30

            # Prediction +60 (index 11)
            pred_60_array = pred_data[11]
            pred_60_raw_data[pred_60_time] = pred_60_array
            fig_60 = compute_figure_gpd(pred_60_array, pred_60_time.strftime("%d/%m/%Y %H:%M"))
            pred_60_figures[pred_60_time] = fig_60

        logger.info(f"✅ Predictions loaded: {len(pred_30_figures)} +30min, {len(pred_60_figures)} +60min")

        # Check if we have predictions
        if not pred_30_figures or not pred_60_figures:
            logger.error(f"Missing prediction data for {model_name} in range {start_dt} to {end_dt}")
            return False

        # ========== STEP 3: Compute difference figures ==========
        logger.info("📊 Computing difference arrays...")
        diff_30_figures = {}
        diff_60_figures = {}

        # Difference +30: target30 - pred30 (signed difference: positive=target higher, negative=pred higher)
        for timestamp in pred_30_figures.keys():
            if timestamp in target30_raw_data and timestamp in pred_30_raw_data:
                try:
                    diff_array = target30_raw_data[timestamp] - pred_30_raw_data[timestamp]
                    fig = compute_figure_gpd(diff_array, timestamp.strftime("%d/%m/%Y %H:%M"), name="diff")
                    diff_30_figures[timestamp] = fig
                except Exception as e:
                    logger.warning(f"Error computing diff +30 for {timestamp}: {e}")

        # Difference +60: target60 - pred60 (signed difference: positive=target higher, negative=pred higher)
        for timestamp in pred_60_figures.keys():
            if timestamp in target60_raw_data and timestamp in pred_60_raw_data:
                try:
                    diff_array = target60_raw_data[timestamp] - pred_60_raw_data[timestamp]
                    fig = compute_figure_gpd(diff_array, timestamp.strftime("%d/%m/%Y %H:%M"), name="diff")
                    diff_60_figures[timestamp] = fig
                except Exception as e:
                    logger.warning(f"Error computing diff +60 for {timestamp}: {e}")

        logger.info(f"✅ Differences computed: {len(diff_30_figures)} +30min, {len(diff_60_figures)} +60min")

        # ========== STEP 4: Get GIF paths ==========
        gif_paths = get_gif_paths(model_name, start_dt, end_dt)

        # ========== STEP 5: Create all 7 GIFs in parallel ==========
        logger.info("🎬 Creating GIFs in parallel...")

        # Prepare GIF tasks
        gif_tasks = [
            ("Groundtruth", gt_figures, gif_paths["gt_t0"]),
            ("Target +30", target30_figures, gif_paths["gt_t6"]),
            ("Target +60", target60_figures, gif_paths["gt_t12"]),
            ("Pred +30", pred_30_figures, gif_paths["pred_t6"]),
            ("Pred +60", pred_60_figures, gif_paths["pred_t12"]),
            ("Diff +30", diff_30_figures, gif_paths["diff_t6"]),
            ("Diff +60", diff_60_figures, gif_paths["diff_t12"]),
        ]

        # Filter out empty tasks
        gif_tasks = [(name, figs, path) for name, figs, path in gif_tasks if figs]

        # Create progress bars for each GIF
        progress_bars = {}
        progress_texts = {}

        with st.container():
            for name, _, _ in gif_tasks:
                col1, col2 = st.columns([0.2, 0.8])
                with col1:
                    st.markdown(f"**{name}:**")
                with col2:
                    progress_bars[name] = st.progress(0)
                    progress_texts[name] = st.empty()

        # Setup multiprocessing queue and processes
        queue = Manager().Queue()
        processes = []

        def create_gif_worker(queue, process_idx, gif_name, figures_dict, save_path, fps_gif=3):
            """Worker function to create a single GIF in parallel."""
            try:
                sorted_keys = sorted(figures_dict.keys())
                frames = []
                total_frames = len(sorted_keys)

                # Convert each figure to a frame
                for idx, key in enumerate(sorted_keys):
                    fig = figures_dict[key]
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
                    buf.seek(0)
                    img = Image.open(buf)
                    frames.append(np.array(img))
                    buf.close()

                    # Send progress update
                    progress = (idx + 1) / total_frames
                    queue.put(("progress", process_idx, gif_name, progress, idx + 1, total_frames))

                # Ensure parent directory exists
                save_path.parent.mkdir(parents=True, exist_ok=True)

                # Save the GIF
                imageio.mimsave(save_path, frames, format="GIF", fps=fps_gif, loop=0)

                # Send completion message
                queue.put(("complete", process_idx, gif_name, str(save_path)))

            except Exception as e:
                queue.put(("error", process_idx, gif_name, str(e)))

        # Start all processes
        for idx, (name, figures, path) in enumerate(gif_tasks):
            p = Process(
                target=create_gif_worker,
                args=(queue, idx, name, figures, path, 3)
            )
            processes.append(p)
            p.start()
            logger.info(f"Started GIF creation process for {name} ({len(figures)} frames)")

        # Monitor queue for updates
        completed_count = 0
        gif_results = []

        while completed_count < len(gif_tasks):
            try:
                msg = queue.get(timeout=1)
                msg_type = msg[0]

                if msg_type == "progress":
                    _, process_idx, gif_name, progress, current, total = msg
                    progress_bars[gif_name].progress(progress)
                    progress_texts[gif_name].text(f"{current}/{total} frames")

                elif msg_type == "complete":
                    _, process_idx, gif_name, result_path = msg
                    completed_count += 1
                    progress_bars[gif_name].progress(1.0)
                    progress_texts[gif_name].text("✅ Complete!")
                    gif_results.append((gif_name, result_path))
                    logger.info(f"✓ {gif_name}: {result_path}")

                elif msg_type == "error":
                    _, process_idx, gif_name, error_msg = msg
                    completed_count += 1
                    progress_texts[gif_name].text(f"❌ Error: {error_msg}")
                    gif_results.append((gif_name, None))
                    logger.error(f"✗ {gif_name}: {error_msg}")

            except:
                # Check if all processes are still alive
                if not any(p.is_alive() for p in processes):
                    break

        # Clean up processes
        for p in processes:
            p.join()

        # Clear progress indicators after a brief display
        time.sleep(1)
        for bar in progress_bars.values():
            bar.empty()
        for text in progress_texts.values():
            text.empty()

        # Log summary
        success_count = sum(1 for _, result in gif_results if result is not None)
        logger.info(f"✅ GIF creation completed: {success_count}/{len(gif_tasks)} GIFs created successfully")

        return gif_paths

    except Exception as e:
        logger.error(f"❌ Error creating GIFs for {model_name}: {e}")

        logger.error(traceback.format_exc())
        return False
