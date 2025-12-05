from nwc_webapp.data.checking import generate_timestamp_range
"""
Prediction data loading and management - shared across all pages.
"""

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np

from nwc_webapp.config.config import get_config
from nwc_webapp.config.environment import is_hpc
from nwc_webapp.logging_config import setup_logger

# Set up logger
logger = setup_logger(__name__)


def get_realtime_prediction_path(model_name: str, date_str: str, time_str: str) -> Path:
    """
    Get the path to real-time prediction data.

    Real-time predictions are stored as:
    real_time_pred/{model_name}/{date}_{time}_prediction.npy

    The data shape is (12, 1400, 1200) where 12 are timesteps with 5-minute intervals.

    Args:
        model_name: Name of the prediction model
        date_str: Date string (format: DDMMYYYY)
        time_str: Time string (format: HHMM)

    Returns:
        Path to the prediction file
    """
    config = get_config()
    pred_file = config.real_time_pred / model_name / f"{date_str}_{time_str}_prediction.npy"
    return pred_file


def extract_timestamp_slices(pred_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract specific timestamp slices from prediction array.

    The prediction array has shape (12, 1400, 1200) with 5-minute intervals:
    - Index 0: t+0 (current)
    - Index 6: t+30 minutes (6 * 5 = 30 min)
    - Index 12: t+60 minutes (12 * 5 = 60 min, but 0-indexed so it's index 11)

    Args:
        pred_array: Prediction array of shape (12, 1400, 1200)

    Returns:
        Tuple of (t0_slice, t30_slice, t60_slice)
    """
    if pred_array.shape[0] != 12:
        raise ValueError(f"Expected 12 timesteps, got {pred_array.shape[0]}")

    t0 = pred_array[0]  # t+0
    t30 = pred_array[6]  # t+30 minutes (index 6)
    t60 = pred_array[11]  # t+60 minutes (index 11, last one)

    return t0, t30, t60


def load_prediction_array(pred_path: Path, model_name: str) -> Optional[np.ndarray]:
    """
    Load prediction array and handle model-specific shapes.

    ED_ConvLSTM outputs shape (1, 12, H, W) while other models output (12, H, W).
    This function normalizes to (12, H, W) for consistent downstream processing.

    Args:
        pred_path: Path to the prediction file
        model_name: Model name

    Returns:
        Prediction array of shape (12, H, W), or None if loading fails
    """
    try:
        pred_array = np.load(pred_path, mmap_mode="r")

        # ED_ConvLSTM has extra batch dimension: (1, 12, H, W)
        if model_name == "ED_ConvLSTM":
            if pred_array.ndim == 4 and pred_array.shape[0] == 1:
                pred_array = pred_array[0]  # Remove batch dimension: (12, H, W)
                logger.debug(f"ED_ConvLSTM: Removed batch dimension, shape now {pred_array.shape}")
            else:
                logger.warning(f"ED_ConvLSTM: Unexpected shape {pred_array.shape}, expected (1, 12, H, W)")

        # Verify final shape
        if pred_array.shape[0] != 12:
            logger.error(f"Unexpected prediction shape: {pred_array.shape}, expected (12, H, W)")
            return None

        return pred_array

    except Exception as e:
        logger.error(f"Error loading prediction from {pred_path}: {e}")
        return None


def load_single_prediction_data(model_name: str, prediction_dt: datetime) -> Tuple[dict, dict, dict]:
    """
    Load groundtruth, target, and prediction data for a single timestamp.

    This function loads data in the format required by init_second_tab_layout():
    - Groundtruth: 12 frames (t0, t+5, t+10, ..., t+55)
    - Target: 12 frames (t+60, t+65, ..., t+115)
    - Prediction: 12 frames (t+60, t+65, ..., t+115) from model

    Args:
        model_name: Name of the prediction model
        prediction_dt: Datetime for the prediction start

    Returns:
        Tuple of (gt_dict, target_dict, pred_dict) with timestamped arrays
    """
    config = get_config()

    # Load radar mask
    mask_path = Path(__file__).resolve().parent.parent / "resources/mask/radar_mask.hdf"
    with h5py.File(mask_path, "r") as f:
        radar_mask = f["mask"][()]

    # Initialize dictionaries
    gt_dict = {}
    target_dict = {}
    pred_dict = {}

    # ========== STEP 1: Load groundtruth (12 frames: t0 to t-55) ==========
    logger.info(f"Loading groundtruth for {prediction_dt.strftime('%d/%m/%Y %H:%M')}")

    for i in reversed(range(12)):
        gt_timestamp = prediction_dt - timedelta(minutes=5 * i)
        gt_filename = gt_timestamp.strftime("%d-%m-%Y-%H-%M") + ".hdf"

        # Determine path based on environment
        gt_path = None

        if is_hpc():
            # HPC: Try data1 first (recent data), then data (archived)
            gt_path_data1 = Path("/davinci-1/work/protezionecivile/data1/SRI_adj") / gt_filename
            year = gt_timestamp.strftime("%Y")
            month = gt_timestamp.strftime("%m")
            day = gt_timestamp.strftime("%d")
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

        if gt_path and gt_path.exists():
            try:
                with h5py.File(gt_path, "r") as hdf:
                    gt_data = hdf["/dataset1/data1/data"][:]
                    # Apply mask and clip
                    gt_data = gt_data * radar_mask
                    gt_data = np.clip(gt_data, 0, 200)

                    timestamp_key = gt_timestamp.strftime("%d/%m/%Y %H:%M")
                    gt_dict[timestamp_key] = gt_data
            except Exception as e:
                logger.warning(f"Error loading GT at {gt_path}: {e}")

    # ========== STEP 2: Load target (12 frames: t+5 to t+60) ==========
    logger.info(f"Loading target for {prediction_dt.strftime('%d/%m/%Y %H:%M')}")

    for i in range(12):
        target_timestamp = prediction_dt + timedelta(minutes=5 * (i + 1))
        target_filename = target_timestamp.strftime("%d-%m-%Y-%H-%M") + ".hdf"

        # Determine path based on environment
        target_path = None

        if is_hpc():
            # HPC: Try data1 first, then data (archived)
            target_path_data1 = Path("/davinci-1/work/protezionecivile/data1/SRI_adj") / target_filename
            year = target_timestamp.strftime("%Y")
            month = target_timestamp.strftime("%m")
            day = target_timestamp.strftime("%d")
            target_path_data = (
                Path(f"/davinci-1/work/protezionecivile/data/{year}/{month}/{day}/SRI_adj") / target_filename
            )

            if target_path_data1.exists():
                target_path = target_path_data1
            elif target_path_data.exists():
                target_path = target_path_data
        else:
            # Local: Use config sri_folder
            target_path_local = config.sri_folder / target_filename
            if target_path_local.exists():
                target_path = target_path_local

        if target_path and target_path.exists():
            try:
                with h5py.File(target_path, "r") as hdf:
                    target_data = hdf["/dataset1/data1/data"][:]
                    # Apply mask and clip
                    target_data = target_data * radar_mask
                    target_data = np.clip(target_data, 0, 200)

                    timestamp_key = target_timestamp.strftime("%d/%m/%Y %H:%M")
                    target_dict[timestamp_key] = target_data
            except Exception as e:
                logger.warning(f"Error loading target at {target_path}: {e}")

    # ========== STEP 3: Load prediction (12 frames from model) ==========
    logger.info(f"Loading prediction for {model_name} at {prediction_dt.strftime('%d/%m/%Y %H:%M')}")

    pred_filename = prediction_dt.strftime("%d-%m-%Y-%H-%M") + ".npy"
    pred_path = config.real_time_pred / model_name / pred_filename

    if pred_path.exists():
        # Use helper function to handle model-specific shapes
        pred_array = load_prediction_array(pred_path, model_name)

        if pred_array is not None:
            # The prediction array contains 12 timesteps at 5-minute intervals
            # Map these to timestamps starting at t+60
            for i in range(12):
                pred_timestamp = prediction_dt + timedelta(minutes=5 * (i + 1))
                pred_data = pred_array[i]

                # Apply mask and clip
                pred_data = pred_data * radar_mask
                pred_data = np.clip(pred_data, 0, 200)

                timestamp_key = pred_timestamp.strftime("%d/%m/%Y %H:%M")
                pred_dict[timestamp_key] = pred_data

            logger.info(f"✅ Loaded {len(pred_dict)} prediction frames")
    else:
        logger.error(f"Prediction file not found: {pred_path}")

    logger.info(f"Loaded: {len(gt_dict)} groundtruth, {len(target_dict)} target, {len(pred_dict)} prediction frames")

    return gt_dict, target_dict, pred_dict


def delete_predictions_in_range(model_name: str, start_dt: datetime, end_dt: datetime) -> int:
    """
    Delete all prediction files in the specified date range.

    Args:
        model_name: Name of the prediction model
        start_dt: Start datetime
        end_dt: End datetime

    Returns:
        Number of files deleted
    """

    config = get_config()
    pred_dir = config.real_time_pred / model_name

    if not pred_dir.exists():
        logger.warning(f"Prediction directory does not exist: {pred_dir}")
        return 0

    # Generate all timestamps in the range
    all_timestamps = generate_timestamp_range(start_dt, end_dt, verbose=False)

    deleted_count = 0
    for timestamp in all_timestamps:
        # Format: DD-MM-YYYY-HH-MM.npy (same as real-time predictions)
        filename = timestamp.strftime("%d-%m-%Y-%H-%M") + ".npy"
        pred_file = pred_dir / filename

        if pred_file.exists():
            try:
                pred_file.unlink()
                deleted_count += 1
                logger.debug(f"Deleted prediction file: {filename}")
            except Exception as e:
                logger.error(f"Failed to delete {filename}: {e}")

    logger.info(f"[{model_name}] Deleted {deleted_count} prediction files from range {start_dt} to {end_dt}")
    return deleted_count
