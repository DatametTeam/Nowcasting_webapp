"""
Utility functions for the nowcasting page workflow.
"""
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Optional
import numpy as np

from nwc_webapp.config.config import get_config
from nwc_webapp.logging_config import setup_logger

# Set up logger
logger = setup_logger(__name__)

# Training cutoff date
TRAINING_CUTOFF_DATE = datetime(2025, 1, 1)


def is_training_date(selected_date: datetime) -> bool:
    """
    Check if the selected date is prior to the training cutoff (Jan 1, 2025).

    Args:
        selected_date: The date selected by the user

    Returns:
        True if date is before Jan 1, 2025 (training data), False otherwise
    """
    return selected_date < TRAINING_CUTOFF_DATE


def get_gif_paths(model_name: str, date_str: str, time_str: str) -> dict:
    """
    Get the paths where GIFs should be stored/loaded.

    Uses the naming convention: {start}_{end}.gif for full sequence,
    {start}_{end}_+30m.gif and {start}_{end}_+60m.gif for time offsets.

    Directory structure:
    - gif_storage/
      - groundtruth/
        - {start}_{end}.gif (full sequence: 0-55 min)
        - {start}_{end}_+30m.gif (from +30: 30-55 min)
        - {start}_{end}_+60m.gif (from +60: 60 min)
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
        date_str: Date string (format: DDMMYYYY)
        time_str: Time string (format: HHMM)

    Returns:
        Dictionary with paths for all GIF types and directories
    """
    from datetime import datetime, timedelta

    config = get_config()
    gif_base = config.gif_storage

    # Create subdirectories
    gt_dir = gif_base / "groundtruth"
    pred_dir = gif_base / "prediction" / model_name
    diff_dir = gif_base / "difference" / model_name

    # Parse the datetime
    start_datetime = datetime.strptime(f"{date_str}_{time_str}", "%d%m%Y_%H%M")

    # Calculate end time (12 frames * 5 min = 60 min total, but display up to frame 11 = 55 min)
    end_datetime = start_datetime + timedelta(minutes=55)

    # Base filename: {start}_{end}
    base_name = f"{start_datetime.strftime('%d%m%Y_%H%M')}_{end_datetime.strftime('%d%m%Y_%H%M')}"

    return {
        'gt_t0': gt_dir / f"{base_name}.gif",           # Full sequence
        'gt_t6': gt_dir / f"{base_name}_+30m.gif",      # From +30 min
        'gt_t12': gt_dir / f"{base_name}_+60m.gif",     # From +60 min
        'pred_t6': pred_dir / f"{base_name}_+30m.gif",
        'pred_t12': pred_dir / f"{base_name}_+60m.gif",
        'diff_t6': diff_dir / f"{base_name}_+30m.gif",
        'diff_t12': diff_dir / f"{base_name}_+60m.gif",
        # Also return the directories for easy access
        'gt_dir': gt_dir,
        'pred_dir': pred_dir,
        'diff_dir': diff_dir,
    }


def check_gifs_exist(gif_paths: dict) -> Tuple[bool, bool, bool]:
    """
    Check if GIFs exist at the specified paths.

    Args:
        gif_paths: Dictionary of GIF paths from get_gif_paths()

    Returns:
        Tuple of (gt_exist, pred_exist, diff_exist) booleans
    """
    gt_exist = (
        gif_paths['gt_t0'].exists() and
        gif_paths['gt_t6'].exists() and
        gif_paths['gt_t12'].exists()
    )

    pred_exist = (
        gif_paths['pred_t6'].exists() and
        gif_paths['pred_t12'].exists()
    )

    diff_exist = (
        gif_paths['diff_t6'].exists() and
        gif_paths['diff_t12'].exists()
    )

    return gt_exist, pred_exist, diff_exist


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


def check_realtime_prediction_exists(model_name: str, date_str: str, time_str: str) -> bool:
    """
    Check if real-time prediction data exists for the requested timestamp.

    Args:
        model_name: Name of the prediction model
        date_str: Date string (format: DDMMYYYY)
        time_str: Time string (format: HHMM)

    Returns:
        True if prediction file exists, False otherwise
    """
    pred_path = get_realtime_prediction_path(model_name, date_str, time_str)
    exists = pred_path.exists()

    if exists:
        logger.info(f"Found real-time prediction: {pred_path}")
    else:
        logger.info(f"Real-time prediction not found: {pred_path}")

    return exists


def load_realtime_prediction(model_name: str, date_str: str, time_str: str) -> Optional[np.ndarray]:
    """
    Load real-time prediction data.

    Args:
        model_name: Name of the prediction model
        date_str: Date string (format: DDMMYYYY)
        time_str: Time string (format: HHMM)

    Returns:
        Prediction array of shape (12, 1400, 1200) or None if not found
    """
    pred_path = get_realtime_prediction_path(model_name, date_str, time_str)

    if not pred_path.exists():
        logger.warning(f"Prediction file not found: {pred_path}")
        return None

    try:
        pred_data = np.load(pred_path)
        logger.info(f"Loaded prediction data from {pred_path}, shape: {pred_data.shape}")
        return pred_data
    except Exception as e:
        logger.error(f"Error loading prediction data: {e}")
        return None


def get_groundtruth_path(date_str: str, time_str: str) -> Path:
    """
    Get the path to ground truth SRI data.

    Args:
        date_str: Date string (format: DDMMYYYY)
        time_str: Time string (format: HHMM)

    Returns:
        Path to the ground truth file
    """
    config = get_config()
    # Assuming SRI files are named like: SRI_DDMMYYYY_HHMM.hdf or similar
    # Adjust the naming pattern based on actual file structure
    gt_file = config.sri_folder / f"SRI_{date_str}_{time_str}.hdf"
    return gt_file


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

    t0 = pred_array[0]      # t+0
    t30 = pred_array[6]     # t+30 minutes (index 6)
    t60 = pred_array[11]    # t+60 minutes (index 11, last one)

    return t0, t30, t60