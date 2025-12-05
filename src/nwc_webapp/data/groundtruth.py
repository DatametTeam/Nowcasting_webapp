"""
Ground truth data loading utilities - shared across all pages.
"""

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


def load_single_groundtruth_frame(timestamp: datetime, radar_mask: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """
    Load a single ground truth frame for a given timestamp.

    This function handles environment-specific path resolution (HPC vs local)
    and applies optional radar masking and clipping.

    Args:
        timestamp: Datetime for the GT frame
        radar_mask: Optional radar mask to apply (if None, no masking)

    Returns:
        np.ndarray of shape (H, W) or None if loading fails
    """
    config = get_config()
    gt_filename = timestamp.strftime("%d-%m-%Y-%H-%M") + ".hdf"

    # Determine path based on environment
    gt_path = None

    if is_hpc():
        # HPC: Try data1 first (recent data), then data (archived)
        gt_path_data1 = Path("/davinci-1/work/protezionecivile/data1/SRI_adj") / gt_filename
        if gt_path_data1.exists():
            gt_path = gt_path_data1
        else:
            # Try archived path
            year = timestamp.strftime("%Y")
            month = timestamp.strftime("%m")
            day = timestamp.strftime("%d")
            gt_path_data = Path(f"/davinci-1/work/protezionecivile/data/{year}/{month}/{day}/SRI_adj") / gt_filename
            if gt_path_data.exists():
                gt_path = gt_path_data
    else:
        # Local: Use config sri_folder
        gt_path_local = config.sri_folder / gt_filename
        if gt_path_local.exists():
            gt_path = gt_path_local

    if gt_path is None or not gt_path.exists():
        logger.debug(f"Ground truth not found: {gt_filename}")
        return None

    try:
        with h5py.File(gt_path, "r") as hdf:
            gt_data = hdf["/dataset1/data1/data"][:]

        # Apply mask if provided
        if radar_mask is not None:
            gt_data = gt_data * radar_mask

        # Clip to valid range
        gt_data = np.clip(gt_data, 0, 200)

        return gt_data

    except Exception as e:
        logger.error(f"Error loading ground truth {gt_filename}: {e}")
        return None


def load_groundtruth_sequence(
    base_timestamp: datetime, num_frames: int = 12, radar_mask: Optional[np.ndarray] = None, fill_missing: bool = True
) -> Optional[np.ndarray]:
    """
    Load a sequence of ground truth frames starting from base_timestamp.

    Loads frames at base_timestamp + 5*(i+1) minutes for i in range(num_frames).
    This corresponds to t+5, t+10, ..., t+60 for 12 frames.

    Args:
        base_timestamp: Starting timestamp
        num_frames: Number of frames to load (default: 12)
        radar_mask: Optional radar mask to apply
        fill_missing: If True, fill missing frames with zeros; if False, return None on any missing frame

    Returns:
        np.ndarray of shape (num_frames, H, W) or None if loading fails (when fill_missing=False)
    """
    gt_frames = []

    for i in range(num_frames):
        # Ground truth at base_timestamp + 5*(i+1) minutes
        gt_time = base_timestamp + timedelta(minutes=5 * (i + 1))
        gt_data = load_single_groundtruth_frame(gt_time, radar_mask)

        if gt_data is None:
            if fill_missing:
                # Fill with zeros for missing data
                gt_frames.append(np.zeros((1400, 1200)))
                logger.debug(f"Missing GT frame at {gt_time.strftime('%d/%m/%Y %H:%M')}, filled with zeros")
            else:
                logger.warning(f"Missing GT frame at {gt_time.strftime('%d/%m/%Y %H:%M')}, aborting sequence load")
                return None
        else:
            gt_frames.append(gt_data)

    return np.array(gt_frames) if gt_frames else None


def load_groundtruth_for_timestamp(timestamp: datetime) -> Optional[np.ndarray]:
    """
    Load all 12 ground truth frames for a given timestamp (t+5 to t+60).

    This is a convenience wrapper around load_groundtruth_sequence() that includes
    radar mask loading. Used by pages that need full GT sequence (e.g., model comparison).

    Args:
        timestamp: Base timestamp

    Returns:
        np.ndarray of shape (12, H, W) or None if loading fails
    """
    # Load radar mask
    mask_path = Path(__file__).resolve().parent.parent / "resources/mask/radar_mask.hdf"
    try:
        with h5py.File(mask_path, "r") as f:
            radar_mask = f["mask"][()]
    except Exception as e:
        logger.error(f"Error loading radar mask: {e}")
        radar_mask = None

    return load_groundtruth_sequence(timestamp, num_frames=12, radar_mask=radar_mask, fill_missing=True)


def check_groundtruth_availability(timestamp: datetime) -> Tuple[bool, int, int]:
    """
    Check if ground truth data is available for a given timestamp sequence.

    Checks for 12 frames (t+5 to t+60) without loading the full data.

    Args:
        timestamp: Base timestamp

    Returns:
        Tuple of (all_exist: bool, found_count: int, total_count: int)
    """
    config = get_config()
    total_count = 12
    found_count = 0

    for i in range(12):
        gt_time = timestamp + timedelta(minutes=5 * (i + 1))
        gt_filename = gt_time.strftime("%d-%m-%Y-%H-%M") + ".hdf"

        # Determine path based on environment
        gt_path = None

        if is_hpc():
            gt_path_data1 = Path("/davinci-1/work/protezionecivile/data1/SRI_adj") / gt_filename
            if gt_path_data1.exists():
                gt_path = gt_path_data1
            else:
                year = gt_time.strftime("%Y")
                month = gt_time.strftime("%m")
                day = gt_time.strftime("%d")
                gt_path_data = Path(f"/davinci-1/work/protezionecivile/data/{year}/{month}/{day}/SRI_adj") / gt_filename
                if gt_path_data.exists():
                    gt_path = gt_path_data
        else:
            gt_path_local = config.sri_folder / gt_filename
            if gt_path_local.exists():
                gt_path = gt_path_local

        if gt_path and gt_path.exists():
            found_count += 1

    all_exist = found_count == total_count
    return all_exist, found_count, total_count
