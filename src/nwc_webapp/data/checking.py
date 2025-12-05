"""
Data availability checking utilities - shared across all pages.
"""

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

from nwc_webapp.config.config import get_config
from nwc_webapp.config.environment import is_hpc
from nwc_webapp.logging_config import setup_logger

# Set up logger
logger = setup_logger(__name__)


def generate_timestamp_range(start_dt: datetime, end_dt: datetime, verbose: bool = True) -> List[datetime]:
    """
    Generate all timestamps in a range with 5-minute intervals.

    Args:
        start_dt: Start datetime
        end_dt: End datetime
        verbose: If True, log the generation (default: True)

    Returns:
        List of datetime objects with 5-minute intervals
    """
    timestamps = []
    current = start_dt

    while current <= end_dt:
        timestamps.append(current)
        current += timedelta(minutes=5)

    if verbose:
        logger.info(f"Generated {len(timestamps)} timestamps from {start_dt} to {end_dt}")
    return timestamps


def check_missing_predictions(
    model_name: str, start_dt: datetime, end_dt: datetime, verbose: bool = True
) -> Tuple[List[datetime], List[datetime]]:
    """
    Check which prediction files are missing in the specified range.

    Args:
        model_name: Name of the prediction model
        start_dt: Start datetime
        end_dt: End datetime
        verbose: If True, log the check results (default: True)

    Returns:
        Tuple of (missing_timestamps, existing_timestamps)
    """
    config = get_config()
    pred_dir = config.real_time_pred / model_name

    # Generate all expected timestamps (don't log during monitoring)
    all_timestamps = generate_timestamp_range(start_dt, end_dt, verbose=verbose)

    missing = []
    existing = []

    # Check directory exists (only log if verbose and directory missing)
    if verbose and not pred_dir.exists():
        logger.warning(f"[{model_name}] Prediction directory does not exist: {pred_dir}")

    for timestamp in all_timestamps:
        # Format: DD-MM-YYYY-HH-MM.npy (same as real-time predictions)
        filename = timestamp.strftime("%d-%m-%Y-%H-%M") + ".npy"
        pred_file = pred_dir / filename

        # Use os.path.exists() to avoid any Path caching issues
        file_exists = os.path.exists(str(pred_file))

        if file_exists:
            existing.append(timestamp)
        else:
            missing.append(timestamp)

    if verbose:
        logger.info(
            f"[{model_name}] Range check: {len(existing)}/{len(all_timestamps)} predictions exist, {len(missing)} "
            f"missing"
        )

    return missing, existing


def check_single_prediction_exists(model_name: str, prediction_dt: datetime) -> bool:
    """
    Check if a single prediction file exists.

    Args:
        model_name: Name of the prediction model
        prediction_dt: Datetime for the prediction

    Returns:
        True if prediction file exists, False otherwise
    """
    config = get_config()
    pred_dir = config.real_time_pred / model_name

    # Format: DD-MM-YYYY-HH-MM.npy
    filename = prediction_dt.strftime("%d-%m-%Y-%H-%M") + ".npy"
    pred_file = pred_dir / filename

    return os.path.exists(str(pred_file))


def check_target_data_exists(prediction_dt: datetime) -> Tuple[bool, int, int]:
    """
    Check if target groundtruth data exists for the prediction.

    Target data is required for t+5 to t+60 (12 frames at 5-min intervals).
    This is needed to compute differences between predictions and targets.

    Args:
        prediction_dt: Datetime for the prediction start

    Returns:
        Tuple of (all_exist: bool, found_count: int, total_count: int)
    """
    config = get_config()

    total_count = 12  # Need 12 target frames (t+5 to t+60)
    found_count = 0

    for i in range(12):
        target_timestamp = prediction_dt + timedelta(minutes=5 + 5 * i)
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
            found_count += 1

    all_exist = found_count == total_count
    return all_exist, found_count, total_count


def check_target_data_for_range(start_dt: datetime, end_dt: datetime) -> Tuple[bool, List[datetime], List[datetime]]:
    """
    Check if target data exists for all timestamps in the date range.

    This checks for target data at +30min and +60min offsets for each timestamp,
    which is needed to create target and difference GIFs.

    Args:
        start_dt: Start datetime
        end_dt: End datetime

    Returns:
        Tuple of (all_exist: bool, missing_timestamps: List[datetime], existing_timestamps: List[datetime])
    """
    config = get_config()
    all_timestamps = generate_timestamp_range(start_dt, end_dt, verbose=False)

    missing_target_timestamps = []
    existing_target_timestamps = []

    logger.info(f"Checking target data availability for {len(all_timestamps)} timestamps...")

    for timestamp in all_timestamps:
        # Check target data at +30min and +60min
        target_30_timestamp = timestamp + timedelta(minutes=30)
        target_60_timestamp = timestamp + timedelta(minutes=60)

        # Check both +30 and +60 targets
        targets_exist = True

        for target_timestamp in [target_30_timestamp, target_60_timestamp]:
            target_filename = target_timestamp.strftime("%d-%m-%Y-%H-%M") + ".hdf"
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

            if not (target_path and target_path.exists()):
                targets_exist = False
                break

        if targets_exist:
            existing_target_timestamps.append(timestamp)
        else:
            missing_target_timestamps.append(timestamp)

    all_exist = len(missing_target_timestamps) == 0

    logger.info(
        f"Target data check: {len(existing_target_timestamps)}/{len(all_timestamps)} complete, "
        f"{len(missing_target_timestamps)} missing target data"
    )

    return all_exist, missing_target_timestamps, existing_target_timestamps


def get_missing_range(missing_timestamps: List[datetime]) -> Tuple[Optional[datetime], Optional[datetime]]:
    """
    Get the start and end of the missing timestamp range.

    Args:
        missing_timestamps: List of missing datetime objects

    Returns:
        Tuple of (first_missing, last_missing) or (None, None) if all exist
    """
    if not missing_timestamps:
        return None, None

    # Sort to ensure correct order
    sorted_missing = sorted(missing_timestamps)
    return sorted_missing[0], sorted_missing[-1]
