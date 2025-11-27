"""
Utility functions for the nowcasting page workflow.
"""

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np

from nwc_webapp.config.config import get_config
from nwc_webapp.config.environment import is_hpc
from nwc_webapp.logging_config import setup_logger
from nwc_webapp.services.parallel_code import create_single_gif_from_dict
from nwc_webapp.utils import compute_figure_gpd

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
        date_str: Date string (format: DDMMYYYY)
        time_str: Time string (format: HHMM)

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
    from datetime import timedelta

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
            f"[{model_name}] Range check: {len(existing)}/{len(all_timestamps)} predictions exist, {len(missing)} missing"
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

    Target data is required for t+60 to t+115 (12 frames at 5-min intervals).
    This is needed to compute differences between predictions and targets.

    Args:
        prediction_dt: Datetime for the prediction start

    Returns:
        Tuple of (all_exist: bool, found_count: int, total_count: int)
    """
    config = get_config()

    total_count = 12  # Need 12 target frames (t+60 to t+115)
    found_count = 0

    for i in range(12):
        target_timestamp = prediction_dt + timedelta(minutes=60 + 5 * i)
        target_filename = target_timestamp.strftime("%d-%m-%Y-%H-%M") + ".hdf"

        # Determine path based on environment
        target_path = None

        if is_hpc():
            # HPC: Try data1 first, then data (archived)
            target_path_data1 = Path("/davinci-1/work/protezionecivile/data1/SRI_adj") / target_filename
            year = target_timestamp.strftime("%Y")
            month = target_timestamp.strftime("%m")
            day = target_timestamp.strftime("%d")
            target_path_data = Path(f"/davinci-1/work/protezionecivile/data/{year}/{month}/{day}/SRI_adj") / target_filename

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

    # ========== STEP 1: Load groundtruth (12 frames: t0 to t+55) ==========
    logger.info(f"Loading groundtruth for {prediction_dt.strftime('%d/%m/%Y %H:%M')}")

    for i in range(12):
        gt_timestamp = prediction_dt + timedelta(minutes=5 * i)
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

    # ========== STEP 2: Load target (12 frames: t+60 to t+115) ==========
    logger.info(f"Loading target for {prediction_dt.strftime('%d/%m/%Y %H:%M')}")

    for i in range(12):
        target_timestamp = prediction_dt + timedelta(minutes=60 + 5 * i)
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
                pred_timestamp = prediction_dt + timedelta(minutes=60 + 5 * i)
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
    import os

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


def modify_yaml_config_for_date_range(model_name: str, start_dt: datetime, end_dt: datetime) -> Path:
    """
    Modify YAML config with start/end dates for date-range predictions.

    Reads the config from start_end_prediction_cfg/{model_name}.yaml,
    modifies the start_date and end_date fields, and overwrites the file.

    Args:
        model_name: Model name (e.g., 'ConvLSTM', 'IAM4VP', 'PredFormer', 'SPROG')
        start_dt: Start datetime
        end_dt: End datetime

    Returns:
        Path to the modified YAML config file
    """
    import yaml

    # Source YAML path
    config_path = Path(__file__).parent.parent / "resources" / "cfg" / "start_end_prediction_cfg" / f"{model_name}.yaml"

    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Read the YAML file
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)

    # Format dates as "YYYY-MM-DD HH:MM"
    start_str = start_dt.strftime("%Y-%m-%d %H:%M")
    end_str = end_dt.strftime("%Y-%m-%d %H:%M")

    # Modify the start_date and end_date fields
    if "dataframe_strategy" in config_data and "args" in config_data["dataframe_strategy"]:
        config_data["dataframe_strategy"]["args"]["start_date"] = start_str
        config_data["dataframe_strategy"]["args"]["end_date"] = end_str
        logger.info(f"Modified {model_name} config: start={start_str}, end={end_str}")
    else:
        logger.warning(f"Could not find dataframe_strategy.args in {model_name} config")

    # Overwrite the original file
    with open(config_path, "w") as f:
        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Overwritten config at: {config_path}")
    return config_path


def submit_date_range_prediction_job(model_name: str, start_dt: datetime, end_dt: datetime) -> Optional[str]:
    """
    Submit PBS job for date-range predictions (HPC) or generate mock predictions (local).

    IMPORTANT: Most models need 1 hour of groundtruth data BEFORE the start_dt to make the
    first prediction at start_dt+5min. Therefore, we adjust the actual job start time
    to be 1 hour earlier than the requested start_dt.

    EXCEPTION: ED_ConvLSTM handles the lookback internally (goes back 12 timesteps from start_dt)
    so it does NOT need the -1 hour adjustment.

    Example: If user selects 12:00 start:
    - ConvLSTM/IAM4VP/etc: use 11:00 as actual start, first prediction at 12:05
    - ED_ConvLSTM: use 12:00 as actual start (goes back 12 timesteps internally)

    HPC mode:
    1. Modifies the YAML config with start/end dates (adjusted by -1 hour for most models)
    2. Modifies the PBS script to use absolute path for config or date parameters
    3. Submits the PBS job using the modified script
    4. Returns the job ID

    Local mode:
    1. Generates mock prediction files instantly
    2. Returns a fake job ID for UI compatibility

    Args:
        model_name: Model name (e.g., 'ConvLSTM', 'ED_ConvLSTM', 'IAM4VP', 'PredFormer', 'SPROG')
        start_dt: Start datetime (user-selected)
        end_dt: End datetime (user-selected)

    Returns:
        Job ID string if successful, None if failed
    """
    # CRITICAL: ED_ConvLSTM handles lookback internally, others need -1 hour adjustment
    if model_name == "ED_ConvLSTM":
        actual_start_dt = start_dt  # No adjustment for ED_ConvLSTM
        logger.info(f"User requested range: {start_dt} to {end_dt}")
        logger.info(f"ED_ConvLSTM will handle lookback internally (12 timesteps before {start_dt})")
    else:
        # Other models: Adjust start time by -1 hour to get required groundtruth data
        actual_start_dt = start_dt - timedelta(hours=1)
        logger.info(f"User requested range: {start_dt} to {end_dt}")
        logger.info(f"Adjusted job range (for GT data): {actual_start_dt} to {end_dt}")
        logger.info(f"First prediction will be at: {start_dt + timedelta(minutes=5)}")

    # Check if running locally
    if not is_hpc():
        logger.info(f"🖥️  Running in LOCAL mode - generating mock predictions for {model_name}")

        try:
            from nwc_webapp.services.mock.mock_data_generator import generate_mock_predictions_for_range

            # Generate mock predictions with ORIGINAL start_dt (not adjusted)
            # The mock generator should handle predictions starting from start_dt
            created_count = generate_mock_predictions_for_range(model_name, start_dt, end_dt)

            if created_count >= 0:
                # Return a fake job ID for UI compatibility
                fake_job_id = f"mock_{int(datetime.now().timestamp())}"
                logger.info(f"✅ Mock predictions generated successfully! Fake job ID: {fake_job_id}")
                return fake_job_id
            else:
                logger.error("Failed to generate mock predictions")
                return None

        except Exception as e:
            logger.error(f"Error generating mock predictions: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return None

    # HPC mode: Submit real PBS job
    import subprocess
    import tempfile

    logger.info(f"🖥️  Running in HPC mode - submitting PBS job for {model_name}")

    # ED_ConvLSTM uses a different interface than other models
    if model_name == "ED_ConvLSTM":
        # ED_ConvLSTM: Pass dates directly as environment variables (format: DD-MM-YYYY-HH-MM)
        start_str = actual_start_dt.strftime("%d-%m-%Y-%H-%M")
        end_str = end_dt.strftime("%d-%m-%Y-%H-%M")

        # Step 1: Get the PBS script path
        pbs_script_path = (
            Path(__file__).parent.parent
            / "pbs_scripts"
            / "start_end_pred_scripts"
            / f"run_{model_name}_inference_startend.sh"
        )

        if not pbs_script_path.exists():
            logger.error(f"PBS script not found: {pbs_script_path}")
            return None

        # Step 2: Modify PBS script to inject START_DATE and END_DATE
        try:
            with open(pbs_script_path, "r") as f:
                script_content = f.read()

            # Replace $START_DATE and $END_DATE with actual values
            modified_script = script_content.replace('"$START_DATE"', f'"{start_str}"')
            modified_script = modified_script.replace('"$END_DATE"', f'"{end_str}"')

            # Write modified script to temp file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as tmp:
                tmp.write(modified_script)
                tmp_script_path = tmp.name

            logger.info(f"Created modified PBS script for ED_ConvLSTM: {tmp_script_path}")
            logger.info(f"START_DATE={start_str}, END_DATE={end_str}")

        except Exception as e:
            logger.error(f"Failed to modify PBS script: {e}")
            return None

        # Step 3: Submit the modified PBS job
        command = ["qsub", tmp_script_path]

        logger.info(f"Submitting PBS job for {model_name} (range: {start_dt} to {end_dt})")
        logger.info(f"Command: {' '.join(command)}")

    else:
        # Other models: Use YAML config approach

        # Step 1: Modify the YAML config with date range (use adjusted start time)
        try:
            config_path = modify_yaml_config_for_date_range(model_name, actual_start_dt, end_dt)
            logger.info(f"Modified config for {model_name}: {config_path}")
            logger.info(f"Config will use adjusted range: {actual_start_dt} to {end_dt}")
        except Exception as e:
            logger.error(f"Failed to modify config for {model_name}: {e}")
            return None

        # Step 2: Get the PBS script path
        pbs_script_path = (
            Path(__file__).parent.parent
            / "pbs_scripts"
            / "start_end_pred_scripts"
            / f"run_{model_name}_inference_startend.sh"
        )

        if not pbs_script_path.exists():
            logger.error(f"PBS script not found: {pbs_script_path}")
            return None

        # Step 3: Modify PBS script to use absolute config path
        try:
            with open(pbs_script_path, "r") as f:
                script_content = f.read()

            # Replace $CFG_PATH with absolute path
            modified_script = script_content.replace('--cfg_path "$CFG_PATH"', f'--cfg_path "{config_path}"')

            # Write modified script to temp file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as tmp:
                tmp.write(modified_script)
                tmp_script_path = tmp.name

            logger.info(f"Created modified PBS script: {tmp_script_path}")

        except Exception as e:
            logger.error(f"Failed to modify PBS script: {e}")
            return None

        # Step 4: Submit the modified PBS job
        command = ["qsub", tmp_script_path]

        logger.info(f"Submitting PBS job for {model_name} (range: {start_dt} to {end_dt})")
        logger.info(f"Command: {' '.join(command)}")
        logger.info(f"Config path: {config_path}")

    # Submit the job (common for both ED_ConvLSTM and other models)
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)

        # Extract job ID from output (format: "123456.davinci-mgt01")
        job_id = result.stdout.strip().split(".")[0]
        logger.info(f"✅ [{model_name}] Job submitted successfully! Job ID: {job_id}")

        # Clean up temp file
        try:
            Path(tmp_script_path).unlink()
        except:
            pass

        return job_id

    except subprocess.CalledProcessError as e:
        logger.error(f"❌ [{model_name}] Failed to submit PBS job!")
        logger.error(f"Error: {e.stderr.strip() if e.stderr else 'Unknown error'}")
        # Clean up temp file
        try:
            Path(tmp_script_path).unlink()
        except:
            pass
        return None
    except Exception as e:
        logger.error(f"❌ [{model_name}] Unexpected error submitting job: {e}")
        # Clean up temp file
        try:
            Path(tmp_script_path).unlink()
        except:
            pass
        return None


def create_groundtruth_figures(all_timestamps, gt_raw_data, gt_figures, gt_found_count, gt_missing_count):
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
                    # Store raw data for difference calculation
                    gt_raw_data[timestamp] = gt_data
                    # Create figure for ground truth (target)
                    fig = compute_figure_gpd(gt_data, timestamp.strftime("%d/%m/%Y %H:%M"))
                    gt_figures[timestamp] = fig
                    gt_found_count += 1
            except Exception as e:
                logger.warning(f"Error loading GT at {gt_path}: {e}")
                gt_missing_count += 1
                return gt_figures, gt_raw_data, gt_found_count, gt_missing_count
        else:
            gt_missing_count += 1
            logger.debug(f"GT not found for {timestamp}: {gt_filename}")

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

        # Difference +30: target30 - pred30
        for timestamp in pred_30_figures.keys():
            if timestamp in target30_raw_data and timestamp in pred_30_raw_data:
                try:
                    diff_array = np.abs(target30_raw_data[timestamp] - pred_30_raw_data[timestamp])
                    fig = compute_figure_gpd(diff_array, timestamp.strftime("%d/%m/%Y %H:%M"), name="diff")
                    diff_30_figures[timestamp] = fig
                except Exception as e:
                    logger.warning(f"Error computing diff +30 for {timestamp}: {e}")

        # Difference +60: target60 - pred60
        for timestamp in pred_60_figures.keys():
            if timestamp in target60_raw_data and timestamp in pred_60_raw_data:
                try:
                    diff_array = np.abs(target60_raw_data[timestamp] - pred_60_raw_data[timestamp])
                    fig = compute_figure_gpd(diff_array, timestamp.strftime("%d/%m/%Y %H:%M"), name="diff")
                    diff_60_figures[timestamp] = fig
                except Exception as e:
                    logger.warning(f"Error computing diff +60 for {timestamp}: {e}")

        logger.info(f"✅ Differences computed: {len(diff_30_figures)} +30min, {len(diff_60_figures)} +60min")

        # ========== STEP 4: Get GIF paths ==========
        gif_paths = get_gif_paths(model_name, start_dt, end_dt)

        # Create sidebar_args for compatibility (not heavily used in new function)
        sidebar_args = {
            "model_name": model_name,
            "start_date": start_dt.date(),
            "start_time": start_dt.time(),
            "end_date": end_dt.date(),
            "end_time": end_dt.time(),
        }

        # ========== STEP 5: Create all 7 GIFs in parallel ==========
        logger.info("🎬 Creating GIFs in parallel...")

        # Import required modules for parallel processing
        import io
        import streamlit as st
        from multiprocessing import Manager, Process
        import imageio
        from PIL import Image

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
        import time
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
        import traceback

        logger.error(traceback.format_exc())
        return False
