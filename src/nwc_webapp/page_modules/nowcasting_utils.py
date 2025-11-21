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


def check_missing_predictions(model_name: str, start_dt: datetime, end_dt: datetime, verbose: bool = True) -> Tuple[List[datetime], List[datetime]]:
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
    import os

    config = get_config()
    pred_dir = config.real_time_pred / model_name

    # Generate all expected timestamps (don't log during monitoring)
    all_timestamps = generate_timestamp_range(start_dt, end_dt, verbose=verbose)

    missing = []
    existing = []

    # DEBUG: On first verbose call, log the directory and list actual files
    if verbose:
        logger.info(f"[{model_name}] Checking prediction directory: {pred_dir}")
        if pred_dir.exists():
            try:
                actual_files = sorted([f for f in os.listdir(pred_dir) if f.endswith('.npy')])
                logger.info(f"[{model_name}] Found {len(actual_files)} .npy files in directory")
                if actual_files:
                    logger.info(f"[{model_name}] Sample files: {actual_files[:5]}")  # Show first 5
            except Exception as e:
                logger.error(f"[{model_name}] Error listing directory: {e}")
        else:
            logger.warning(f"[{model_name}] Prediction directory does not exist: {pred_dir}")

    for timestamp in all_timestamps:
        # Format: DD-MM-YYYY-HH-MM.npy (same as real-time predictions)
        filename = timestamp.strftime('%d-%m-%Y-%H-%M') + '.npy'
        pred_file = pred_dir / filename

        # Use os.path.exists() to avoid any Path caching issues
        file_exists = os.path.exists(str(pred_file))

        if file_exists:
            existing.append(timestamp)
        else:
            missing.append(timestamp)

        # DEBUG: Log first few checks when verbose
        if verbose and len(existing) + len(missing) <= 3:
            logger.debug(f"[{model_name}] Checking {filename}: {'EXISTS' if file_exists else 'MISSING'}")

    if verbose:
        logger.info(f"[{model_name}] Range check: {len(existing)}/{len(all_timestamps)} predictions exist, {len(missing)} missing")

    return missing, existing


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
        filename = timestamp.strftime('%d-%m-%Y-%H-%M') + '.npy'
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
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)

    # Format dates as "YYYY-MM-DD HH:MM"
    start_str = start_dt.strftime("%Y-%m-%d %H:%M")
    end_str = end_dt.strftime("%Y-%m-%d %H:%M")

    # Modify the start_date and end_date fields
    if 'dataframe_strategy' in config_data and 'args' in config_data['dataframe_strategy']:
        config_data['dataframe_strategy']['args']['start_date'] = start_str
        config_data['dataframe_strategy']['args']['end_date'] = end_str
        logger.info(f"Modified {model_name} config: start={start_str}, end={end_str}")
    else:
        logger.warning(f"Could not find dataframe_strategy.args in {model_name} config")

    # Overwrite the original file
    with open(config_path, 'w') as f:
        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Overwritten config at: {config_path}")
    return config_path


def submit_date_range_prediction_job(model_name: str, start_dt: datetime, end_dt: datetime) -> Optional[str]:
    """
    Submit PBS job for date-range predictions.

    This function:
    1. Modifies the YAML config with start/end dates
    2. Modifies the PBS script to use absolute path for config
    3. Submits the PBS job using the modified script
    4. Returns the job ID

    Args:
        model_name: Model name (e.g., 'ConvLSTM', 'IAM4VP', 'PredFormer', 'SPROG')
        start_dt: Start datetime
        end_dt: End datetime

    Returns:
        Job ID string if successful, None if failed
    """
    import subprocess
    import tempfile

    # Step 1: Modify the YAML config with date range
    try:
        config_path = modify_yaml_config_for_date_range(model_name, start_dt, end_dt)
        logger.info(f"Modified config for {model_name}: {config_path}")
    except Exception as e:
        logger.error(f"Failed to modify config for {model_name}: {e}")
        return None

    # Step 2: Get the PBS script path
    pbs_script_path = Path(__file__).parent.parent / "pbs_scripts" / "start_end_pred_scripts" / f"run_{model_name}_inference_startend.sh"

    if not pbs_script_path.exists():
        logger.error(f"PBS script not found: {pbs_script_path}")
        return None

    # Step 3: Modify PBS script to use absolute config path
    try:
        with open(pbs_script_path, 'r') as f:
            script_content = f.read()

        # Replace $CFG_PATH with absolute path
        modified_script = script_content.replace('--cfg_path "$CFG_PATH"', f'--cfg_path "{config_path}"')

        # Write modified script to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as tmp:
            tmp.write(modified_script)
            tmp_script_path = tmp.name

        logger.info(f"Created modified PBS script: {tmp_script_path}")

    except Exception as e:
        logger.error(f"Failed to modify PBS script: {e}")
        return None

    # Step 4: Submit the modified PBS job
    command = ["qsub", tmp_script_path]

    try:
        logger.info(f"Submitting PBS job for {model_name} (range: {start_dt} to {end_dt})")
        logger.info(f"Command: {' '.join(command)}")
        logger.info(f"Config path: {config_path}")

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


def create_gifs_from_prediction_range(model_name: str, start_dt: datetime, end_dt: datetime, sri_folder_dir: str) -> bool:
    """
    Create sliding window GIFs from a range of prediction files.

    This function:
    1. Loads all predictions in the range
    2. Loads corresponding ground truth data
    3. Creates sliding window GIFs for the entire sequence
    4. Saves GIFs to gif_storage location

    Args:
        model_name: Model name
        start_dt: Start datetime
        end_dt: End datetime
        sri_folder_dir: Path to SRI folder

    Returns:
        True if successful, False otherwise
    """
    from datetime import timedelta
    from nwc_webapp.services.parallel_code import create_sliding_window_gifs
    from nwc_webapp.utils import compute_figure_gpd
    import h5py

    config = get_config()

    logger.info(f"Creating GIFs for {model_name} from {start_dt} to {end_dt}")

    try:
        # Generate all timestamps in the range
        all_timestamps = generate_timestamp_range(start_dt, end_dt)

        # Dictionary to store figures for ground truth and predictions
        gt_figures = {}
        pred_figures = {}

        # Load all predictions and ground truth data
        for timestamp in all_timestamps:
            # Format: DD-MM-YYYY-HH-MM.npy (same as check_missing_predictions)
            filename = timestamp.strftime('%d-%m-%Y-%H-%M') + '.npy'
            pred_path = config.real_time_pred / model_name / filename

            if not pred_path.exists():
                logger.warning(f"Prediction not found: {pred_path}")
                continue

            pred_data = np.load(pred_path)  # Shape: (12, 1400, 1200)

            # Also get date_str and time_str for ground truth paths (old format for SRI files)
            date_str = timestamp.strftime('%d%m%Y')
            time_str = timestamp.strftime('%H%M')

            # Load ground truth (SRI data) for all 12 timesteps
            for t in range(12):
                # Calculate timestamp for this frame (t * 5 minutes ahead)
                frame_time = timestamp + timedelta(minutes=t * 5)
                frame_date_str = frame_time.strftime('%d%m%Y')
                frame_time_str = frame_time.strftime('%H%M')

                # Ground truth path
                gt_path = config.sri_folder / f"SRI_{frame_date_str}_{frame_time_str}.hdf"

                if gt_path.exists():
                    try:
                        with h5py.File(gt_path, 'r') as hdf:
                            gt_data = hdf['/dataset1/data1/data'][:]
                            # Create figure for ground truth
                            fig = compute_figure_gpd(gt_data, frame_time.strftime('%d/%m/%Y %H:%M'))
                            gt_figures[frame_time] = fig
                    except Exception as e:
                        logger.warning(f"Error loading GT at {gt_path}: {e}")

                # Create figure for prediction
                try:
                    fig = compute_figure_gpd(pred_data[t], frame_time.strftime('%d/%m/%Y %H:%M'))
                    pred_figures[frame_time] = fig
                except Exception as e:
                    logger.warning(f"Error creating pred figure at {frame_time}: {e}")

        # Check if we loaded any predictions
        if not pred_figures:
            logger.error(f"No prediction data loaded for {model_name} in range {start_dt} to {end_dt}")
            logger.error("Cannot create GIFs without prediction data")
            return False

        # Create sidebar_args for GIF creation
        sidebar_args = {
            'model_name': model_name,
            'start_date': start_dt.date(),
            'start_time': start_dt.time(),
            'end_date': end_dt.date(),
            'end_time': end_dt.time(),
        }

        # Create sliding window GIFs
        logger.info(f"Creating sliding window GIFs: {len(gt_figures)} GT frames, {len(pred_figures)} pred frames")

        # Create GIFs for ground truth
        if gt_figures:
            create_sliding_window_gifs(
                figures_dict=gt_figures,
                sidebar_args=sidebar_args,
                start_positions=[0, 6, 12],
                save_on_disk=True,
                fps_gif=3,
                name="gt"
            )
            logger.info("Ground truth GIFs created")

        # Create GIFs for predictions
        if pred_figures:
            create_sliding_window_gifs(
                figures_dict=pred_figures,
                sidebar_args=sidebar_args,
                start_positions=[6, 12],
                save_on_disk=True,
                fps_gif=3,
                name="pred"
            )
            logger.info("Prediction GIFs created")

        # Create difference GIFs
        # TODO: Implement difference GIF creation
        logger.info("Difference GIFs creation not yet implemented")

        logger.info(f"✅ GIF creation completed for {model_name}")
        return True

    except Exception as e:
        logger.error(f"❌ Error creating GIFs for {model_name}: {e}")
        return False