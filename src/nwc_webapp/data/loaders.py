"""
Data loading utilities for ground truth and prediction data.
Handles loading from HDF5 and NumPy files with environment-aware paths.
"""
import os
from pathlib import Path
from datetime import datetime, timedelta
import h5py
import numpy as np
import streamlit as st

from nwc_webapp.config.config import get_config
from nwc_webapp.visualization.colormaps import configure_colorbar
from nwc_webapp.geo.warping import warp_map
from nwc_webapp.logging_config import setup_logger

# Set up logger
logger = setup_logger(__name__)

# Get configuration
config = get_config()

# Initialize colormap
cmap, norm, vmin, vmax, null_color, void_color, discrete, ticks = (
    configure_colorbar('R', min_val=None, max_val=None)
)


def read_groundtruth_and_target_data(selected_key, selected_model):
    """
    Read ground truth and target data for a specific timestamp and model.

    Args:
        selected_key: Timestamp key in format "ddmmYYYY_HHMM"
        selected_model: Model name

    Returns:
        Tuple of (gt_dict, target_dict, pred_dict) with timestamped arrays
    """
    # Use config for paths
    out_dir = config.prediction_output / selected_model
    test_dir = config.prediction_output / "Test"

    # Load arrays
    gt_array = np.load(test_dir / "predictions.npy", mmap_mode='r')[0:12, 0]
    target_array = np.load(test_dir / "predictions.npy", mmap_mode='r')[12:24, 0]
    pred_array = np.load(out_dir / "predictions.npy", mmap_mode='r')[12]

    if selected_model == 'Test':
        pred_array = np.load(out_dir / "predictions.npy", mmap_mode='r')[12:24, 0]

    # Load radar mask
    mask_path = Path(__file__).resolve().parent.parent / "resources/mask/radar_mask.hdf"
    with h5py.File(mask_path, "r") as f:
        radar_mask = f["mask"][()]

    # Apply mask
    pred_array = pred_array * radar_mask
    target_array = target_array * radar_mask
    gt_array = gt_array * radar_mask

    # Clean and normalize arrays
    gt_array = np.clip(gt_array, 0, 200)
    pred_array = np.clip(pred_array, 0, 200)
    target_array = np.clip(target_array, 0, 200)

    # Convert selected_key to a datetime object
    selected_time = datetime.strptime(selected_key, "%d%m%Y_%H%M")

    # Create dictionaries for ground truth and predictions
    gt_dict = {}
    pred_dict = {}
    target_dict = {}

    # Fill ground truth dictionary
    for i in range(len(gt_array)):
        timestamp = (selected_time + timedelta(minutes=5 * i)).strftime("%d%m%Y_%H%M")
        gt_dict[timestamp] = gt_array[i]

    for i in range(len(target_array)):
        timestamp = (selected_time + timedelta(minutes=5 * (12 + i))).strftime("%d%m%Y_%H%M")
        target_dict[timestamp] = target_array[i]

    # Fill prediction dictionary
    for i in range(len(pred_array)):
        timestamp = (selected_time + timedelta(minutes=5 * (12 + i))).strftime("%d%m%Y_%H%M")
        pred_dict[timestamp] = pred_array[i]

    return gt_dict, target_dict, pred_dict


def load_prediction_data(st, time_options, latest_file):
    """
    Load prediction data for real-time visualization.

    Args:
        st: Streamlit module
        time_options: List of time options
        latest_file: Latest input file name

    Returns:
        RGBA image array or None
    """
    if not (st.session_state.selected_model and st.session_state.selected_time):
        return None

    selected_model = st.session_state.selected_model
    selected_time = st.session_state.selected_time

    latest_npy = Path(latest_file).stem + '.npy'

    try:
        # Use config for prediction paths
        if selected_model == 'ED_ConvLSTM':
            pred_path = config.real_time_pred / selected_model / latest_npy
            img1 = np.load(pred_path)[0, time_options.index(selected_time)]
        else:
            pred_path = config.prediction_output / selected_model / "predictions.npy"
            img1 = np.load(pred_path)[0, time_options.index(selected_time)]
    except FileNotFoundError:
        logger.warning(f"Prediction file not present yet: {pred_path}")
        return None

    img1[img1 < 0] = 0

    # Load radar mask
    mask_path = Path(__file__).resolve().parent.parent / "resources/mask/radar_mask.hdf"
    with h5py.File(mask_path, "r") as f:
        radar_mask = f["mask"][()]

    img1 = img1 * radar_mask

    # Warp image using warping function (reads grid params from config)
    img1 = warp_map(img1)
    img1 = np.nan_to_num(img1, nan=0)

    img1[img1 < 0] = 0
    img1 = img1.astype(float)

    # Apply colormap
    img_norm = norm(img1)
    rgba_img = cmap(img_norm)
    return rgba_img


def load_all_predictions(st, time_options, latest_file):
    """
    Load ground truth and prediction timesteps for real-time animated visualization.

    Args:
        st: Streamlit module
        time_options: List of time options (e.g., ["-30min", ..., "+5min", ..., "+60min"])
        latest_file: Latest input file name

    Returns:
        Tuple of (rgba_images_dict, status_info) where:
        - rgba_images_dict: Dictionary mapping time_option -> RGBA image array, or None if loading fails
        - status_info: Dictionary with keys 'ground_truth_available', 'predictions_available', 'error'
    """
    if not st.session_state.selected_model:
        return None, {'ground_truth_available': False, 'predictions_available': False, 'error': 'No model selected'}

    # Split time_options into ground truth (negative/zero) and predictions (positive)
    ground_truth_times = [t for t in time_options if t.startswith('-') or t == "0min"]
    prediction_times = [t for t in time_options if t.startswith('+')]

    # Load radar mask once
    mask_path = Path(__file__).resolve().parent.parent / "resources/mask/radar_mask.hdf"
    with h5py.File(mask_path, "r") as f:
        radar_mask = f["mask"][()]

    rgba_images = {}
    status_info = {
        'ground_truth_available': False,
        'ground_truth_count': 0,
        'predictions_available': False,
        'error': None
    }

    # ===== Load Ground Truth Data (past SRI files) =====
    ground_truth_loaded = 0
    if ground_truth_times:
        try:
            # Parse latest file timestamp (format: DD-MM-YYYY-HH-MM.hdf)
            filename = Path(latest_file).stem
            dt = datetime.strptime(filename, "%d-%m-%Y-%H-%M")

            # Get SRI folder from config and ensure it's a Path object
            sri_folder = config.sri_folder
            if not isinstance(sri_folder, Path):
                sri_folder = Path(sri_folder)

            logger.debug(f"SRI folder type: {type(sri_folder)}, value: {sri_folder}")

            # Load past SRI files
            for time_option in ground_truth_times:
                # Extract minutes offset (e.g., "-30min" -> -30)
                minutes_offset = int(time_option.replace("min", ""))

                # Calculate past timestamp
                past_dt = dt + timedelta(minutes=minutes_offset)
                past_filename = past_dt.strftime("%d-%m-%Y-%H-%M") + ".hdf"
                past_filepath = sri_folder / past_filename

                logger.debug(f"Looking for ground truth file: {past_filepath} (type: {type(past_filepath)})")

                if past_filepath.exists():
                    # Load SRI file
                    with h5py.File(past_filepath, "r") as f:
                        # Try common dataset names
                        if 'dataset1/data1/data' in f:
                            img = f['dataset1/data1/data'][()].astype(float)

                    img[img < 0] = 0

                    # Apply mask
                    img = img * radar_mask

                    # Warp image
                    img = warp_map(img)
                    img = np.nan_to_num(img, nan=0)
                    img[img < 0] = 0

                    # Flip image vertically to correct orientation
                    img = np.flipud(img)

                    # Apply colormap
                    img_norm = norm(img)
                    rgba_img = cmap(img_norm)

                    rgba_images[time_option] = rgba_img
                    ground_truth_loaded += 1
                    logger.debug(f"Loaded ground truth {time_option}: {past_filename}")
                else:
                    logger.error(f"Ground truth file not found: {past_filepath}")
                    # Create empty/zero image as fallback
                    img = np.zeros_like(radar_mask).astype(float)
                    img_norm = norm(img)
                    rgba_img = cmap(img_norm)
                    rgba_images[time_option] = rgba_img

            # Check if we got any ground truth data
            if ground_truth_loaded > 0:
                status_info['ground_truth_available'] = True
                status_info['ground_truth_count'] = ground_truth_loaded
            else:
                status_info['error'] = f"No ground truth data found (checked {len(ground_truth_times)} files)"
                logger.error(f"No ground truth data found! Expected {len(ground_truth_times)} files in {sri_folder}")

        except Exception as e:
            status_info['error'] = f"Failed to load ground truth: {str(e)}"
            logger.error(f"Failed to load ground truth data: {e}", exc_info=True)

    # ===== Load Prediction Data =====
    selected_model = st.session_state.selected_model
    latest_npy = Path(latest_file).stem + '.npy'

    try:
        # Special handling for TEST model - use test directory
        if selected_model == "TEST":
            pred_path = config.real_time_pred.parent / "test_predictions" / selected_model / latest_npy
        else:
            # All other models use the same path structure in real_time_pred
            pred_path = config.real_time_pred / selected_model / latest_npy

        pred_array = np.load(pred_path)[0]  # Load all 12 timesteps
        logger.debug(f"Loaded predictions from: {pred_path}")
    except FileNotFoundError:
        logger.warning(f"Prediction file not present yet: {pred_path}")
        # Return ground truth if available, otherwise None
        if ground_truth_times and ground_truth_loaded > 0:
            return rgba_images, status_info
        else:
            return None, status_info

    # Process all 12 prediction timesteps
    status_info['predictions_available'] = True
    for i, time_option in enumerate(prediction_times):
        img = pred_array[i].copy()
        img[img < 0] = 0

        # Apply mask
        img = img * radar_mask

        # Warp image
        img = warp_map(img)
        img = np.nan_to_num(img, nan=0)
        img[img < 0] = 0
        img = img.astype(float)

        # Flip image vertically to correct orientation
        img = np.flipud(img)

        # Debug: Check if images are different
        if i == 0:
            logger.debug(f"Prediction {time_option}: min={img.min():.2f}, max={img.max():.2f}, mean={img.mean():.2f}")
        elif i == len(prediction_times) - 1:
            logger.debug(f"Prediction {time_option}: min={img.min():.2f}, max={img.max():.2f}, mean={img.mean():.2f}")

        # Apply colormap
        img_norm = norm(img)
        rgba_img = cmap(img_norm)

        rgba_images[time_option] = rgba_img

    logger.debug(f"Loaded {ground_truth_loaded}/{len(ground_truth_times)} ground truth + {len(prediction_times)} prediction timesteps = {len(rgba_images)} total")
    return rgba_images, status_info