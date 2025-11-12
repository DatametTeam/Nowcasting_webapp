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

# Lazy import of sou_py to avoid circular import issues
# sou_py has legacy imports that expect 'sou_py' not 'nwc_webapp.sou_py'
def _get_dpg():
    """Lazy import of dpg module."""
    import sys
    # Temporarily add src to path for sou_py compatibility
    src_path = str(Path(__file__).resolve().parent.parent)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    import sou_py.dpg as dpg
    return dpg

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
            pred_path = config.prediction_output / "real_time_pred" / selected_model / latest_npy
            img1 = np.load(pred_path)[0, time_options.index(selected_time)]
        else:
            pred_path = config.prediction_output / selected_model / "predictions.npy"
            img1 = np.load(pred_path)[0, time_options.index(selected_time)]
    except FileNotFoundError:
        print(f"Prediction file not present yet: {pred_path}")
        return None

    img1[img1 < 0] = 0

    # Load radar mask
    mask_path = Path(__file__).resolve().parent.parent / "resources/mask/radar_mask.hdf"
    with h5py.File(mask_path, "r") as f:
        radar_mask = f["mask"][()]

    img1 = img1 * radar_mask

    # Load nodes for warping
    dpg = _get_dpg()
    root_dir = Path(__file__).resolve().parent.parent.parent.parent
    sourceNode = dpg.tree.createTree(str(root_dir / "data/nodes/sourceNode"))
    destNode = dpg.tree.createTree(str(root_dir / "data/nodes/destNode"))
    img1 = dpg.warp.warp_map(sourceNode, destNode=destNode, source_data=img1)
    img1 = np.nan_to_num(img1, nan=0)

    img1[img1 < 0] = 0
    img1 = img1.astype(float)

    # Apply colormap
    img_norm = norm(img1)
    rgba_img = cmap(img_norm)
    return rgba_img