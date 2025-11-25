"""
Data loading utilities for predictions.
Handles loading from HDF5, NPY, and other file formats.
"""

import os
from pathlib import Path

import h5py
import numpy as np

from nwc_webapp.logging_config import setup_logger
from nwc_webapp.utils import get_latest_file_once

# Set up logger
logger = setup_logger(__name__)


def get_prediction_results_test(folder_path, sidebar_args, get_only_pred=False):
    """
    Load prediction results for testing.

    Args:
        folder_path: Path to data folder
        sidebar_args: Dictionary with model configuration
        get_only_pred: If True, only load predictions not ground truth

    Returns:
        Tuple of (gt_array, pred_array)
    """
    model_name = sidebar_args["model_name"]
    gt_array = None
    pred_array = None

    if not get_only_pred:
        logger.info("Loading GT data")

        # NEW for test
        gt_array = get_latest_file_once()
        logger.info("GT data loaded")

        gt_array = np.array(gt_array)
        gt_array[gt_array < 0] = 0

    logger.info("Loading pred data")

    # NEW for test
    pred_array = get_latest_file_once()
    if model_name == "Test":  # TODO: sistemare
        # NEW for test
        pred_array = get_latest_file_once()
    pred_array = np.array(pred_array)
    pred_array[pred_array < 0] = 0
    logger.info("Loaded pred data")

    logger.debug("Loading radar mask")
    src_fold = Path(__file__).resolve().parent.parent
    with h5py.File(os.path.join(src_fold, "resources/mask/radar_mask.hdf"), "r") as f:
        radar_mask = f["mask"][()]
    logger.info("Radar mask loaded")

    pred_array = pred_array * radar_mask

    logger.info("*** LOADED DATA ***")

    return gt_array, pred_array


def get_prediction_results(out_dir, sidebar_args, get_only_pred=False):
    """
    Load prediction results from disk.

    Args:
        out_dir: Output directory path
        sidebar_args: Dictionary with model configuration
        get_only_pred: If True, only load predictions not ground truth

    Returns:
        Tuple of (gt_array, pred_array)
    """
    # TODO: da fixare
    model_name = sidebar_args["model_name"]
    pred_out_dir = Path(f"/davinci-1/work/protezionecivile/sole24/pred_teo/Test")
    model_out_dir = Path(f"/davinci-1/work/protezionecivile/sole24/pred_teo/{model_name}")
    gt_array = None

    if not get_only_pred:
        logger.debug("Loading GT data")
        gt_array = np.load(pred_out_dir / "predictions.npy", mmap_mode="r")[12:36]
        logger.info("GT data loaded")
        gt_array = np.array(gt_array)
        gt_array[gt_array < 0] = 0
        # gt_array[gt_array > 200] = 200
        # gt_array = (gt_array - np.min(gt_array)) / (np.max(gt_array) - np.min(gt_array))

    logger.debug("Loading pred data")
    pred_array = np.load(model_out_dir / "predictions.npy", mmap_mode="r")[0:24]
    if model_name == "Test":  # TODO: sistemare
        pred_array = np.load(model_out_dir / "predictions.npy", mmap_mode="r")[12:36]
    pred_array = np.array(pred_array)
    pred_array[pred_array < 0] = 0
    logger.info("Loaded pred data")
    # pred_array[pred_array > 200] = 200
    # pred_array = (pred_array - np.min(pred_array)) / (np.max(pred_array) - np.min(pred_array))
    logger.debug("Loading radar mask")
    src_fold = Path(__file__).resolve().parent.parent
    with h5py.File(os.path.join(src_fold, "resources/mask/radar_mask.hdf"), "r") as f:
        radar_mask = f["mask"][()]
    logger.info("Radar mask loaded")

    pred_array = pred_array * radar_mask

    logger.info("*** LOADED DATA ***")

    return gt_array, pred_array
