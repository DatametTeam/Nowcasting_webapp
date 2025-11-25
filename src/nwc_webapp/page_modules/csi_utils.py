"""
Utility functions for CSI analysis.
"""
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import pandas as pd

from nwc_webapp.config.config import get_config
from nwc_webapp.config.environment import is_hpc
from nwc_webapp.evaluation.metrics import compute_CSI
from nwc_webapp.logging_config import setup_logger
from nwc_webapp.page_modules.nowcasting_utils import generate_timestamp_range

# Set up logger
logger = setup_logger(__name__)


def load_range_prediction_data(
    model_name: str,
    start_dt: datetime,
    end_dt: datetime
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Load groundtruth and prediction data for a date range.

    For each timestamp in the range, loads:
    - Target data (groundtruth) at t+60, t+65, ..., t+115 (12 frames)
    - Prediction data from model (12 frames)

    Args:
        model_name: Model name
        start_dt: Start datetime
        end_dt: End datetime

    Returns:
        Tuple of (target_dict, pred_dict) where each is {timestamp_key: array}
        Keys are in format "ddmmYYYY_HHMM" to match existing CSI functions
    """
    config = get_config()

    # Load radar mask
    mask_path = Path(__file__).resolve().parent.parent / "resources/mask/radar_mask.hdf"
    with h5py.File(mask_path, "r") as f:
        radar_mask = f["mask"][()]

    # Generate all timestamps in range
    all_timestamps = generate_timestamp_range(start_dt, end_dt, verbose=False)

    target_dict = {}
    pred_dict = {}

    logger.info(f"Loading data for {len(all_timestamps)} timestamps ({start_dt} to {end_dt})")

    for timestamp in all_timestamps:
        # Load prediction file (contains 12 timesteps at 5-min intervals)
        pred_filename = timestamp.strftime('%d-%m-%Y-%H-%M') + '.npy'
        pred_path = config.real_time_pred / model_name / pred_filename

        if not pred_path.exists():
            logger.warning(f"Prediction not found: {pred_path}")
            continue

        try:
            pred_array = np.load(pred_path, mmap_mode='r')  # Shape: (12, 1400, 1200)

            # For each of the 12 prediction timesteps (corresponding to t+60 to t+115)
            for i in range(12):
                # Prediction at timestamp + 60 + (i*5) minutes
                pred_time = timestamp + timedelta(minutes=60 + 5 * i)
                pred_data = pred_array[i]

                # Apply mask and clip
                pred_data = pred_data * radar_mask
                pred_data = np.clip(pred_data, 0, 200)

                # Store with key in format "ddmmYYYY_HHMM"
                pred_key = pred_time.strftime("%d%m%Y_%H%M")
                pred_dict[pred_key] = pred_data

                # Load corresponding target (groundtruth) data
                target_filename = pred_time.strftime('%d-%m-%Y-%H-%M') + '.hdf'

                # Determine path based on environment
                target_path = None

                if is_hpc():
                    # HPC: Try data1 first, then data (archived)
                    target_path_data1 = Path('/davinci-1/work/protezionecivile/data1/SRI_adj') / target_filename
                    year = pred_time.strftime('%Y')
                    month = pred_time.strftime('%m')
                    day = pred_time.strftime('%d')
                    target_path_data = Path(f'/davinci-1/work/protezionecivile/data/{year}/{month}/{day}/SRI_adj') / target_filename

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
                        with h5py.File(target_path, 'r') as hdf:
                            target_data = hdf['/dataset1/data1/data'][:]
                            # Apply mask and clip
                            target_data = target_data * radar_mask
                            target_data = np.clip(target_data, 0, 200)

                            target_dict[pred_key] = target_data
                    except Exception as e:
                        logger.warning(f"Error loading target at {target_path}: {e}")
                else:
                    logger.warning(f"Target data not found for {pred_time}: {target_filename}")

        except Exception as e:
            logger.error(f"Error loading prediction {pred_path}: {e}")

    logger.info(f"Loaded {len(target_dict)} target frames and {len(pred_dict)} prediction frames")

    return target_dict, pred_dict


def compute_csi_for_models(
    models: List[str],
    start_dt: datetime,
    end_dt: datetime,
    thresholds: List[float] = None
) -> Dict[str, pd.DataFrame]:
    """
    Compute CSI scores for multiple models over a date range, preserving lead time dimension.

    For each model, computes CSI averaged across all base timestamps in the interval,
    but keeps lead times separate (+5min, +10min, ..., +60min).

    Args:
        models: List of model names
        start_dt: Start datetime
        end_dt: End datetime
        thresholds: CSI thresholds (default: [1, 5, 10, 20, 50])

    Returns:
        Dictionary mapping model names to DataFrames where:
        - Rows: Thresholds (mm/h)
        - Columns: Lead times (5, 10, 15, ..., 60 minutes)
        - Values: CSI scores averaged across all timestamps in the interval
    """
    if thresholds is None:
        config = get_config()
        thresholds = config.csi_thresholds if hasattr(config, 'csi_thresholds') else [1, 5, 10, 20, 50]

    logger.info(f"Computing CSI for {len(models)} models from {start_dt} to {end_dt}")

    # Store CSI dataframes (one per model)
    model_csi_dfs = {}

    for model in models:
        logger.info(f"Processing {model}...")

        try:
            config = get_config()

            # Load radar mask
            mask_path = Path(__file__).resolve().parent.parent / "resources/mask/radar_mask.hdf"
            with h5py.File(mask_path, "r") as f:
                radar_mask = f["mask"][()]

            # Generate all base timestamps in range
            all_timestamps = generate_timestamp_range(start_dt, end_dt, verbose=False)

            # Initialize storage for CSI by lead time
            # Structure: csi_by_leadtime[lead_time_idx][threshold] = [csi values across timestamps]
            csi_by_leadtime = {lt: {th: [] for th in thresholds} for lt in range(12)}

            logger.info(f"Loading and computing CSI for {len(all_timestamps)} timestamps...")

            for timestamp in all_timestamps:
                # Load prediction file
                pred_filename = timestamp.strftime('%d-%m-%Y-%H-%M') + '.npy'
                pred_path = config.real_time_pred / model / pred_filename

                if not pred_path.exists():
                    logger.warning(f"Prediction not found: {pred_path}")
                    continue

                try:
                    pred_array = np.load(pred_path, mmap_mode='r')  # Shape: (12, 1400, 1200)

                    # For each of the 12 lead times
                    for lead_time_idx in range(12):
                        # Target time = timestamp + 60 + (lead_time_idx * 5) minutes
                        target_time = timestamp + timedelta(minutes=60 + 5 * lead_time_idx)
                        target_filename = target_time.strftime('%d-%m-%Y-%H-%M') + '.hdf'

                        # Load target (groundtruth)
                        target_path = None
                        if is_hpc():
                            target_path_data1 = Path('/davinci-1/work/protezionecivile/data1/SRI_adj') / target_filename
                            year = target_time.strftime('%Y')
                            month = target_time.strftime('%m')
                            day = target_time.strftime('%d')
                            target_path_data = Path(f'/davinci-1/work/protezionecivile/data/{year}/{month}/{day}/SRI_adj') / target_filename

                            if target_path_data1.exists():
                                target_path = target_path_data1
                            elif target_path_data.exists():
                                target_path = target_path_data
                        else:
                            target_path_local = config.sri_folder / target_filename
                            if target_path_local.exists():
                                target_path = target_path_local

                        if target_path and target_path.exists():
                            try:
                                with h5py.File(target_path, 'r') as hdf:
                                    target_data = hdf['/dataset1/data1/data'][:]
                                    target_data = target_data * radar_mask
                                    target_data = np.clip(target_data, 0, 200)

                                    # Get prediction data for this lead time
                                    pred_data = pred_array[lead_time_idx]
                                    pred_data = pred_data * radar_mask
                                    pred_data = np.clip(pred_data, 0, 200)

                                    # Compute CSI for each threshold
                                    from nwc_webapp.evaluation.metrics import CSI
                                    for th in thresholds:
                                        csi_value = CSI(target_data, pred_data, threshold=th)
                                        if csi_value is not None:
                                            csi_by_leadtime[lead_time_idx][th].append(csi_value)

                            except Exception as e:
                                logger.warning(f"Error loading target at {target_path}: {e}")

                except Exception as e:
                    logger.error(f"Error processing prediction {pred_path}: {e}")

            # Average CSI across all timestamps for each lead time
            lead_time_labels = [f"{5 * (i + 1)}" for i in range(12)]  # "5", "10", ..., "60"
            csi_matrix = []

            for th in thresholds:
                row = []
                for lead_time_idx in range(12):
                    csi_values = csi_by_leadtime[lead_time_idx][th]
                    if csi_values:
                        avg_csi = np.mean(csi_values)
                        row.append(avg_csi)
                    else:
                        row.append(0.0)  # Or np.nan
                csi_matrix.append(row)

            # Create DataFrame: rows=thresholds, columns=lead times
            model_df = pd.DataFrame(csi_matrix, index=thresholds, columns=lead_time_labels)
            model_df.index.name = "Threshold (mm/h)"
            model_df.columns.name = "Lead Time (min)"

            model_csi_dfs[model] = model_df

            logger.info(f"✅ Computed CSI for {model}")

        except Exception as e:
            logger.error(f"Error computing CSI for {model}: {e}")
            import traceback
            logger.error(traceback.format_exc())

    if not model_csi_dfs:
        logger.error("No CSI data computed for any model")
        return None

    logger.info(f"CSI computation completed for {len(model_csi_dfs)} models")

    return model_csi_dfs