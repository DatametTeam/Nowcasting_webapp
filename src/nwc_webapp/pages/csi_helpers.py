from nwc_webapp.evaluation.metrics import CSI, POD, FAR, FSS
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from nwc_webapp.data.checking import generate_timestamp_range
from nwc_webapp.data.predictions import load_prediction_array

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
            # Use helper to handle model-specific shapes (ED_ConvLSTM: (1,12,H,W) vs others: (12,H,W))
            pred_array = load_prediction_array(pred_path, model_name)

            if pred_array is None:
                logger.warning(f"Failed to load prediction: {pred_path}")
                continue

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


def compute_csi_for_single_model(
    model: str,
    start_dt: datetime,
    end_dt: datetime,
    thresholds: List[float],
    window_sizes: List[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[float, pd.Series]]:
    """
    Compute CSI, POD, FAR, and FSS for a single model (helper function for parallel processing).

    Args:
        model: Model name
        start_dt: Start datetime
        end_dt: End datetime
        thresholds: CSI thresholds
        window_sizes: FSS window sizes (default: [5, 10, 20, 40, 80])

    Returns:
        Tuple of (CSI_df, POD_df, FAR_df, FSS_dict) where:
        - CSI_df, POD_df, FAR_df: DataFrames with rows=thresholds, columns=lead times
        - FSS_dict: Dict[threshold, Series] where Series has window_sizes as index
    """
    logger.info(f"🔄 [{model}] Starting CSI/POD/FAR/FSS computation...")

    try:
        config = get_config()

        # Set default window sizes if not provided
        if window_sizes is None:
            window_sizes = config.fss_window_sizes if hasattr(config, 'fss_window_sizes') else [5, 10, 20, 40, 80]

        # Load radar mask
        mask_path = Path(__file__).resolve().parent.parent / "resources/mask/radar_mask.hdf"
        with h5py.File(mask_path, "r") as f:
            radar_mask = f["mask"][()]

        # Generate all base timestamps in range
        all_timestamps = generate_timestamp_range(start_dt, end_dt, verbose=False)

        # Initialize storage for CSI, POD, FAR by lead time
        csi_by_leadtime = {lt: {th: [] for th in thresholds} for lt in range(12)}
        pod_by_leadtime = {lt: {th: [] for th in thresholds} for lt in range(12)}
        far_by_leadtime = {lt: {th: [] for th in thresholds} for lt in range(12)}

        # Initialize storage for FSS: {threshold: {window_size: [values]}}
        fss_storage = {th: {ws: [] for ws in window_sizes} for th in thresholds}

        logger.info(f"📊 [{model}] Loading predictions for {len(all_timestamps)} timestamps...")

        for timestamp in all_timestamps:
            # Load prediction file
            pred_filename = timestamp.strftime('%d-%m-%Y-%H-%M') + '.npy'
            pred_path = config.real_time_pred / model / pred_filename

            if not pred_path.exists():
                continue

            try:
                # Use helper to handle model-specific shapes (ED_ConvLSTM: (1,12,H,W) vs others: (12,H,W))
                pred_array = load_prediction_array(pred_path, model)

                if pred_array is None:
                    logger.warning(f"[{model}] Failed to load prediction: {pred_path}")
                    continue

                # For each of the 12 lead times
                for lead_time_idx in range(12):
                    target_time = timestamp + timedelta(minutes=5 * (lead_time_idx + 1))
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

                                pred_data = pred_array[lead_time_idx]
                                pred_data = pred_data * radar_mask
                                pred_data = np.clip(pred_data, 0, 200)

                                # Compute CSI, POD, FAR, FSS for each threshold
                                for th in thresholds:
                                    csi_value = CSI(target_data, pred_data, threshold=th)
                                    pod_value = POD(target_data, pred_data, threshold=th)
                                    far_value = FAR(target_data, pred_data, threshold=th)

                                    if csi_value is not None:
                                        csi_by_leadtime[lead_time_idx][th].append(csi_value)
                                    if pod_value is not None:
                                        pod_by_leadtime[lead_time_idx][th].append(pod_value)
                                    if far_value is not None:
                                        far_by_leadtime[lead_time_idx][th].append(far_value)

                                    # Compute FSS for each window size (averaged across all lead times)
                                    for ws in window_sizes:
                                        fss_value = FSS(target_data, pred_data, threshold=th, window_size=ws)
                                        if fss_value is not None:
                                            fss_storage[th][ws].append(fss_value)

                        except Exception as e:
                            logger.warning(f"[{model}] Error loading target at {target_path}: {e}")

            except Exception as e:
                logger.error(f"[{model}] Error processing prediction {pred_path}: {e}")

        # Average CSI, POD, FAR across all timestamps for each lead time
        lead_time_labels = [f"{5 * (i + 1)}" for i in range(12)]
        csi_matrix = []
        pod_matrix = []
        far_matrix = []

        for th in thresholds:
            csi_row = []
            pod_row = []
            far_row = []

            for lead_time_idx in range(12):
                # CSI
                csi_values = csi_by_leadtime[lead_time_idx][th]
                if csi_values:
                    csi_row.append(np.mean(csi_values))
                else:
                    csi_row.append(0.0)

                # POD
                pod_values = pod_by_leadtime[lead_time_idx][th]
                if pod_values:
                    pod_row.append(np.mean(pod_values))
                else:
                    pod_row.append(0.0)

                # FAR
                far_values = far_by_leadtime[lead_time_idx][th]
                if far_values:
                    far_row.append(np.mean(far_values))
                else:
                    far_row.append(0.0)

            csi_matrix.append(csi_row)
            pod_matrix.append(pod_row)
            far_matrix.append(far_row)

        # Create DataFrames
        csi_df = pd.DataFrame(csi_matrix, index=thresholds, columns=lead_time_labels)
        csi_df.index.name = "Threshold (mm/h)"
        csi_df.columns.name = "Lead Time (min)"

        pod_df = pd.DataFrame(pod_matrix, index=thresholds, columns=lead_time_labels)
        pod_df.index.name = "Threshold (mm/h)"
        pod_df.columns.name = "Lead Time (min)"

        far_df = pd.DataFrame(far_matrix, index=thresholds, columns=lead_time_labels)
        far_df.index.name = "Threshold (mm/h)"
        far_df.columns.name = "Lead Time (min)"

        # Create FSS results: Dict[threshold, Series(window_sizes)]
        fss_results = {}
        for th in thresholds:
            fss_row = []
            for ws in window_sizes:
                fss_values = fss_storage[th][ws]
                if fss_values:
                    fss_row.append(np.mean(fss_values))
                else:
                    fss_row.append(0.0)

            fss_results[th] = pd.Series(fss_row, index=window_sizes, name=model)

        logger.info(f"✅ [{model}] CSI/POD/FAR/FSS computation completed!")
        return csi_df, pod_df, far_df, fss_results

    except Exception as e:
        logger.error(f"❌ [{model}] Error computing CSI/POD/FAR/FSS: {e}")
        logger.error(traceback.format_exc())
        return None, None, None, None


def compute_csi_for_models(
    models: List[str],
    start_dt: datetime,
    end_dt: datetime,
    thresholds: List[float] = None,
    window_sizes: List[int] = None
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[float, pd.DataFrame]]:
    """
    Compute CSI, POD, FAR, and FSS scores for multiple models over a date range in parallel.

    For each model, computes metrics averaged across all base timestamps in the interval,
    but keeps lead times separate (+5min, +10min, ..., +60min).

    Args:
        models: List of model names
        start_dt: Start datetime
        end_dt: End datetime
        thresholds: CSI thresholds (default: [1, 5, 10, 20, 50])
        window_sizes: FSS window sizes (default: [5, 10, 20, 40, 80])

    Returns:
        Tuple of (CSI_dict, POD_dict, FAR_dict, FSS_dict) where:
        - CSI_dict, POD_dict, FAR_dict: Dict[model, DataFrame(thresholds × lead_times)]
        - FSS_dict: Dict[threshold, DataFrame(window_sizes × models)]
    """
    if thresholds is None:
        config = get_config()
        thresholds = config.csi_thresholds if hasattr(config, 'csi_thresholds') else [1, 5, 10, 20, 50]

    if window_sizes is None:
        config = get_config()
        window_sizes = config.fss_window_sizes if hasattr(config, 'fss_window_sizes') else [5, 10, 20, 40, 80]

    logger.info(f"🚀 Computing CSI/POD/FAR/FSS for {len(models)} models in parallel from {start_dt} to {end_dt}")

    # Store dataframes (one per model for each metric)
    model_csi_dfs = {}
    model_pod_dfs = {}
    model_far_dfs = {}
    model_fss_dicts = {}  # Dict[model, Dict[threshold, Series]]

    # Use ThreadPoolExecutor for parallel processing

    with ThreadPoolExecutor(max_workers=min(len(models), 4)) as executor:
        # Submit all tasks
        future_to_model = {
            executor.submit(compute_csi_for_single_model, model, start_dt, end_dt, thresholds, window_sizes): model
            for model in models
        }

        # Collect results as they complete
        for future in as_completed(future_to_model):
            model = future_to_model[future]
            try:
                csi_df, pod_df, far_df, fss_dict = future.result()
                if csi_df is not None and pod_df is not None and far_df is not None and fss_dict is not None:
                    model_csi_dfs[model] = csi_df
                    model_pod_dfs[model] = pod_df
                    model_far_dfs[model] = far_df
                    model_fss_dicts[model] = fss_dict
            except Exception as e:
                logger.error(f"❌ [{model}] Exception during CSI/POD/FAR/FSS computation: {e}")

    if not model_csi_dfs:
        logger.error("No CSI data computed for any model")
        return None, None, None, None

    # Reorganize FSS results: from Dict[model, Dict[threshold, Series]]
    # to Dict[threshold, DataFrame(window_sizes × models)]
    fss_by_threshold = {}
    for th in thresholds:
        # Collect Series from all models for this threshold
        series_list = []
        for model in models:
            if model in model_fss_dicts:
                series = model_fss_dicts[model][th]
                series.name = model  # Rename series to model name
                series_list.append(series)

        if series_list:
            # Combine into DataFrame: rows=window_sizes, columns=models
            fss_df = pd.concat(series_list, axis=1)
            fss_df.index.name = "Window Size (px)"
            fss_by_threshold[th] = fss_df

    logger.info(f"CSI/POD/FAR/FSS computation completed for {len(model_csi_dfs)} models")

    return model_csi_dfs, model_pod_dfs, model_far_dfs, fss_by_threshold