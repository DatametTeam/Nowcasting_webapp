"""
Metrics API endpoints - CSI, POD, FAR, FSS computation.

This replaces the huge show_csi_analysis_page() in csi_analysis.py.
The computation logic (in csi_helpers.py) is pure Python and reused directly.
The UI (tables, charts, buttons) moves to the Vue frontend.
"""
from datetime import datetime
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/metrics", tags=["metrics"])


class MetricsRequest(BaseModel):
    models: List[str]
    start: str  # ISO format
    end: str    # ISO format


class ComparisonRequest(BaseModel):
    models: List[str]
    timestamp: str  # ISO format


@router.post("/compute")
async def compute_metrics(request: MetricsRequest):
    """
    Compute CSI, POD, FAR, FSS for selected models in a date range.

    In Streamlit (csi_analysis.py lines ~464-475):
        csi_results, pod_results, far_results, fss_results = compute_csi_for_models(
            models=models_with_predictions, start_dt=start_datetime, end_dt=end_datetime
        )

    Here: same function call, but we return the results as JSON.

    NOTE: This can be slow for large date ranges. In Phase 1.6, we'll
    make it a background task with progress updates via WebSocket.
    For now, it runs synchronously (the frontend shows a loading spinner).
    """
    try:
        start_dt = datetime.fromisoformat(request.start)
        end_dt = datetime.fromisoformat(request.end)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid datetime format: {e}")

    if not request.models:
        raise HTTPException(status_code=400, detail="At least one model is required")

    try:
        from nwc_webapp.pages.csi_helpers import compute_csi_for_models

        csi_results, pod_results, far_results, fss_results, nmse_results, beta_results = compute_csi_for_models(
            models=request.models,
            start_dt=start_dt,
            end_dt=end_dt,
        )

        if csi_results is None:
            raise HTTPException(status_code=500, detail="Failed to compute metrics")

        # Convert DataFrames to JSON-serializable dicts
        # Each result is Dict[model_name, DataFrame]
        # DataFrame has: index=thresholds, columns=lead_times
        response = {
            "models": request.models,
            "start": request.start,
            "end": request.end,
            "csi": {model: df.to_dict() for model, df in csi_results.items()},
            "pod": {model: df.to_dict() for model, df in pod_results.items()},
            "far": {model: df.to_dict() for model, df in far_results.items()},
            "fss": {},
            "regression": {},
        }

        # FSS has a different structure: Dict[threshold, DataFrame]
        # where DataFrame has: index=window_sizes, columns=models
        if fss_results:
            for threshold, df in fss_results.items():
                response["fss"][str(threshold)] = df.to_dict()

        # NMSE and beta: Dict[model, Series(lead_times)]
        if nmse_results and beta_results:
            for model in request.models:
                model_reg = {}
                if model in nmse_results:
                    model_reg["nmse"] = nmse_results[model].to_dict()
                if model in beta_results:
                    model_reg["beta"] = beta_results[model].to_dict()
                if model_reg:
                    response["regression"][model] = model_reg

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error computing metrics: {str(e)}")


@router.post("/comparison")
async def compute_comparison_metrics(request: ComparisonRequest):
    """
    Compute CSI for multiple models at a single timestamp, across all 12 lead times.

    Used by the Model Comparison tab to show per-row CSI tables alongside
    the GT + prediction images. Returns CSI at each threshold for every
    (model, lead_time) combination.
    """
    try:
        dt = datetime.fromisoformat(request.timestamp)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid datetime format: {e}")

    if len(request.models) < 1:
        raise HTTPException(status_code=400, detail="At least one model is required")

    try:
        from nwc_webapp.config.config import get_config
        from nwc_webapp.data.groundtruth import load_groundtruth_for_timestamp
        from nwc_webapp.data.predictions import load_prediction_array
        from nwc_webapp.evaluation.metrics import CSI

        config = get_config()
        thresholds = config.csi_threshold

        # Load groundtruth: shape (12, H, W)
        gt_data = load_groundtruth_for_timestamp(dt)
        if gt_data is None:
            raise HTTPException(status_code=404, detail="Groundtruth data not available for this timestamp")

        # Load predictions for each model: {model_name: (12, H, W)}
        predictions = {}
        for model in request.models:
            pred_filename = dt.strftime("%d-%m-%Y-%H-%M") + ".npy"
            pred_path = config.real_time_pred / model / pred_filename
            if pred_path.exists():
                pred_array = load_prediction_array(pred_path, model)
                if pred_array is not None:
                    predictions[model] = pred_array

        if not predictions:
            raise HTTPException(status_code=404, detail="No prediction data found for any model")

        # Compute CSI for each lead time
        lead_times = []
        for lt_idx in range(12):
            minutes = (lt_idx + 1) * 5
            gt_frame = gt_data[lt_idx]

            csi_per_model = {}
            for model_name, pred_array in predictions.items():
                pred_frame = pred_array[lt_idx]
                model_csi = {}
                values = []
                for th in thresholds:
                    csi_val = CSI(gt_frame, pred_frame, threshold=th)
                    # Convert to float for JSON serialization (numpy types aren't serializable)
                    if csi_val is not None:
                        csi_val = round(float(csi_val), 4)
                        values.append(csi_val)
                    model_csi[str(th)] = csi_val
                model_csi["avg"] = round(float(np.mean(values)), 4) if values else None
                csi_per_model[model_name] = model_csi

            lead_times.append({
                "index": lt_idx,
                "minutes": minutes,
                "csi": csi_per_model,
            })

        return {
            "thresholds": thresholds,
            "models": list(predictions.keys()),
            "lead_times": lead_times,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error computing comparison metrics: {str(e)}")