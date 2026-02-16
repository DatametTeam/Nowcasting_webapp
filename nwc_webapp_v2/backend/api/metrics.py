"""
Metrics API endpoints - CSI, POD, FAR, FSS computation.

This replaces the huge show_csi_analysis_page() in csi_analysis.py.
The computation logic (in csi_helpers.py) is pure Python and reused directly.
The UI (tables, charts, buttons) moves to the Vue frontend.
"""
from datetime import datetime
from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/metrics", tags=["metrics"])


class MetricsRequest(BaseModel):
    models: List[str]
    start: str  # ISO format
    end: str    # ISO format


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

        csi_results, pod_results, far_results, fss_results = compute_csi_for_models(
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
        }

        # FSS has a different structure: Dict[threshold, DataFrame]
        # where DataFrame has: index=window_sizes, columns=models
        if fss_results:
            for threshold, df in fss_results.items():
                response["fss"][str(threshold)] = df.to_dict()

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error computing metrics: {str(e)}")