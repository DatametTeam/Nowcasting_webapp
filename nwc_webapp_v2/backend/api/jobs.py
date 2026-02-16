"""
Jobs API endpoints - submit and monitor prediction jobs.

This replaces the inline job submission + monitoring loops that live
in nowcasting.py, prediction_by_date.py, model_comparison.py, and csi_analysis.py.

In Streamlit, job monitoring was a while-loop with time.sleep(2) that blocked
the page. Here, the frontend polls GET /api/jobs/{id}/status every few seconds,
or (later) receives updates via WebSocket.
"""
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from nwc_webapp.config.environment import is_hpc
from nwc_webapp.hpc.jobs import submit_date_range_prediction_job
from nwc_webapp.hpc.pbs import is_pbs_available

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


# ============================================================================
# Request/Response models
# ============================================================================
# Pydantic models define the shape of data the API accepts and returns.
# This is like defining a dataclass, but FastAPI uses it to:
# 1. Validate incoming data (reject bad requests automatically)
# 2. Generate API documentation
# 3. Provide type hints


class JobSubmitRequest(BaseModel):
    """
    What the frontend sends when submitting a job.

    In Streamlit this was spread across sidebar_args dict and function params.
    Here it's a clean, typed, validated model.
    """
    model: str
    start: str  # ISO format: "2025-01-01T12:00"
    end: str    # ISO format: "2025-01-01T13:00"


class JobSubmitResponse(BaseModel):
    job_id: str | None
    model: str
    start: str
    end: str
    is_mock: bool
    success: bool
    error: str | None = None


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/submit", response_model=JobSubmitResponse)
async def submit_job(request: JobSubmitRequest):
    """
    Submit a prediction job (PBS on HPC, mock locally).

    In Streamlit (nowcasting.py lines ~533-537):
        with st.spinner(f"Submitting job for {model_name}..."):
            job_id = submit_date_range_prediction_job(model_name, start_dt, end_dt)
        st.success(f"✅ Job submitted! Job ID: {job_id}")

    Here: POST /api/jobs/submit with JSON body, returns job ID.
    The frontend handles the UI (spinner, success message) itself.
    """
    try:
        start_dt = datetime.fromisoformat(request.start)
        end_dt = datetime.fromisoformat(request.end)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid datetime format: {e}")

    if end_dt < start_dt:
        raise HTTPException(status_code=400, detail="End must be after start")

    # Submit the job (uses PBS on HPC, mock locally - same as Streamlit)
    job_id = submit_date_range_prediction_job(request.model, start_dt, end_dt)

    is_mock = not is_hpc() or (job_id and job_id.startswith("mock_"))

    return JobSubmitResponse(
        job_id=job_id,
        model=request.model,
        start=request.start,
        end=request.end,
        is_mock=is_mock,
        success=job_id is not None,
        error=None if job_id else f"Failed to submit job for {request.model}",
    )


@router.get("/status")
async def get_job_status(
    model: str = Query(..., description="Model name"),
):
    """
    Check PBS job status for a model.

    In Streamlit (scattered everywhere):
        current_status = get_model_job_status(model)
        if current_status == "Q": ...
        elif current_status == "R": ...

    Here: returns the status as JSON. The frontend polls this endpoint.

    Returns:
        status: "Q" (queued), "R" (running), null (not in queue / completed)
    """
    if not is_pbs_available():
        return {"model": model, "status": None, "pbs_available": False}

    try:
        from nwc_webapp.hpc.pbs import get_model_job_status
        status = get_model_job_status(model)
    except Exception as e:
        return {"model": model, "status": None, "error": str(e)}

    return {
        "model": model,
        "status": status,
        "pbs_available": True,
    }