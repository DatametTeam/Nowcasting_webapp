"""
Jobs API endpoints - submit and monitor prediction jobs.

This replaces the inline job submission + monitoring loops that live
in nowcasting.py, prediction_by_date.py, model_comparison.py, and csi_analysis.py.

In Streamlit, job monitoring was a while-loop with time.sleep(2) that blocked
the page. Here, the frontend polls GET /api/jobs/{id}/status every few seconds,
or (later) receives updates via WebSocket.
"""
import os
from datetime import datetime
from pathlib import Path

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

    # is_mock=True tells the frontend to skip PBS status polling and just watch
    # the predictions folder. This covers three cases:
    #   - local mock mode (job_id starts with "mock_")
    #   - watch-folder mode (submit_jobs=false, job_id == "watch_only")
    #   - anything that isn't a real PBS job
    is_mock = not is_hpc() or bool(job_id and (job_id.startswith("mock_") or job_id == "watch_only"))

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
    job_id: str = Query(None, description="PBS job ID for direct lookup (avoids matching wrong jobs)"),
):
    """
    Check PBS job status for a model.

    When job_id is provided, checks that specific job directly via qstat -f.
    This avoids the old model-name substring search which could match
    unrelated jobs (e.g. a real-time job matching a range job's model name).

    Without job_id, falls back to searching by model name (legacy behavior).

    Returns:
        status: "Q" (queued), "R" (running), null (not in queue / completed)
    """
    if not is_pbs_available():
        return {"model": model, "status": None, "pbs_available": False}

    try:
        if job_id:
            # Direct lookup by job ID — fast and accurate
            from nwc_webapp.hpc.pbs import get_job_status as pbs_get_job_status
            status = pbs_get_job_status(job_id)
            # get_job_status returns "ended" when job is no longer in queue
            if status == "ended":
                status = None
        else:
            # Fallback: search by model name (legacy, may match wrong jobs)
            from nwc_webapp.hpc.pbs import get_model_job_status
            status = get_model_job_status(model)
    except Exception as e:
        return {"model": model, "status": None, "error": str(e)}

    return {
        "model": model,
        "status": status,
        "pbs_available": True,
    }


@router.get("/error-log")
async def get_job_error_log(
    model: str = Query(..., description="Model name"),
    job_id: str = Query(..., description="PBS job ID (numeric part)"),
):
    """
    Read the PBS output file for a failed job.

    PBS creates output files named like:
      ~/nwc_{model}_range.o{job_id}

    This endpoint reads that file (last 200 lines) so the frontend
    can show the user what went wrong.
    """
    # Construct the PBS output file path
    # PBS job names are like "nwc_ConvLSTM_range", output file: ~/nwc_ConvLSTM_range.o12345
    job_name = f"nwc_{model}_range"
    filename = f"{job_name}.o{job_id}"
    log_path = Path.home() / filename

    if not log_path.exists():
        # Also try the configured pbs_logs directory as fallback
        fallback = Path.home() / "pbs_logs" / "pbs.log"
        if fallback.exists():
            log_path = fallback
        else:
            return {
                "found": False,
                "log": None,
                "path": str(log_path),
            }

    try:
        text = log_path.read_text(errors="replace")
        # Truncate to last 200 lines to avoid huge responses
        lines = text.splitlines()
        if len(lines) > 200:
            lines = lines[-200:]
            text = "... (truncated, showing last 200 lines) ...\n" + "\n".join(lines)
        else:
            text = "\n".join(lines)

        return {
            "found": True,
            "log": text,
            "path": str(log_path),
        }
    except Exception as e:
        return {
            "found": False,
            "log": f"Error reading log file: {e}",
            "path": str(log_path),
        }