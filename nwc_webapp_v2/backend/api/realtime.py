"""
Real-time prediction API endpoints.

Three endpoints that control the backend RealtimeService:
  POST /api/realtime/start   → start the background prediction loop
  POST /api/realtime/stop    → stop it cleanly
  GET  /api/realtime/status  → poll current state (models, SRI, notification)
"""

from fastapi import APIRouter, HTTPException

from services.realtime import RealtimeService
from nwc_webapp.config.environment import is_server

router = APIRouter(prefix="/api/realtime", tags=["realtime"])


@router.post("/start")
async def start_realtime():
    """Start the real-time prediction loop (HPC or local mock)."""
    if is_server():
        # In server mode the loop is always running (auto-started on uvicorn startup).
        return {"ok": True, "reason": "always_running"}
    service = RealtimeService()
    return service.start()


@router.post("/stop")
async def stop_realtime():
    """Stop the real-time prediction loop."""
    if is_server():
        raise HTTPException(status_code=403, detail="Real-time loop cannot be stopped in server mode.")
    service = RealtimeService()
    return service.stop()


@router.get("/status")
async def get_realtime_status():
    """Get the current real-time state (polled by frontend every 3s)."""
    service = RealtimeService()
    return service.get_state()
