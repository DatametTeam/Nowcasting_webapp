"""
Real-time prediction API endpoints.

Three endpoints that control the backend RealtimeService:
  POST /api/realtime/start   → start the background prediction loop
  POST /api/realtime/stop    → stop it cleanly
  GET  /api/realtime/status  → poll current state (models, SRI, notification)
"""

from fastapi import APIRouter

from services.realtime import RealtimeService

router = APIRouter(prefix="/api/realtime", tags=["realtime"])


@router.post("/start")
async def start_realtime():
    """Start the real-time prediction loop (HPC or local mock)."""
    service = RealtimeService()
    return service.start()


@router.post("/stop")
async def stop_realtime():
    """Stop the real-time prediction loop."""
    service = RealtimeService()
    return service.stop()


@router.get("/status")
async def get_realtime_status():
    """Get the current real-time state (polled by frontend every 3s)."""
    service = RealtimeService()
    return service.get_state()
