"""
Real-time prediction API endpoints.

Three endpoints that control the backend RealtimeService:
  POST /api/realtime/start   → start the background prediction loop
  POST /api/realtime/stop    → stop it cleanly
  GET  /api/realtime/status  → poll current state (models, SRI, notification)
"""

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from services.realtime import RealtimeService
from nwc_webapp.config.environment import is_server
from ws.manager import ws_manager

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


@router.websocket("/ws")
async def realtime_ws(ws: WebSocket):
    """
    WebSocket endpoint — pushes state_update messages to the client whenever
    new SRI data arrives or model statuses change.  The client uses this to
    trigger its search window immediately instead of waiting for the 5-min
    clock-aligned poll.

    On connect the current state is sent once so the client is in sync.
    The connection stays open indefinitely; the client may send any message
    (e.g. a ping) and it is silently ignored.
    """
    await ws_manager.connect(ws)
    service = RealtimeService()
    try:
        # Send current state immediately so the client bootstraps without an extra HTTP call
        await ws.send_json({"type": "state_update", "data": service.get_state()})
        # Keep the connection alive — we don't expect messages from the client,
        # but we must await something to block (otherwise the handler returns and
        # FastAPI closes the socket).
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        ws_manager.disconnect(ws)
