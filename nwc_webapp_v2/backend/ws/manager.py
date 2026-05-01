"""
WebSocket connection manager — broadcast realtime state to all connected clients.

Usage:
    from ws.manager import ws_manager
    await ws_manager.broadcast({"type": "state_update", "data": state})
    await ws_manager.connect(websocket)
    ws_manager.disconnect(websocket)
"""

import asyncio
import logging
from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    def __init__(self):
        self._connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self._connections.append(ws)
        logger.debug("WS client connected (%d total)", len(self._connections))

    def disconnect(self, ws: WebSocket):
        if ws in self._connections:
            self._connections.remove(ws)
        logger.debug("WS client disconnected (%d remaining)", len(self._connections))

    async def broadcast(self, message: dict):
        if not self._connections:
            return
        dead = []
        for ws in list(self._connections):
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    def broadcast_sync(self, message: dict):
        """Thread-safe broadcast from the RealtimeService background thread."""
        if not self._connections:
            return
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.run_coroutine_threadsafe(self.broadcast(message), loop)
        except RuntimeError:
            pass  # No event loop — server shutting down


ws_manager = ConnectionManager()
