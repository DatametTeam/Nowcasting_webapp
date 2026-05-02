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
        # Captured once from the async startup context so background threads
        # can schedule coroutines onto the correct (running) uvicorn event loop.
        # In Python 3.10+, asyncio.get_event_loop() from a background thread
        # returns a new non-running loop, making broadcast_sync a silent no-op.
        self._loop: asyncio.AbstractEventLoop | None = None

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Call once from an async context (e.g. FastAPI startup) to capture the loop."""
        self._loop = loop

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
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        asyncio.run_coroutine_threadsafe(self.broadcast(message), loop)


ws_manager = ConnectionManager()
