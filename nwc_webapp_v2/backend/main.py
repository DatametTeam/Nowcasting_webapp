"""
FastAPI backend for the Weather Nowcasting webapp.

HOW THIS WORKS (for learning):
===============================
FastAPI is a Python web framework. Unlike Streamlit (which re-runs your entire
script on every interaction), FastAPI defines "endpoints" - functions that respond
to HTTP requests.

Key concepts:
- @app.get("/api/something") → defines what happens when the browser requests that URL
- @app.post("/api/something") → defines what happens when the browser SENDS data
- async def → the function can handle multiple requests simultaneously (non-blocking)
- uvicorn → the server that runs FastAPI (like 'streamlit run' runs Streamlit)

To run:
    cd nwc_webapp_v2/backend
    uvicorn main:app --reload --port 8000

Then open http://localhost:8000 in your browser.
FastAPI also auto-generates API docs at http://localhost:8000/docs (very useful!)
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# Import API routers
# Each router is a separate file with related endpoints (like pages/ in Streamlit)
from api.config import router as config_router
from api.data import router as data_router
from api.jobs import router as jobs_router
from api.rendering import router as rendering_router
from api.metrics import router as metrics_router
from api.realtime import router as realtime_router
from api.wind import router as wind_router
from api.lk import router as lk_router, lk_ws_manager
from api.wr10 import router as wr10_router, wr10_ws_manager
from api.fss import router as fss_router, fss_ws_manager

# Create the FastAPI application instance
app = FastAPI(
    title="Weather Nowcasting API",
    description="Backend API for the weather radar nowcasting webapp",
    version="2.0.0",
)

# CORS middleware - allows the Vue frontend (on a different port during development)
# to talk to this backend. Without this, the browser blocks cross-origin requests.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vue dev server (Vite default port)
        "http://localhost:8000",   # Same origin
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Register API routers
# ============================================================================
# Each router handles a group of related endpoints:
#   config_router  → /api/config/*     (app configuration, model list)
#   data_router    → /api/data/*       (predictions, groundtruth, SRI files)
#   jobs_router    → /api/jobs/*       (submit and monitor prediction jobs)
#   rendering_router → /api/render/*   (images, overlays, GIFs)
#   metrics_router → /api/metrics/*    (CSI, POD, FAR, FSS computation)

app.include_router(config_router)
app.include_router(data_router)
app.include_router(jobs_router)
app.include_router(rendering_router)
app.include_router(metrics_router)
app.include_router(realtime_router)
app.include_router(wind_router)
app.include_router(lk_router)
app.include_router(wr10_router)
app.include_router(fss_router)


# ============================================================================
# Health check (stays in main.py - it's the simplest endpoint)
# ============================================================================

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "app": "Weather Nowcasting API",
        "version": "2.0.0",
    }


# ============================================================================
# Static file serving (for the Vue frontend in production)
# ============================================================================
# When the frontend is built (npm run build), it produces static HTML/JS/CSS
# files in backend/static/. FastAPI serves these directly, so we only need
# ONE server running (no separate frontend server needed in production).

# SPA fallback: serve index.html for any non-API route so that Vue Router
# handles client-side paths like /realtime, /nowcasting, etc.
# Without this, refreshing the page on /realtime returns 404 because FastAPI
# looks for a literal file instead of letting the SPA router take over.
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    # Serve actual static assets (JS, CSS, images)
    app.mount("/assets", StaticFiles(directory=str(static_dir / "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """Catch-all: serve the file if it exists, otherwise index.html."""
        file_path = static_dir / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(static_dir / "index.html")


# ============================================================================
# Startup event
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Runs once when the server starts."""
    import asyncio
    from nwc_webapp.config.config import get_config
    from nwc_webapp.config.environment import is_hpc, is_server

    # Capture the running uvicorn event loop so broadcast_sync (called from
    # background threads) can schedule coroutines onto it correctly.
    # asyncio.get_event_loop() from a non-async thread returns a new
    # non-running loop in Python 3.10+, making it a silent no-op.
    from ws.manager import ws_manager
    ws_manager.set_event_loop(asyncio.get_running_loop())
    wr10_ws_manager.set_event_loop(asyncio.get_running_loop())
    fss_ws_manager.set_event_loop(asyncio.get_running_loop())
    lk_ws_manager.set_event_loop(asyncio.get_running_loop())

    # Initialise the config singleton with the explicit path before any
    # route handler calls get_config(). This makes nwc_webapp_v2/cfg.yaml
    # the single source of truth for the new backend.
    cfg_path = Path(__file__).parent.parent / "cfg.yaml"
    config = get_config(config_path=cfg_path)
    env = "HPC" if is_hpc() else ("Server" if is_server() else "Local")

    print("=" * 60)
    print(f"  Weather Nowcasting API v2.0 ({env} mode)")
    print(f"  Models: {', '.join(config.models)}")
    print(f"  Docs:   http://localhost:8000/docs")
    print(f"  Health: http://localhost:8000/api/health")
    print("=" * 60)

    # In server mode, auto-start the real-time loop so it's always running.
    # Users only view predictions — no start/stop controls are shown in the UI.
    if is_server():
        from services.realtime import RealtimeService
        result = RealtimeService().start()
        print(f"  Real-time loop: auto-started ({result})")

        from services.wr10 import WR10Service
        WR10Service().start()
        print(f"  WR10 watcher:   auto-started")
        print("=" * 60)