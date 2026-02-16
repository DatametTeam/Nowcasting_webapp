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

static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="frontend")


# ============================================================================
# Startup event
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Runs once when the server starts."""
    from nwc_webapp.config.config import get_config
    from nwc_webapp.config.environment import is_hpc

    config = get_config()
    env = "HPC" if is_hpc() else "Local"

    print("=" * 60)
    print(f"  Weather Nowcasting API v2.0 ({env} mode)")
    print(f"  Models: {', '.join(config.models)}")
    print(f"  Docs:   http://localhost:8000/docs")
    print(f"  Health: http://localhost:8000/api/health")
    print("=" * 60)