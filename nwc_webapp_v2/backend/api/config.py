"""
Config API endpoints.

HOW FASTAPI ROUTERS WORK:
==========================
Instead of putting all endpoints in main.py (which would get huge),
we split them into "routers" — separate files for related endpoints.

This is like how your Streamlit app has separate files in pages/.

A router is created with APIRouter(), then endpoints are added to it.
In main.py, we "include" the router so FastAPI knows about these endpoints.

The prefix="/api/config" means all routes here start with /api/config:
  @router.get("/")       → GET /api/config/
  @router.get("/models") → GET /api/config/models
"""
from fastapi import APIRouter

from nwc_webapp.config.config import get_config
from nwc_webapp.config.environment import is_hpc, is_local

router = APIRouter(prefix="/api/config", tags=["config"])


@router.get("/")
async def get_app_config():
    """
    Return full application configuration.

    This replaces the module-level `app_config = get_config()` in your
    Streamlit app.py. Instead of loading config at import time, the
    frontend requests it when needed.
    """
    config = get_config()

    return {
        "models": config.models,
        "environment": "hpc" if is_hpc() else "local",
        "sri_folder": str(config.sri_folder),
        "gif_storage": str(config.gif_storage),
        "real_time_pred": str(config.real_time_pred),
        "csi_thresholds": config.csi_threshold,
    }


@router.get("/models")
async def get_models():
    """
    Return list of available models.

    In Streamlit: model_list = app_config.models
    Here: GET /api/config/models → {"models": ["ConvLSTM", ...]}
    """
    config = get_config()
    return {"models": config.models}


@router.get("/environment")
async def get_environment():
    """
    Return environment info (HPC vs local).

    In Streamlit: from nwc_webapp.config.environment import is_hpc
    Here: GET /api/config/environment → {"is_hpc": false, "is_local": true}
    """
    return {
        "is_hpc": is_hpc(),
        "is_local": is_local(),
        "mode": "hpc" if is_hpc() else "local",
    }