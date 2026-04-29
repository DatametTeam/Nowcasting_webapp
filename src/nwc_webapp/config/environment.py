"""
Environment detection for running on HPC vs dedicated server vs local development.
"""

import os
import subprocess
from pathlib import Path
from typing import Literal

import yaml

EnvironmentType = Literal["hpc", "server", "local"]


def _read_mode_from_cfg() -> str:
    """
    Read the 'mode' field from cfg.yaml without going through get_config()
    (which would create a circular import since config.py imports this module).
    """
    try:
        v2_cfg = Path(__file__).parent.parent.parent.parent / "nwc_webapp_v2" / "cfg.yaml"
        cfg_path = v2_cfg if v2_cfg.exists() else Path(__file__).parent / "cfg.yaml"
        with open(cfg_path) as f:
            raw = yaml.safe_load(f)
        return raw.get("mode", "")
    except Exception:
        return ""


def detect_environment() -> EnvironmentType:
    """
    Detect the deployment environment.

    Priority:
    1. Explicit 'mode' field in cfg.yaml (hpc / server / local)
    2. Auto-detection via filesystem and process checks (legacy fallback)
    """
    mode = _read_mode_from_cfg()
    if mode in ("hpc", "server", "local"):
        return mode

    # Legacy auto-detection fallback
    if Path("/davinci-1").exists():
        return "hpc"

    try:
        result = subprocess.run(["qstat"], capture_output=True, timeout=2)
        if result.returncode == 0:
            return "hpc"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    hpc_indicators = ["PBS_JOBID", "SLURM_JOB_ID", "PBS_O_WORKDIR"]
    if any(os.environ.get(var) for var in hpc_indicators):
        return "hpc"

    return "local"


def is_hpc() -> bool:
    """Check if running on HPC cluster (PBS job scheduler)."""
    return detect_environment() == "hpc"


def is_server() -> bool:
    """Check if running on a dedicated GPU server (no job scheduler, real data)."""
    return detect_environment() == "server"


def is_local() -> bool:
    """Check if running in local development mode (mock data)."""
    return detect_environment() == "local"


# Global environment detection
ENVIRONMENT = detect_environment()
IS_HPC = ENVIRONMENT == "hpc"
IS_SERVER = ENVIRONMENT == "server"
IS_LOCAL = ENVIRONMENT == "local"


def get_data_root() -> Path:
    """Get the root data directory based on environment."""
    if IS_HPC:
        return Path("/davinci-1/work/protezionecivile")
    else:
        return Path(__file__).parent.parent.parent / "data"


def get_sri_folder() -> Path:
    """Get the SRI data folder based on environment."""
    if IS_HPC:
        return Path("/davinci-1/work/protezionecivile/data1/SRI_adj")
    else:
        return get_data_root() / "mock_sri"


def get_prediction_output_dir() -> Path:
    """Get the prediction output directory based on environment."""
    if IS_HPC:
        return Path("/davinci-1/work/protezionecivile/sole24/pred_teo")
    else:
        return get_data_root() / "predictions"