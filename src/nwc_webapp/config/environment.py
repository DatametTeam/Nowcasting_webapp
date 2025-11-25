"""
Environment detection for running on HPC vs local development.
"""

import os
import subprocess
from pathlib import Path
from typing import Literal

EnvironmentType = Literal["hpc", "local"]


def detect_environment() -> EnvironmentType:
    """
    Detect if we're running on HPC or local machine.

    Returns:
        "hpc" if running on HPC cluster, "local" otherwise
    """
    # Check for HPC-specific paths
    if Path("/davinci-1").exists():
        return "hpc"

    # Check if PBS is available
    try:
        result = subprocess.run(["qstat"], capture_output=True, timeout=2)
        if result.returncode == 0:
            return "hpc"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Check environment variables that might indicate HPC
    hpc_indicators = ["PBS_JOBID", "SLURM_JOB_ID", "PBS_O_WORKDIR"]
    if any(os.environ.get(var) for var in hpc_indicators):
        return "hpc"

    return "local"


def is_hpc() -> bool:
    """Check if running on HPC."""
    return detect_environment() == "hpc"


def is_local() -> bool:
    """Check if running on local machine."""
    return detect_environment() == "local"


# Global environment detection
ENVIRONMENT = detect_environment()
IS_HPC = ENVIRONMENT == "hpc"
IS_LOCAL = ENVIRONMENT == "local"


def get_data_root() -> Path:
    """Get the root data directory based on environment."""
    if IS_HPC:
        return Path("/davinci-1/work/protezionecivile")
    else:
        # Use local data directory
        return Path(__file__).parent.parent.parent / "data"


def get_sri_folder() -> Path:
    """Get the SRI data folder based on environment."""
    if IS_HPC:
        return Path("/davinci-1/work/protezionecivile/data1/SRI_adj")
    else:
        # Use local mock data
        return get_data_root() / "mock_sri"


def get_prediction_output_dir() -> Path:
    """Get the prediction output directory based on environment."""
    if IS_HPC:
        return Path("/davinci-1/work/protezionecivile/sole24/pred_teo")
    else:
        return get_data_root() / "predictions"
