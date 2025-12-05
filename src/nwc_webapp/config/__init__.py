"""
Configuration module for nowcasting webapp.
Centralizes all configuration, environment settings, constants, and logging.
"""

# Main config management
from nwc_webapp.config.config import (
    AutoRefreshConfig,
    Config,
    DestGridConfig,
    LoggingConfig,
    PBSConfig,
    PredictionConfig,
    SourceGridConfig,
    VisualizationConfig,
    get_config,
    reload_config,
)

# Constants
from nwc_webapp.config.constants import OUTPUT_DATA_DIR, TARGET_GPU

# Environment detection and paths
from nwc_webapp.config.environment import (
    detect_environment,
    get_data_root,
    get_prediction_output_dir,
    get_sri_folder,
    is_hpc,
    is_local,
)

# Logging setup
from nwc_webapp.logging_config import setup_logger

__all__ = [
    # Config classes and functions
    "get_config",
    "reload_config",
    "Config",
    "VisualizationConfig",
    "SourceGridConfig",
    "DestGridConfig",
    "PredictionConfig",
    "PBSConfig",
    "AutoRefreshConfig",
    "LoggingConfig",
    # Environment
    "detect_environment",
    "is_hpc",
    "is_local",
    "get_data_root",
    "get_sri_folder",
    "get_prediction_output_dir",
    # Constants
    "OUTPUT_DATA_DIR",
    "TARGET_GPU",
    # Logging
    "setup_logger",
]
