"""
Configuration module for nowcasting webapp.
Centralizes all configuration, environment settings, constants, and logging.
"""

# Main config management
from nwc_webapp.config.config import (
    get_config,
    reload_config,
    Config,
    VisualizationConfig,
    ProjectionConfig,
    PredictionConfig,
    PBSConfig,
    AutoRefreshConfig,
    LoggingConfig
)

# Environment detection and paths
from nwc_webapp.config.environment import (
    detect_environment,
    is_hpc,
    is_local,
    get_data_root,
    get_sri_folder,
    get_prediction_output_dir
)

# Constants
from nwc_webapp.config.constants import (
    OUTPUT_DATA_DIR,
    TARGET_GPU
)

# Logging setup
from nwc_webapp.config.logging_config import (
    setup_logging,
    get_logger
)

__all__ = [
    # Config classes and functions
    'get_config',
    'reload_config',
    'Config',
    'VisualizationConfig',
    'ProjectionConfig',
    'PredictionConfig',
    'PBSConfig',
    'AutoRefreshConfig',
    'LoggingConfig',
    # Environment
    'detect_environment',
    'is_hpc',
    'is_local',
    'get_data_root',
    'get_sri_folder',
    'get_prediction_output_dir',
    # Constants
    'OUTPUT_DATA_DIR',
    'TARGET_GPU',
    # Logging
    'setup_logging',
    'get_logger',
]