"""
Configuration management for the nowcasting application.
Loads settings from YAML and provides easy access with type hints.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from dataclasses import dataclass

from nwc_webapp.config.environment import is_hpc, is_local
from nwc_webapp.logging_config import setup_logger

# Set up logger
logger = setup_logger(__name__)

@dataclass
class VisualizationConfig:
    """Visualization settings."""
    italy_bounds: Dict[str, float]
    map_center: List[float]
    zoom_start: int
    data_shape: Dict[str, int]
    colormap: Dict[str, Any]
    gif: Dict[str, Any]


@dataclass
class SourceGridConfig:
    """Source grid navigation parameters (Transverse Mercator)."""
    projection: str
    prj_lat: float
    prj_lon: float
    ncols: int
    nlines: int
    cOff: int
    lOff: int
    cRes: int
    lRes: int


@dataclass
class DestGridConfig:
    """Destination grid navigation parameters (Geographic lat/lon)."""
    projection: str
    prj_lat: float
    prj_lon: float
    ncols: int
    nlines: int
    minLon: float
    maxLon: float
    minLat: float
    maxLat: float


@dataclass
class PredictionConfig:
    """Prediction settings."""
    num_input_timesteps: int
    num_forecast_timesteps: int
    timestep_minutes: int
    num_sequences: int
    display_times: List[int]
    time_options: List[str]


@dataclass
class PBSConfig:
    """PBS/HPC settings."""
    queue: str
    target_gpu: str
    walltime: str
    log_folder: str
    environments: Dict[str, str]


@dataclass
class AutoRefreshConfig:
    """Auto-refresh settings."""
    interval_seconds: int
    check_interval: int
    refresh_on_minute_multiple: int


@dataclass
class LoggingConfig:
    """Logging settings."""
    level: str
    log_to_file: bool
    log_to_console: bool


class Config:
    """
    Main configuration class.
    Loads settings from YAML and provides structured access.
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize configuration.

        Args:
            config_path: Path to config YAML file. If None, uses default.
        """
        if config_path is None:
            # config.py is in src/nwc_webapp/config/, resources is in src/nwc_webapp/resources/
            config_path = Path(__file__).parent.parent / "resources" / "cfg" / "cfg.yaml"

        self.config_path = config_path
        self._config: Dict[str, Any] = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            self._config = yaml.safe_load(f)

    def reload(self) -> None:
        """Reload configuration from file."""
        self._load_config()

    # Models
    @property
    def models(self) -> List[str]:
        """Get list of available models."""
        return self._config.get("models", [])

    @property
    def csi_threshold(self) -> List[int]:
        """Get CSI thresholds."""
        return self._config.get("csi_threshold", [])

    # Paths (environment-aware)
    def get_paths(self) -> Dict[str, str]:
        """Get paths based on current environment."""
        if is_hpc():
            return self._config.get("hpc_paths", {})
        else:
            return self._config.get("local_paths", {})

    @property
    def sri_folder(self) -> Path:
        """Get SRI data folder path."""
        return Path(self.get_paths().get("sri_folder", "data/mock_sri"))

    @property
    def prediction_output(self) -> Path:
        """Get prediction output folder path."""
        return Path(self.get_paths().get("prediction_output", "data/predictions"))

    @property
    def data_root(self) -> Path:
        """Get data root folder path."""
        return Path(self.get_paths().get("data_root", "data"))

    @property
    def real_time_pred(self) -> Path:
        """Get real-time prediction folder path."""
        return Path(self.get_paths().get("real_time_pred", "data/predictions/real_time_pred"))

    @property
    def radar_mask_path(self) -> Path:
        """Get radar mask file path (relative to package root)."""
        rel_path = self._config.get("radar_mask_path", "resources/mask/radar_mask.hdf")
        # Resolve relative to package root (config.py is in config/ subfolder)
        return Path(__file__).parent.parent / rel_path

    @property
    def shapefiles_folder(self) -> Path:
        """Get shapefiles folder path (relative to package root)."""
        rel_path = self._config.get("shapefiles_folder", "resources/shapefiles")
        # Resolve relative to package root (config.py is in config/ subfolder)
        return Path(__file__).parent.parent / rel_path

    @property
    def legends_folder(self) -> Path:
        """Get legends folder path (relative to package root)."""
        rel_path = self._config.get("legends_folder", "resources/legends")
        # Resolve relative to package root (config.py is in config/ subfolder)
        return Path(__file__).parent.parent / rel_path

    # Structured configs
    @property
    def visualization(self) -> VisualizationConfig:
        """Get visualization configuration."""
        viz_config = self._config.get("visualization", {})
        return VisualizationConfig(**viz_config)

    @property
    def source_grid(self) -> SourceGridConfig:
        """Get source grid configuration."""
        grid_config = self._config.get("source_grid", {})
        return SourceGridConfig(**grid_config)

    @property
    def dest_grid(self) -> DestGridConfig:
        """Get destination grid configuration."""
        grid_config = self._config.get("dest_grid", {})
        return DestGridConfig(**grid_config)

    @property
    def prediction(self) -> PredictionConfig:
        """Get prediction configuration."""
        pred_config = self._config.get("prediction", {})
        return PredictionConfig(**pred_config)

    @property
    def pbs(self) -> PBSConfig:
        """Get PBS configuration."""
        pbs_config = self._config.get("pbs", {})
        return PBSConfig(**pbs_config)

    @property
    def auto_refresh(self) -> AutoRefreshConfig:
        """Get auto-refresh configuration."""
        refresh_config = self._config.get("auto_refresh", {})
        return AutoRefreshConfig(**refresh_config)

    @property
    def logging(self) -> LoggingConfig:
        """Get logging configuration."""
        log_config = self._config.get("logging", {})
        return LoggingConfig(**log_config)

    # Convenience methods
    def get_model_pbs_env(self, model_name: str) -> str:
        """
        Get PBS environment for a specific model.

        Args:
            model_name: Name of the model

        Returns:
            Environment name
        """
        environments = self.pbs.environments
        return environments.get(model_name, environments.get("default", "sole24_310"))

    def __repr__(self) -> str:
        """String representation."""
        return f"Config(path={self.config_path}, environment={'HPC' if is_hpc() else 'Local'})"


# Global config instance
_config: Optional[Config] = None


def get_config(config_path: Optional[Path] = None) -> Config:
    """
    Get the global configuration instance.

    Args:
        config_path: Optional path to config file

    Returns:
        Config instance
    """
    global _config

    if _config is None:
        _config = Config(config_path)

    return _config


def reload_config() -> None:
    """Reload the global configuration from file."""
    global _config
    if _config is not None:
        _config.reload()


if __name__ == "__main__":
    # Test the configuration
    config = get_config()

    logger.info(f"Configuration: {config}")
    logger.(f"\nModels: {config.models}")
    logger.(f"SRI Folder: {config.sri_folder}")
    logger.(f"Prediction Output: {config.prediction_output}")
    logger.(f"Map Center: {config.visualization.map_center}")
    logger.(f"Source Grid: {config.source_grid.projection} at ({config.source_grid.prj_lat}, {config.source_grid.prj_lon})")
    logger.(f"Dest Grid: {config.dest_grid.minLat}-{config.dest_grid.maxLat}, {config.dest_grid.minLon}-{config.dest_grid.maxLon}")
    logger.(f"PBS Queue: {config.pbs.queue}")
    logger.(f"ED_ConvLSTM Environment: {config.get_model_pbs_env('ED_ConvLSTM')}")
