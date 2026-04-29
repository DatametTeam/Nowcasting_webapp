"""
Configuration management for the nowcasting application.
Loads settings from YAML and provides easy access with type hints.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from nwc_webapp.config.environment import is_hpc, is_local, is_server
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
    difference: Dict[str, Any]  # Difference plot settings (vmin, vmax, colormap)
    min_value_threshold: float
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
class ServerConfig:
    """Settings for dedicated GPU server mode (no job scheduler)."""

    inference_script_path: str
    conda_env: str


@dataclass
class PBSConfig:
    """PBS/HPC settings."""

    queue: str
    walltime: str
    inference_script_path: str
    ed_convlstm_script_path: str
    environments: Dict[str, str]
    submit_jobs: bool = True


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
            # Prefer nwc_webapp_v2/cfg.yaml (single source of truth for v2).
            # config.py is at src/nwc_webapp/config/config.py, so the project
            # root is 4 levels up and nwc_webapp_v2/cfg.yaml is relative to that.
            v2_cfg = Path(__file__).parent.parent.parent.parent / "nwc_webapp_v2" / "cfg.yaml"
            config_path = v2_cfg if v2_cfg.exists() else Path(__file__).parent / "cfg.yaml"

        self.config_path = config_path
        self._config: Dict[str, Any] = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        with open(self.config_path, "r") as f:
            self._config = yaml.safe_load(f)

    def reload(self) -> None:
        """Reload configuration from file."""
        self._load_config()

    @property
    def mode(self) -> str:
        """Deployment mode: hpc | server | local."""
        return self._config.get("mode", "local")

    # Models
    @property
    def enabled_tabs(self) -> List[str]:
        """Get list of tabs to show in the navbar."""
        return self._config.get("enabled_tabs", ["realtime", "nowcasting", "explorer", "comparison", "metrics"])

    @property
    def models(self) -> List[str]:
        """Get list of available models."""
        return self._config.get("models", [])

    @property
    def csi_threshold(self) -> List[int]:
        """Get CSI thresholds."""
        return self._config.get("csi_threshold", [])

    @property
    def fss_window_sizes(self) -> List[int]:
        """Get FSS window sizes (in pixels)."""
        return self._config.get("fss_window_sizes", [5, 10, 20, 40, 80])

    # Paths (environment-aware)
    def get_paths(self) -> Dict[str, str]:
        """Get paths based on current environment."""
        if is_hpc():
            return self._config.get("hpc_paths", {})
        elif is_server():
            return self._config.get("server_paths", self._config.get("local_paths", {}))
        else:
            return self._config.get("local_paths", {})

    @property
    def sri_folder(self) -> Path:
        """Get SRI data folder path — derived from the SRI_adj product folder."""
        return self.get_product_folder("SRI_adj") or (self.data_root / "SRI_adj")

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
    def explorer_max_hours(self) -> int:
        """Maximum date range in hours allowed for the Data Explorer."""
        return self._config.get("explorer_max_hours", 48)

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

    @property
    def radar_products(self) -> Dict[str, Any]:
        """Get radar product configurations for the Data Explorer."""
        return self._config.get("radar_products", {})

    def get_product_folder(self, product_name: str) -> Optional[Path]:
        """
        Get the data folder for a radar product, resolving {data_root} templates.

        Args:
            product_name: Product key (e.g. 'SRI_adj', 'VMI')

        Returns:
            Path to product folder, or None if not configured.
        """
        products = self.radar_products
        if product_name not in products:
            return None
        product = products[product_name]
        folder = product.get("folder", "")
        if not folder:
            return None
        resolved = folder.replace("{data_root}", str(self.data_root))
        return Path(resolved)

    @property
    def data_archive_folder(self) -> Optional[Path]:
        """
        Get the archive base folder for radar products older than ~3 days.

        Files in the archive are organised as:
            {archive_folder}/YYYY/MM/DD/{product}/{filename}
        """
        folder = self.get_paths().get("archive_folder", "")
        return Path(folder) if folder else None

    def find_product_file(self, product_name: str, dt: datetime, filename: str) -> Optional[Path]:
        """
        Resolve the correct path for a radar product file at a given datetime.

        Checks the recent flat folder first, then falls back to the archive
        directory structure (YYYY/MM/DD/product/).

        Args:
            product_name: Product key (e.g. 'SRI_adj', 'VMI')
            dt: Datetime of the file
            filename: Filename including extension (e.g. '22-03-2026-15-30.hdf')

        Returns:
            Path to the existing file, or None if not found in either location.
        """
        # 1. Try recent flat folder
        flat_folder = self.get_product_folder(product_name)
        if flat_folder:
            path = flat_folder / filename
            if path.exists():
                return path

        # 2. Try archive directory: {archive_base}/YYYY/MM/DD/{product}/{filename}
        archive_base = self.data_archive_folder
        if archive_base:
            path = archive_base / dt.strftime("%Y") / dt.strftime("%m") / dt.strftime("%d") / product_name / filename
            if path.exists():
                return path

        return None

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
    def server(self) -> ServerConfig:
        """Get server-mode inference configuration."""
        server_config = self._config.get("server", {})
        return ServerConfig(
            inference_script_path=server_config.get("inference_script_path", ""),
            conda_env=server_config.get("conda_env", "nowcasting3.12_webapp"),
        )

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

    # Difference plot settings
    @property
    def diff_vmin(self) -> float:
        """Get minimum value for difference plots."""
        return self._config.get("visualization", {}).get("difference", {}).get("vmin", -20)

    @property
    def diff_vmax(self) -> float:
        """Get maximum value for difference plots."""
        return self._config.get("visualization", {}).get("difference", {}).get("vmax", 20)

    @property
    def diff_colormap(self) -> str:
        """Get colormap for difference plots."""
        return self._config.get("visualization", {}).get("difference", {}).get("colormap", "RdBu_r")

    # Convenience methods
    @property
    def model_configs_path(self) -> Path:
        """
        Root folder containing model inference configs (real_time/ and start_end/).

        If the path in cfg.yaml is relative, it is resolved relative to cfg.yaml itself
        so that nwc_webapp_v2/cfg.yaml → nwc_webapp_v2/model_configs/.
        Falls back to the legacy location inside the package.
        """
        raw = self._config.get("model_configs_path", "")
        if not raw:
            # Legacy fallback: configs live next to config.py
            return Path(__file__).parent / "model_configs"
        p = Path(raw)
        if p.is_absolute():
            return p
        # Resolve relative to the directory that contains cfg.yaml
        return (self.config_path.parent / p).resolve()

    @property
    def submit_jobs(self) -> bool:
        """Whether to submit PBS jobs or run in watch-folder mode."""
        return self._config.get("pbs", {}).get("submit_jobs", True)

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
    logger.info(f"\nModels: {config.models}")
    logger.info(f"SRI Folder: {config.sri_folder}")
    logger.info(f"Prediction Output: {config.prediction_output}")
    logger.info(f"Map Center: {config.visualization.map_center}")
    logger.info(
        f"Source Grid: {config.source_grid.projection} at ({config.source_grid.prj_lat}, {config.source_grid.prj_lon})"
    )
    logger.info(
        f"Dest Grid: {config.dest_grid.minLat}-{config.dest_grid.maxLat}, {config.dest_grid.minLon}-{config.dest_grid.maxLon}"
    )
    logger.info(f"PBS Queue: {config.pbs.queue}")
    logger.info(f"ED_ConvLSTM Environment: {config.get_model_pbs_env('ED_ConvLSTM')}")
