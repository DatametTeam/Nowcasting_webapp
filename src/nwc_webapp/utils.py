"""
Backwards-compatible utils.py that re-exports from new modular structure.
This file maintains compatibility with existing imports while the codebase transitions.

TODO: Eventually remove this file and update all imports to use the new modules directly.
"""

# Coordinate transformations
from nwc_webapp.geo.coordinates import (
    lincol_2_yx,
    lincol_2_radyx,
    yx_2_latlon,
    par,
    lat_0,
    lon_0,
    map_
)

# Shapefiles
from nwc_webapp.geo.shapefiles import (
    get_italian_region_shapefile,
    load_italy_shape,
    SHAPEFILE_FOLDER
)

# Colormaps
from nwc_webapp.visualization.colormaps import (
    get_legend_data,
    build_legend_file_path,
    configure_colorbar,
    create_colormap_from_legend,
    CustomNorm,
    forward,
    inverse
)

# Figures
from nwc_webapp.visualization.figures import (
    compute_figure_gpd,
    create_colorbar_fig
)

# Data loaders
from nwc_webapp.data.loaders import (
    read_groundtruth_and_target_data,
    load_prediction_data
)

# GIF utilities
from nwc_webapp.data.gif_utils import (
    check_if_gif_present,
    load_gif_as_bytesio
)

# Workers
from nwc_webapp.services.workers import (
    get_latest_file,
    worker_thread,
    worker_thread_test,
    launch_thread_execution,
    load_prediction_thread
)

# Time utilities (inline since we removed utils/ directory)
from datetime import datetime
import yaml
import numpy as np
from nwc_webapp.logging_config import setup_logger

# Set up logger
logger = setup_logger(__name__)

def get_closest_5_minute_time():
    """Get the closest earlier 5-minute mark from current time."""
    now = datetime.now()
    minutes = now.minute - (now.minute % 5)
    return now.replace(minute=minutes, second=0, microsecond=0).time()


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def generate_splotchy_image_main_(batch_size=24, channels=12, height=1400, width=1200,
                                num_clusters=5, cluster_radius=100,
                                min_value=0, max_value=100):
    """
    Generate random splotchy image for testing (legacy function).

    Args:
        batch_size: Number of batches
        channels: Number of channels per batch
        height: Image height
        width: Image width
        num_clusters: Number of precipitation clusters
        cluster_radius: Radius of each cluster
        min_value: Minimum value
        max_value: Maximum value

    Returns:
        4D numpy array of shape (batch_size, channels, height, width)
    """
    # Initialize the 4D array
    images = np.zeros((batch_size, channels, height, width), dtype=np.float32)
    intensity_scale = max_value

    for b in range(batch_size):
        for c in range(channels):
            # Create a unique image for each channel
            image = np.zeros((height, width), dtype=np.float32)

            # Generate different cluster centers for each channel for variety
            cluster_centers = np.random.randint(0, [height, width], size=(num_clusters, 2))

            for cx, cy in cluster_centers:
                y_grid, x_grid = np.ogrid[:height, :width]
                dist = np.sqrt((x_grid - cy) ** 2 + (y_grid - cx) ** 2)
                mask = dist < cluster_radius
                noise = np.random.rand(height, width).astype(np.float32)

                # Create blob with intensity that decreases with distance
                blob = intensity_scale * noise * (1 - dist / cluster_radius)
                blob *= mask  # apply mask
                image += blob

            # Assign the generated image to this batch and channel
            images[b, c] = image

    return images


def generate_splotchy_image_realTime(height, width, num_clusters, cluster_radius):
    """
    Generate random splotchy image for real-time testing (legacy function).

    Args:
        height: Image height
        width: Image width
        num_clusters: Number of precipitation clusters
        cluster_radius: Radius of each cluster

    Returns:
        2D numpy array
    """
    intensity_scale = 5
    image = np.zeros((height, width))
    cluster_centers = np.random.randint(0, min(height, width), size=(num_clusters, 2))

    for center in cluster_centers:
        cluster_x, cluster_y = center
        for i in range(height):
            for j in range(width):
                dist = np.sqrt((i - cluster_x) ** 2 + (j - cluster_y) ** 2)
                if dist < cluster_radius:
                    image[i, j] += intensity_scale * np.random.random() * (1 - dist / cluster_radius)

    image = np.clip(image, 0, 1)
    return image


def get_latest_file_once():
    """
    Generate random image for testing (legacy function).

    Returns:
        Random image array
    """
    logger.debug("GENERAZIONE randomica IMMAGINE")
    img = generate_splotchy_image_main_()
    return img

# Export global variables from figures module for backwards compatibility
from nwc_webapp.visualization.figures import (
    cmap,
    norm,
    vmin,
    vmax,
    ticks,
    italy_shape,
    x,
    y
)

# Bounding box constants
ll_lat = 35
ur_lat = 47
ll_lon = 6.5
ur_lon = 20

# Re-export commonly used items
__all__ = [
    # Coordinates
    'lincol_2_yx',
    'lincol_2_radyx',
    'yx_2_latlon',
    'par',
    'lat_0',
    'lon_0',
    'map_',
    # Shapefiles
    'get_italian_region_shapefile',
    'load_italy_shape',
    'SHAPEFILE_FOLDER',
    # Colormaps
    'get_legend_data',
    'build_legend_file_path',
    'configure_colorbar',
    'create_colormap_from_legend',
    'CustomNorm',
    'forward',
    'inverse',
    # Figures
    'compute_figure_gpd',
    'create_colorbar_fig',
    # Data loaders
    'read_groundtruth_and_target_data',
    'load_prediction_data',
    # GIF utilities
    'check_if_gif_present',
    'load_gif_as_bytesio',
    # Workers
    'get_latest_file',
    'worker_thread',
    'worker_thread_test',
    'launch_thread_execution',
    'load_prediction_thread',
    # Time utilities
    'get_closest_5_minute_time',
    # Config
    'load_config',
    # Mock data generation (legacy)
    'generate_splotchy_image_main_',
    'generate_splotchy_image_realTime',
    'get_latest_file_once',
    # Global variables
    'cmap',
    'norm',
    'vmin',
    'vmax',
    'ticks',
    'italy_shape',
    'x',
    'y',
    'll_lat',
    'ur_lat',
    'll_lon',
    'ur_lon',
]
