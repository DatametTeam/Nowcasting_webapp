"""
Map warping utilities for reprojecting radar data.

This module provides functionality to warp radar data from one projection to another,
using pyproj for coordinate transformations.
"""

import numpy as np
from pyproj import Proj

from nwc_webapp.config.config import get_config


def warp_map(source_data: np.ndarray) -> np.ndarray:
    """
    Warp a 2D array from source projection to destination projection.

    This function reprojects radar data from a Transverse Mercator projection
    to a geographic lat/lon grid suitable for display on interactive maps.

    Grid parameters are read from the application configuration.

    Args:
        source_data: 2D numpy array to warp (nlines x ncols)

    Returns:
        Warped 2D numpy array in destination projection

    Example:
        >>> import numpy as np
        >>> data = np.random.rand(1400, 1200)
        >>> warped = warp_map(data)
    """
    # Get configuration
    config = get_config()
    source_params = config.source_grid
    dest_params = config.dest_grid

    # Create source projection (Transverse Mercator)
    source_proj = Proj(
        proj="tmerc", lat_0=source_params.prj_lat, lon_0=source_params.prj_lon, x_0=0, y_0=0, ellps="WGS84"
    )

    # Get grid dimensions
    nlines_src = source_params.nlines
    ncols_src = source_params.ncols
    nlines_dst = dest_params.nlines
    ncols_dst = dest_params.ncols

    # Source grid parameters
    cOff = source_params.cOff
    lOff = source_params.lOff
    cRes = source_params.cRes
    lRes = source_params.lRes

    # Create destination grid in lat/lon space
    dest_lons = np.linspace(dest_params.minLon, dest_params.maxLon, ncols_dst)
    dest_lats = np.linspace(dest_params.maxLat, dest_params.minLat, nlines_dst)

    # Create meshgrid for destination
    dest_lon_grid, dest_lat_grid = np.meshgrid(dest_lons, dest_lats)

    # Convert destination lat/lon to source projection coordinates (Transverse Mercator)
    dest_x_in_src, dest_y_in_src = source_proj(dest_lon_grid, dest_lat_grid)

    # Convert source projection coordinates to source grid indices
    # Formula: x = (col - cOff) * cRes  =>  col = (x / cRes) + cOff
    dest_col_in_src = (dest_x_in_src / cRes) + cOff
    dest_line_in_src = (dest_y_in_src / lRes) + lOff

    # Use nearest-neighbor sampling
    # Convert to integer indices using truncation (matching sou_py behavior)
    dest_col_indices = dest_col_in_src.astype(int)
    dest_line_indices = dest_line_in_src.astype(int)

    # Create output array filled with NaN
    warped_data = np.full((nlines_dst, ncols_dst), np.nan, dtype=source_data.dtype)

    # Create mask for valid indices (within source grid bounds)
    valid_mask = (
        (dest_col_indices >= 0)
        & (dest_col_indices < ncols_src)
        & (dest_line_indices >= 0)
        & (dest_line_indices < nlines_src)
    )

    # Sample source data at valid locations using nearest-neighbor
    warped_data[valid_mask] = source_data[dest_line_indices[valid_mask], dest_col_indices[valid_mask]]

    # Flip vertically to match expected orientation
    warped_data = np.flipud(warped_data)

    return warped_data
