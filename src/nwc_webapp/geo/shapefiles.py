"""
Geographic shapefile utilities.
Handles loading and processing of Italian region shapefiles.
"""

import os
from pathlib import Path

import geopandas as gpd

from nwc_webapp.geo.coordinates import lat_0, lon_0

# Root path for resources
ROOT_PATH = Path(__file__).parent.parent.absolute()
SHAPEFILE_FOLDER = ROOT_PATH / "resources/shapefiles"


def get_italian_region_shapefile() -> Path:
    """
    Get the path to the Italian regions shapefile.

    Returns:
        Path to the shapefile (without extension)
    """
    italian_regions_folder_path = SHAPEFILE_FOLDER / "italian_regions"
    files_in_folder = list(italian_regions_folder_path.glob("*"))
    filename = files_in_folder[0].stem
    return italian_regions_folder_path / filename


def load_italy_shape():
    """
    Load and reproject the Italy shapefile to custom Transverse Mercator projection.

    Returns:
        GeoDataFrame with Italy regions in custom CRS
    """
    # Define the custom Transverse Mercator projection
    custom_crs = {
        "proj": "tmerc",  # Transverse Mercator projection
        "lat_0": lat_0,  # Latitude of the origin
        "lon_0": lon_0,  # Longitude of the origin
        "k": 1,  # Scale factor
        "x_0": 0,  # False easting (no shift applied)
        "y_0": 0,  # False northing (no shift applied)
        "datum": "WGS84",  # Geodetic datum
        "units": "m",  # Units in meters
        "no_defs": True,  # Do not use external defaults
    }

    # Load shapefile
    shapefile_path = SHAPEFILE_FOLDER / "italian_regions/gadm41_ITA_1.shp"
    italy_shape = gpd.read_file(shapefile_path)

    # Reproject to custom CRS
    italy_shape = italy_shape.to_crs(crs=custom_crs)

    return italy_shape
