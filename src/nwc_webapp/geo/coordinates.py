"""
Coordinate transformation utilities for radar data.
Handles conversions between linear/column coordinates, radar coordinates, and lat/lon.
"""
import numpy as np
import pyproj
from numba import njit

# Map projection constants
par = np.array([600., 1000., 650., -1000.])
lat_0 = 42.0
lon_0 = 12.5
map_ = pyproj.Proj({"proj": 'tmerc', "lat_0": lat_0, "lon_0": lon_0})


def lincol_2_yx(
        lin: np.ndarray,
        col: np.ndarray,
        params: list,
        dim=None,
        az_coords: tuple = (),
        el_coords=None,
        set_center: bool = False,
):
    """
    Convert linear and column coordinates to y, x coordinates.

    Args:
        lin: Linear indices
        col: Column indices
        params: Parameters from NAVIGATION.txt
        az_coords: Azimuth coordinates from AZIMUTH.txt
        el_coords: Elevation coordinates from ELEVATION.txt
        dim: Optional dimension parameter for mosaicking
        set_center: Whether to set center offset

    Returns:
        Tuple of (y, x) coordinates
    """
    if dim is None:
        dim = []

    if len(params) >= 10:
        # Polar data case
        x, y = lincol_2_radyx(
            lin, col, params, az_coords, el_coords, set_center=set_center
        )
    else:
        # Non-polar data case
        xoff = params[0]
        xres = params[1]
        yoff = params[2]
        yres = params[3]

        if set_center:
            xoff -= 0.5
            yoff -= 0.5
        if xres == 0 or yres == 0:
            x = 0.0
            y = 0.0
            return

        x = (col - xoff) * xres
        y = (lin - yoff) * yres

    if len(dim) != 2:
        return y, x

    # Check for requests outside the matrix
    ind = np.where((lin < 0) | lin >= dim[1])[0]

    if len(ind) > 0:
        notValid = ind

    ind = np.where((col < 0) | col >= dim[0])[0]
    if len(ind) > 0:
        if len(notValid) > 0:
            notValid = np.concatenate((notValid, ind))
        else:
            notValid = ind

    if len(notValid) <= 0:
        return

    y[notValid] = np.nan
    x[notValid] = np.nan

    return y, x


@njit(cache=True, parallel=True, fastmath=True)
def lincol_2_radyx(
        lin,
        col,
        par: dict,
        az_coords=np.array(()),
        el_coords: np.ndarray = None,
        set_az: bool = False,
        set_center: bool = False,
        lev=np.array(()),
        set_z: bool = False,
        radz=np.array(()),
):
    """
    Converts linear and column coordinates to radar coordinates in azimuth and range dimensions.

    This function transforms linear and column indices (lin, col) into radar coordinates (rady, radx)
    based on specified parameters (par) and optional azimuth and elevation coordinates.

    Args:
        lin: Linear indices or a single linear index
        col: Column indices or a single column index
        par: Parameters containing offsets and resolutions for the transformation
        az_coords: Azimuth coordinates corresponding to the linear indices
        el_coords: Elevation coordinates
        set_az: Whether to set azimuth
        set_center: Whether to set center offset
        lev: Level coordinates
        set_z: Whether to set z coordinates
        radz: Radar z coordinates

    Returns:
        Tuple of (radx, rady) radar coordinates
    """
    azres = 0
    polres = 0
    azoff = 0
    poloff = 0
    if az_coords is None:
        az_coords = np.array(())

    if len(par) >= 10:
        poloff = par[6]
        polres = par[7]
        azoff = par[8]
        azres = par[9]

    if polres == 0:
        radx = col
        rady = lin
        return

    az = azoff

    if len(az_coords) == 0:
        if azres != 0:
            az += lin * azres
    else:
        if len(lin) == 1:
            if 0 <= lin <= len(az_coords):
                az = az_coords[lin]
        else:
            az = az_coords[lin]

    if len(az) == 1 or set_az:
        azimuth = az
        if len(azimuth) == 1:
            if azimuth >= 360:
                azimuth -= 360

    ro = col * polres

    if set_center:
        if len(az_coords) == 0:
            az += azres / 2.0
        ro += polres / 2.0

    # Angles in polar start from 0 at north and go clockwise
    # opposite to normal trigonometry
    az = 450 - az  # make counter-clockwise

    if poloff > 0:
        ro += poloff

    az *= np.pi / 180

    cosaz = np.cos(az)
    sinaz = np.sin(az)

    radx = ro * cosaz
    rady = ro * sinaz

    if azres == 0 and len(lev) == 0:
        lev = lin
    if not set_z:
        return radx, rady
    if radz is not None:
        return radx, rady

    return radx, rady


def yx_2_latlon(y, x, map_proj):
    """
    Converts map projection coordinates to latitude and longitude.

    Args:
        y: The y-coordinate(s) in the map projection
        x: The x-coordinate(s) in the map projection
        map_proj: The map projection object used for the conversion

    Returns:
        Tuple of (lat, lon) in degrees
    """
    lon, lat = map_proj(longitude=x, latitude=y, inverse=True)
    return lat, lon