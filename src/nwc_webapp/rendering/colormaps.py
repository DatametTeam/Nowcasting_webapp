"""
Colormap and legend management for radar visualization.
Handles custom colormaps from legend files.
"""

from pathlib import Path

import matplotlib.colors as mcolors
import numpy as np

ROOT_PATH = Path(__file__).parent.parent.absolute()


def get_legend_data(filepath) -> dict:
    """
    Parse legend file and extract color information.

    Args:
        filepath: Path to legend.txt file

    Returns:
        Dictionary containing thresholds, RGB colors, and other legend metadata
    """
    legend_data = {
        "Thresh": [],
        "rgb": [],
        "null_color": (0, 0, 0, 0),
        "void_color": (0, 0, 0, 0),
        "discrete": 0,
        "label": [],
    }

    with open(filepath, "r") as file:
        lines = file.readlines()

    for line in lines:
        parts = line.strip().split()
        key = parts[0].lower()
        if key == "thresh":
            legend_data["Thresh"].append(float(parts[2]))
        elif key == "rgb":
            color = tuple(float(c) / 255.0 for c in parts[2:])
            legend_data["rgb"].append(color)
        elif key == "null_color":
            legend_data["null_color"] = tuple(float(c) / 255.0 for c in parts[2:])
        elif key == "void_color":
            legend_data["void_color"] = tuple(float(c) / 255.0 for c in parts[2:])
        elif key == "discrete":
            legend_data["discrete"] = int(parts[2])
        elif key == "label":
            legend_data["label"].append(" ".join((parts[2:])))

    return legend_data


def build_legend_file_path(parname):
    """
    Build path to legend file for a given parameter name.

    Args:
        parname: Parameter name (e.g., 'R' for rainfall)

    Returns:
        Path to legend.txt file
    """
    legend_file_path = ROOT_PATH / "resources/legends" / parname / "legend.txt"
    return legend_file_path


def forward(x, thresh):
    """Map the threshold values to a [0, 1] scale."""
    return np.interp(x, thresh, np.linspace(0, 1, len(thresh)))


def inverse(x, thresh):
    """Map normalized values [0, 1] back to the original threshold values."""
    return np.interp(x, np.linspace(0, 1, len(thresh)), thresh)


class CustomNorm(mcolors.Normalize):
    """Custom normalization to handle forward and inverse functions with thresholds."""

    def __init__(self, thresh, vmin=None, vmax=None):
        super().__init__(vmin, vmax)
        self.thresh = thresh

    def __call__(self, value, clip=None):
        return forward(value, self.thresh)

    def inverse(self, value):
        return inverse(value, self.thresh)


def create_colormap_from_legend(legend_data, parname, min_value, max_value):
    """
    Create a custom colormap from legend data.

    Args:
        legend_data: Dictionary with legend information
        parname: Parameter name
        min_value: Minimum value for colormap range
        max_value: Maximum value for colormap range

    Returns:
        Tuple of (cmap, norm, extended_thresh)
    """
    cmap_name = "colormap_from_legend"

    if legend_data["discrete"] == 0:
        thresh = legend_data["Thresh"]
        extended_thresh = thresh
        rgb_colors = legend_data["rgb"]
        cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, rgb_colors, N=256)

        norm = CustomNorm(thresh, vmin=thresh[0], vmax=thresh[-1])
    else:
        extended_thresh = legend_data["Thresh"]
        rgb_colors = legend_data["rgb"]
        cmap = mcolors.ListedColormap(rgb_colors)
        norm = mcolors.BoundaryNorm(extended_thresh, cmap.N)

    return cmap, norm, extended_thresh


def configure_colorbar(parameter_name, min_val, max_val):
    """
    Configure colorbar for a given parameter.

    Args:
        parameter_name: Name of the parameter (e.g., 'R')
        min_val: Minimum value
        max_val: Maximum value

    Returns:
        Tuple of (cmap, norm, vmin, vmax, null_color, void_color, discrete, ticks)
    """
    legend_file_path = build_legend_file_path(parameter_name)
    if legend_file_path.exists():
        legend_data = get_legend_data(legend_file_path)
        cmap, norm, extended_thresh = create_colormap_from_legend(
            legend_data, parameter_name, min_value=min_val, max_value=max_val
        )
        if legend_data["discrete"] == 1:
            vmin = 0
        else:
            vmin = min(extended_thresh)
        vmax = max(extended_thresh)
        null_color = legend_data.get("null_color", (0, 0, 0, 0))
        void_color = legend_data.get("void_color", (0, 0, 0, 0))
        discrete = legend_data.get("discrete")
        ticks = extended_thresh
    else:
        cmap = "jet"
        norm = None
        vmin = None
        vmax = None
        null_color = (0, 0, 0, 0)
        void_color = (0, 0, 0, 0)
        discrete = 0
        ticks = []
    return cmap, norm, vmin, vmax, null_color, void_color, discrete, ticks


# Create default colormap and normalization for rainfall ('R') parameter
# These are used by ui/maps.py for the default precipitation colormap
cmap, norm, _, _, _, _, _, _ = configure_colorbar('R', min_val=0, max_val=100)
