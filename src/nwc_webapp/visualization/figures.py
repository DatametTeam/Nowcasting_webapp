"""
Figure generation utilities.
Creates matplotlib figures with radar data and colorbars.
"""

import io
import warnings

import matplotlib.pyplot as plt
import numpy as np

from nwc_webapp.geo.coordinates import lincol_2_yx, par
from nwc_webapp.geo.shapefiles import load_italy_shape
from nwc_webapp.visualization.colormaps import configure_colorbar

# Suppress specific matplotlib warnings
warnings.filterwarnings(
    "ignore", category=UserWarning, message="The input coordinates to pcolormesh are interpreted as cell centers.*"
)

# Initialize globals (loaded once at module import)
cmap, norm, vmin, vmax, null_color, void_color, discrete, ticks = configure_colorbar("R", min_val=None, max_val=None)

destlines = 1400
destcols = 1200
y = np.arange(destlines).reshape(-1, 1) * np.ones((1, destcols))
x = np.ones((destlines, 1)) * np.arange(destcols).reshape(1, -1).astype(int)
y, x = lincol_2_yx(lin=y, col=x, params=par, set_center=True)

italy_shape = load_italy_shape()


def compute_figure_gpd(img1, timestamp, name=""):
    """
    Compute a figure with radar data overlaid on Italy map.

    Args:
        img1: 2D array with radar data
        timestamp: Timestamp string for the title
        name: Optional name (if "diff", uses diverging red/blue colormap)

    Returns:
        Matplotlib figure object
    """
    global x, y

    fig, ax = plt.subplots(figsize=(10, 10))
    italy_shape.plot(ax=ax, edgecolor="black", color="white")

    # Use diverging colormap for differences (red=positive/target higher, blue=negative/pred higher)
    if name == "diff":
        from nwc_webapp.config.config import get_config
        config = get_config()

        cmap_ = plt.get_cmap(config.diff_colormap)  # RdBu_r from config
        diff_vmin = config.diff_vmin  # -20 from config
        diff_vmax = config.diff_vmax  # 20 from config

        mesh = ax.pcolormesh(
            x,
            y,
            img1,
            shading="auto",
            cmap=cmap_,
            vmin=diff_vmin,
            vmax=diff_vmax,
            snap=True,
            linewidths=0,
        )
    else:
        mesh = ax.pcolormesh(
            x,
            y,
            img1,
            shading="auto",
            cmap=cmap,
            norm=norm,
            vmin=None if norm else vmin,
            vmax=None if norm else vmax,
            snap=True,
            linewidths=0,
        )

    # Remove the axis
    plt.axis("off")

    # Set a white background
    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(1.0)

    # Adjust the suptitle to be closer to the image
    plt.suptitle(timestamp, y=0.92, fontsize=14)
    plt.close()
    return fig


def create_colorbar_fig(top_adj=None, bot_adj=None):
    """
    Create a colorbar figure.

    Args:
        top_adj: Top adjustment for subplot
        bot_adj: Bottom adjustment for subplot

    Returns:
        BytesIO buffer with colorbar PNG
    """
    # Create a figure
    fig, ax = plt.subplots(figsize=(2, 25))
    fig.subplots_adjust(right=0.5, top=top_adj, bottom=bot_adj)

    # Create a colorbar
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax,
        orientation="vertical",
        ticks=ticks,
    )

    cbar.ax.tick_params(labelsize=25, length=10, width=3)
    product_unit = "mm/h"
    cbar.ax.set_title(product_unit, fontsize=30, pad=50)

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)

    return buf
