"""
GIF utilities for prediction visualization.
Handles checking for existing GIFs and loading them.
"""

import io
import os
from datetime import datetime, timedelta
from pathlib import Path

from nwc_webapp.config.config import get_config


def check_if_gif_present(sidebar_args):
    """
    Check if GIF files are already generated for the given sidebar arguments.
    Uses config-based paths instead of hardcoded paths.

    Args:
        sidebar_args: Dictionary with model_name, start_date, start_time

    Returns:
        Tuple of (groundtruth_present, prediction_present, difference_present,
                 groundtruth_paths, prediction_paths, difference_paths)
    """
    config = get_config()
    model = sidebar_args["model_name"]

    # Use config-based output directory
    gif_dir = config.prediction_output.parent / "gifs" / model

    start_date = sidebar_args["start_date"]
    start_time = sidebar_args["start_time"]

    # Generate datetime objects for start, +30 mins, and +60 mins
    start_datetime = datetime.combine(start_date, start_time)
    datetime_plus_30 = start_datetime + timedelta(minutes=30)
    datetime_plus_60 = start_datetime + timedelta(minutes=60)

    # File names for groundtruths
    groundtruth_files = [
        f"{start_datetime.strftime('%d%m%Y_%H%M')}_"
        f"{(start_datetime + timedelta(minutes=55)).strftime('%d%m%Y_%H%M')}.gif",
        f"{datetime_plus_30.strftime('%d%m%Y_%H%M')}_"
        f"{(datetime_plus_30 + timedelta(minutes=55)).strftime('%d%m%Y_%H%M')}.gif",
        f"{datetime_plus_60.strftime('%d%m%Y_%H%M')}_"
        f"{(datetime_plus_60 + timedelta(minutes=55)).strftime('%d%m%Y_%H%M')}.gif",
    ]

    # File names for predictions
    prediction_files = [
        f"{start_datetime.strftime('%d%m%Y_%H%M')}_"
        f"{(start_datetime + timedelta(minutes=55)).strftime('%d%m%Y_%H%M')}_+30 mins.gif",
        f"{start_datetime.strftime('%d%m%Y_%H%M')}_"
        f"{(start_datetime + timedelta(minutes=55)).strftime('%d%m%Y_%H%M')}_+60 mins.gif",
    ]

    # File names for differences (charged like gt gifs)
    difference_files = [
        f"{start_datetime.strftime('%d%m%Y_%H%M')}_"
        f"{(start_datetime + timedelta(minutes=55)).strftime('%d%m%Y_%H%M')}.gif",
        f"{start_datetime.strftime('%d%m%Y_%H%M')}_"
        f"{(start_datetime + timedelta(minutes=55)).strftime('%d%m%Y_%H%M')}.gif",
    ]

    groundtruth_paths = [os.path.join(gif_dir, "gt", file) for file in groundtruth_files]
    prediction_paths = [os.path.join(gif_dir, "pred", file) for file in prediction_files]
    difference_paths = [os.path.join(gif_dir, "diff", file) for file in difference_files]

    # Check presence of groundtruth files
    groundtruth_present = all(os.path.exists(path) for path in groundtruth_paths)

    # Check presence of prediction files
    prediction_present = all(os.path.exists(path) for path in prediction_paths)

    # Check presence of difference files
    difference_present = all(os.path.exists(path) for path in difference_paths)

    return (
        groundtruth_present,
        prediction_present,
        difference_present,
        groundtruth_paths,
        prediction_paths,
        difference_paths,
    )


def load_gif_as_bytesio(gif_paths):
    """
    Loads GIFs from specified paths into io.BytesIO objects.

    Args:
        gif_paths: List of paths to GIF files

    Returns:
        List of BytesIO objects with GIF data
    """
    gifs = []
    for gif_path in gif_paths:
        with open(gif_path, "rb") as f:
            gif_data = f.read()
        gifs.append(io.BytesIO(gif_data))

    return gifs
