"""
Prediction visualization and display functions.
"""

import io
from pathlib import Path

import imageio
import numpy as np
import streamlit as st
from PIL import Image

from nwc_webapp.logging_config import setup_logger
from nwc_webapp.page_modules.nowcasting_utils import extract_timestamp_slices
from nwc_webapp.prediction.jobs import submit_prediction_job
from nwc_webapp.prediction.loaders import get_prediction_results
from nwc_webapp.services.parallel_code import (
    create_diff_dict_in_parallel,
    create_fig_dict_in_parallel,
    create_sliding_window_gifs,
    create_sliding_window_gifs_for_predictions,
)
from nwc_webapp.ui.layouts import init_prediction_visualization_layout
from nwc_webapp.utils import check_if_gif_present, compute_figure_gpd, create_colorbar_fig, load_gif_as_bytesio

# Set up logger
logger = setup_logger(__name__)


def update_prediction_visualization(gt0_gif, gt6_gif, gt12_gif, pred_gif_6, pred_gif_12, diff_gif_6, diff_gif_12):
    """
    Update the prediction visualization layout with GIF data.

    Args:
        gt0_gif: Ground truth at t=0
        gt6_gif: Ground truth at t+30min
        gt12_gif: Ground truth at t+60min
        pred_gif_6: Prediction at t+30min
        pred_gif_12: Prediction at t+60min
        diff_gif_6: Difference at t+30min
        diff_gif_12: Difference at t+60min
    """
    (
        gt_current,
        pred_current,
        gt_plus_30,
        pred_plus_30,
        gt_plus_60,
        pred_plus_60,
        colorbar30,
        colorbar60,
        diff_plus_30,
        diff_plus_60,
    ) = init_prediction_visualization_layout()

    # Display the GIF using Streamlit
    gt_current.image(gt0_gif, caption="Current data", width="content")
    pred_current.image(gt0_gif, caption="Current data", width="content")
    gt_plus_30.image(gt6_gif, caption="Data +30 minutes", width="content")
    pred_plus_30.image(pred_gif_6, caption="Prediction +30 minutes", width="content")
    gt_plus_60.image(gt12_gif, caption="Data +60 minutes", width="content")
    pred_plus_60.image(pred_gif_12, caption="Prediction +60 minutes", width="content")
    diff_plus_30.image(diff_gif_6, caption="Differences +30 minutes", width="content")
    diff_plus_60.image(diff_gif_12, caption="Differences +60 minutes", width="content")
    colorbar30.image(create_colorbar_fig(top_adj=0.96, bot_adj=0.12))
    colorbar60.image(create_colorbar_fig(top_adj=0.96, bot_adj=0.12))


def display_results(gt_gifs, pred_gifs, diff_gifs):
    """
    Display prediction results and store them in session state.

    Args:
        gt_gifs: List of ground truth GIFs
        pred_gifs: List of prediction GIFs
        diff_gifs: List of difference GIFs
    """
    gt0_gif = gt_gifs[0]  # Full sequence
    gt_gif_6 = gt_gifs[1]  # Starts from frame 6
    gt_gif_12 = gt_gifs[2]  # Starts from frame 12
    pred_gif_6 = pred_gifs[0]
    pred_gif_12 = pred_gifs[1]
    diff_gif_6 = diff_gifs[0]  # diff from frame 6
    diff_gif_12 = diff_gifs[1]  # diff from frame 12

    # Store results in session state
    st.session_state.prediction_result = {
        "gt0_gif": gt0_gif,
        "gt6_gif": gt_gif_6,
        "gt12_gif": gt_gif_12,
        "pred6_gif": pred_gif_6,
        "pred12_gif": pred_gif_12,
        "diff6_gif": diff_gif_6,
        "diff12_gif": diff_gif_12,
    }
    st.session_state.tab1_gif = gt0_gif.getvalue()
    update_prediction_visualization(gt0_gif, gt_gif_6, gt_gif_12, pred_gif_6, pred_gif_12, diff_gif_6, diff_gif_12)


def compute_prediction_results(sidebar_args, folder_path):
    """
    Compute prediction results: submit job, load data, create visualizations.

    Args:
        sidebar_args: Dictionary with prediction parameters
        folder_path: Path to data folder
    """
    error, out_dir = submit_prediction_job(sidebar_args)
    if not error:
        with st.status(f":hammer_and_wrench: **Loading results...**", expanded=True) as status:

            prediction_placeholder = st.empty()

            with prediction_placeholder:
                status.update(label="🔄 Loading results...", state="running", expanded=True)

                gt_gif_ok, pred_gif_ok, _, gt_paths, pred_paths, _ = check_if_gif_present(sidebar_args)
                if gt_gif_ok:
                    gt_gifs = load_gif_as_bytesio(gt_paths)

                gt_array, pred_array = get_prediction_results(folder_path, sidebar_args)

                # calculating differences
                diff_array = gt_array[:, :, :, :] - pred_array[:, :, :, :]

                # only for test --> amplification
                # diff_array = np.clip(diff_array * 10, -1, 1)

                status.update(label="🔄 Creating dictionaries...", state="running", expanded=True)

                gt_dict, pred_dict = create_fig_dict_in_parallel(gt_array, pred_array, sidebar_args)

                logger.info("CREATING DIFF DICT")
                diff_dict = create_diff_dict_in_parallel(np.abs(diff_array), sidebar_args)

                if not gt_gif_ok:
                    status.update(label="🔄 Creating GT GIFs...", state="running", expanded=True)
                    gt_gifs = create_sliding_window_gifs(gt_dict, sidebar_args, fps_gif=3, save_on_disk=True)

                status.update(label="🔄 Creating Pred GIFs...", state="running", expanded=True)

                pred_gifs = create_sliding_window_gifs_for_predictions(
                    pred_dict, sidebar_args, fps_gif=3, save_on_disk=True
                )

                diff_gifs = create_sliding_window_gifs(
                    diff_dict, sidebar_args, fps_gif=3, save_on_disk=True, name="diff"
                )

                status.update(label=f"Done!", state="complete", expanded=True)

                display_results(gt_gifs, pred_gifs, diff_gifs)
    else:
        st.error(error)


def create_simple_gif_from_array(data_array: np.ndarray, title_prefix: str, fps: int = 3) -> io.BytesIO:
    """
    Create a simple GIF from a 3D numpy array.

    Args:
        data_array: Array of shape (timesteps, height, width)
        title_prefix: Prefix for the title (e.g., "Ground Truth", "Prediction")
        fps: Frames per second for the GIF

    Returns:
        BytesIO buffer containing the GIF
    """
    from datetime import datetime, timedelta

    buf = io.BytesIO()
    frames = []

    logger.info(f"Creating GIF from array with shape {data_array.shape}")

    for i in range(data_array.shape[0]):
        # Create figure for this timestep
        time_offset = i * 5  # 5 minutes per timestep
        title = f"{title_prefix} +{time_offset}min"

        fig = compute_figure_gpd(data_array[i], title)

        # Convert figure to image
        buf_temp = io.BytesIO()
        fig.savefig(buf_temp, format="png", bbox_inches="tight", pad_inches=0)
        buf_temp.seek(0)
        img = Image.open(buf_temp)
        frames.append(np.array(img))

        # Close the figure to free memory
        import matplotlib.pyplot as plt

        plt.close(fig)

    # Create GIF
    imageio.mimsave(buf, frames, format="GIF", fps=fps, loop=0)
    buf.seek(0)

    logger.info(f"GIF created with {len(frames)} frames")
    return buf


def create_gifs_from_realtime_data(pred_data: np.ndarray, sidebar_args: dict, sri_folder_dir: str, gif_paths: dict):
    """
    Create GIFs from real-time prediction data.

    The pred_data has shape (12, 1400, 1200) with 5-minute intervals.
    We need to create:
    - Ground truth GIFs: t0 (frames 0-11), t+30 (frames 6-11), t+60 (frames 11)
    - Prediction GIFs: t+30 (frame 6), t+60 (frame 11)
    - Difference GIFs: t+30, t+60

    Args:
        pred_data: Prediction array of shape (12, 1400, 1200)
        sidebar_args: Sidebar arguments with model info
        sri_folder_dir: Path to SRI data folder
        gif_paths: Dictionary of GIF paths from get_gif_paths()
    """
    with st.status("🔄 Creating GIFs from real-time data...", expanded=True) as status:

        # Step 1: Load ground truth data
        status.update(label="📂 Loading ground truth data...", state="running")

        try:
            gt_array, _ = get_prediction_results(sri_folder_dir, sidebar_args)
            logger.info(f"Loaded ground truth data with shape {gt_array.shape}")
        except Exception as e:
            logger.error(f"Error loading ground truth: {e}")
            st.error(f"Failed to load ground truth data: {e}")
            return

        # Step 2: Extract specific timesteps from prediction
        status.update(label="🔍 Extracting prediction timesteps...", state="running")

        # For predictions: we want the full sequence for sliding window
        # But we'll focus on t+30 (index 6) and t+60 (index 11)

        # Step 3: Create ground truth GIFs (sliding windows)
        status.update(label="🎬 Creating ground truth GIFs...", state="running")

        # GT GIF 1: Full sequence (0-11)
        gt_t0_gif = create_simple_gif_from_array(gt_array[:12, 0], "Ground Truth", fps=3)

        # GT GIF 2: From t+30 onwards (frames 6-11)
        gt_t6_gif = create_simple_gif_from_array(gt_array[6:12, 0], "Ground Truth", fps=3)

        # GT GIF 3: Just t+60 (frame 11)
        gt_t12_gif = create_simple_gif_from_array(gt_array[11:12, 0], "Ground Truth", fps=3)

        # Save GT GIFs to disk
        for gif_key, gif_buf in [("gt_t0", gt_t0_gif), ("gt_t6", gt_t6_gif), ("gt_t12", gt_t12_gif)]:
            gif_path = gif_paths[gif_key]
            gif_path.parent.mkdir(parents=True, exist_ok=True)

            with open(gif_path, "wb") as f:
                f.write(gif_buf.getvalue())
            logger.info(f"Saved {gif_key} to {gif_path}")

        # Step 4: Create prediction GIFs
        status.update(label="🎬 Creating prediction GIFs...", state="running")

        # Prediction GIF 1: t+30 (frame 6)
        pred_t6_gif = create_simple_gif_from_array(pred_data[6:7], "Prediction", fps=3)

        # Prediction GIF 2: t+60 (frame 11)
        pred_t12_gif = create_simple_gif_from_array(pred_data[11:12], "Prediction", fps=3)

        # Save prediction GIFs
        for gif_key, gif_buf in [("pred_t6", pred_t6_gif), ("pred_t12", pred_t12_gif)]:
            gif_path = gif_paths[gif_key]
            gif_path.parent.mkdir(parents=True, exist_ok=True)

            with open(gif_path, "wb") as f:
                f.write(gif_buf.getvalue())
            logger.info(f"Saved {gif_key} to {gif_path}")

        # Step 5: Create difference GIFs
        status.update(label="🎬 Creating difference GIFs...", state="running")

        # Difference at t+30
        diff_t6 = np.abs(gt_array[6:7, 0] - pred_data[6:7])
        diff_t6_gif = create_simple_gif_from_array(diff_t6, "Difference", fps=3)

        # Difference at t+60
        diff_t12 = np.abs(gt_array[11:12, 0] - pred_data[11:12])
        diff_t12_gif = create_simple_gif_from_array(diff_t12, "Difference", fps=3)

        # Save difference GIFs
        for gif_key, gif_buf in [("diff_t6", diff_t6_gif), ("diff_t12", diff_t12_gif)]:
            gif_path = gif_paths[gif_key]
            gif_path.parent.mkdir(parents=True, exist_ok=True)

            with open(gif_path, "wb") as f:
                f.write(gif_buf.getvalue())
            logger.info(f"Saved {gif_key} to {gif_path}")

        # Step 6: Display results
        status.update(label="✅ Done! Displaying results...", state="complete")

        # Prepare GIF lists for display
        gt_gifs = [gt_t0_gif, gt_t6_gif, gt_t12_gif]
        pred_gifs = [pred_t6_gif, pred_t12_gif]
        diff_gifs = [diff_t6_gif, diff_t12_gif]

        display_results(gt_gifs, pred_gifs, diff_gifs)
