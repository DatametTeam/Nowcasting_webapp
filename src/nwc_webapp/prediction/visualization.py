"""
Prediction visualization and display functions.
"""
import numpy as np
import streamlit as st

from nwc_webapp.ui.layouts import init_prediction_visualization_layout
from nwc_webapp.utils import check_if_gif_present, load_gif_as_bytesio, create_colorbar_fig
from nwc_webapp.services.parallel_code import (
    create_fig_dict_in_parallel,
    create_sliding_window_gifs,
    create_sliding_window_gifs_for_predictions,
    create_diff_dict_in_parallel
)
from nwc_webapp.prediction.jobs import submit_prediction_job
from nwc_webapp.prediction.loaders import get_prediction_results
from nwc_webapp.logging_config import setup_logger

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
    gt_current, pred_current, gt_plus_30, pred_plus_30, gt_plus_60, pred_plus_60, colorbar30, colorbar60, diff_plus_30, diff_plus_60 = \
        init_prediction_visualization_layout()

    # Display the GIF using Streamlit
    gt_current.image(gt0_gif, caption="Current data", width='content')
    pred_current.image(gt0_gif, caption="Current data", width='content')
    gt_plus_30.image(gt6_gif, caption="Data +30 minutes", width='content')
    pred_plus_30.image(pred_gif_6, caption="Prediction +30 minutes", width='content')
    gt_plus_60.image(gt12_gif, caption="Data +60 minutes", width='content')
    pred_plus_60.image(pred_gif_12, caption="Prediction +60 minutes", width='content')
    diff_plus_30.image(diff_gif_6, caption="Differences +30 minutes", width='content')
    diff_plus_60.image(diff_gif_12, caption="Differences +60 minutes", width='content')
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
        'gt0_gif': gt0_gif,
        'gt6_gif': gt_gif_6,
        'gt12_gif': gt_gif_12,
        'pred6_gif': pred_gif_6,
        'pred12_gif': pred_gif_12,
        'diff6_gif': diff_gif_6,
        'diff12_gif': diff_gif_12,
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
        with st.status(f':hammer_and_wrench: **Loading results...**', expanded=True) as status:

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
                    gt_gifs = create_sliding_window_gifs(gt_dict, sidebar_args, fps_gif=3,
                                                         save_on_disk=True)

                status.update(label="🔄 Creating Pred GIFs...", state="running", expanded=True)

                pred_gifs = create_sliding_window_gifs_for_predictions(pred_dict, sidebar_args,
                                                                       fps_gif=3, save_on_disk=True)

                diff_gifs = create_sliding_window_gifs(diff_dict, sidebar_args, fps_gif=3,
                                                       save_on_disk=True, name="diff")

                status.update(label=f"Done!", state="complete", expanded=True)

                display_results(gt_gifs, pred_gifs, diff_gifs)
    else:
        st.error(error)