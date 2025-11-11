"""
Nowcasting page - main prediction interface.
"""
import streamlit as st

from nwc_webapp.utils import check_if_gif_present, load_gif_as_bytesio
from nwc_webapp.prediction.visualization import compute_prediction_results, display_results, update_prediction_visualization


def main_page(sidebar_args, sri_folder_dir) -> None:
    """
    Main nowcasting page with prediction submission and visualization.

    Args:
        sidebar_args: Dictionary with sidebar configuration
        sri_folder_dir: Path to SRI folder directory
    """
    # Only run prediction if not already done
    if 'prediction_result' not in st.session_state or st.session_state.prediction_result == {}:
        if 'submitted' not in st.session_state:
            submitted = sidebar_args['submitted']
            if submitted:
                st.session_state.submitted = True
        if 'submitted' in st.session_state and st.session_state.submitted:
            gt_gif_ok, pred_gif_ok, diff_gif_ok, gt_paths, pred_paths, diff_paths = check_if_gif_present(sidebar_args)

            if gt_gif_ok and pred_gif_ok:
                st.warning("Prediction data already present. Do you want to recompute?")
                col1, _, col2, _ = st.columns([1, 0.5, 1, 3])  # Ensure both buttons take half the page
                with col1:
                    compute_ok = False
                    if st.button("YES", width='content'):
                        compute_ok = True
                if compute_ok:
                    compute_prediction_results(sidebar_args, sri_folder_dir)

                with col2:
                    compute_nok = False
                    if st.button("NO", width='content'):
                        compute_nok = True
                if compute_nok:
                    gt_gifs = load_gif_as_bytesio(gt_paths)
                    pred_gifs = load_gif_as_bytesio(pred_paths)
                    diff_gifs = load_gif_as_bytesio(diff_paths)
                    display_results(gt_gifs, pred_gifs, diff_gifs)

            else:
                compute_prediction_results(sidebar_args, sri_folder_dir)
            return
    else:
        # If prediction results already exist, reuse them
        gt0_gif = st.session_state.prediction_result['gt0_gif']
        gt_gif_6 = st.session_state.prediction_result['gt6_gif']
        gt_gif_12 = st.session_state.prediction_result['gt12_gif']
        pred_gif_6 = st.session_state.prediction_result['pred6_gif']
        pred_gif_12 = st.session_state.prediction_result['pred12_gif']
        diff_gif_6 = st.session_state.prediction_result['diff6_gif']
        diff_gif_12 = st.session_state.prediction_result['diff12_gif']
        update_prediction_visualization(gt0_gif, gt_gif_6, gt_gif_12, pred_gif_6, pred_gif_12, diff_gif_6, diff_gif_12)