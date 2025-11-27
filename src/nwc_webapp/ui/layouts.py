import io
import os.path
from datetime import datetime, time, timedelta
from pathlib import Path

import numpy as np
import streamlit as st
from folium import folium
from PIL import Image
from streamlit_folium import st_folium

from nwc_webapp.evaluation.plots import generate_metrics_plot
from nwc_webapp.ui.maps import create_map
from nwc_webapp.utils import compute_figure_gpd, create_colorbar_fig


def round_to_previous_5_minutes():
    """Round current time to the previous 5-minute interval"""
    now = datetime.now()
    minutes = (now.minute // 5) * 5
    return time(now.hour, minutes)


def configure_sidebar(model_list):
    with st.sidebar:
        root_dir = Path(__file__).resolve().parent.parent.parent
        st.image(
            os.path.join(root_dir, "imgs/LDO_logo_transp.png"), width="content"
        )  # Replace with the path to your logo

        st.markdown("<h1 style='font-size: 32px; font-weight: bold;'>NOWCASTING</h1>", unsafe_allow_html=True)
        with st.form("weather_prediction_form"):
            # Get current date and rounded time as fallback
            current_date = datetime.now().date()
            rounded_time = round_to_previous_5_minutes()

            # Use session state to persist values across reloads, or use current date/time as default
            default_start_date = st.session_state.get("sidebar_start_date", current_date)
            default_start_time = st.session_state.get("sidebar_start_time", rounded_time)
            default_end_date = st.session_state.get("sidebar_end_date", current_date)
            default_end_time = st.session_state.get("sidebar_end_time", rounded_time)
            default_model = st.session_state.get("sidebar_model", model_list[0] if model_list else None)

            # Date inputs
            start_date = st.date_input(
                "Select a start date",
                value=default_start_date,
                format="DD/MM/YYYY",
                max_value=datetime.today().date(),
            )
            # Time inputs
            start_time = st.time_input(
                "Select a start time", value=default_start_time, step=timedelta(minutes=5)  # 5-minute intervals
            )
            end_date = st.date_input("Select an end date", value=default_end_date, format="DD/MM/YYYY")

            end_time = st.time_input(
                "Select an end time", value=default_end_time, step=timedelta(minutes=5)  # 5-minute intervals
            )

            # Model selection
            model_name = st.selectbox(
                "Select a model",
                model_list,
                index=model_list.index(default_model) if default_model in model_list else 0,
            )

            # Form submission
            submitted = st.form_submit_button("Submit", type="primary", width="content")

            # Store values in session state when form is submitted
            if submitted:
                st.session_state["sidebar_start_date"] = start_date
                st.session_state["sidebar_start_time"] = start_time
                st.session_state["sidebar_end_date"] = end_date
                st.session_state["sidebar_end_time"] = end_time
                st.session_state["sidebar_model"] = model_name
                # Switch to nowcasting tab when form is submitted
                st.session_state["active_tab"] = "Nowcasting"

        return {
            "start_date": start_date,
            "end_date": end_date,
            "start_time": start_time,
            "end_time": end_time,
            "model_name": model_name,
            "submitted": submitted,
        }


def init_prediction_visualization_layout():
    with st.container():
        row1 = st.columns([3, 0.3, 3, 3, 3, 0.4], vertical_alignment="center")
        row2 = st.columns([3, 0.3, 3, 3, 3, 0.4], vertical_alignment="center")
        row3 = st.columns([3, 0.3, 3, 3, 3, 0.4], vertical_alignment="center")

        with row1[0]:
            st.markdown("<h3 style='text-align: center;'>Current Time</h3>", unsafe_allow_html=True)
        with row1[2]:
            st.markdown("<h3 style='text-align: center;'>Groundtruths</h3>", unsafe_allow_html=True)
        with row1[3]:
            st.markdown("<h3 style='text-align: center;'>Predictions</h3>", unsafe_allow_html=True)
        with row1[4]:
            st.markdown("<h3 style='text-align: center;'>Differences</h3>", unsafe_allow_html=True)

        with row2[0]:
            gt_current = st.empty()

        with row3[0]:
            pred_current = st.empty()

        with row2[1]:
            st.markdown(
                """
                <div style="display: flex; align-items: center; justify-content: center; height: 100%;">
                    <div style="transform: rotate(-90deg); font-weight: bold; font-size: 1.5em; white-space: nowrap;">
                        +30min
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with row3[1]:
            st.markdown(
                """
                <div style="position: relative; height: 100%; width: 100%;">
                    <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%) rotate(
                    270deg);
                    font-weight: bold; font-size: 1.5em; white-space: nowrap;">
                        +60min
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with row2[2]:
            gt_plus_30 = st.empty()
        with row3[2]:
            gt_plus_60 = st.empty()

        with row2[3]:
            pred_plus_30 = st.empty()
        with row3[3]:
            pred_plus_60 = st.empty()

        with row2[4]:
            diff_plus_30 = st.empty()  # Added difference container for +30min
        with row3[4]:
            diff_plus_60 = st.empty()  # Added difference container for +60min

        with row2[5]:
            colorbar30 = st.empty()
        with row3[5]:
            colorbar60 = st.empty()

    return (
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
    )


def precompute_images(frame_dict):
    """
    Precomputes images from frame data and stores them in a list.
    """
    precomputed_images = []
    total_frames = len(frame_dict)

    # Initialize a progress bar
    progress = st.progress(0)
    progress_text = st.empty()

    for idx, (timestamp, frame) in enumerate(frame_dict.items()):
        if frame is not None:
            fig = compute_figure_gpd(frame, timestamp)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            precomputed_images.append((timestamp, Image.open(buf)))
        progress.progress((idx + 1) / total_frames)
        progress_text.text(f"Processing image {idx + 1}/{total_frames}")

    progress.empty()

    return precomputed_images


def init_second_tab_layout(groundtruth_images, target_frames, pred_frames):
    with st.spinner("🔄 Loading layout..", show_time=True):
        # Define the layout with 5 columns (1 for the label and 4 for images)
        groundtruth_rows = st.columns([0.2] + [1] * 4 + [0.2], vertical_alignment="center")

        # First column spanning all 3 rows (label column)
        with groundtruth_rows[0]:
            st.markdown(
                """
                <div style="position: relative; height: 100%; width: 100%;">
                    <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%) rotate(270deg);
                    font-weight: bold; font-size: 1.5em; white-space: nowrap;">
                        Groundtruths
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with groundtruth_rows[-1]:
            st.image(create_colorbar_fig())

        # Loop through the 4 columns for the images
        for row_idx in range(3):  # Loop through rows 1 to 3
            row_offset = row_idx * 4  # 4 frames per row
            for col_idx in range(1, 5):  # Columns 1 to 4 (skip index 0)
                with groundtruth_rows[col_idx]:
                    timestamp_idx = col_idx - 1 + row_offset
                    if timestamp_idx < len(groundtruth_images):
                        timestamp, image = groundtruth_images[timestamp_idx]
                        st.image(image, width="content")

        # Additional layout for TARGET and PREDICTION columns
        target_pred_rows = []

        for i in range(13):
            target_pred_rows.append(st.columns([0.5, 0.2, 1.5, 1.5, 1.5, 0.2, 0.5], vertical_alignment="center"))

    # Titles for TARGET and PREDICTION
    with target_pred_rows[0][2]:
        st.markdown(
            """<div style="text-align: center; font-weight: bold; font-size: 2em;">Target</div>""",
            unsafe_allow_html=True,
        )

        with target_pred_rows[0][3]:
            st.markdown(
                """<div style="text-align: center; font-weight: bold; font-size: 2em;">Prediction</div>""",
                unsafe_allow_html=True,
            )

        # Titles for DIFFERENCES
        with target_pred_rows[0][4]:
            st.markdown(
                """<div style="text-align: center; font-weight: bold; font-size: 2em;">Differences</div>""",
                unsafe_allow_html=True,
            )

        # Fill TARGET and PREDICTION frames row by row
        for row_idx in range(1, 13):  # Skip the title row
            # Left empty column with +Xmins labels
            with target_pred_rows[row_idx][1]:
                st.markdown(
                    f"""<div style="position: relative; height: 100%; width: 100%;">
                    <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%) rotate(270deg);
                    font-size: 1em; font-weight: bold;">+{row_idx * 5}mins</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

            # TARGET frames
            with target_pred_rows[row_idx][2]:
                if row_idx - 1 < len(target_frames):
                    timestamp = list(target_frames.keys())[row_idx - 1]
                    frame = target_frames.get(timestamp, None)
                    if frame is not None:
                        fig = compute_figure_gpd(frame, timestamp)
                        buf = io.BytesIO()
                        fig.savefig(buf, format="png", bbox_inches="tight")
                        buf.seek(0)
                        image = Image.open(buf)
                        st.image(image, width="content")

            # PREDICTION frames
            with target_pred_rows[row_idx][3]:
                if row_idx - 1 < len(pred_frames):
                    timestamp = list(pred_frames.keys())[row_idx - 1]
                    frame = pred_frames.get(timestamp, None)
                    if frame is not None:
                        fig = compute_figure_gpd(frame, "PRED @ " + timestamp)
                        buf = io.BytesIO()
                        fig.savefig(buf, format="png", bbox_inches="tight")
                        buf.seek(0)
                        image = Image.open(buf)
                        st.image(image, width="content")

            # DIFFERENCE frames
            with target_pred_rows[row_idx][4]:
                if row_idx - 1 < len(pred_frames) and row_idx - 1 < len(target_frames):
                    timestamp_pred = list(pred_frames.keys())[row_idx - 1]
                    timestamp_target = list(target_frames.keys())[row_idx - 1]
                    frame_pred = pred_frames.get(timestamp_pred, None)
                    frame_target = target_frames.get(timestamp_target, None)

                    if frame_pred is not None and frame_target is not None:
                        # calculate differences
                        frame_diff = np.abs(frame_target - frame_pred)

                        if frame_diff is not None:
                            fig_diff = compute_figure_gpd(frame_diff, "DIFF @ " + timestamp_pred, name="diff")
                            buf = io.BytesIO()
                            fig_diff.savefig(buf, format="png", bbox_inches="tight")
                            buf.seek(0)
                            image = Image.open(buf)
                            st.image(image, width="content")

            with target_pred_rows[row_idx][-2]:
                st.image(create_colorbar_fig(top_adj=0.85, bot_adj=0.07))

    return


def show_metrics_page(model_list):
    # Select time
    selected_date = st.date_input(
        "Select date", value=datetime(2025, 1, 31).date(), format="DD/MM/YYYY"
    )  # TODO: rimettere now

    # Select date
    selected_time = st.time_input("Select time", value=time(1, 0), step=timedelta(minutes=5))  # 5-minute intervals

    # Initialize session state for selected models and plots
    if "selected_models" not in st.session_state:
        st.session_state["selected_models"] = []
    if "plotted_metrics" not in st.session_state:
        st.session_state["plotted_metrics"] = []

    # Display checkboxes for all models
    st.subheader("Select Models to Display Data")

    selected_models = []
    num_columns = 5  # Adjust the number of models per row here

    # Create rows of checkboxes
    for i in range(0, len(model_list), num_columns):
        cols = st.columns(num_columns)
        for col, model in zip(cols, model_list[i : i + num_columns]):
            with col:
                if st.checkbox(model, value=model in st.session_state["selected_models"]):
                    selected_models.append(model)

    st.session_state["selected_models"] = selected_models

    # Button to generate the plot
    if st.button("Generate Plot"):
        if selected_models:

            empty_space = st.empty()
            with empty_space.container():
                with st.status(f":hammer_and_wrench: **Loading results...**", expanded=True) as status:
                    plotted_metrics = generate_metrics_plot(selected_date, selected_time, selected_models, config)
                    # status.update(label=f"Done!", state="complete", expanded=True)
                    st.session_state["plotted_metrics"] = plotted_metrics
            empty_space.empty()

        else:
            st.warning("Please select at least one model.")

    # Display the formula and plots if they exist in session state
    if st.session_state["plotted_metrics"]:
        with st.status(f"Done!", state="complete", expanded=True) as status:
            columns = st.columns([0.5, 0.05, 0.3])
            with columns[2]:
                st.markdown(
                    r"""
                    ### CSI Formula
                    The Critical Success Index (CSI) is calculated as:

                    ### $$CSI = \frac{TP}{TP + FP + FN}$$

                    Where:
                    - **TP** is the True Positives
                    - **FP** is the False Positives
                    - **FN** is the False Negatives
                """,
                    unsafe_allow_html=True,
                )

            with columns[0]:
                for i, plot_buffer in enumerate(st.session_state["plotted_metrics"]):
                    st.image(plot_buffer)


def display_map_layout(model_options, time_options, st, columns):
    with columns[0]:
        internal_columns = st.columns([0.3, 0.1, 0.3])
        with internal_columns[0]:
            # Select model, bound to session state
            st.selectbox("Select a model", options=model_options, key="selected_model")

        with internal_columns[2]:
            # Select time, bound to session state
            st.selectbox(
                "Select a prediction time",
                options=time_options,
                key="selected_time",
            )

        # TODO: da fixare
        st.markdown(
            "<div style='text-align: center; font-size: 18px;'>"
            f"<b>Current Date: {st.session_state.latest_file}</b>"
            "</div>",
            unsafe_allow_html=True,
        )

        map = create_map()

        return map
