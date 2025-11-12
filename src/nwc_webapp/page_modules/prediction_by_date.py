"""
Prediction by date and time page.
"""
import streamlit as st
from datetime import time as dt_time
from datetime import datetime, timedelta

from nwc_webapp.ui.layouts import precompute_images, init_second_tab_layout
from nwc_webapp.utils import read_groundtruth_and_target_data
from nwc_webapp.prediction.jobs import submit_prediction_job


def show_prediction_page(model_list):
    """
    Display prediction page with date/time selection.

    Args:
        model_list: List of available models
    """
    st.title("Select Date and Time for Prediction")

    # Date and time selection
    selected_date = st.date_input(
        "Select Date", min_value=datetime(2020, 1, 1).date(), max_value=datetime.today().date(), format="DD/MM/YYYY",
        value=datetime(2025, 2, 6).date())
    selected_time = st.time_input("Select Time", value=dt_time(15, 00))  # get_closest_5_minute_time(), s
    selected_model = st.selectbox("Select model", model_list)

    if st.button("Submit"):
        # Combine selected date and time
        selected_datetime = datetime.combine(selected_date, selected_time)
        prediction_start_datetime = selected_datetime - timedelta(hours=1)
        selected_key = prediction_start_datetime.strftime("%d%m%Y_%H%M")

        args = {'start_date': selected_datetime, 'start_time': prediction_start_datetime, 'model_name': selected_model,
                'submitted': True}

        print("submit prediction")
        submit_prediction_job(args)
        print("submit prediction DONE")

        # Check if groundtruths are already in session state
        if selected_key not in st.session_state:
            print("read ground thruth and target data")
            print("selected key --> " + str(selected_key))
            print("selected model --> " + str(selected_model))
            groundtruth_dict, target_dict, pred_dict = read_groundtruth_and_target_data(selected_key, selected_model)
            print("read ground thruth and target data DONE")

            # Precompute and cache images for groundtruths
            st.session_state[selected_key] = {
                "groundtruths": precompute_images(groundtruth_dict),
                "target_dict": target_dict,
                "pred_dict": pred_dict,
            }
        else:
            # If groundtruths exist, just update the target and prediction dictionaries for the new model
            _, target_dict, pred_dict = read_groundtruth_and_target_data(selected_key, selected_model)
            st.session_state[selected_key]["target_dict"] = target_dict
            st.session_state[selected_key]["pred_dict"] = pred_dict

        # Use cached groundtruths, targets, and predictions
        groundtruth_images = st.session_state[selected_key]["groundtruths"]
        target_dict = st.session_state[selected_key]["target_dict"]
        pred_dict = st.session_state[selected_key]["pred_dict"]

        # Initialize the second tab layout with precomputed images
        print("init second tab layout")
        init_second_tab_layout(groundtruth_images, target_dict, pred_dict)