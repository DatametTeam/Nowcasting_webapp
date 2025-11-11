"""
Status indicators and spinners for background operations.
"""
import streamlit as st


def background_checker_spinner(columns):
    """
    Display spinner for background file checker.

    Args:
        columns: Streamlit columns to render in
    """
    print("BACKGROUND checker spinner")
    with columns[1]:
        # with st.spinner("🔄 Running background file **CHECKER**..", show_time=False):
        #     while True:
        #         time.sleep(5)
        st.write("🔄 Running background file **CHECKER**..")


def background_prediction_calculator_spinner(columns):
    """
    Display spinner for background prediction calculator.

    Args:
        columns: Streamlit columns to render in
    """
    with columns[1]:
        st.write("🚀 new data file **FOUND**..")
        # with st.spinner("🛠️ Running background prediction **CALCULATOR**..", show_time=False):
        #     while True:
        #         time.sleep(5)
        st.write("🛠️ Running background prediction **CALCULATOR**..")


def background_prediction_loader_spinner(columns):
    """
    Display spinner for background prediction loader.

    Args:
        columns: Streamlit columns to render in
    """
    with columns[1]:
        # with st.spinner("⚙️ Running background prediction **LOADER**..", show_time=False):
        #     while True:
        #         time.sleep(5)
        st.write("⚙️ Running background prediction **LOADER**..")