"""
Session state management utilities.
"""
import streamlit as st


def initial_state_management(COUNT=None):
    """
    Initialize all required session state variables.

    Args:
        COUNT: Optional count value for auto-refresh
    """
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = None
    if 'selected_time' not in st.session_state:
        st.session_state.selected_time = None
    if 'latest_file' not in st.session_state:
        st.session_state.latest_file = None
    if 'rgba_image' not in st.session_state:
        st.session_state.rgba_image = None
    if 'thread_started' not in st.session_state:
        st.session_state.thread_started = None
    if 'old_count' not in st.session_state:
        st.session_state.old_count = COUNT
    if 'previous_model' not in st.session_state:
        st.session_state.previous_model = None
    if 'previous_time' not in st.session_state:
        st.session_state.previous_time = None
    if not "latest_thread" in st.session_state:
        st.session_state["latest_thread"] = None
    if "launch_prediction_thread" not in st.session_state:
        st.session_state["launch_prediction_thread"] = None
    if "computing_models" not in st.session_state:
        st.session_state["computing_models"] = set()  # Track multiple models being computed