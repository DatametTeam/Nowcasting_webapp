"""
Common utilities shared across multiple pages.
"""

from datetime import datetime

import streamlit as st

from nwc_webapp.logging_config import setup_logger

# Set up logger
logger = setup_logger(__name__)

# Training cutoff date
TRAINING_CUTOFF_DATE = datetime(2025, 1, 1)


def is_training_date(selected_date: datetime) -> bool:
    """
    Check if the selected date is prior to the training cutoff (Jan 1, 2025).

    Args:
        selected_date: The date selected by the user

    Returns:
        True if date is before Jan 1, 2025 (training data), False otherwise
    """
    return selected_date < TRAINING_CUTOFF_DATE


def show_training_date_warning(key_prefix: str) -> bool:
    """
    Show warning dialog for dates prior to Jan 1, 2025.

    Args:
        key_prefix: Prefix for button keys to avoid conflicts (e.g., "nowcasting", "comparison", "third_tab")

    Returns:
        True if user wants to proceed, False otherwise
    """
    st.warning(
        "⚠️ **Training Data Warning**\n\n"
        "Dates prior to **1st January 2025** were used for model training. "
        "The prediction results will not be accountable and may not reflect real-world performance.\n\n"
        "**Are you sure you want to proceed?**"
    )

    col1, col2, _ = st.columns([1, 1, 3])

    with col1:
        if st.button("✅ YES, Proceed", key=f"training_yes_{key_prefix}", width='stretch'):
            st.session_state[f"training_warning_accepted_{key_prefix}"] = True
            return True

    with col2:
        if st.button("❌ NO, Cancel", key=f"training_no_{key_prefix}", width='stretch'):
            st.session_state[f"training_warning_accepted_{key_prefix}"] = False
            st.info("Operation cancelled. Please select a different date.")
            return False

    return False