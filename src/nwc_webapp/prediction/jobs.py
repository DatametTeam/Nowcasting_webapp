"""
Job submission and management for predictions.
"""
import time
import numpy as np
import streamlit as st

# Import the appropriate submit_inference based on environment
from nwc_webapp.environment import is_hpc

if is_hpc():
    from pbs import submit_inference
else:
    from nwc_webapp.mock import submit_inference


def submit_prediction_job(sidebar_args):
    """
    Submit a prediction job and show progress.

    Args:
        sidebar_args: Dictionary with prediction parameters

    Returns:
        Tuple of (error, out_dir)
    """
    error = None
    with st.status(f':hammer_and_wrench: **Running prediction...**', expanded=True) as status:
        prediction_placeholder = st.empty()
        with prediction_placeholder:
            with prediction_placeholder.container():
                pbs_job_id, out_dir = submit_inference(sidebar_args)
                # if pbs_job_id is None:
                #     error = "Error submitting prediction job"
                #     status.update(label="❌ Prediction failed!", state="error", expanded=True)
                #     return error
                # while get_job_status(pbs_job_id)=="R":
                #     sleep(1)
                #     status.update(label="🔄 Prediction in progress...", state="running", expanded=True)
                progress_bar = st.progress(0)
                your_array = np.zeros(100)
                status_placeholder = st.empty()
                for i in range(len(your_array)):
                    # Your processing logic here
                    progress = (i + 1) / len(your_array)
                    progress_bar.progress(progress)
                    status_placeholder.write(f"Processing item {i + 1}/{len(your_array)}")
                    time.sleep(0.01)
                # status1.update(label="Processing complete!")

        status.update(label="✅ Prediction completed!", state="complete", expanded=False)
    return error, out_dir