"""
Mock PBS job submission for local development.
Simulates HPC job submission and generates mock predictions.
"""

import logging
import multiprocessing
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from nwc_webapp.config.config import get_config
from nwc_webapp.logging_config import setup_logger
from nwc_webapp.mock.generator import create_mock_prediction_file

# Set up logger
logger = setup_logger(__name__)

# Global process tracker
_IN_PROCESS: Optional[multiprocessing.Process] = None
_JOB_COUNTER = 0


def get_job_status(job_id: str) -> str:
    """
    Get the status of a mock job.

    Args:
        job_id: Job ID to check

    Returns:
        "R" if running, "C" if complete
    """
    global _IN_PROCESS

    if _IN_PROCESS is not None and _IN_PROCESS.is_alive():
        return "R"  # Running
    else:
        return "C"  # Complete


def submit_inference(inf_args: dict) -> Tuple[str, Path]:
    """
    Submit a mock inference job (simulates PBS qsub).

    Args:
        inf_args: Dictionary with inference arguments

    Returns:
        Tuple of (job_id, output_dir)
    """
    global _IN_PROCESS, _JOB_COUNTER

    _JOB_COUNTER += 1
    job_id = f"mock_{_JOB_COUNTER}_{int(time.time())}"

    config = get_config()
    model_name = inf_args.get("model_name", "Test")

    # Create output directory
    pred_folder = config.prediction_output / model_name
    pred_folder.mkdir(parents=True, exist_ok=True)

    logger.info(f"📝 Mock job submitted: {job_id} (model: {model_name})")

    # Start background process to simulate job execution
    p = multiprocessing.Process(
        target=_mock_inference_worker, args=(job_id, model_name, pred_folder, inf_args), daemon=True
    )
    p.start()
    _IN_PROCESS = p

    return job_id, pred_folder


def _mock_inference_worker(job_id: str, model_name: str, output_dir: Path, inf_args: dict):
    """
    Background worker that simulates inference job.

    Args:
        job_id: Mock job ID
        model_name: Name of the model
        output_dir: Where to save predictions
        inf_args: Inference arguments
    """
    try:
        logger.info(f"🚀 Mock job {job_id} started")

        # Simulate processing time (5-10 seconds)
        processing_time = 5 + (hash(model_name) % 6)
        logger.info(f"⏳ Simulating {model_name} inference ({processing_time}s)...")
        time.sleep(processing_time)

        # Check if this is a real-time prediction
        is_real_time = inf_args.get("real_time", False)

        if is_real_time:
            # Use the specified output file for real-time predictions
            pred_file = inf_args.get("output_file")
            if not pred_file:
                raise ValueError("Real-time prediction requires output_file in inf_args")
        else:
            # Standard predictions go to predictions.npy
            pred_file = output_dir / "predictions.npy"

        logger.info(f"📊 Generating mock predictions for {model_name}...")

        # For real-time: single sequence (1, 12, H, W)
        # For standard: multiple sequences (24, 12, H, W)
        num_sequences = 1 if is_real_time else 24

        create_mock_prediction_file(
            output_path=pred_file,
            model_name=model_name,
            start_time=datetime.now(),
            num_sequences=num_sequences,
            sequence_length=12,
        )

        logger.info(f"✅ Mock job {job_id} completed successfully")
        logger.info(f"📁 Predictions saved to: {pred_file}")

    except Exception as e:
        logger.error(f"❌ Mock job {job_id} failed: {e}", exc_info=True)


# For backward compatibility
def inference_mock(inf_args):
    """Legacy interface for mock inference."""
    return submit_inference(inf_args)


def start_prediction_job(model: str, latest_data: str) -> str:
    """
    Start a prediction job for real-time predictions (mock version).
    Matches the PBS interface for compatibility.

    Args:
        model: Model name to use for prediction
        latest_data: Latest data file name (e.g., "11-11-2025-14-35.hdf" or just the stem)
                     This is the starting point - the inference will use the last 12 timesteps
                     (1 hour of data at 5-min intervals) ending at this time.

    Returns:
        Job ID
    """
    # Clean the filename if needed
    if "." in latest_data:
        latest_data = latest_data.split(".")[0]

    logger.info(f"📍 Starting prediction from data: {latest_data}")
    logger.info(f"   (Will use last 1 hour of data: 12 timesteps at 5-min intervals)")

    config = get_config()

    # For real-time predictions, create output in real_time_pred folder
    output_dir = config.real_time_pred / model
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output filename matches the input data timestamp
    output_file = output_dir / f"{latest_data}.npy"

    # Create inference args
    inf_args = {
        "model_name": model,
        "start_time": datetime.now(),
        "latest_data": latest_data,
        "output_file": output_file,
        "submitted": True,
        "real_time": True,  # Flag for real-time prediction
    }

    job_id, _ = submit_inference(inf_args)
    logger.info(f"📝 Real-time prediction job started: {job_id} for {model}")
    logger.info(f"📁 Output will be saved to: {output_file}")

    return job_id


if __name__ == "__main__":
    # Test the mock system
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    logger.info("Testing mock PBS system...\n")

    # Submit a test job
    test_args = {"model_name": "TestModel", "start_time": datetime.now(), "submitted": True}

    job_id, output_dir = submit_inference(test_args)
    logger.info(f"Job submitted: {job_id}")
    logger.info(f"Output dir: {output_dir}")

    # Monitor job status
    while get_job_status(job_id) == "R":
        logger.info("Job running...")
        time.sleep(1)

    logger.info("Job complete!")
