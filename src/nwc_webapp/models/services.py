"""
Prediction service for handling model inference and job management.
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from nwc_webapp.config.config import get_config
from nwc_webapp.logging_config import setup_logger
from nwc_webapp.hpc.pbs import is_pbs_available

# Import PBS or mock based on environment
if is_pbs_available():
    from nwc_webapp.hpc.pbs import get_job_status, submit_inference
else:
    from nwc_webapp.mock.predictions import get_job_status, submit_inference

logger = get_logger(__name__)


class PredictionService:
    """Service for managing prediction jobs and inference."""

    def __init__(self):
        """Initialize prediction service."""
        self.config = get_config()
        self.logger = logger

    def submit_job(self, args: Dict[str, Any]) -> Tuple[Optional[str], Optional[Path]]:
        """
        Submit a prediction job.

        Args:
            args: Dictionary with job parameters:
                - start_date: Start date
                - end_date: End date
                - start_time: Start time
                - end_time: End time
                - model_name: Model name
                - submitted: Whether form was submitted

        Returns:
            Tuple of (job_id, output_dir) or (None, None) on error
        """
        model_name = args.get("model_name", "Test")
        start_date = args.get("start_date")
        end_date = args.get("end_date")

        self.logger.info(f"Submitting prediction job for model '{model_name}'")
        self.logger.debug(f"Job parameters: {args}")

        try:
            job_id, output_dir = submit_inference(args)

            if job_id:
                self.logger.info(f"Job submitted successfully: {job_id}")
                self.logger.debug(f"Output directory: {output_dir}")
                return job_id, output_dir
            else:
                self.logger.error("Job submission failed: No job ID returned")
                return None, None

        except Exception as e:
            self.logger.error(f"Error submitting job: {e}", exc_info=True)
            return None, None

    def wait_for_job_completion(self, job_id: str, timeout: int = 3600, check_interval: int = 5) -> bool:
        """
        Wait for a job to complete.

        Args:
            job_id: Job ID to monitor
            timeout: Maximum time to wait in seconds
            check_interval: How often to check status in seconds

        Returns:
            True if job completed, False if timeout or error
        """
        self.logger.info(f"Waiting for job {job_id} to complete...")

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                status = get_job_status(job_id)

                if status == "R":
                    # Job is still running
                    self.logger.debug(f"Job {job_id} still running...")
                    time.sleep(check_interval)
                else:
                    # Job completed
                    self.logger.info(f"Job {job_id} completed with status: {status}")
                    return True

            except Exception as e:
                self.logger.error(f"Error checking job status: {e}", exc_info=True)
                return False

        self.logger.warning(f"Job {job_id} timed out after {timeout}s")
        return False

    def get_job_status(self, job_id: str) -> str:
        """
        Get the status of a job.

        Args:
            job_id: Job ID

        Returns:
            Job status string ("R" for running, "ended" for completed)
        """
        try:
            status = get_job_status(job_id)
            return status
        except Exception as e:
            self.logger.error(f"Error getting job status: {e}", exc_info=True)
            return "unknown"

    def get_prediction_paths(self, model_name: str, timestamp: Optional[datetime] = None) -> Dict[str, Path]:
        """
        Get paths to prediction outputs for a model.

        Args:
            model_name: Model name
            timestamp: Optional timestamp for specific prediction

        Returns:
            Dictionary with paths to prediction files
        """
        pred_folder = self.config.prediction_output / model_name

        paths = {
            "predictions": pred_folder / "predictions.npy",
            "model_folder": pred_folder,
        }

        if timestamp:
            timestamp_str = timestamp.strftime("%d%m%Y_%H%M")
            paths["timestamped"] = pred_folder / f"{timestamp_str}.npy"

        return paths

    def is_pbs_available(self) -> bool:
        """Check if PBS is available."""
        return is_pbs_available()
