"""
Mock services for local development without HPC resources.
"""

from nwc_webapp.services.mock.mock import start_prediction_job
from nwc_webapp.services.mock.mock_data_generator import (
    setup_mock_sri_data,
    setup_mock_prediction_data
)

__all__ = [
    'start_prediction_job',
    'setup_mock_sri_data',
    'setup_mock_prediction_data',
]