#!/usr/bin/env python
"""
Setup script to initialize mock data for local development.
Run this before starting the Streamlit app locally.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from nwc_webapp.config.environment import get_data_root, get_prediction_output_dir, get_sri_folder, is_local
from nwc_webapp.logging_config import setup_logger
from nwc_webapp.mock.generator import setup_mock_prediction_data, setup_mock_sri_data

# Set up logger
logger = setup_logger(__name__)


def main():
    if not is_local():
        logger.info("This script is only needed for local development.")
        logger.info("Detected HPC environment - no mock data needed.")
        return

    logger.info("=" * 60)
    logger.info("Setting up mock data for local development...")
    logger.info("=" * 60)

    data_root = get_data_root()
    logger.info(f"\nData root: {data_root}")

    # Set up mock SRI data (radar input files)
    sri_folder = get_sri_folder()
    logger.info(f"\nCreating mock SRI data in: {sri_folder}")
    setup_mock_sri_data(sri_folder, num_files=20)

    # Set up mock predictions for all models
    pred_folder = get_prediction_output_dir()
    logger.info(f"\nCreating mock prediction data in: {pred_folder}")
    models = ["ConvLSTM", "ED_ConvLSTM", "DynamicUnet", "pystep", "Test"]
    setup_mock_prediction_data(pred_folder, models)

    logger.info("\n" + "=" * 60)
    logger.info("Mock data setup complete!")
    logger.info("=" * 60)
    logger.info("\nYou can now run the app with:")
    logger.info("  streamlit run src/nwc_webapp/hello.py")
    logger.info()


if __name__ == "__main__":
    main()
