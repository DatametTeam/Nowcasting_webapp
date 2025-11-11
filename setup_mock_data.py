#!/usr/bin/env python
"""
Setup script to initialize mock data for local development.
Run this before starting the Streamlit app locally.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from nwc_webapp.mock_data_generator import setup_mock_sri_data, setup_mock_prediction_data
from nwc_webapp.environment import get_data_root, get_sri_folder, get_prediction_output_dir, is_local

def main():
    if not is_local():
        print("This script is only needed for local development.")
        print("Detected HPC environment - no mock data needed.")
        return

    print("=" * 60)
    print("Setting up mock data for local development...")
    print("=" * 60)

    data_root = get_data_root()
    print(f"\nData root: {data_root}")

    # Set up mock SRI data (radar input files)
    sri_folder = get_sri_folder()
    print(f"\nCreating mock SRI data in: {sri_folder}")
    setup_mock_sri_data(sri_folder, num_files=20)

    # Set up mock predictions for all models
    pred_folder = get_prediction_output_dir()
    print(f"\nCreating mock prediction data in: {pred_folder}")
    models = ["ConvLSTM", "ED_ConvLSTM", "DynamicUnet", "pystep", "Test"]
    setup_mock_prediction_data(pred_folder, models)

    print("\n" + "=" * 60)
    print("Mock data setup complete!")
    print("=" * 60)
    print("\nYou can now run the app with:")
    print("  streamlit run src/nwc_webapp/hello.py")
    print()

if __name__ == "__main__":
    main()