"""
Mock data generator for local development.
Creates realistic-looking weather radar data without HPC/GPU.
"""

import os
from datetime import datetime, timedelta
from pathlib import Path

import h5py
import numpy as np

from nwc_webapp.logging_config import setup_logger

# Set up logger
logger = setup_logger(__name__)


def generate_realistic_precipitation_field(shape=(1400, 1200), intensity_scale=50.0, num_cells=5, seed=None):
    """
    Generate a realistic-looking precipitation field.

    Args:
        shape: Shape of the output array (height, width)
        intensity_scale: Maximum intensity value (mm/h)
        num_cells: Number of precipitation cells to generate
        seed: Random seed for reproducibility

    Returns:
        2D numpy array with precipitation values
    """
    if seed is not None:
        np.random.seed(seed)

    # Create empty field
    field = np.zeros(shape, dtype=np.float32)

    # Generate multiple precipitation cells
    for i in range(num_cells):
        # Random center position
        center_y = np.random.randint(shape[0] // 4, 3 * shape[0] // 4)
        center_x = np.random.randint(shape[1] // 4, 3 * shape[1] // 4)

        # Random cell parameters
        radius = np.random.randint(80, 200)
        intensity = np.random.uniform(0.3, 1.0) * intensity_scale

        # Create meshgrid for this cell
        y, x = np.ogrid[: shape[0], : shape[1]]

        # Calculate distance from center
        dist = np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)

        # Gaussian-like precipitation cell
        cell = intensity * np.exp(-((dist / radius) ** 2))

        # Add some noise for realism
        noise = np.random.normal(0, 0.05 * intensity, shape)
        cell += noise

        # Add to field
        field = np.maximum(field, cell)

    # Clip negative values
    field = np.clip(field, 0, intensity_scale)

    # Add some general noise
    field += np.random.uniform(0, 0.5, shape)
    field = np.clip(field, 0, intensity_scale)

    return field


def generate_temporal_sequence(num_timesteps=12, shape=(1400, 1200), base_seed=None):
    """
    Generate a temporal sequence of precipitation fields.
    Fields evolve smoothly over time.

    Args:
        num_timesteps: Number of time steps
        shape: Shape of each field
        base_seed: Base random seed

    Returns:
        3D numpy array (timesteps, height, width)
    """
    sequence = []

    for t in range(num_timesteps):
        # Use time-varying seed for smooth evolution
        seed = base_seed + t if base_seed is not None else None

        # Generate field with time-varying intensity
        intensity = 30 + 20 * np.sin(2 * np.pi * t / num_timesteps)
        field = generate_realistic_precipitation_field(
            shape=shape, intensity_scale=intensity, num_cells=3 + t % 3, seed=seed  # Varying number of cells
        )

        sequence.append(field)

    return np.array(sequence, dtype=np.float32)


def create_mock_hdf_file(output_path: Path, timestamp: datetime):
    """
    Create a mock HDF5 file matching real SRI data structure.

    Real SRI structure: /dataset1/data1/data

    Args:
        output_path: Path where to save the HDF5 file
        timestamp: Timestamp for the data
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate realistic precipitation field
    data = generate_realistic_precipitation_field(shape=(1400, 1200), seed=int(timestamp.timestamp()))

    with h5py.File(output_path, "w") as f:
        # Create nested group structure to match real SRI files
        dataset1 = f.create_group("dataset1")
        data1 = dataset1.create_group("data1")

        # Create the data dataset at the correct path
        data1.create_dataset("data", data=data, compression="gzip")

        # Add metadata
        f.attrs["timestamp"] = timestamp.isoformat()
        f.attrs["format"] = "mock_sri"
        f.attrs["units"] = "mm/h"

        # Add coordinate information
        # Italy bounding box approximately
        f.create_dataset("latitude", data=np.linspace(35.0, 47.6, 1400))
        f.create_dataset("longitude", data=np.linspace(4.5, 20.5, 1200))


def create_mock_prediction_file(
    output_path: Path, model_name: str, start_time: datetime, num_sequences=24, sequence_length=12
):
    """
    Create a mock prediction NPY file.

    Args:
        output_path: Path where to save the NPY file
        model_name: Name of the model
        start_time: Start time for predictions
        num_sequences: Number of prediction sequences
        sequence_length: Length of each sequence (forecast horizon)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate temporal predictions
    predictions = []

    for i in range(num_sequences):
        sequence = generate_temporal_sequence(
            num_timesteps=sequence_length, shape=(1400, 1200), base_seed=int(start_time.timestamp()) + i * 1000
        )
        predictions.append(sequence)

    predictions = np.array(predictions, dtype=np.float32)

    # Save as NPY file
    np.save(output_path, predictions)

    return predictions


def setup_mock_sri_data(sri_folder: Path, num_files=10):
    """
    Set up mock SRI HDF files for real-time testing.

    Args:
        sri_folder: Folder where to create mock SRI files
        num_files: Number of files to create
    """
    sri_folder.mkdir(parents=True, exist_ok=True)

    # Create files with 5-minute intervals
    base_time = datetime.now() - timedelta(minutes=5 * num_files)

    for i in range(num_files):
        timestamp = base_time + timedelta(minutes=5 * i)
        filename = timestamp.strftime("%d-%m-%Y-%H-%M.hdf")
        filepath = sri_folder / filename

        if not filepath.exists():
            create_mock_hdf_file(filepath, timestamp)
            logger.info(f"Created mock SRI file: {filename}")


def generate_mock_predictions_for_range(model_name: str, start_dt: datetime, end_dt: datetime) -> int:
    """
    Generate mock prediction files for a date range (for local development).

    Creates prediction files in format: DD-MM-YYYY-HH-MM.npy
    Each file has shape (12, 1400, 1200) with 5-minute forecast intervals.

    Args:
        model_name: Model name
        start_dt: Start datetime
        end_dt: End datetime

    Returns:
        Number of prediction files created
    """
    from nwc_webapp.config.config import get_config

    config = get_config()
    pred_folder = config.real_time_pred / model_name
    pred_folder.mkdir(parents=True, exist_ok=True)

    # Generate all timestamps with 5-minute intervals
    timestamps = []
    current = start_dt
    while current <= end_dt:
        timestamps.append(current)
        current += timedelta(minutes=5)

    logger.info(f"Generating {len(timestamps)} mock prediction files for {model_name}...")

    created_count = 0
    for timestamp in timestamps:
        # Create filename in format: DD-MM-YYYY-HH-MM.npy
        filename = timestamp.strftime("%d-%m-%Y-%H-%M") + ".npy"
        filepath = pred_folder / filename

        # Skip if file already exists
        if filepath.exists():
            logger.debug(f"Skipping existing file: {filename}")
            continue

        # Generate mock prediction sequence (12 timesteps, shape: 12x1400x1200)
        prediction = generate_temporal_sequence(
            num_timesteps=12, shape=(1400, 1200), base_seed=int(timestamp.timestamp())
        )

        # Save as NPY file
        np.save(filepath, prediction)
        created_count += 1
        logger.debug(f"Created mock prediction: {filename}")

    logger.info(f"✅ Created {created_count}/{len(timestamps)} mock prediction files for {model_name}")
    return created_count


def setup_mock_prediction_data(pred_folder: Path, model_names: list):
    """
    Set up mock prediction data for all models.

    Args:
        pred_folder: Base folder for predictions
        model_names: List of model names to create data for
    """
    from nwc_webapp.config.config import get_config

    pred_folder.mkdir(parents=True, exist_ok=True)
    config = get_config()

    for model_name in model_names:
        model_folder = pred_folder / model_name
        model_folder.mkdir(parents=True, exist_ok=True)

        # Create test predictions
        pred_file = model_folder / "predictions.npy"
        if not pred_file.exists():
            create_mock_prediction_file(pred_file, model_name, datetime.now(), num_sequences=24, sequence_length=12)
            logger.info(f"Created mock prediction file for {model_name}")

        # Create real-time predictions folder using config
        realtime_folder = config.real_time_pred / model_name
        realtime_folder.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Example usage: set up mock data
    from environment import get_data_root, get_prediction_output_dir, get_sri_folder

    logger.info("Setting up mock data for local development...")

    # Set up mock SRI data
    sri_folder = get_sri_folder()
    setup_mock_sri_data(sri_folder, num_files=20)

    # Set up mock predictions
    pred_folder = get_prediction_output_dir()
    models = ["ConvLSTM", "ED_ConvLSTM", "DynamicUnet", "pystep", "Test"]
    setup_mock_prediction_data(pred_folder, models)

    logger.info("Mock data setup complete!")
