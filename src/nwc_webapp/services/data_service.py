"""
Data service for loading and processing radar and prediction data.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import h5py
import numpy as np

from nwc_webapp.config.config import get_config
from nwc_webapp.config.logging_config import get_logger

logger = get_logger(__name__)


class DataService:
    """Service for data loading and processing."""

    def __init__(self):
        """Initialize data service."""
        self.config = get_config()
        self.logger = logger
        self._radar_mask: Optional[np.ndarray] = None

    @property
    def radar_mask(self) -> np.ndarray:
        """
        Get radar mask (lazy loaded).

        Returns:
            Radar mask array
        """
        if self._radar_mask is None:
            self._radar_mask = self.load_radar_mask()
        return self._radar_mask

    def load_radar_mask(self) -> np.ndarray:
        """
        Load radar mask from HDF5 file.

        Returns:
            Radar mask array
        """
        mask_path = self.config.radar_mask_path

        try:
            self.logger.debug(f"Loading radar mask from: {mask_path}")
            with h5py.File(mask_path, "r") as f:
                mask = f["mask"][()]
            self.logger.info(f"Radar mask loaded: shape={mask.shape}")
            return mask
        except Exception as e:
            self.logger.error(f"Error loading radar mask: {e}", exc_info=True)
            # Return default mask (all ones)
            shape = self.config.visualization.data_shape
            return np.ones((shape["height"], shape["width"]), dtype=np.float32)

    def load_prediction_data(
        self, model_name: str, start_idx: int = 0, end_idx: Optional[int] = None, apply_mask: bool = True
    ) -> np.ndarray:
        """
        Load prediction data for a model.

        Args:
            model_name: Model name
            start_idx: Start index for slicing
            end_idx: End index for slicing (None for all)
            apply_mask: Whether to apply radar mask

        Returns:
            Prediction array
        """
        pred_path = self.config.prediction_output / model_name / "predictions.npy"

        try:
            self.logger.debug(f"Loading predictions from: {pred_path}")

            # Load with memory mapping for efficiency
            data = np.load(pred_path, mmap_mode="r")

            # Slice the data
            if end_idx is not None:
                data = data[start_idx:end_idx]
            else:
                data = data[start_idx:]

            # Convert to array (from mmap)
            data = np.array(data, dtype=np.float32)

            # Clip negative values
            data = np.clip(data, 0, None)

            # Apply radar mask if requested
            if apply_mask:
                data = data * self.radar_mask

            self.logger.info(f"Loaded {model_name} predictions: shape={data.shape}")
            return data

        except FileNotFoundError:
            self.logger.error(f"Prediction file not found: {pred_path}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading predictions: {e}", exc_info=True)
            raise

    def load_ground_truth(self, start_idx: int = 12, end_idx: int = 36, apply_mask: bool = False) -> np.ndarray:
        """
        Load ground truth data (from Test model).

        Args:
            start_idx: Start index for slicing
            end_idx: End index for slicing
            apply_mask: Whether to apply radar mask

        Returns:
            Ground truth array
        """
        return self.load_prediction_data(model_name="Test", start_idx=start_idx, end_idx=end_idx, apply_mask=apply_mask)

    def load_sri_file(self, file_path: Path) -> np.ndarray:
        """
        Load SRI (radar) data from HDF5 file.

        Args:
            file_path: Path to HDF5 file

        Returns:
            Precipitation data array
        """
        try:
            self.logger.debug(f"Loading SRI file: {file_path}")

            with h5py.File(file_path, "r") as f:
                # Try common dataset names
                if "precipitation" in f:
                    data = f["precipitation"][()]
                elif "data" in f:
                    data = f["data"][()]
                else:
                    # Get first dataset
                    dataset_name = list(f.keys())[0]
                    data = f[dataset_name][()]

            self.logger.debug(f"Loaded SRI data: shape={data.shape}")
            return data

        except Exception as e:
            self.logger.error(f"Error loading SRI file: {e}", exc_info=True)
            raise

    def get_latest_sri_file(self) -> Optional[str]:
        """
        Get the filename of the latest SRI file.

        Returns:
            Latest SRI filename or None if no files found
        """
        sri_folder = self.config.sri_folder

        try:
            # Get all .hdf files
            files = [f.name for f in sri_folder.glob("*.hdf")]

            if not files:
                self.logger.warning(f"No SRI files found in: {sri_folder}")
                return None

            # Sort by timestamp in filename (format: DD-MM-YYYY-HH-MM.hdf)
            files.sort(key=lambda x: datetime.strptime(x.split(".")[0], "%d-%m-%Y-%H-%M"), reverse=True)

            latest = files[0]
            self.logger.debug(f"Latest SRI file: {latest}")
            return latest

        except Exception as e:
            self.logger.error(f"Error getting latest SRI file: {e}", exc_info=True)
            return None

    def preprocess_data(
        self, data: np.ndarray, clip_range: Optional[Tuple[float, float]] = None, normalize: bool = False
    ) -> np.ndarray:
        """
        Preprocess data array.

        Args:
            data: Input data
            clip_range: Optional (min, max) values to clip to
            normalize: Whether to normalize to [0, 1]

        Returns:
            Preprocessed data
        """
        data = data.copy()

        # Clip values
        if clip_range is not None:
            data = np.clip(data, clip_range[0], clip_range[1])

        # Normalize
        if normalize:
            data_min = np.min(data)
            data_max = np.max(data)
            if data_max > data_min:
                data = (data - data_min) / (data_max - data_min)

        return data
