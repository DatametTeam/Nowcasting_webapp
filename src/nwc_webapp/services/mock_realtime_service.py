"""
Real-time mock data service for local development.
Continuously generates radar data files to simulate live data stream.
"""
import time
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import logging

from nwc_webapp.mock_data_generator import create_mock_hdf_file, create_mock_prediction_file
from nwc_webapp.config import get_config

logger = logging.getLogger(__name__)


class MockRealtimeService:
    """
    Background service that generates mock radar data at regular intervals.
    Simulates the real-time data stream for local development.
    """

    def __init__(self, interval_seconds: int = 60):
        """
        Initialize the mock realtime service.

        Args:
            interval_seconds: How often to check/generate new data files (default: 60 seconds)
                            Note: Files are always created at 5-minute intervals regardless of check frequency
        """
        self.interval_seconds = interval_seconds
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.config = get_config()
        self.sri_folder = self.config.sri_folder

        # Start from the last 5-minute mark
        now = datetime.now()
        self.current_time = now.replace(
            minute=(now.minute // 5) * 5,
            second=0,
            microsecond=0
        )

        # Ensure folders exist
        self.sri_folder.mkdir(parents=True, exist_ok=True)

        logger.info(f"MockRealtimeService initialized (check interval: {interval_seconds}s)")
        logger.info(f"SRI folder: {self.sri_folder}")

    def _count_existing_files(self) -> int:
        """Count how many HDF files exist in the SRI folder."""
        return len(list(self.sri_folder.glob("*.hdf")))

    def _generate_next_file(self):
        """
        Generate the next radar data file.
        Always adds 5 minutes to current_time to maintain proper intervals.
        """
        # Move to next 5-minute mark
        self.current_time = self.current_time + timedelta(minutes=5)

        # Create filename in format: dd-mm-yyyy-HH-MM.hdf
        filename = self.current_time.strftime("%d-%m-%Y-%H-%M.hdf")
        filepath = self.sri_folder / filename

        if not filepath.exists():
            try:
                create_mock_hdf_file(filepath, self.current_time)
                logger.info(f"✅ Generated mock SRI file: {filename}")
                return filepath
            except Exception as e:
                logger.error(f"❌ Failed to generate mock file {filename}: {e}")
                return None
        else:
            logger.info(f"📄 File {filename} already exists, using existing data")
            return filepath

    def _run_loop(self):
        """Main loop that generates files at regular intervals."""
        logger.info("🚀 Mock realtime service started")

        while self.running:
            try:
                # Generate next file
                filepath = self._generate_next_file()

                if filepath:
                    logger.info(f"📡 New radar data available: {filepath.name}")

                # Wait for next interval
                time.sleep(self.interval_seconds)

            except Exception as e:
                logger.error(f"Error in mock service loop: {e}", exc_info=True)
                time.sleep(5)  # Wait a bit before retrying

        logger.info("🛑 Mock realtime service stopped")

    def start(self):
        """Start the mock service in a background thread."""
        if self.running:
            logger.warning("Mock service is already running")
            return

        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True, name="MockRealtimeService")
        self.thread.start()
        logger.info("Mock realtime service thread started")

    def stop(self):
        """Stop the mock service."""
        if not self.running:
            logger.warning("Mock service is not running")
            return

        logger.info("Stopping mock realtime service...")
        self.running = False

        if self.thread:
            self.thread.join(timeout=5)
            if self.thread.is_alive():
                logger.warning("Mock service thread did not stop gracefully")
            else:
                logger.info("Mock service stopped successfully")

    def generate_initial_history(self, num_files: int = 12):
        """
        Generate initial historical files for testing.
        Files are created at proper 5-minute intervals (:00, :05, :10, :15, etc.)

        Args:
            num_files: Number of historical files to create (default: 12 = 1 hour at 5min intervals)
        """
        logger.info(f"Generating {num_files} historical files...")

        # Round current time DOWN to nearest 5-minute mark
        now = datetime.now()
        rounded_now = now.replace(
            minute=(now.minute // 5) * 5,
            second=0,
            microsecond=0
        )

        # Go back in time by num_files * 5 minutes
        base_time = rounded_now - timedelta(minutes=5 * num_files)

        for i in range(num_files):
            timestamp = base_time + timedelta(minutes=5 * i)
            filename = timestamp.strftime("%d-%m-%Y-%H-%M.hdf")
            filepath = self.sri_folder / filename

            if not filepath.exists():
                create_mock_hdf_file(filepath, timestamp)
                logger.info(f"Created historical file: {filename}")

        logger.info("Historical files generation complete")


# Global service instance
_mock_service: Optional[MockRealtimeService] = None


def get_mock_service(interval_seconds: int = 30) -> MockRealtimeService:
    """
    Get or create the global mock service instance.

    Args:
        interval_seconds: Interval for generating new files

    Returns:
        MockRealtimeService instance
    """
    global _mock_service

    if _mock_service is None:
        _mock_service = MockRealtimeService(interval_seconds=interval_seconds)

    return _mock_service


def start_mock_service(interval_seconds: int = 60, generate_history: bool = True):
    """
    Start the mock realtime service.

    Args:
        interval_seconds: How often to check/generate new files
        generate_history: Whether to generate initial historical files (only if < 12 exist)
    """
    service = get_mock_service(interval_seconds)

    # Check if we need to generate history
    existing_files = service._count_existing_files()
    logger.info(f"Found {existing_files} existing mock files")

    if generate_history and existing_files < 12:
        logger.info(f"Generating {12 - existing_files} historical files to reach minimum of 12...")
        service.generate_initial_history(num_files=12)
    elif existing_files >= 12:
        logger.info("✅ Sufficient mock data already exists, skipping history generation")

    service.start()

    return service


def stop_mock_service():
    """Stop the mock realtime service."""
    global _mock_service

    if _mock_service:
        _mock_service.stop()


if __name__ == "__main__":
    # Test the service
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("Starting mock realtime service for testing...")
    print("Press Ctrl+C to stop\n")

    service = start_mock_service(interval_seconds=10, generate_history=True)

    try:
        # Keep running until interrupted
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nStopping service...")
        stop_mock_service()
        print("Service stopped")
        sys.exit(0)