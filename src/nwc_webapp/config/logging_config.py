"""
Centralized logging configuration for the nowcasting application.
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# Create logs directory
LOGS_DIR = Path(__file__).parent.parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'

    def format(self, record):
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{self.BOLD}{levelname}{self.RESET}"

        # Format the message
        result = super().format(record)

        # Reset levelname for other handlers
        record.levelname = levelname

        return result


def setup_logging(
    name: str = "nowcasting",
    level: str = "INFO",
    log_to_file: bool = True,
    log_to_console: bool = True
) -> logging.Logger:
    """
    Setup and configure logging for the application.

    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to log to file
        log_to_console: Whether to log to console

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Only configure if not already configured
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper()))
    logger.propagate = False

    # Console handler with colors
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = ColoredFormatter(
            fmt='%(levelname)s | %(name)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # File handler with detailed information
    if log_to_file:
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = LOGS_DIR / f"{name}_{timestamp}.log"

        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Logger name (uses calling module name if not provided)

    Returns:
        Logger instance
    """
    if name is None:
        # Try to get the calling module name
        import inspect
        frame = inspect.currentframe()
        if frame and frame.f_back:
            caller_module = inspect.getmodule(frame.f_back)
            if caller_module:
                name = caller_module.__name__
            else:
                name = "nowcasting"
        else:
            name = "nowcasting"

    # If logger doesn't exist or has no handlers, set it up
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger = setup_logging(name)

    return logger


# Create default logger
logger = setup_logging()


# Convenience functions for quick logging
def debug(msg: str, **kwargs):
    """Log debug message."""
    logger.debug(msg, **kwargs)


def info(msg: str, **kwargs):
    """Log info message."""
    logger.info(msg, **kwargs)


def warning(msg: str, **kwargs):
    """Log warning message."""
    logger.warning(msg, **kwargs)


def error(msg: str, **kwargs):
    """Log error message."""
    logger.error(msg, **kwargs)


def critical(msg: str, **kwargs):
    """Log critical message."""
    logger.critical(msg, **kwargs)


if __name__ == "__main__":
    # Test the logging system
    test_logger = setup_logging("test", level="DEBUG")

    test_logger.debug("This is a debug message")
    test_logger.info("This is an info message")
    test_logger.warning("This is a warning message")
    test_logger.error("This is an error message")
    test_logger.critical("This is a critical message")

    print(f"\nLog file created at: {LOGS_DIR}")