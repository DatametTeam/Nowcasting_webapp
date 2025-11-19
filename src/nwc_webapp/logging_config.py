"""
Centralized logging configuration with colored output for the nowcasting webapp.
"""
import logging
import sys


# ANSI color codes
class Colors:
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


class ColoredFormatter(logging.Formatter):
    """
    Custom formatter that adds colors to log messages.
    - ERROR: entire message in RED
    - WARNING: entire message in YELLOW
    - DEBUG: only "DEBUG" prefix in BLUE
    - INFO: normal (no color)
    """

    def __init__(self):
        super().__init__(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    def format(self, record):
        # Store original values
        original_levelname = record.levelname
        original_msg = record.msg

        # Format the base message
        formatted = super().format(record)

        # Apply colors based on level
        if record.levelno == logging.ERROR:
            # Entire message in RED
            return f"{Colors.RED}{formatted}{Colors.RESET}"
        elif record.levelno == logging.WARNING:
            # Entire message in YELLOW
            return f"{Colors.YELLOW}{formatted}{Colors.RESET}"
        elif record.levelno == logging.DEBUG:
            # Only "DEBUG" prefix in BLUE
            return formatted.replace('DEBUG', f"{Colors.BLUE}DEBUG{Colors.RESET}")
        else:
            # INFO and others: no color
            return formatted


def setup_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """
    Create and configure a logger with colored output.

    Args:
        name: Logger name (usually __name__ from calling module)
        level: Logging level (default: DEBUG)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # Set colored formatter
    console_handler.setFormatter(ColoredFormatter())

    # Add handler to logger
    logger.addHandler(console_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


# Create a default logger for quick imports
default_logger = setup_logger('nwc_webapp')