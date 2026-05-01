"""Logging helpers for the Autonomous Finance Tutor application."""

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Create and return a configured logger instance.

    Args:
        name: Logger name, usually __name__ from the calling module.

    Returns:
        Configured logger that logs to stdout.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
