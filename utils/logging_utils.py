"""
JR MineralForge – Logging Utilities
=====================================
Centralised structured logging for all agents and modules.
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

from config.settings import LOG_LEVEL, LOG_FILE, BRAND_NAME


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given module name."""
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # already configured

    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Rotating file handler (10 MB × 5 backups)
    try:
        fh = RotatingFileHandler(LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=5)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except (OSError, PermissionError):
        pass  # gracefully skip file logging if directory not writable

    logger.propagate = False
    return logger


# Module-level logger for utilities themselves
log = get_logger(BRAND_NAME)
