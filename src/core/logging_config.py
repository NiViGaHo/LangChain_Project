from __future__ import annotations

import logging
from typing import Optional


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Create and return a configured logger.

    Uses a simple structured format suitable for terminals and CI logs.
    """
    logger = logging.getLogger(name or "app")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


