"""
src/utils/logger.py

WHY?
    Print statements scattered across a project are hard to filter, silence, or redirect.
    A single, shared logger configuration means every module in this framework reports
    progress and warnings in the same format, and you can turn verbosity up or down in
    one place instead of hunting down `print()` calls.

WHAT?
    `get_logger(name)` returns a standard Python `logging.Logger`, pre-configured with a
    simple, readable console format. Every module in this framework calls
    `get_logger(__name__)` at import time.

HOW?
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Loaded dataset with shape (100, 5)")
"""
import logging
import sys

_CONFIGURED = False


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger. Safe to call repeatedly across modules --
    the console handler is only attached once per process."""
    global _CONFIGURED

    logger = logging.getLogger(name)
    logger.setLevel(level)

    root = logging.getLogger()
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        root.addHandler(handler)
        root.setLevel(level)
        _CONFIGURED = True

    return logger
