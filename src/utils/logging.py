"""Observability helpers: structured logging + a latency timer context manager.

Satisfies the NFR "Basic logging for failed image uploads or high-latency queries".
"""
from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from typing import Iterator

from loguru import logger

from src import config

_CONFIGURED = False


def get_logger():
    """Return a process-wide configured loguru logger (idempotent)."""
    global _CONFIGURED
    if not _CONFIGURED:
        logger.remove()
        logger.add(sys.stderr, level="INFO",
                   format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}")
        logger.add(config.LOG_DIR / "app.log", level="DEBUG", rotation="5 MB",
                   retention="7 days", enqueue=True)
        _CONFIGURED = True
    return logger


@contextmanager
def timed(operation: str, slow_ms: float = config.SLOW_QUERY_MS) -> Iterator[dict]:
    """Time a block; warn if it exceeds ``slow_ms``. Yields a dict with 'ms' after exit.

    Usage:
        with timed("search") as t:
            ...
        print(t["ms"])
    """
    log = get_logger()
    holder: dict = {"ms": 0.0}
    start = time.perf_counter()
    try:
        yield holder
    finally:
        elapsed = (time.perf_counter() - start) * 1000.0
        holder["ms"] = round(elapsed, 1)
        if elapsed >= slow_ms:
            log.warning(f"[slow] {operation} took {holder['ms']} ms (>{slow_ms} ms)")
        else:
            log.debug(f"{operation} took {holder['ms']} ms")
