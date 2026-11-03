"""Logging setup shared by the CLI, notebooks and the app."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging(level: str | int = "INFO", log_file: str | Path | None = None) -> None:
    """Configure the root `sopverify` logger once. Safe to call repeatedly."""
    logger = logging.getLogger("sopverify")
    logger.setLevel(level)
    if not any(isinstance(h, logging.StreamHandler) and getattr(h, "_sopverify", False) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_FORMAT, "%H:%M:%S"))
        handler._sopverify = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(path)
        fh.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(fh)
    logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name if name.startswith("sopverify") else f"sopverify.{name}")
