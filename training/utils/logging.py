"""Rank-aware logger.

By default only rank-0 emits to stdout; other ranks log to rank-suffixed files if
`MERIT_LOG_DIR` is set. Works with torchrun / deepspeed / accelerate launches.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path


def _detect_rank() -> int:
    # Works with torchrun / deepspeed / accelerate
    for env in ("RANK", "LOCAL_RANK", "SLURM_PROCID"):
        v = os.environ.get(env)
        if v is not None and v.isdigit():
            return int(v)
    return 0


_FORMAT = "%(asctime)s [%(levelname)s] %(name)s :: %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str = "merit", level: int | str = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if getattr(logger, "_merit_configured", False):
        return logger

    logger.setLevel(level)
    logger.propagate = False

    rank = _detect_rank()
    fmt = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    if rank == 0:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    log_dir = os.environ.get("MERIT_LOG_DIR")
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(Path(log_dir) / f"rank{rank}.log", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger._merit_configured = True  # type: ignore[attr-defined]
    return logger
