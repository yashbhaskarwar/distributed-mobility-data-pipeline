from __future__ import annotations

import logging
import os
from datetime import datetime


def setup_logger(name: str, log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicates
    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler 
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d")
    log_path = os.path.join(log_dir, f"{ts}.log")

    fh = logging.FileHandler(log_path)
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.propagate = False
    return logger
