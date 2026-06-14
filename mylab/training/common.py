"""Common training/playing helpers.

Lightweight utilities shared by ``train.py`` and ``play.py``: argument
parsing for the algorithm + checkpoint resolution, log-directory
construction, and so on.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

from mylab.training.run import resolve_checkpoint_path


def make_run_dir(log_root: str, env_name: str, suffix: Optional[str] = None) -> str:
    """Create a timestamped run directory and return its absolute path.

    Args:
        log_root: Root logs directory (typically ``"logs"``).
        env_name: Environment name (sub-directory of ``log_root``).
        suffix: Optional suffix appended to the timestamp (e.g. ``"ppo-seed42"``).
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = f"{timestamp}_{suffix}" if suffix else timestamp
    run_dir = os.path.join(log_root, env_name, name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def resolve_load_path(
    log_root: str,
    env_name: str,
    checkpoint_arg: Optional[str] = None,
    load_run_arg: Optional[str] = None,
) -> Optional[str]:
    """Thin wrapper around :func:`mylab.training.run.resolve_checkpoint_path`."""
    return resolve_checkpoint_path(checkpoint_arg, load_run_arg, log_root, env_name)
