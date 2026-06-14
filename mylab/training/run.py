"""Run / checkpoint resolution helpers.

The training runner writes one timestamped sub-directory per run inside
``logs/<env_name>/``. These helpers make it easy to:

* find the most recent run directory for a given env,
* find the highest-numbered ``model_<N>.pt`` inside a run,
* resolve a ``--load-run`` argument (either a literal path or ``-1`` meaning
  "the latest run for this env").
"""

from __future__ import annotations

import os
import re
from glob import glob
from typing import Optional


_MODEL_PATTERN = re.compile(r"model_(\d+)\.pt$")


def get_latest_run(log_root: str, env_name: str) -> Optional[str]:
    """Return the most-recently-created run directory for ``env_name``.

    Args:
        log_root: Root logs directory (typically ``"logs"``).
        env_name: Environment name (sub-directory of ``log_root``).

    Returns:
        Absolute path to the latest run directory, or ``None`` if none exists.
    """
    env_dir = os.path.join(log_root, env_name)
    if not os.path.isdir(env_dir):
        return None
    candidates = sorted(
        (d for d in os.listdir(env_dir) if os.path.isdir(os.path.join(env_dir, d))),
        reverse=True,
    )
    if not candidates:
        return None
    return os.path.join(env_dir, candidates[0])


def get_latest_checkpoint(run_dir: str) -> Optional[str]:
    """Return the highest-numbered ``model_<N>.pt`` inside ``run_dir``."""
    if not os.path.isdir(run_dir):
        return None
    files = glob(os.path.join(run_dir, "model_*.pt"))
    if not files:
        return None

    def _index(path: str) -> int:
        m = _MODEL_PATTERN.search(path)
        return int(m.group(1)) if m else -1

    return max(files, key=_index)


def resolve_checkpoint_path(
    checkpoint_arg: Optional[str],
    load_run_arg: Optional[str],
    log_root: str,
    env_name: str,
) -> Optional[str]:
    """Resolve a ``--checkpoint`` / ``--load-run`` pair to a concrete file path.

    Precedence:
        1. ``--checkpoint`` (explicit file path) is always honoured.
        2. ``--load-run -1`` resolves to ``get_latest_run(log_root, env_name)``
           + ``get_latest_checkpoint``.
        3. ``--load-run <name>`` resolves to ``log_root/env_name/<name>``
           + ``get_latest_checkpoint``.
    """
    if checkpoint_arg is not None:
        return checkpoint_arg

    if load_run_arg is None:
        return None

    if str(load_run_arg) == "-1":
        run_dir = get_latest_run(log_root, env_name)
    else:
        run_dir = os.path.join(log_root, env_name, str(load_run_arg))

    if run_dir is None:
        return None
    return get_latest_checkpoint(run_dir)
