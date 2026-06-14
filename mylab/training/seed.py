"""Seed helpers for deterministic training and inference.

Provides a single entry point that sets the seed for Python's ``random``,
``numpy`` and ``torch`` (including CUDA) so that a training run is fully
reproducible.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def apply_training_seed(
    seed: int,
    torch_deterministic: bool = False,
    cuda: bool = True,
) -> None:
    """Apply the given ``seed`` to Python, NumPy and Torch.

    Args:
        seed: Integer seed value.
        torch_deterministic: When ``True``, enables the slower-but-deterministic
            algorithms on CUDA via :func:`torch.use_deterministic_algorithms`.
        cuda: When ``True`` and CUDA is available, also seed all CUDA devices
            and (optionally) the cuDNN benchmark flag.
    """
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Avoid cuDNN picking a different algorithm each iteration.
        torch.backends.cudnn.deterministic = torch_deterministic
        torch.backends.cudnn.benchmark = not torch_deterministic
    if torch_deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
