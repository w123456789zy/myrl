# Copyright (c) 2025, Your Name
# All rights reserved.

"""Backwards-compatible alias for the (now vision-enabled) :class:`Go2RoughEnv`.

The rough-terrain env is configured with ``use_vision=True`` by default
(see :file:`go2_rough.py`), so this class is just a thin alias. The env
name ``"go2-rough-vision"`` is preserved for backward compatibility —
the env factory in :file:`__init__.py` will return a :class:`Go2RoughEnv`
when asked for ``go2-rough-vision``.

Trainable with :py:class:`mylab.rl.alg.ppo.VisionPPO` (CNN encoder on
the image, MLP actor/critic on the state+history).
"""

from __future__ import annotations

from envs.genesis.go2_rough import Go2RoughEnv


class Go2RoughVisionEnv(Go2RoughEnv):
    """Alias of :class:`Go2RoughEnv` (which has vision enabled by default)."""

    def __init__(self, num_envs: int, show_viewer: bool = False, eval_mode: bool = False) -> None:
        super().__init__(num_envs=num_envs, show_viewer=show_viewer, eval_mode=eval_mode)
        self.name = "go2_rough_vision"


def get_env(num_envs: int, eval_mode: bool = False) -> Go2RoughVisionEnv:
    try:
        import genesis as gs
        gs.init(logging_level="warning")
    except Exception:
        pass
    return Go2RoughVisionEnv(num_envs=num_envs, eval_mode=eval_mode)
