"""Velocity command sampler with curriculum support.

This is a thin re-implementation of
``isaaclab.envs.mdp.UniformVelocityCommandCfg`` adapted to the mylab
env API. It supports:

* ``rel_standing_envs`` — fraction of envs forced to stand still
* ``resampling_time`` — seconds between command resamples
* ``rel_heading_envs`` — fraction of envs tracking a heading (not used)
* curriculum via ``limit_ranges`` — used by ``lin_vel_cmd_levels``

The command buffer is stored on the env as ``env.commands`` with shape
``(num_envs, num_commands)``. ``num_commands`` defaults to 3
(lin_vel_x, lin_vel_y, ang_vel_z).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch

from envs.genesis.mdp.math_utils import rand_float


@dataclass
class _Range:
    lin_vel_x: tuple[float, float] = (-1.0, 1.0)
    lin_vel_y: tuple[float, float] = (-1.0, 1.0)
    ang_vel_z: tuple[float, float] = (-1.0, 1.0)

    def to_dict(self):
        return {
            "lin_vel_x": self.lin_vel_x,
            "lin_vel_y": self.lin_vel_y,
            "ang_vel_z": self.ang_vel_z,
        }


@dataclass
class UniformLevelVelocityCommand:
    """Velocity command with curriculum-friendly ``limit_ranges``."""

    num_commands: int = 3
    resampling_time: float = 10.0
    rel_standing_envs: float = 0.0
    debug_vis: bool = False

    ranges: _Range = field(default_factory=_Range)
    limit_ranges: _Range = field(default_factory=_Range)

    @classmethod
    def Ranges(cls, lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0)) -> _Range:
        return _Range(lin_vel_x=lin_vel_x, lin_vel_y=lin_vel_y, ang_vel_z=ang_vel_z)


def sample_commands(
    env,
    env_ids: Sequence[int] | None = None,
    ranges: _Range | None = None,
    rel_standing_envs: float = 0.0,
) -> None:
    """Re-sample velocity commands for the selected envs.

    The env must already have:
        * ``env.commands`` of shape ``(num_envs, 3)`` filled with the current
          commands (we only overwrite ``env_ids``).
        * ``env.num_envs`` int
        * ``env.device`` torch.device
    """
    if ranges is None:
        # Default to the env_cfg-defined ranges if available.
        cfg = env._command_cfg
        ranges = _Range(
            lin_vel_x=tuple(cfg.get("lin_vel_x_range", (-1.0, 1.0))),
            lin_vel_y=tuple(cfg.get("lin_vel_y_range", (-1.0, 1.0))),
            ang_vel_z=tuple(cfg.get("ang_vel_range", (-1.0, 1.0))),
        )

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    n = len(env_ids)
    if n == 0:
        return

    # Standing mask
    n_standing = int(round(rel_standing_envs * n))
    if n_standing > 0:
        standing_idx = env_ids[:n_standing]
        env.commands[standing_idx, 0] = 0.0
        env.commands[standing_idx, 1] = 0.0
        env.commands[standing_idx, 2] = 0.0
        moving_idx = env_ids[n_standing:]
    else:
        moving_idx = env_ids

    n_moving = len(moving_idx)
    if n_moving > 0:
        env.commands[moving_idx, 0] = rand_float(*ranges.lin_vel_x, (n_moving,), env.device)
        env.commands[moving_idx, 1] = rand_float(*ranges.lin_vel_y, (n_moving,), env.device)
        env.commands[moving_idx, 2] = rand_float(*ranges.ang_vel_z, (n_moving,), env.device)

        # Force a non-zero linear velocity if both axes are too small
        norms = torch.norm(env.commands[moving_idx, :2], dim=1)
        env.commands[moving_idx, :2] *= (norms > 0.2).unsqueeze(1)


def step_commands(env, dt: float) -> torch.Tensor:
    """Return the env_ids whose commands should be re-sampled at this step.

    Uses ``env._command_cfg['resampling_time_s']`` (or a sensible default)
    to decide when commands expire.
    """
    resampling_time_s = env._command_cfg.get("resampling_time_s", 10.0)
    period = int(round(resampling_time_s / dt))
    if period <= 0:
        period = 1
    env_ids = (env.episode_length_buf % period == 0).nonzero(as_tuple=False).flatten()
    return env_ids
