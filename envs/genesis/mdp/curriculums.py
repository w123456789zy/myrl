"""Curriculum learning utilities for Genesis Go2 environments.

Ported from ``unitree_rl_lab.tasks.locomotion.mdp.curriculums`` and
adapted to the mylab env API.

Each curriculum function takes the env and the env_ids being processed,
and returns a scalar tensor (for logging) reflecting the current curriculum
level.
"""

from __future__ import annotations

from typing import Sequence

import torch


def lin_vel_cmd_levels(
    env,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
    delta_range: tuple[float, float] = (-0.1, 0.1),
) -> torch.Tensor:
    """Linearly expand ``lin_vel_x``/``lin_vel_y`` ranges when the policy
    is performing above 80% of the configured weight for the given reward.

    The ``ranges`` here are pulled from ``env._command_cfg`` and clamped to
    ``env._command_cfg['lin_vel_x_limit']`` / ``lin_vel_y_limit`` if those
    keys are present.
    """
    if len(env_ids) == 0:
        return torch.tensor(0.0, device=env.device)

    cfg = env._command_cfg
    reward = torch.mean(env.episode_sums[reward_term_name][env_ids]) / max(env.max_episode_length_s, 1e-6)
    target_weight = env._reward_cfg["reward_scales"].get(reward_term_name, 1.0)

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > target_weight * 0.8:
            low, high = cfg.get("lin_vel_x_range", (-1.0, 1.0))
            limit_low, limit_high = cfg.get("lin_vel_x_limit", (low, high))
            new_low = max(limit_low, low + delta_range[0])
            new_high = min(limit_high, high + delta_range[1])
            cfg["lin_vel_x_range"] = (new_low, new_high)

            low, high = cfg.get("lin_vel_y_range", (-1.0, 1.0))
            limit_low, limit_high = cfg.get("lin_vel_y_limit", (low, high))
            new_low = max(limit_low, low + delta_range[0])
            new_high = min(limit_high, high + delta_range[1])
            cfg["lin_vel_y_range"] = (new_low, new_high)

    return torch.tensor(cfg["lin_vel_x_range"][1], device=env.device)


def ang_vel_cmd_levels(
    env,
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z",
    delta_range: tuple[float, float] = (-0.1, 0.1),
) -> torch.Tensor:
    if len(env_ids) == 0:
        return torch.tensor(0.0, device=env.device)

    cfg = env._command_cfg
    reward = torch.mean(env.episode_sums[reward_term_name][env_ids]) / max(env.max_episode_length_s, 1e-6)
    target_weight = env._reward_cfg["reward_scales"].get(reward_term_name, 1.0)

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > target_weight * 0.8:
            low, high = cfg.get("ang_vel_range", (-1.0, 1.0))
            limit_low, limit_high = cfg.get("ang_vel_limit", (low, high))
            new_low = max(limit_low, low + delta_range[0])
            new_high = min(limit_high, high + delta_range[1])
            cfg["ang_vel_range"] = (new_low, new_high)

    return torch.tensor(cfg["ang_vel_range"][1], device=env.device)


def terrain_levels_vel(
    env,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    """Track the current terrain difficulty level.

    The actual terrain-row assignment is done by the env (which moves
    robots between rows when they succeed). This function only returns
    a scalar for logging.
    """
    if not hasattr(env, "terrain_levels"):
        return torch.tensor(0.0, device=env.device)
    return torch.tensor(float(env.terrain_levels.mean().item()), device=env.device)
