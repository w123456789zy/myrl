"""Termination conditions for Genesis Go2 environments.

Ported from ``unitree_rl_lab.tasks.locomotion.mdp.terminations``. Each
function takes the *env* as its first argument and returns a
``torch.Tensor`` of shape ``(num_envs,)`` with a boolean done flag per env.

The env is expected to expose:
    * ``link_contact_forces`` (num_envs, n_links, 3)
    * ``base_euler`` (num_envs, 3) — roll/pitch/yaw
    * ``episode_length_buf`` (num_envs,)
    * ``max_episode_length`` int
    * ``base_pos`` (num_envs, 3)
    * ``dt`` float
    * ``device`` torch.device
"""

from __future__ import annotations

import math

import torch


def time_out(env) -> torch.Tensor:
    """Terminate on episode-length timeout."""
    return env.episode_length_buf > env.max_episode_length


def illegal_contact(env, sensor_cfg, threshold: float = 1.0) -> torch.Tensor:
    """Terminate when any monitored body has a contact force above threshold.

    ``sensor_cfg`` is a mapping with ``body_indices`` (or ``body_names``) — for
    the mylab envs we pass a simple dict like
    ``{"body_indices": env.base_link_index, "threshold": 1.0}``.
    """
    body_indices = sensor_cfg["body_indices"]
    if not isinstance(body_indices, (list, tuple)):
        body_indices = [body_indices]
    threshold = sensor_cfg.get("threshold", threshold)
    forces = env.link_contact_forces[:, body_indices, :]
    return torch.any(torch.norm(forces, dim=-1) > threshold, dim=1)


def bad_orientation(env, limit_angle: float = 0.8) -> torch.Tensor:
    """Terminate when the base roll or pitch exceeds ``limit_angle`` (radians)."""
    return torch.logical_or(
        torch.abs(env.base_euler[:, 1]) > limit_angle,
        torch.abs(env.base_euler[:, 0]) > limit_angle,
    )


def root_height_below_minimum(env, minimum_height: float = 0.2) -> torch.Tensor:
    """Terminate when the base z position falls below a minimum height."""
    return env.base_pos[:, 2] < minimum_height
