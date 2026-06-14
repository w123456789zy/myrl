"""Reward functions for Genesis Go2 environments.

Ported from ``unitree_rl_lab.tasks.locomotion.mdp.rewards`` and
adapted to the mylab env API. Each function takes the env (and any
relevant parameters) and returns a ``torch.Tensor`` of shape
``(num_envs,)``.

The env is expected to expose:
    * ``base_lin_vel``, ``base_ang_vel``, ``base_euler`` (num_envs, 3)
    * ``projected_gravity`` (num_envs, 3)
    * ``dof_pos``, ``dof_vel``, ``default_dof_pos`` (num_envs, num_dof)
    * ``last_dof_vel`` (num_envs, num_dof)
    * ``dof_pos_limits`` (num_dof, 2) lower/upper
    * ``torques`` (num_envs, num_dof) — the current applied torques
    * ``last_actions``, ``actions`` (num_envs, num_actions)
    * ``commands_body`` (num_envs, 3)
    * ``commands`` (num_envs, 3)
    * ``link_contact_forces`` (num_envs, n_links, 3)
    * ``feet_link_indices`` (num_envs, n_feet)
    * ``feet_air_time`` (num_envs, n_feet)
    * ``last_contacts`` (num_envs, n_feet)
    * ``foot_positions`` (num_envs, n_feet, 3)
    * ``foot_velocities`` (num_envs, n_feet, 3)
    * ``device``, ``dt``
"""

from __future__ import annotations

from typing import Sequence

import torch


# ---------------------------------------------------------------------------
# Task rewards
# ---------------------------------------------------------------------------

def track_lin_vel_xy_exp(env, command_name: str = "base_velocity", std: float = 0.25) -> torch.Tensor:
    err = torch.sum(torch.square(env.commands_body[:, :2] - env.base_lin_vel[:, :2]), dim=1)
    return torch.exp(-err / std)


def track_ang_vel_z_exp(env, command_name: str = "base_velocity", std: float = 0.25) -> torch.Tensor:
    err = torch.square(env.commands_body[:, 2] - env.base_ang_vel[:, 2])
    return torch.exp(-err / std)


# ---------------------------------------------------------------------------
# Base penalties
# ---------------------------------------------------------------------------

def lin_vel_z_l2(env) -> torch.Tensor:
    return torch.square(env.base_lin_vel[:, 2])


def ang_vel_xy_l2(env) -> torch.Tensor:
    return torch.sum(torch.square(env.base_ang_vel[:, :2]), dim=1)


def joint_vel_l2(env) -> torch.Tensor:
    return torch.sum(torch.square(env.dof_vel), dim=1)


def joint_acc_l2(env) -> torch.Tensor:
    """Penalize joint accelerations (numerical). Requires ``last_dof_vel``."""
    if not hasattr(env, "last_dof_vel"):
        return torch.zeros(env.num_envs, device=env.device)
    acc = (env.dof_vel - env.last_dof_vel) / max(env.dt, 1e-6)
    return torch.sum(torch.square(acc), dim=1)


def joint_torques_l2(env) -> torch.Tensor:
    if not hasattr(env, "torques"):
        return torch.zeros(env.num_envs, device=env.device)
    return torch.sum(torch.square(env.torques), dim=1)


def action_rate_l2(env) -> torch.Tensor:
    return torch.sum(torch.square(env.last_actions - env.actions), dim=1)


def joint_pos_limits(env) -> torch.Tensor:
    """Penalize joint positions outside their (soft) limits."""
    out_of_limits = -torch.min(env.dof_pos - env.dof_pos_limits[:, 0], torch.zeros_like(env.dof_pos))
    out_of_limits += torch.max(env.dof_pos - env.dof_pos_limits[:, 1], torch.zeros_like(env.dof_pos))
    return torch.sum(out_of_limits, dim=1)


def energy(env) -> torch.Tensor:
    """Penalize ``|qvel| * |torque|``."""
    if not hasattr(env, "torques"):
        return torch.zeros(env.num_envs, device=env.device)
    return torch.sum(torch.abs(env.dof_vel) * torch.abs(env.torques), dim=1)


# ---------------------------------------------------------------------------
# Posture / orientation
# ---------------------------------------------------------------------------

def flat_orientation_l2(env) -> torch.Tensor:
    """Squared L2 of the xy components of projected gravity."""
    return torch.sum(torch.square(env.projected_gravity[:, :2]), dim=1)


def base_height(env, target_height: float = 0.30) -> torch.Tensor:
    """Penalize base height deviation from ``target_height``."""
    return torch.square(env.base_pos[:, 2] - target_height)


def smoothness(env) -> torch.Tensor:
    """Penalize action jerk (second-order difference).  Requires ``last_actions`` and ``last_last_actions``."""
    if not hasattr(env, "last_last_actions"):
        return torch.zeros(env.num_envs, device=env.device)
    return torch.sum(torch.square(env.actions - 2.0 * env.last_actions + env.last_last_actions), dim=1)


def upward(env) -> torch.Tensor:
    """Reward ``projected_gravity[:, 2]`` ≈ 1 (i.e. gravity along body -z)."""
    return torch.square(1.0 - env.projected_gravity[:, 2])


def joint_position_penalty(
    env,
    stand_still_scale: float = 5.0,
    velocity_threshold: float = 0.3,
) -> torch.Tensor:
    """Penalize joint deviation from default; extra penalty when not moving."""
    cmd = torch.linalg.norm(env.commands, dim=1)
    body_vel = torch.linalg.norm(env.base_lin_vel[:, :2], dim=1)
    err = torch.linalg.norm(env.dof_pos - env.default_dof_pos, dim=1)
    mask = torch.logical_or(cmd > 0.0, body_vel > velocity_threshold)
    return torch.where(mask, err, stand_still_scale * err)


# ---------------------------------------------------------------------------
# Foot / contact rewards
# ---------------------------------------------------------------------------

def feet_air_time(
    env,
    threshold: float = 0.5,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Reward feet being in the air for at least ``threshold`` seconds."""
    contact = env.link_contact_forces[:, env.feet_link_indices, 2] > 1.0
    contact_filt = torch.logical_or(contact, env.last_contacts)
    first_contact = (env.feet_air_time > 0.0) * contact_filt
    env.last_contacts = contact
    env.feet_air_time += env.dt
    rew = torch.sum((env.feet_air_time - threshold) * first_contact, dim=1)
    rew *= torch.norm(env.commands[:, :2], dim=1) > 0.1
    env.feet_air_time *= ~contact_filt
    return rew


def air_time_variance_penalty(env) -> torch.Tensor:
    """Penalize variance in air/contact time across feet (encourages sync)."""
    if not hasattr(env, "feet_air_time"):
        return torch.zeros(env.num_envs, device=env.device)
    if not hasattr(env, "feet_contact_time"):
        env.feet_contact_time = torch.zeros_like(env.feet_air_time)
    # We approximate ``last_contact_time`` using a per-step counter.
    contact = env.link_contact_forces[:, env.feet_link_indices, 2] > 1.0
    env.feet_contact_time = env.feet_contact_time * contact.float() + env.dt * contact.float()
    return (
        torch.var(torch.clip(env.feet_air_time, max=0.5), dim=1)
        + torch.var(torch.clip(env.feet_contact_time, max=0.5), dim=1)
    )


def feet_slide(env) -> torch.Tensor:
    """Penalize feet sliding (xy velocity while in contact)."""
    contact = env.link_contact_forces[:, env.feet_link_indices, 2] > 1.0
    foot_vel_xy = env.foot_velocities[:, :, :2]
    slip = torch.sum(torch.square(torch.norm(foot_vel_xy, dim=-1)) * contact.float(), dim=1)
    return slip


def undesired_contacts(env, threshold: float = 1.0, body_indices: Sequence[int] = ()) -> torch.Tensor:
    """Penalize contacts on non-foot body parts (thigh/calf/base)."""
    indices = body_indices if body_indices else getattr(env, "penalized_contact_link_indices", [])
    if not indices:
        return torch.zeros(env.num_envs, device=env.device)
    forces = env.link_contact_forces[:, indices, :]
    return torch.sum((torch.norm(forces, dim=-1) > threshold).float(), dim=1)


def feet_stumble(env, sensor_cfg=None) -> torch.Tensor:
    """Penalize feet hitting vertical surfaces (large xy force relative to z)."""
    contact = env.link_contact_forces[:, env.feet_link_indices, :]
    fz = torch.abs(contact[:, :, 2])
    fxy = torch.linalg.norm(contact[:, :, :2], dim=2)
    return torch.any(fxy > 4 * fz, dim=1).float()


def feet_too_near(env, threshold: float = 0.2) -> torch.Tensor:
    """Penalize foot–foot distance below ``threshold``."""
    feet_pos = env.foot_positions[:, :, :]
    # Distance between front feet (0, 1) and between rear feet (2, 3) — keep it simple
    d_front = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    d_rear = torch.norm(feet_pos[:, 2] - feet_pos[:, 3], dim=-1)
    return (threshold - d_front).clamp(min=0) + (threshold - d_rear).clamp(min=0)


def feet_contact_without_cmd(env, command_name: str = "base_velocity") -> torch.Tensor:
    """Reward feet contact when the command is zero."""
    contact = env.link_contact_forces[:, env.feet_link_indices, 2] > 1.0
    rew = torch.sum(contact.float(), dim=-1)
    cmd_norm = torch.norm(env.commands, dim=1)
    return rew * (cmd_norm < 0.1)


def feet_gait(
    env,
    period: float = 0.8,
    offset: Sequence[float] = (0.0, 0.5, 0.5, 0.0),
    threshold: float = 0.5,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Reward a target contact pattern (e.g. trot: diagonal pairs)."""
    contact = env.link_contact_forces[:, env.feet_link_indices, 2] > 1.0
    global_phase = ((env.episode_length_buf * env.dt) % period / period).unsqueeze(1)
    leg_phase = (global_phase + torch.tensor(offset, device=env.device).unsqueeze(0)) % 1.0
    is_stance = leg_phase < threshold
    reward = torch.zeros(env.num_envs, device=env.device)
    for i in range(leg_phase.shape[1]):
        reward += (~(is_stance[:, i] ^ contact[:, i])).float()
    if command_name is not None:
        cmd_norm = torch.norm(env.commands, dim=1)
        reward *= (cmd_norm > 0.1).float()
    return reward


def feet_height_body(
    env,
    target_height: float = 0.08,
    tanh_mult: float = 2.0,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Penalize foot z-position error relative to a target clearance height.

    Re-implemented without the for-loop in unitree_rl_lab's reference to
    be vectorized across feet.
    """
    foot_z = env.foot_positions[:, :, 2] - env.base_pos[:, 2:3]  # (num_envs, n_feet)
    err = torch.square(foot_z - target_height)  # (num_envs, n_feet)
    foot_vel_xy = env.foot_velocities[:, :, :2]
    vel = torch.tanh(tanh_mult * torch.norm(foot_vel_xy, dim=2))  # (num_envs, n_feet)
    rew = torch.sum(err * vel, dim=1)
    rew *= torch.norm(env.commands[:, :2], dim=1) > 0.1
    return rew


def foot_clearance_reward(
    env,
    target_height: float = 0.08,
    std: float = 0.05,
    tanh_mult: float = 2.0,
) -> torch.Tensor:
    foot_z_target_error = torch.square(env.foot_positions[:, :, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(env.foot_velocities[:, :, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


def feet_impact_vel(env, threshold: float = 1.0) -> torch.Tensor:
    """Penalize feet hitting ground with high downward velocity (encourages cushioning).

    Requires ``env.prev_foot_velocities`` to be populated in the base env.
    """
    if not hasattr(env, "prev_foot_velocities"):
        return torch.zeros(env.num_envs, device=env.device)
    prev_foot_vel_z = env.prev_foot_velocities[:, :, 2]
    contact = env.link_contact_forces[:, env.feet_link_indices, 2] > threshold
    return torch.sum(contact * torch.square(torch.clip(prev_foot_vel_z, -100.0, 0.0)), dim=1)


def feet_clearance(env, target_height: float = 0.08) -> torch.Tensor:
    """Penalize low foot clearance during swing phase (encourages lifting legs).

    Returns a *penalty* (positive value = bad).  Use a negative scale in
    ``reward_scales`` to turn it into a reward.
    """
    foot_z = env.foot_positions[:, :, 2] - env.base_pos[:, 2:3]
    err = torch.square(foot_z - target_height)
    contact = env.link_contact_forces[:, env.feet_link_indices, 2] > 1.0
    return torch.sum(err * (~contact).float(), dim=1)


def stand_still(env, command_name: str = "base_velocity") -> torch.Tensor:
    err = torch.sum(torch.abs(env.dof_pos - env.default_dof_pos), dim=1)
    cmd_norm = torch.norm(env.commands, dim=1)
    return err * (cmd_norm < 0.1)
