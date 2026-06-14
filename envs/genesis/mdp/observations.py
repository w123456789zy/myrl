"""Observation functions for Genesis Go2 environments.

Ported from ``unitree_rl_lab.tasks.locomotion.mdp.observations``.
Each function takes the env and returns a tensor of shape
``(num_envs, k)``.

The env is expected to expose:
    * ``base_lin_vel``, ``base_ang_vel``, ``projected_gravity`` (num_envs, 3)
    * ``dof_pos``, ``dof_vel`` (num_envs, num_dof)
    * ``actions``, ``last_actions`` (num_envs, num_actions)
    * ``commands`` (num_envs, 3) and ``commands_body`` (num_envs, 3)
    * ``default_dof_pos`` (num_dof,)
    * ``device``
    * ``terrain_heights`` (num_envs,) when terrain is enabled
    * ``height_field`` (Hx, Hy) when terrain is enabled
    * ``terrain_cfg`` dict with ``horizontal_scale`` key
"""

from __future__ import annotations

import math

import torch


# ---------------------------------------------------------------------------
# Single-term observations
# ---------------------------------------------------------------------------

def base_lin_vel(env) -> torch.Tensor:
    return env.base_lin_vel


def base_ang_vel(env) -> torch.Tensor:
    return env.base_ang_vel


def projected_gravity(env) -> torch.Tensor:
    return env.projected_gravity


def joint_pos_rel(env) -> torch.Tensor:
    return env.dof_pos - env.default_dof_pos


def joint_vel_rel(env) -> torch.Tensor:
    return env.dof_vel


def last_action(env) -> torch.Tensor:
    return env.last_actions


def generated_commands(env, command_name: str = "base_velocity") -> torch.Tensor:
    """Return the current velocity command (the first 3 components)."""
    return env.commands[:, :3]


# ---------------------------------------------------------------------------
# Height scan
# ---------------------------------------------------------------------------

def height_scan(
    env,
    resolution: float = 0.1,
    size_x: float = 1.6,
    size_y: float = 1.0,
    height_offset: float = 0.5,
) -> torch.Tensor:
    """Return a 2D height grid sampled around each env's base position.

    The scan is performed in the BASE frame: the X axis points forward, the
    Y axis points to the left. The grid has shape
    ``(num_envs, n_x, n_y)`` flattened to ``(num_envs, n_x * n_y)`` to match
    the convention used by unitree_rl_lab (1D concatenated obs).

    For envs without a height field (e.g. flat ground) this returns a
    zero tensor of the correct shape.
    """
    device = env.device
    n_x = int(round(size_x / resolution)) + 1
    n_y = int(round(size_y / resolution)) + 1

    # Skip the scan entirely when terrain is not enabled. ``height_field``
    # / ``terrain_cfg`` are *always* attributes on the env (set in
    # ``Go2BaseEnv._build_scene``), but they're ``None`` for flat-ground
    # envs. Use a single truthy check to cover both cases.
    height_field = getattr(env, "height_field", None)
    terrain_cfg = getattr(env, "terrain_cfg", None)
    if height_field is None or terrain_cfg is None:
        return torch.zeros(env.num_envs, n_x * n_y, device=device)

    # Build sample grid in base frame (forward, left)
    xs = torch.linspace(-size_x / 2.0, size_x / 2.0, n_x, device=device)
    ys = torch.linspace(-size_y / 2.0, size_y / 2.0, n_y, device=device)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")  # (n_x, n_y)
    grid_x = grid_x.reshape(-1)  # (n_x*n_y,)
    grid_y = grid_y.reshape(-1)
    n_pts = grid_x.shape[0]

    h_scale = terrain_cfg["horizontal_scale"]

    # Rotate the base-frame offsets by the base yaw into the world frame
    yaw = env.base_euler[:, 2]  # (num_envs,)
    cos_y = torch.cos(yaw)
    sin_y = torch.sin(yaw)
    dx_w = cos_y[:, None] * grid_x[None, :] - sin_y[:, None] * grid_y[None, :]
    dy_w = sin_y[:, None] * grid_x[None, :] + cos_y[:, None] * grid_y[None, :]

    # World-frame sample points
    px = env.base_pos[:, 0:1] + dx_w
    py = env.base_pos[:, 1:2] + dy_w
    px = torch.clamp(px, min=0.0, max=env.terrain_margin[0] - h_scale)
    py = torch.clamp(py, min=0.0, max=env.terrain_margin[1] - h_scale)

    # Convert to height-field indices
    h_ids = (px / h_scale - 0.5).floor().long().clamp(min=0, max=env.height_field.shape[0] - 1)
    w_ids = (py / h_scale - 0.5).floor().long().clamp(min=0, max=env.height_field.shape[1] - 1)
    heights = env.height_field[h_ids, w_ids]  # (num_envs, n_pts)

    # Subtract the base height to make the scan relative to the robot.
    # ``height_offset`` is the *optional* height (above the base) at which
    # a real ray-caster would emit the ray. We don't actually ray-cast
    # (we just sample the height-field at the requested x/y), but we
    # subtract it from the relative heights so callers that use a
    # non-zero offset (e.g. a head-mounted sensor) get a consistent
    # signed-height signal.
    rel = heights - env.base_pos[:, 2:3] - height_offset
    return rel  # (num_envs, n_pts)


# ---------------------------------------------------------------------------
# Gait phase
# ---------------------------------------------------------------------------

def gait_phase(env, period: float = 0.8) -> torch.Tensor:
    """Return ``(sin(2pi t / T), cos(2pi t / T))`` as a ``(num_envs, 2)`` tensor."""
    if not hasattr(env, "episode_length_buf"):
        return torch.zeros(env.num_envs, 2, device=env.device)
    global_phase = (env.episode_length_buf * env.dt) % period / period
    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * 2 * math.pi)
    phase[:, 1] = torch.cos(global_phase * 2 * math.pi)
    return phase
