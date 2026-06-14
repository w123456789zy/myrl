"""Domain randomization events for Genesis Go2 environments.

Ported from ``unitree_rl_lab.tasks.locomotion.mdp.events`` and adapted
to the mylab env API. Each function takes the env, a list of env_ids to
operate on, and a parameter dict.

The env is expected to expose:
    * ``num_envs`` int
    * ``device`` torch.device
    * ``n_links`` int
    * ``link_start`` int
    * ``rigid_solver`` with the right genesis rigid API
    * ``robot`` (the genesis articulated entity)
    * ``base_init_pos`` (3,) tensor
    * ``base_init_quat`` (4,) tensor
    * ``motor_dofs`` list[int]
    * ``num_dof`` int
    * ``default_dof_pos`` (num_dof,) tensor
    * ``motor_strengths``, ``motor_offsets``, ``batched_p_gains``, ``batched_d_gains``
    * ``link_contact_forces`` (num_envs, n_links, 3)
"""

from __future__ import annotations

from typing import Sequence

import torch

from envs.genesis.mdp.math_utils import rand_float


def reset_root_state_uniform(
    env,
    env_ids: Sequence[int],
    pose_range: dict,
    velocity_range: dict,
) -> None:
    """Reset the base pose + velocity uniformly in a small box.

    ``pose_range`` / ``velocity_range`` follow the isaac-lab convention:
    each value is a ``(low, high)`` tuple applied independently per axis.
    """
    if len(env_ids) == 0:
        return
    device = env.device
    n = len(env_ids)

    def _sample(rng, default=(0.0, 0.0)):
        low, high = rng if rng is not None else default
        return rand_float(float(low), float(high), (n,), device)

    def _sample2(rng, default=(0.0, 0.0)):
        low, high = rng if rng is not None else default
        return rand_float(float(low), float(high), (n, 2), device)

    # position
    px = _sample(pose_range.get("x"))
    py = _sample(pose_range.get("y"))
    pz = _sample(pose_range.get("z", (0.0, 0.0)))
    # rotation
    roll = _sample(pose_range.get("roll", (0.0, 0.0)))
    pitch = _sample(pose_range.get("pitch", (0.0, 0.0)))
    yaw = _sample(pose_range.get("yaw", (0.0, 0.0)))

    base_pos = env.base_init_pos.unsqueeze(0).repeat(n, 1).clone()
    base_pos[:, 0] += px
    base_pos[:, 1] += py
    base_pos[:, 2] += pz

    from envs.genesis.mdp.math_utils import euler_to_quat, quat_mul
    base_quat = euler_to_quat(torch.stack([roll, pitch, yaw], dim=-1))
    base_quat = quat_mul(base_quat, env.base_init_quat.unsqueeze(0).repeat(n, 1))

    env.base_pos[env_ids] = base_pos
    env.base_quat[env_ids] = base_quat
    env.robot.set_pos(base_pos, zero_velocity=False, envs_idx=env_ids)
    env.robot.set_quat(base_quat, zero_velocity=False, envs_idx=env_ids)

    # velocity (linear xyz + angular xyz)
    vx = _sample(velocity_range.get("x", (0.0, 0.0)))
    vy = _sample(velocity_range.get("y", (0.0, 0.0)))
    vz = _sample(velocity_range.get("z", (0.0, 0.0)))
    wx = _sample(velocity_range.get("roll", (0.0, 0.0)))
    wy = _sample(velocity_range.get("pitch", (0.0, 0.0)))
    wz = _sample(velocity_range.get("yaw", (0.0, 0.0)))

    env.base_lin_vel[env_ids, 0] = vx
    env.base_lin_vel[env_ids, 1] = vy
    env.base_lin_vel[env_ids, 2] = vz
    env.base_ang_vel[env_ids, 0] = wx
    env.base_ang_vel[env_ids, 1] = wy
    env.base_ang_vel[env_ids, 2] = wz


def reset_joints_by_scale(
    env,
    env_ids: Sequence[int],
    position_range: tuple[float, float] = (0.5, 1.5),
    velocity_range: tuple[float, float] = (-1.0, 1.0),
) -> None:
    """Reset joint positions to ``default * Uniform(position_range)``.

    Resets the joint velocities to ``Uniform(velocity_range)``.
    """
    if len(env_ids) == 0:
        return
    device = env.device
    n = len(env_ids)
    scale = rand_float(*position_range, (n, env.num_dof), device)
    vel = rand_float(*velocity_range, (n, env.num_dof), device)
    env.dof_pos[env_ids] = env.default_dof_pos.unsqueeze(0) * scale
    env.dof_vel[env_ids] = vel
    env.robot.set_dofs_position(
        position=env.dof_pos[env_ids],
        dofs_idx_local=env.motor_dofs,
        zero_velocity=True,
        envs_idx=env_ids,
    )
    env.robot.zero_all_dofs_velocity(env_ids)


def randomize_rigid_body_material(
    env,
    env_ids: Sequence[int],
    static_friction_range: tuple[float, float] = (0.3, 1.2),
    dynamic_friction_range: tuple[float, float] = (0.3, 1.2),
    restitution_range: tuple[float, float] = (0.0, 0.15),
    num_buckets: int = 64,
) -> None:
    """Randomize the friction / restitution coefficients of robot geoms.

    Genesis does not expose a per-env friction override in the public API
    today, so we fall back to a global randomization applied to all geoms
    in proportion to the per-env bucket. This matches the spirit of
    isaac-lab's bucket-based event.
    """
    if len(env_ids) == 0:
        return
    solver = env.rigid_solver
    n_geoms = solver.n_geoms
    sf = rand_float(*static_friction_range, (len(env_ids), 1), env.device)
    df = rand_float(*dynamic_friction_range, (len(env_ids), 1), env.device)
    # Average scale used per env (1 friction ratio applied to all geoms per env)
    ratios = sf.repeat(1, n_geoms) * df.repeat(1, n_geoms)
    solver.set_geoms_friction_ratio(ratios, torch.arange(0, n_geoms), env_ids)


def randomize_rigid_body_mass(
    env,
    env_ids: Sequence[int],
    body_names: Sequence[str] | None = None,
    mass_distribution_params: tuple[float, float] = (-1.0, 3.0),
    operation: str = "add",
) -> None:
    """Randomize the mass of specific bodies. Defaults to the base link.

    ``operation`` can be ``"add"`` (additive) or ``"scale"`` (multiplicative).
    """
    if len(env_ids) == 0:
        return
    if body_names is None:
        # Default: the base link (link index 1 in the Go2 URDF).
        link_idx = [1]
    else:
        link_idx = []
        for name in body_names:
            for link in env.robot.links:
                if name in link.name:
                    link_idx.append(link.idx - env.robot.link_start)
        if not link_idx:
            link_idx = [1]
    if operation == "add":
        delta = rand_float(*mass_distribution_params, (len(env_ids), 1), env.device)
        env.rigid_solver.set_links_mass_shift(delta, link_idx, env_ids)
    else:  # scale
        scale = rand_float(*mass_distribution_params, (len(env_ids), 1), env.device)
        env.rigid_solver.set_links_mass_shift(scale, link_idx, env_ids)


def push_by_setting_velocity(
    env,
    env_ids: Sequence[int],
    velocity_range: dict,
) -> None:
    """Apply an instantaneous base-velocity push to the selected envs."""
    if len(env_ids) == 0:
        return
    vx = rand_float(*velocity_range.get("x", (0.0, 0.0)), (len(env_ids),), env.device)
    vy = rand_float(*velocity_range.get("y", (0.0, 0.0)), (len(env_ids),), env.device)
    # Apply to the first 2 base dofs (lin vel x, y) at the rigid-solver level.
    dofs_vel = env.robot.get_dofs_velocity()
    dofs_vel[env_ids, 0] += vx
    dofs_vel[env_ids, 1] += vy
    env.robot.set_dofs_velocity(dofs_vel)


def apply_external_force_torque(
    env,
    env_ids: Sequence[int],
    force_range: tuple[float, float] = (0.0, 0.0),
    torque_range: tuple[float, float] = (0.0, 0.0),
) -> None:
    """Stub: external force/torque application is a no-op in the
    current genesis build. Implementations can be added when the rigid
    API exposes a per-step force-injection hook.
    """
    return
