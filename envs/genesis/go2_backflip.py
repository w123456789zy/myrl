# Copyright (c) 2025, Your Name
# All rights reserved.

"""Go2 backflip.

The robot must complete a full back-flip rotation around its lateral
(y) axis and land upright. Originally adapted from the
ziyanx02/Genesis-backflip reference.

The env is a subclass of :class:`Go2WalkEnv` (which provides the
state + height scan + MDP plumbing). It overrides ``_default_configs``
to use a short episode (2s), a custom reward formulation for the flip
and a custom ``compute_observations`` that adds a phase signal
(sin/cos of episode time) — the same trick used in periodic locomotion
paper references.
"""

from __future__ import annotations

import math

import torch

from envs.genesis.go2_base import (
    gs_inv_quat,
    gs_quat_from_angle_axis,
    gs_quat_mul,
    gs_transform_by_quat,
    gs_quat_conjugate,
    gs_quat_apply,
)
from envs.genesis.go2_walk import Go2WalkEnv


class Go2BackflipEnv(Go2WalkEnv):
    """Go2 backflip: 360° pitch rotation in 2 seconds and land upright."""

    def __init__(self, num_envs: int, show_viewer: bool = False, eval_mode: bool = False) -> None:
        super().__init__(num_envs=num_envs, show_viewer=show_viewer, eval_mode=eval_mode)
        self.name = "go2_backflip"

    @classmethod
    def _default_configs(cls):
        env_cfg = {
            "urdf_path": "urdf/go2/urdf/go2.urdf",
            "links_to_keep": ["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
            "num_actions": 12,
            "num_dofs": 12,
            "default_joint_angles": {
                "FL_hip_joint": 0.0, "FR_hip_joint": 0.0, "RL_hip_joint": 0.0, "RR_hip_joint": 0.0,
                "FL_thigh_joint": 0.8, "FR_thigh_joint": 0.8, "RL_thigh_joint": 1.0, "RR_thigh_joint": 1.0,
                "FL_calf_joint": -1.5, "FR_calf_joint": -1.5, "RL_calf_joint": -1.5, "RR_calf_joint": -1.5,
            },
            "dof_names": [
                "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
                "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
                "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
                "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            ],
            "termination_contact_link_names": ["base"],
            "penalized_contact_link_names": ["base", "thigh", "calf"],
            "feet_link_names": ["foot"],
            "base_link_name": ["base"],
            "PD_stiffness": {"joint": 70.0},
            "PD_damping": {"joint": 3.0},
            "use_implicit_controller": False,
            "termination_if_roll_greater_than": 0.4,
            "termination_if_pitch_greater_than": 0.4,
            "termination_if_height_lower_than": 0.2,
            "base_init_pos": [0.0, 0.0, 0.36],
            "base_init_quat": [1.0, 0.0, 0.0, 0.0],
            "push_interval_s": -1,
            "max_push_vel_xy": 1.0,
            "episode_length_s": 2.0,
            "resampling_time_s": 4.0,
            "command_type": "ang_vel_yaw",
            "action_scale": 0.5,
            "action_latency": 0.02,
            "action_range": 6.0,
            "send_timeouts": True,
            "control_freq": 50,
            "decimation": 4,
            "feet_geom_offset": 1,
            # No terrain / no height scan for the backflip (it's a
            # vertical aerial task — terrain awareness is irrelevant).
            "use_terrain": False,
            "use_height_scan": False,
            # No velocity curriculum either.
            "curriculum_terms": {},
            # Standard randomization, but slightly more aggressive so
            # the policy learns a robust flip.
            "randomize_friction": True,
            "friction_range": [0.2, 1.5],
            "randomize_base_mass": True,
            "added_mass_range": [-1.0, 3.0],
            "randomize_com_displacement": True,
            "com_displacement_range": [-0.01, 0.01],
            "randomize_motor_strength": False,
            "motor_strength_range": [0.9, 1.1],
            "randomize_motor_offset": True,
            "motor_offset_range": [-0.02, 0.02],
            "randomize_kp_scale": True,
            "kp_scale_range": [0.8, 1.2],
            "randomize_kd_scale": True,
            "kd_scale_range": [0.8, 1.2],
            "coupling": False,
        }
        # Backflip adds a phase signal to obs (sin/cos of episode time
        # at multiple harmonics), so the obs dimension is different from
        # the canonical walk env.
        n_phase = 2 * 4  # 4 harmonics, sin + cos each
        obs_cfg = {
            "num_obs": 3 + 3 + 12 + 12 + 12 + 12 + n_phase,  # 66
            "num_history_obs": 1,
            "obs_noise": {"ang_vel": 0.1, "gravity": 0.02, "dof_pos": 0.01, "dof_vel": 0.5},
            "obs_scales": {"lin_vel": 2.0, "ang_vel": 0.25, "dof_pos": 1.0, "dof_vel": 0.05},
            # Match the actual ``privileged_obs_buf`` produced in
            # ``Go2BackflipEnv.compute_observations``:
            # ``[obs_buf (num_obs), base_pos[:, 2:3] (1), base_lin_vel (3)]``.
            "num_priv_obs": 3 + 3 + 12 + 12 + 12 + 12 + n_phase + 1 + 3,
        }
        reward_cfg = {
            "soft_dof_pos_limit": 0.9,
            "reward_scales": {
                "ang_vel_y": 5.0, "ang_vel_z": -1.0, "lin_vel_z": 20.0,
                "orientation_control": -1.0, "feet_height_before_backflip": -30.0,
                "height_control": -10.0, "actions_symmetry": -0.1, "gravity_y": -10.0,
                "feet_distance": -1.0, "action_rate": -0.001,
            },
        }
        command_cfg = {
            "num_commands": 4,
            "lin_vel_x_range": [-0.0, 0.0], "lin_vel_y_range": [-0.0, 0.0], "ang_vel_range": [-0.0, 0.0],
        }
        return env_cfg, obs_cfg, reward_cfg, command_cfg

    # ------------------------------------------------------------------
    # Phase-augmented observation
    # ------------------------------------------------------------------

    def compute_observations(self):
        """Phase-augmented obs (no height scan, no temporal history)."""
        phase = math.pi * self.episode_length_buf[:, None] * self.dt / 2
        self.obs_buf = torch.cat(
            [
                self.base_ang_vel * self.obs_scales["ang_vel"],
                self.projected_gravity,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],
                self.dof_vel * self.obs_scales["dof_vel"],
                self.actions,
                self.last_actions,
                torch.sin(phase), torch.cos(phase),
                torch.sin(phase / 2), torch.cos(phase / 2),
                torch.sin(phase / 4), torch.cos(phase / 4),
            ],
            axis=-1,
        )
        # Backflip uses no temporal history: obs_buf is the live obs.
        self.obs_history_buf = self.obs_buf.detach()
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.cat(
                [
                    self.obs_buf,
                    self.base_pos[:, 2:3],
                    self.base_lin_vel * self.obs_scales["lin_vel"],
                ],
                axis=-1,
            )
            # Same invariant as Go2BaseEnv: actual buffer dim must match
            # the declared ``num_privileged_obs``.
            actual = self.privileged_obs_buf.shape[-1]
            if actual != self.num_privileged_obs:
                raise RuntimeError(
                    f"{type(self).__name__}: privileged_obs_buf dim "
                    f"({actual}) != num_privileged_obs "
                    f"({self.num_privileged_obs}). Update "
                    f"obs_cfg['num_priv_obs'] to {actual} (the env "
                    f"concatenates [obs_buf, base_pos[:, 2:3], "
                    f"base_lin_vel*scale])."
                )

    # ------------------------------------------------------------------
    # Termination (time-out only — flipping past the end is fine)
    # ------------------------------------------------------------------

    def check_termination(self):
        self.reset_buf = self.episode_length_buf > self.max_episode_length

    # ------------------------------------------------------------------
    # Custom rewards
    # ------------------------------------------------------------------

    def _reward_orientation_control(self):
        current_time = self.episode_length_buf * self.dt
        phase = (current_time - 0.5).clamp(min=0, max=0.5)
        quat_pitch = gs_quat_from_angle_axis(
            4 * phase * math.pi,
            torch.tensor([0, 1, 0], device=self.device, dtype=torch.float),
        )
        desired_base_quat = gs_quat_mul(quat_pitch, self.base_init_quat.reshape(1, -1).repeat(self.num_envs, 1))
        inv_desired_base_quat = gs_inv_quat(desired_base_quat)
        desired_projected_gravity = gs_transform_by_quat(self.global_gravity, inv_desired_base_quat)
        orientation_diff = torch.sum(torch.square(self.projected_gravity - desired_projected_gravity), dim=1)
        return orientation_diff

    def _reward_ang_vel_y(self):
        current_time = self.episode_length_buf * self.dt
        ang_vel = -self.base_ang_vel[:, 1].clamp(max=7.2, min=-7.2)
        return ang_vel * torch.logical_and(current_time > 0.5, current_time < 1.0)

    def _reward_ang_vel_z(self):
        return torch.abs(self.base_ang_vel[:, 2])

    def _reward_lin_vel_z(self):
        current_time = self.episode_length_buf * self.dt
        lin_vel = self.robot.get_vel()[:, 2].clamp(max=3)
        return lin_vel * torch.logical_and(current_time > 0.5, current_time < 0.75)

    def _reward_height_control(self):
        current_time = self.episode_length_buf * self.dt
        target_height = 0.3
        height_diff = torch.square(target_height - self.base_pos[:, 2]) * torch.logical_or(
            current_time < 0.4, current_time > 1.4
        )
        return height_diff

    def _reward_actions_symmetry(self):
        actions_diff = torch.square(self.actions[:, 0] + self.actions[:, 3])
        actions_diff += torch.square(self.actions[:, 1:3] - self.actions[:, 4:6]).sum(dim=-1)
        actions_diff += torch.square(self.actions[:, 6] + self.actions[:, 9])
        actions_diff += torch.square(self.actions[:, 7:9] - self.actions[:, 10:12]).sum(dim=-1)
        return actions_diff

    def _reward_gravity_y(self):
        return torch.square(self.projected_gravity[:, 1])

    def _reward_feet_distance(self):
        cur_footsteps_translated = self.foot_positions - self.base_pos.unsqueeze(1)
        footsteps_in_body_frame = torch.zeros(self.num_envs, 4, 3, device=self.device)
        for i in range(4):
            footsteps_in_body_frame[:, i, :] = gs_quat_apply(
                gs_quat_conjugate(self.base_quat), cur_footsteps_translated[:, i, :]
            )
        stance_width = 0.3 * torch.zeros([self.num_envs, 1], device=self.device)
        desired_ys = torch.cat([stance_width / 2, -stance_width / 2, stance_width / 2, -stance_width / 2], dim=1)
        stance_diff = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1]).sum(dim=1)
        return stance_diff

    def _reward_feet_height_before_backflip(self):
        current_time = self.episode_length_buf * self.dt
        foot_height = (self.foot_positions[:, :, 2]).view(self.num_envs, -1) - 0.02
        return foot_height.clamp(min=0).sum(dim=1) * (current_time < 0.5)

    def _reward_collision(self):
        return (
            1.0 * (torch.norm(self.link_contact_forces[:, self.penalized_contact_link_indices, :], dim=-1) > 0.1)
        ).sum(dim=1)


def get_env(num_envs: int, eval_mode: bool = False) -> Go2BackflipEnv:
    try:
        import genesis as gs
        gs.init(logging_level="warning")
    except Exception:
        pass
    return Go2BackflipEnv(num_envs=num_envs, eval_mode=eval_mode)
