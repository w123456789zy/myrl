# Copyright (c) 2025, Your Name
# All rights reserved.

"""Go2 hand-stand (rear feet on ground, front feet lifted).

The robot balances on its REAR feet with the body pitched ~180°
backward (nose-up).

Subclass of the canonical :class:`Go2WalkEnv`. Inherits the state +
height scan + MDP machinery and only overrides the env config to:
    * disable the velocity-range curriculum (commands are zero)
    * swap the locomotion reward formulation for posture rewards
"""

from __future__ import annotations

import torch

from envs.genesis.go2_walk import Go2WalkEnv


# Foot ordering matches go2_walk.py dof_names: FR=0, FL=1, RR=2, RL=3.
_HANDSTAND_FRONT_FEET = (0, 1)
_HANDSTAND_REAR_FEET = (2, 3)


class Go2HandStandEnv(Go2WalkEnv):
    """Go2 hand-stand: rear feet on ground, front feet lifted, body nose-up."""

    def __init__(self, num_envs: int, show_viewer: bool = False, eval_mode: bool = False) -> None:
        super().__init__(num_envs=num_envs, show_viewer=show_viewer, eval_mode=eval_mode)
        self.name = "go2_handstand"

    @classmethod
    def _default_configs(cls):
        env_cfg, obs_cfg, reward_cfg, command_cfg = super()._default_configs()

        env_cfg["base_init_pos"] = [0.0, 0.0, 0.4]
        env_cfg["base_init_quat"] = [1.0, 0.0, 0.0, 0.0]
        env_cfg["command_type"] = "ang_vel_yaw"
        env_cfg["termination_if_roll_greater_than"] = 1.0
        env_cfg["termination_if_pitch_greater_than"] = 1.0
        env_cfg["episode_length_s"] = 15.0
        env_cfg["use_contact_termination"] = False
        # Hand-stand needs no terrain awareness.
        env_cfg["use_height_scan"] = False

        command_cfg["num_commands"] = 3
        command_cfg["lin_vel_x_range"] = [0.0, 0.0]
        command_cfg["lin_vel_y_range"] = [0.0, 0.0]
        command_cfg["ang_vel_range"] = [0.0, 0.0]

        reward_cfg["tracking_sigma"] = 0.25
        reward_cfg["reward_scales"] = {
            "handstand_orientation": 2.0,
            "handstand_front_height": 1.5,
            "handstand_rear_contact": 0.5,
            "action_rate": -0.005,
            "termination": -2.0,
        }
        # No velocity curriculum in this env.
        env_cfg["curriculum_terms"] = {}
        return env_cfg, obs_cfg, reward_cfg, command_cfg

    # ------------------------------------------------------------------
    # Custom rewards
    # ------------------------------------------------------------------

    def _reward_handstand_orientation(self):
        """Reward body nose-up hand-stand pose.

        In a hand-stand, world gravity should be along body -x (body +x
        pointing up), with body +z pointing forward. So
        ``projected_gravity`` should be roughly (-1, 0, 0).
        """
        pg = self.projected_gravity
        align_x = (1.0 - pg[:, 0]) * 0.5  # 0 when g along +x, 1 when g along -x
        align_z = 1.0 - pg[:, 2].abs()
        return align_x * align_z

    def _reward_handstand_front_height(self):
        """Reward the front feet being lifted to a target world-z height."""
        front_z = self.foot_positions[:, _HANDSTAND_FRONT_FEET, 2]
        target = 0.50
        return torch.exp(-((front_z - target) ** 2).mean(dim=1) / 0.1)

    def _reward_handstand_rear_contact(self):
        """Reward the rear feet being in contact with the ground."""
        contact_force = self.link_contact_forces[:, _HANDSTAND_REAR_FEET, 2]
        in_contact = (contact_force > 1.0).float().mean(dim=1)
        return in_contact
