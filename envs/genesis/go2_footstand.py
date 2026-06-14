# Copyright (c) 2025, Your Name
# All rights reserved.

"""Go2 foot-stand (front feet on ground, rear feet lifted).

The robot's body is pitched ~90° forward so the FRONT feet support
the weight and the REAR feet are in the air at a target height. The
base is forced upright in the WORLD frame only along the local y-axis
(no roll), so the posture is a 1-DoF pitch + heading control task.

Subclass of the canonical :class:`Go2WalkEnv`. Inherits the state +
height scan + MDP machinery and only overrides the env config to:
    * disable the velocity-range curriculum (commands are zero)
    * swap the locomotion reward formulation for posture rewards
"""

from __future__ import annotations

import torch

from envs.genesis.go2_base import wrap_to_pi
from envs.genesis.go2_walk import Go2WalkEnv


# Foot ordering in the go2 URDF (matches go2_walk.py dof_names):
#   0: FR_foot   1: FL_foot   2: RR_foot   3: RL_foot
_FOOTSTAND_FRONT_FEET = (0, 1)
_FOOTSTAND_REAR_FEETS = (2, 3)


class Go2FootStandEnv(Go2WalkEnv):
    """Go2 foot-stand: front feet on ground, rear feet lifted.

    Commands are zero (stand still + face the commanded heading). The
    policy must learn to keep the body pitched forward ~90° while
    maintaining balance on the front feet.
    """

    def __init__(self, num_envs: int, show_viewer: bool = False, eval_mode: bool = False) -> None:
        super().__init__(num_envs=num_envs, show_viewer=show_viewer, eval_mode=eval_mode)
        self.name = "go2_footstand"

    @classmethod
    def _default_configs(cls):
        env_cfg, obs_cfg, reward_cfg, command_cfg = super()._default_configs()

        # Stand still in place; keep a heading command so the policy
        # learns to keep the body straight along an axis.
        env_cfg["base_init_pos"] = [0.0, 0.0, 0.4]
        env_cfg["base_init_quat"] = [1.0, 0.0, 0.0, 0.0]
        env_cfg["command_type"] = "ang_vel_yaw"
        env_cfg["termination_if_roll_greater_than"] = 1.0
        env_cfg["termination_if_pitch_greater_than"] = 1.0
        env_cfg["episode_length_s"] = 15.0
        env_cfg["use_contact_termination"] = False
        # Foot-stand needs no terrain awareness — disable height scan.
        env_cfg["use_height_scan"] = False

        command_cfg["num_commands"] = 3
        command_cfg["lin_vel_x_range"] = [0.0, 0.0]
        command_cfg["lin_vel_y_range"] = [0.0, 0.0]
        command_cfg["ang_vel_range"] = [0.0, 0.0]

        # Footstand-specific reward: disable walking rewards, add posture rewards.
        reward_cfg["tracking_sigma"] = 0.25
        reward_cfg["reward_scales"] = {
            # Posture: keep base pitched forward ~90° and not rolled.
            "footstand_orientation": 2.0,
            "footstand_rear_height": 1.5,
            "footstand_front_contact": 0.5,
            # Smooth / safe behaviour
            "action_rate": -0.005,
            "termination": -2.0,
        }
        # No velocity curriculum in this env.
        env_cfg["curriculum_terms"] = {}
        return env_cfg, obs_cfg, reward_cfg, command_cfg

    # ------------------------------------------------------------------
    # Custom rewards (registered automatically by ``Go2BaseEnv``).
    # ------------------------------------------------------------------

    def _reward_footstand_orientation(self):
        """Reward the base being in a foot-stand orientation.

        In a foot-stand pose the world-gravity vector should be along
        the body +x (paws up). We reward projecting gravity toward
        body +x and penalizing roll.
        """
        pg = self.projected_gravity
        # Body +x points up: g_body should be (1, 0, 0).
        align_x = (pg[:, 0] + 1.0) * 0.5  # 0 when g along -x, 1 when g along +x
        # No roll: gravity should be along body z=0
        align_z = 1.0 - pg[:, 2].abs()
        return align_x * align_z

    def _reward_footstand_rear_height(self):
        """Reward the rear feet being lifted to a target world-z height."""
        rear_z = self.foot_positions[:, _FOOTSTAND_REAR_FEETS, 2]
        target = 0.30
        return torch.exp(-((rear_z - target) ** 2).mean(dim=1) / 0.05)

    def _reward_footstand_front_contact(self):
        """Reward the front feet being in contact (vertical force > threshold)."""
        contact_force = self.link_contact_forces[:, _FOOTSTAND_FRONT_FEET, 2]
        in_contact = (contact_force > 1.0).float().mean(dim=1)
        return in_contact
