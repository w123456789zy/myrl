# Copyright (c) 2025, Your Name
# All rights reserved.

"""Go2 walk on stairs terrain.

Subclass of the canonical :class:`Go2WalkEnv` with a procedural
2-tile terrain (flat → stairs) and locked forward commands.

The robot spawns in the central flat area and is commanded to walk in
+ x direction (heading mode) so it must turn to face the stairs and
then climb them. The height scan is enabled and slightly widened so
the policy can "see" the upcoming steps.
"""

from __future__ import annotations

import torch

from envs.genesis.go2_base import gs_rand_float
from envs.genesis.go2_walk import Go2WalkEnv


class Go2WalkStairsEnv(Go2WalkEnv):
    """Go2 walk on stairs terrain.

    A flat starting area (6m x 6m) followed by upward stairs (6m wide,
    15 steps). Commands use heading mode so the robot turns toward +x
    (into the stairs).
    """

    def __init__(self, num_envs: int, show_viewer: bool = False, eval_mode: bool = False) -> None:
        super().__init__(num_envs=num_envs, show_viewer=show_viewer, eval_mode=eval_mode)
        self.name = "go2_walk_stairs"

    @classmethod
    def _default_configs(cls):
        env_cfg, obs_cfg, reward_cfg, command_cfg = super()._default_configs()

        # --- Terrain: flat start (6m) then upward stairs (6m) ---
        env_cfg["use_terrain"] = True
        env_cfg["terrain_cfg"] = {
            "n_subterrains": (2, 1),
            "subterrain_size": (6.0, 6.0),
            "horizontal_scale": 0.1,
            "vertical_scale": 0.005,
            "subterrain_types": [
                ["flat_terrain"],
                ["pyramid_stairs_terrain"],
            ],
            "subterrain_parameters": {
                "flat_terrain": {},
                "pyramid_stairs_terrain": {
                    "step_width": 0.6,
                    "step_height": 0.04,
                },
            },
        }

        # Spawn in the flat area (x=1~5, y=1~5) so ±1m random offset stays inside
        env_cfg["base_init_pos"] = [3.0, 3.0, 0.42]

        # Use heading command mode so the robot turns toward the target direction
        env_cfg["command_type"] = "heading"
        command_cfg["num_commands"] = 4

        # Termination: only base contact terminates (aligned with go2-rough).
        env_cfg["termination_if_roll_greater_than"] = 1.0
        env_cfg["termination_if_pitch_greater_than"] = 1.0
        env_cfg["use_contact_termination"] = False

        # A wider height scan helps "see" the upcoming steps.
        env_cfg["height_scan_cfg"] = {
            "resolution": 0.1,
            "size_x": 1.6,
            "size_y": 1.0,
        }
        # Recompute obs dims because we changed the height scan above.
        scan_cfg = env_cfg["height_scan_cfg"]
        n_x = int(round(scan_cfg["size_x"] / scan_cfg["resolution"])) + 1
        n_y = int(round(scan_cfg["size_y"] / scan_cfg["resolution"])) + 1
        scan_dim = n_x * n_y
        base_obs = 12 + 3 * env_cfg["num_dofs"]
        obs_cfg["num_obs"] = base_obs + scan_dim
        # Match the actual ``privileged_obs_buf`` produced in
        # ``Go2BaseEnv.compute_observations``:
        # ``[obs_buf (num_obs), base_lin_vel (3), last_actions (num_dofs)]``.
        obs_cfg["num_priv_obs"] = obs_cfg["num_obs"] + 3 + env_cfg["num_dofs"]

        # Enable vision so checkpoints trained with vision_ppo (e.g. go2-rough)
        # can be loaded and evaluated on this env.
        env_cfg["use_vision"] = True
        env_cfg["vision_cfg"] = {
            "res": (96, 96),
            "fov": 100.0,
            "offset": (0.3, 0.0, 0.25),
            "lookat_offset": (2.0, 0.0, -0.4),
        }
        # Match go2-rough (the env used to train the checkpoints we
        # evaluate here for zero-shot stairs transfer).  Same
        # ``num_history_obs`` keeps the actor's MoE input dim identical
        # so the trained weights load cleanly.
        obs_cfg["num_history_obs"] = 5

        # Stair climbing requires stronger feet-clearance and feet-air-time.
        # Align reward scales with go2-rough so cross-env evaluation is fair.
        reward_cfg["reward_scales"]["track_lin_vel_xy"] = 2.5
        reward_cfg["reward_scales"]["feet_air_time"] = 2.0
        reward_cfg["reward_scales"]["feet_slide"] = -0.02
        reward_cfg["reward_scales"]["undesired_contacts"] = -0.1
        reward_cfg["reward_scales"]["feet_impact_vel"] = -0.1
        reward_cfg["reward_scales"]["feet_clearance"] = -0.5
        # Disable velocity-range curriculum — commands are locked to +x
        env_cfg["curriculum_terms"] = {}

        return env_cfg, obs_cfg, reward_cfg, command_cfg

    def _resample_commands(self, envs_idx):
        """Override: commands always target heading=0 (+x, into the stairs)."""
        if len(envs_idx) == 0:
            return
        self.commands[envs_idx, 0] = gs_rand_float(0.5, 1.0, (len(envs_idx),), self.device)
        self.commands[envs_idx, 1] = 0.0
        # heading target = 0.0 rad → +x direction
        self.commands[envs_idx, 3] = 0.0


def get_env(num_envs: int, eval_mode: bool = False) -> Go2WalkStairsEnv:
    try:
        import genesis as gs
        gs.init(logging_level="warning")
    except Exception:
        pass
    return Go2WalkStairsEnv(num_envs=num_envs, eval_mode=eval_mode)
