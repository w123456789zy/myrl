# Copyright (c) 2025, Your Name
# All rights reserved.

"""Go2 zero-shot test track (long multi-terrain corridor).

This environment is **not** used for training. It is a fixed deterministic
13-tile corridor that evaluates a policy trained on :class:`Go2RoughEnv`
(``go2-rough``) against **all** the rough-terrain shapes the policy saw
during training, in sequence.  The track layout, in walking direction
(+x):

    tile  0: flat                     (6 m start buffer)
    tile  1: stairs UP                (30 steps × 0.10 m rise = 3.00 m)
    tile  2: platform at 3.00 m       (landing between flights)
    tile  3: stairs DOWN              (30 steps × 0.10 m drop back to ground)
    tile  4: flat                     (6 m buffer)
    tile  5: discrete obstacles       (6 m obstacle field, 0.15 m boxes)
    tile  6: flat                     (6 m buffer)
    tile  7: stepping stones          (6 m stone field, 0.4 m stones, 0.1 m gap)
    tile  8: flat                     (6 m buffer)
    tile  9: square gap               (6 m tile, 0.5 m hole in the middle)
    tile 10: flat                     (6 m buffer)
    tile 11: square pit               (6 m tile, 0.5 m deep pit in the middle)
    tile 12: flat                     (6 m end buffer)

Total track length: 78 m.  The robot spawns at ``(3, 3)`` and is commanded
to walk straight in +x (heading = 0) for the entire episode.  Commands
are locked (no curriculum).

The corridor is intentionally long: at 0.6 m/s it takes ~130 s to walk
end-to-end, so we give a 180 s episode budget.  Every individual
terrain section is sized 6 m × 6 m (the same as the original go2-test
single-staircase layout) so we can directly compare go2-test
performance against the smaller per-terrain corridors that the policy
saw during go2-rough training.
"""

from __future__ import annotations

import torch

from envs.genesis.go2_base import gs_rand_float
from envs.genesis.go2_walk import Go2WalkEnv


# ---------------------------------------------------------------------------
# Terrain layout (13 tiles, 6 m each, walking along +x).
# A single realistic staircase followed by the four other rough-terrain
# types extracted from ``third_party/legged_robot_competition``.
# ---------------------------------------------------------------------------
_TERRAIN_TYPES = [
    ["flat_terrain"],                # 0:  start buffer (6 m)
    ["stairs_terrain"],              # 1:  ascending staircase
    ["flat_terrain_at_height"],      # 2:  landing at the top of the staircase
    ["down_stairs_terrain"],         # 3:  descending staircase
    ["flat_terrain"],                # 4:  buffer
    ["discrete_obstacles_terrain"],  # 5:  obstacle field
    ["flat_terrain"],                # 6:  buffer
    ["stepping_stones_terrain"],     # 7:  stepping-stone field
    ["flat_terrain"],                # 8:  buffer
    ["gap_terrain"],                 # 9:  square gap
    ["flat_terrain"],                # 10: buffer
    ["pit_terrain"],                 # 11: square pit
    ["flat_terrain"],                # 12: end buffer (6 m)
]

# Real residential stair (per China GB 50096, ~0.18 m rise × ~0.25 m
# going), but capped at 0.10 m rise because the go2's trunk is 0.32 m
# and it cannot physically clear a taller single step.  Total rise of
# 3.0 m × 30 steps across a 6 m tile is the largest realistic flight a
# go2 can plausibly climb.
#
# ``step_width`` is chosen so ``step_width / horizontal_scale`` is an
# **exact** integer (genesis uses ``int()`` truncation, not rounding,
# so 0.30/0.10 = int(2.999...) = 2 cells/step — NOT 3).  We use
# 0.20 m / 0.10 m = 2 cells/step × 30 = 30 steps × 0.10 m = 3.00 m.
_STEP_HEIGHT = 0.10
_STEP_WIDTH = 0.20
_NUM_STEPS = 30                  # 30 × 0.20 m = 6 m tile exactly
_TOP_HEIGHT = _STEP_HEIGHT * _NUM_STEPS  # = 3.00 m

_TERRAIN_PARAMS = {
    "flat_terrain": {},
    "stairs_terrain": {
        # Genesis ``stairs_terrain`` accumulates ``step_height`` per
        # step; a *positive* value here is what makes the staircase
        # rise in the world frame.
        "step_width": _STEP_WIDTH,
        "step_height": _STEP_HEIGHT,
    },
    "flat_terrain_at_height": {
        # Raise the landing platform to the top of the ascending flight
        # so the descending flight starts at the same world height.
        "height": _TOP_HEIGHT,
    },
    "down_stairs_terrain": {
        "step_width": _STEP_WIDTH,
        "step_height": _STEP_HEIGHT,
        "base_height": _TOP_HEIGHT,  # start descending from the landing
    },
    "discrete_obstacles_terrain": {
        # 0.15 m boxes, 12 of them, sized 0.3-1.0 m — same as the
        # ``go2_test_obstacles`` corridor (and slightly taller than
        # the 0.12 m boxes seen during go2-rough training, so this is
        # mildly OOD).
        "max_height": 0.15,
        "min_size": 0.3,
        "max_size": 1.0,
        "num_rects": 12,
    },
    "stepping_stones_terrain": {
        # 0.4 m stones, 0.1 m gap between them, flat — same as
        # go2-rough training so this section is in-distribution.
        "stone_size": 0.4,
        "stone_distance": 0.1,
        "max_height": 0.0,
        "platform_size": 1.0,
    },
    "gap_terrain": {
        # 0.5 m square hole, 2 m solid platform around it.  Same
        # parameters as go2-rough training.
        "gap_size": 0.5,
        "platform_size": 2.0,
    },
    "pit_terrain": {
        # 0.5 m deep square pit.  Falling in terminates the episode.
        "depth": 0.5,
        "platform_size": 2.0,
    },
}


class Go2TestEnv(Go2WalkEnv):
    """Go2 zero-shot test track (long multi-terrain corridor)."""

    def __init__(self, num_envs: int, show_viewer: bool = False, eval_mode: bool = False) -> None:
        super().__init__(num_envs=num_envs, show_viewer=show_viewer, eval_mode=eval_mode)
        self.name = "go2-test"

    @classmethod
    def _default_configs(cls):
        env_cfg, obs_cfg, reward_cfg, command_cfg = super()._default_configs()

        # --- Terrain: 13x1 multi-terrain corridor ---
        env_cfg["use_terrain"] = True
        env_cfg["terrain_cfg"] = {
            "n_subterrains": (13, 1),
            "subterrain_size": (6.0, 6.0),
            "horizontal_scale": 0.1,
            "vertical_scale": 0.005,
            "subterrain_types": _TERRAIN_TYPES,
            "subterrain_parameters": _TERRAIN_PARAMS,
        }

        # Spawn in the first flat tile (x in [0, 6]) so the robot
        # starts on the ground level — the staircase begins only after
        # the 6 m buffer.
        env_cfg["base_init_pos"] = [3.0, 3.0, 0.42]
        env_cfg["base_init_pos_sampling_range"] = [-1.0, 1.0]

        # Heading mode, locked to +x (heading = 0) for the entire episode.
        env_cfg["command_type"] = "heading"
        command_cfg["num_commands"] = 4

        # Long episode: 78 m / ~0.6 m/s ≈ 130 s, give it 180 s so the
        # robot has time to recover from any single-terrain stumble and
        # still finish the run.
        env_cfg["episode_length_s"] = 180.0

        # Termination: align with go2-rough for fair zero-shot eval.
        env_cfg["termination_if_roll_greater_than"] = 1.0
        env_cfg["termination_if_pitch_greater_than"] = 1.0
        env_cfg["use_contact_termination"] = False

        # Height scan: same as go2-rough so the policy can see the
        # stairs / obstacles / stones / gap / pit.
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

        # Vision matches go2-rough so vision_ppo checkpoints load cleanly.
        env_cfg["use_vision"] = True
        env_cfg["vision_cfg"] = {
            "res": (96, 96),
            "fov": 100.0,
            "offset": (0.3, 0.0, 0.25),
            "lookat_offset": (2.0, 0.0, -0.4),
        }
        # Stack 5 obs steps, matching go2-rough (the env used to train the
        # checkpoints we evaluate).  Without this, the actor's MoE block
        # sees ``num_single_obs`` (235) instead of
        # ``5 * num_single_obs`` (1175) and the checkpoint cannot be
        # loaded.
        obs_cfg["num_history_obs"] = 5

        # Reward scales: mirror go2-rough so cross-env evaluation is fair.
        reward_cfg["reward_scales"]["track_lin_vel_xy"] = 2.5
        reward_cfg["reward_scales"]["feet_air_time"] = 0.5
        reward_cfg["reward_scales"]["feet_slide"] = -0.02
        reward_cfg["reward_scales"]["undesired_contacts"] = -0.1
        reward_cfg["reward_scales"]["termination"] = -2.0

        # Disable curriculum — commands are locked to +x.
        env_cfg["curriculum_terms"] = {}

        return env_cfg, obs_cfg, reward_cfg, command_cfg

    def _resample_commands(self, envs_idx):
        """Lock commands to forward walk in +x (heading = 0)."""
        if len(envs_idx) == 0:
            return
        self.commands[envs_idx, 0] = gs_rand_float(0.5, 1.0, (len(envs_idx),), self.device)
        self.commands[envs_idx, 1] = 0.0
        # heading target = 0.0 rad → +x direction
        self.commands[envs_idx, 3] = 0.0


def get_env(num_envs: int, eval_mode: bool = False) -> Go2TestEnv:
    try:
        import genesis as gs
        gs.init(logging_level="warning")
    except Exception:
        pass
    return Go2TestEnv(num_envs=num_envs, eval_mode=eval_mode)
