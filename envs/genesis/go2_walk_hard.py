# Copyright (c) 2025, Your Name
# All rights reserved.

"""Go2 walk on a hard obstacle course (vision-friendly).

Subclass of the canonical :class:`Go2VisionEnv` (which inherits the
state + height scan + MDP machinery from :class:`Go2WalkEnv`). The
``_build_scene`` override is the only piece that differs from the
parent class: it builds a custom obstacle course of stair pyramids
and gap blocks rather than the procedural terrain generator.

Terrain layout (total ~20m along +x, 4m wide):

    1. Start flat:   x=0~3m
    2. Step down:    x=3~6m   (3-level stepped pit, depth 0.3m)
    3. Step up:      x=6~9m   (3-level stepped climb back to flat)
    4. Ramp up:      x=9~12m  (gentle ramp to height 0.5m)
    5. Gap blocks:   x=12~15m (two blocks with a gap in between, height 0.5m)
    6. Ramp down:    x=15~18m (gentle ramp back to flat)
    7. End flat:     x=18~20m

Training with a vision policy (``vision_ppo``) lets the robot "see"
the upcoming obstacles and pre-shape its gait.
"""

from __future__ import annotations

import numpy as np
import torch

from envs.genesis.go2_base import gs_rand_float
from envs.genesis.go2_vision import Go2VisionEnv


class Go2WalkHardEnv(Go2VisionEnv):
    """Go2 hard obstacle course with a front camera."""

    def __init__(self, num_envs: int, show_viewer: bool = False, eval_mode: bool = False) -> None:
        super().__init__(num_envs=num_envs, show_viewer=show_viewer, eval_mode=eval_mode)
        self.name = "go2_walk_hard"

    @classmethod
    def _default_configs(cls):
        env_cfg, obs_cfg, reward_cfg, command_cfg = super()._default_configs()

        # Disable the procedural terrain generator; the obstacle course
        # is built in ``_build_scene`` below.
        env_cfg["use_terrain"] = False

        # Spawn closer to the first obstacle so the camera can see it head-on.
        env_cfg["base_init_pos"] = [1.5, 2.0, 0.42]
        env_cfg["base_init_quat"] = [1.0, 0.0, 0.0, 0.0]

        # Fixed forward command (heading mode for explicit direction tracking).
        env_cfg["command_type"] = "heading"
        env_cfg["use_contact_termination"] = True
        env_cfg["sim_substeps"] = 4

        # Tighter termination on rough terrain (allow some recovery before terminating)
        env_cfg["termination_if_roll_greater_than"] = 1.22  # ~70 degrees, aligned with mjlab
        env_cfg["termination_if_pitch_greater_than"] = 1.22

        command_cfg["num_commands"] = 4
        command_cfg["lin_vel_x_range"] = [0.2, 0.5]
        command_cfg["lin_vel_y_range"] = [0.0, 0.0]
        command_cfg["ang_vel_range"] = [0.0, 0.0]

        # Reward shaping for hard terrain: add progress reward,
        # boost velocity tracking to encourage forward movement on obstacles.
        reward_cfg["reward_scales"]["progress"] = 5.0
        reward_cfg["reward_scales"]["track_lin_vel_xy"] = 4.0

        # No velocity-range curriculum on a hard course — commands are locked to +x
        env_cfg["curriculum_terms"] = {}

        return env_cfg, obs_cfg, reward_cfg, command_cfg

    def _build_scene(self, num_envs: int, show_viewer: bool) -> None:
        """Override to add custom obstacle terrain with Boxes."""
        import genesis as gs

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.sim_dt,
                substeps=self.sim_substeps,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(10.0, 5.0, 8.0),
                camera_lookat=(10.0, 2.0, 0.0),
                max_FPS=60,
            ),
            show_viewer=not self.headless,
            rigid_options=gs.options.RigidOptions(
                dt=self.sim_dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
        )

        self.rigid_solver = self.scene.rigid_solver
        self.base_init_pos = torch.tensor(self._env_cfg["base_init_pos"], device=self.device)
        self.base_init_quat = torch.tensor(self._env_cfg["base_init_quat"], device=self.device)

        # ---- Ground plane as base ----
        self.scene.add_entity(gs.morphs.Plane())

        terrain_surface = gs.surfaces.Rough(color=(0.35, 0.35, 0.35))

        stair_w = 4.0          # y-direction width
        step_depth = 1.0       # x-direction depth per step
        step_rise = 0.1        # height per step
        n_steps = 3            # steps per staircase
        stair_height = step_rise * n_steps  # 0.3m

        # ---- 1. Start flat (covered by Plane) ----

        # ---- 2. Stair up (x=2~5m) ----
        for i in range(n_steps):
            x_center = 2.0 + step_depth * (i + 0.5)
            h = step_rise * (i + 1)
            self.scene.add_entity(
                gs.morphs.Box(
                    size=(step_depth, stair_w, 0.05),
                    pos=(x_center, 2.0, h + 0.025),
                    fixed=True,
                ),
                surface=terrain_surface,
            )

        # ---- 3. Stair down (x=5~8m) ----
        for i in range(n_steps):
            x_center = 5.0 + step_depth * (i + 0.5)
            h = stair_height - step_rise * i
            self.scene.add_entity(
                gs.morphs.Box(
                    size=(step_depth, stair_w, 0.05),
                    pos=(x_center, 2.0, h + 0.025),
                    fixed=True,
                ),
                surface=terrain_surface,
            )

        # ---- 4. Stair up again (x=8~11m) ----
        for i in range(n_steps):
            x_center = 8.0 + step_depth * (i + 0.5)
            h = step_rise * (i + 1)
            self.scene.add_entity(
                gs.morphs.Box(
                    size=(step_depth, stair_w, 0.05),
                    pos=(x_center, 2.0, h + 0.025),
                    fixed=True,
                ),
                surface=terrain_surface,
            )

        # ---- 5. Two blocks with gap (x=11~15m, height=stair_height) ----
        block_depth = 1.5      # x-direction length of each block
        gap_depth = 0.5        # gap between blocks
        block1_x = 11.0 + block_depth / 2
        block2_x = block1_x + block_depth / 2 + gap_depth + block_depth / 2
        for bx in (block1_x, block2_x):
            self.scene.add_entity(
                gs.morphs.Box(
                    size=(block_depth, stair_w, stair_height),
                    pos=(bx, 2.0, stair_height / 2),
                    fixed=True,
                ),
                surface=terrain_surface,
            )

        # ---- 6. Stair down to flat (x=15~18m) ----
        for i in range(n_steps):
            x_center = 15.0 + step_depth * (i + 0.5)
            h = stair_height - step_rise * i
            self.scene.add_entity(
                gs.morphs.Box(
                    size=(step_depth, stair_w, 0.05),
                    pos=(x_center, 2.0, h + 0.025),
                    fixed=True,
                ),
                surface=terrain_surface,
            )

        # ---- 7. End flat (covered by Plane) ----

        # ---- Robot ----
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=self._env_cfg["urdf_path"],
                merge_fixed_links=True,
                links_to_keep=self._env_cfg["links_to_keep"],
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            ),
            visualize_contact=self.debug,
        )

        # ---- Front camera (also created by Go2BaseEnv._build_scene, but
        # this env skipped the parent _build_scene entirely). ----
        self._front_camera = self.scene.add_camera(
            pos=(0.0, 0.0, 0.0),
            lookat=(1.0, 0.0, 0.0),
            res=self._vision_res,
            fov=self._vision_fov,
            GUI=False,
        )

        import sys
        if sys.platform != "darwin":
            self._set_camera()

        self.scene.build(n_envs=num_envs)

        if not self.headless and self.debug:
            self._setup_camera()

    def post_physics_step(self):
        """Hook: update the front-camera pose every step so it tracks the base."""
        super().post_physics_step()
        if self._use_vision and hasattr(self, "_front_camera"):
            front_pos = self.base_pos[0].cpu().numpy() + np.array(self._vision_offset)
            # Slight downward tilt to keep obstacles centered in frame
            front_lookat = front_pos + np.array(self._vision_lookat_offset)
            self._front_camera.set_pose(pos=front_pos, lookat=front_lookat)

    def _resample_commands(self, envs_idx):
        """Override: commands always target heading=0 (+x)."""
        if len(envs_idx) == 0:
            return
        self.commands[envs_idx, 0] = gs_rand_float(0.5, 1.0, (len(envs_idx),), self.device)
        self.commands[envs_idx, 1] = 0.0
        self.commands[envs_idx, 3] = 0.0

    def _reward_progress(self):
        """Reward forward displacement along +x."""
        return self.base_pos[:, 0] - self.last_base_pos[:, 0]


def get_env(num_envs: int, eval_mode: bool = False) -> Go2WalkHardEnv:
    try:
        import genesis as gs
        gs.init(logging_level="warning")
    except Exception:
        pass
    return Go2WalkHardEnv(num_envs=num_envs, eval_mode=eval_mode)
