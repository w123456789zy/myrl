# Copyright (c) 2025, Your Name
# All rights reserved.

"""Base class for all Go2 Genesis environments.

The class is responsible for:

* Building the genesis ``Scene`` (terrain + URDF robot + optional height
  scanner + optional front camera).
* Managing per-env state buffers (positions, velocities, contact forces,
  foot heights, action history, last observations, height scan, etc.).
* Implementing the ``VecEnv`` contract (reset, step, get_observations,
  get_rewards, seed, close).
* Composing the *reward / termination / event / command / curriculum* MDP
  by name — concrete envs only override :py:meth:`_default_configs` to
  describe what they want.
* Optional *vision* support: a front camera that renders ``images`` in
  :py:meth:`get_observations` for use with ``VisionPPO`` / ``VisionFlashSAC``.

Originally ported from the mjlab ``Go2Env`` and the Genesis-backflip
example, then aligned with the isaac-lab-style MDP terminology exposed in
``envs.genesis.mdp``.
"""

from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET
from typing import Any

import genesis as gs
import numpy as np
import torch
import torch.nn.functional as F

from mylab.env.vec_env import VecEnv, VecEnvObs

from envs.genesis.mdp import math_utils as mutils
from envs.genesis.mdp.math_utils import (
    gs_rand_float,
    gs_inv_quat,
    gs_transform_by_quat,
    gs_quat2euler,
    gs_quat_apply,
    gs_quat_conjugate,
    gs_quat_from_angle_axis,
    gs_quat_mul,
    gs_euler2quat,
    inv_quat as _inv_quat,
    normalize,
    quat_apply as _quat_apply,
    quat_apply_yaw,
    quat_conjugate as _quat_conjugate,
    quat_to_euler,
    transform_by_quat,
    wrap_to_pi,
)


class Go2BaseEnv(VecEnv):
    """Base class for the Go2 family of genesis environments."""

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        num_envs: int,
        show_viewer: bool = False,
        eval_mode: bool = False,
        debug: bool = False,
        device: str = "cuda",
    ) -> None:
        self.name = "go2_base"
        self.num_envs = 1 if num_envs == 0 else num_envs
        self.num_build_envs = num_envs

        # Load internal configs (subclass-provided).
        self._env_cfg, self._obs_cfg, self._reward_cfg, self._command_cfg = self._default_configs()

        # ------------------- sizes -------------------
        self.num_single_obs = self._obs_cfg["num_obs"]
        self.num_history = self._obs_cfg.get("num_history_obs", 1)
        self.num_obs = self.num_single_obs * self.num_history
        self.num_privileged_obs = self._obs_cfg.get("num_priv_obs")
        self.num_actions = self._env_cfg["num_actions"]
        self.num_commands = self._command_cfg["num_commands"]

        # Optional obs: height scan (ray-cast over the terrain height-field)
        self._use_height_scan = bool(self._env_cfg.get("use_height_scan", False))
        if self._use_height_scan:
            scan_cfg = self._env_cfg.get("height_scan_cfg", {})
            self._height_resolution = float(scan_cfg.get("resolution", 0.1))
            self._height_size_x = float(scan_cfg.get("size_x", 1.6))
            self._height_size_y = float(scan_cfg.get("size_y", 1.0))
            n_x = int(round(self._height_size_x / self._height_resolution)) + 1
            n_y = int(round(self._height_size_y / self._height_resolution)) + 1
            self._height_scan_dim = n_x * n_y
        else:
            self._height_scan_dim = 0
            self._height_resolution = 0.1
            self._height_size_x = 0.0
            self._height_size_y = 0.0

        # Optional obs: vision (front camera image)
        self._use_vision = bool(self._env_cfg.get("use_vision", False))
        if self._use_vision:
            vision_cfg = self._env_cfg.get("vision_cfg", {})
            self._vision_res = tuple(vision_cfg.get("res", (64, 64)))
            self._vision_fov = float(vision_cfg.get("fov", 60.0))
            self._vision_offset = tuple(vision_cfg.get("offset", (0.3, 0.0, 0.15)))
            self._vision_lookat_offset = tuple(vision_cfg.get("lookat_offset", (1.0, 0.0, -0.15)))

        # ------------------- mode flags -------------------
        self.headless = not show_viewer
        self.eval = eval_mode
        self.debug = debug

        # ------------------- time -------------------
        self.dt = 1 / self._env_cfg["control_freq"]
        sim_dt = self.dt / self._env_cfg["decimation"]
        sim_substeps = self._env_cfg.get("sim_substeps", 1)
        self.max_episode_length_s = self._env_cfg["episode_length_s"]
        self.max_episode_length = int(np.ceil(self.max_episode_length_s / self.dt))

        # ------------------- scales -------------------
        self.obs_scales = self._obs_cfg["obs_scales"]
        self.reward_scales = self._reward_cfg["reward_scales"].copy()
        self.command_type = self._env_cfg.get("command_type", "ang_vel_yaw")
        assert self.command_type in ["heading", "ang_vel_yaw"]

        self.action_latency = self._env_cfg.get("action_latency", 0)
        assert self.action_latency in [0, 0.02]

        # ------------------- device -------------------
        self.num_dof = self._env_cfg["num_dofs"]
        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            assert device in ["cpu", "cuda"]
            self.device = torch.device(device)

        self.sim_dt = sim_dt
        self.sim_substeps = sim_substeps

        # ------------------- build scene & buffers -------------------
        self._build_scene(num_envs=num_envs, show_viewer=show_viewer)
        self._init_buffers()
        self._prepare_reward_function()
        self._randomize_controls()
        self._randomize_rigids()

    # ------------------------------------------------------------------
    # Scene construction (overridable)
    # ------------------------------------------------------------------

    def _build_scene(self, num_envs: int, show_viewer: bool) -> None:
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.sim_dt, substeps=self.sim_substeps),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(1 / self.dt * self._env_cfg["decimation"]),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
            rigid_options=gs.options.RigidOptions(
                dt=self.sim_dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_self_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )

        self.rigid_solver = self.scene.rigid_solver

        # ----- Terrain vs plane -----
        if self._env_cfg.get("use_terrain", False):
            self.terrain_cfg = self._env_cfg["terrain_cfg"]
            # Build the Terrain morph.  Two ways to drive it:
            #   1. ``subterrain_types`` (Genesis built-in types only) —
            #      Genesis will call the appropriate
            #      ``genesis.ext.isaacgym.terrain_utils`` function for
            #      each cell.
            #   2. ``height_field`` (pre-computed) — Genesis skips
            #      ``subterrain_types`` entirely and just rasterises
            #      the heightfield.  This is the only way to use
            #      **local** terrain types (e.g. our
            #      ``stairs_terrain_y`` in
            #      ``envs.genesis.terrain``) without patching Genesis
            #      source code.
            precomputed_hf = self.terrain_cfg.get("height_field", None)
            terrain_morph_kwargs = dict(
                n_subterrains=self.terrain_cfg["n_subterrains"],
                horizontal_scale=self.terrain_cfg["horizontal_scale"],
                vertical_scale=self.terrain_cfg["vertical_scale"],
                subterrain_size=self.terrain_cfg["subterrain_size"],
            )
            # ``add_bottom`` is an optional memory-optimisation flag on
            # newer (un-released / patched) Genesis builds — when
            # ``False`` the watertight trimesh skips the bottom plane
            # and the 4 side walls, saving a few hundred thousand
            # vertices.  Official Genesis releases do NOT have this
            # field, so we only forward it if the user explicitly set
            # it in their config and the field is recognised.
            if "add_bottom" in self.terrain_cfg:
                terrain_morph_kwargs["add_bottom"] = bool(self.terrain_cfg["add_bottom"])
            if precomputed_hf is not None:
                terrain_morph_kwargs["height_field"] = precomputed_hf
            else:
                terrain_morph_kwargs["subterrain_types"] = self.terrain_cfg["subterrain_types"]
                terrain_morph_kwargs["subterrain_parameters"] = self.terrain_cfg.get(
                    "subterrain_parameters", {}
                )
            self.terrain = self.scene.add_entity(gs.morphs.Terrain(**terrain_morph_kwargs))
            terrain_margin_x = self.terrain_cfg["n_subterrains"][0] * self.terrain_cfg["subterrain_size"][0]
            terrain_margin_y = self.terrain_cfg["n_subterrains"][1] * self.terrain_cfg["subterrain_size"][1]
            self.terrain_margin = torch.tensor(
                [terrain_margin_x, terrain_margin_y], device=self.device, dtype=gs.tc_float
            )
            height_field = self.terrain.geoms[0].metadata["height_field"]
            self.height_field = (
                torch.tensor(height_field, device=self.device, dtype=gs.tc_float)
                * self.terrain_cfg["vertical_scale"]
            )

            # Per-cell difficulty rating (float in [1.0, 5.0]) as a
            # torch tensor of shape ``(n_subterrains[0], n_subterrains[1])``.
            # Indexed by ``(cell_x, cell_y)`` where cell 0 is at the
            # grid's min-x / min-y corner.  The forward-progress
            # reward in :class:`Go2RoughEnv` uses this to weight each
            # step's reward by the difficulty of the cell the robot
            # is *currently* walking on, and the spawn sampler uses
            # it to bias new spawns toward the harder cells.
            difficulty_map_cfg = self.terrain_cfg.get("difficulty_map", None)
            if difficulty_map_cfg is not None:
                self.difficulty_map = torch.as_tensor(
                    difficulty_map_cfg, device=self.device, dtype=gs.tc_float
                )
            else:
                self.difficulty_map = None
        else:
            self.scene.add_entity(gs.morphs.Plane())
            self.terrain_cfg = None
            self.height_field = None
            self.terrain_margin = None
            self.difficulty_map = None

        # ----- Robot -----
        self.base_init_pos = torch.tensor(self._env_cfg["base_init_pos"], device=self.device)
        self.base_init_quat = torch.tensor(self._env_cfg["base_init_quat"], device=self.device)
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=self._env_cfg["urdf_path"],
                merge_fixed_links=True,
                links_to_keep=self._env_cfg["links_to_keep"],
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
                # Force convex decomposition of every URDF link. The default
                # `decompose_robot_error_threshold=float("inf")` would skip
                # decomposition entirely, leaving the ~48k-face visual mesh
                # (base.dae / thigh.dae / calf.dae) to flow into the SDF
                # pre-processor, which fires the
                # "Beware that SDF pre-processing of mesh having more than
                # 50000 vertices may take a very long time (>10min)" warning
                # and adds ~30s+ of startup time per env-batch.
                decimate=True,
                decimate_face_num=200,
                decimate_aggressiveness=5,
                convexify=True,
                decompose_robot_error_threshold=0.0,
                decompose_object_error_threshold=0.0,
            ),
            visualize_contact=self.debug,
        )

        # ----- Optional vision camera -----
        if self._use_vision:
            self._front_camera = self.scene.add_camera(
                pos=(0.0, 0.0, 0.0),
                lookat=(1.0, 0.0, 0.0),
                res=self._vision_res,
                fov=self._vision_fov,
                GUI=False,
            )

        if sys.platform != "darwin":
            self._set_camera()

        self.scene.build(n_envs=num_envs)

    @classmethod
    def _default_configs(cls) -> tuple[dict, dict, dict, dict]:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Reward plumbing
    # ------------------------------------------------------------------

    def _prepare_reward_function(self):
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        self.reward_functions = []
        self.reward_names = []
        for name in self.reward_scales:
            if name == "termination":
                continue
            self.reward_names.append(name)
            method_name = "_reward_" + name
            if hasattr(self, method_name):
                self.reward_functions.append(getattr(self, method_name))
            else:
                # Fallback: dynamic dispatch into envs.genesis.mdp.rewards
                self.reward_functions.append(self._dispatch_mdp_reward(name))
        self.episode_sums = {
            name: torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
            for name in self.reward_scales.keys()
        }

    def _dispatch_mdp_reward(self, name: str):
        """Bind a reward function name to ``envs.genesis.mdp.rewards``.

        Returns a closure that takes no arguments (apart from the env which
        is bound by attribute access on the env object).
        """
        from envs.genesis.mdp import rewards as mdp_rewards
        from envs.genesis.mdp.terminations import illegal_contact

        # Special handling for body-aware rewards: the user provides the
        # body indices in ``env.penalized_contact_link_indices``.
        if name == "undesired_contacts":
            def fn():
                return mdp_rewards.undesired_contacts(
                    self, threshold=1.0, body_indices=self.penalized_contact_link_indices
                )
            return fn
        if name == "illegal_contact":
            def fn():
                return illegal_contact(self, {"body_indices": 1, "threshold": 1.0})
            return fn
        if name == "feet_stumble":
            def fn():
                return mdp_rewards.feet_stumble(self)
            return fn
        if name == "feet_too_near":
            def fn():
                return mdp_rewards.feet_too_near(self)
            return fn
        if name == "flat_orientation_l2":
            def fn():
                return mdp_rewards.flat_orientation_l2(self)
            return fn
        if name == "joint_torques_l2":
            def fn():
                return mdp_rewards.joint_torques_l2(self)
            return fn
        if name == "energy":
            def fn():
                return mdp_rewards.energy(self)
            return fn
        if name == "joint_vel_l2":
            def fn():
                return mdp_rewards.joint_vel_l2(self)
            return fn
        if name == "joint_acc_l2":
            def fn():
                return mdp_rewards.joint_acc_l2(self)
            return fn
        if name == "joint_pos_limits":
            def fn():
                return mdp_rewards.joint_pos_limits(self)
            return fn
        if name == "lin_vel_z_l2":
            def fn():
                return mdp_rewards.lin_vel_z_l2(self)
            return fn
        if name == "ang_vel_xy_l2":
            def fn():
                return mdp_rewards.ang_vel_xy_l2(self)
            return fn
        if name == "action_rate_l2":
            def fn():
                return mdp_rewards.action_rate_l2(self)
            return fn
        if name == "track_lin_vel_xy":
            def fn():
                # Pass ``tracking_sigma`` directly — the mdp function
                # computes ``exp(-err / std)``, so we want ``std=0.25``
                # to reproduce isaac-lab's
                # ``mdp.track_lin_vel_xy_exp(std=sqrt(0.25))`` followed
                # by an internal ``/ std**2`` (net effect: 0.25).
                return mdp_rewards.track_lin_vel_xy_exp(
                    self, std=self._reward_cfg.get("tracking_sigma", 0.25)
                )
            return fn
        if name == "track_ang_vel_z":
            def fn():
                return mdp_rewards.track_ang_vel_z_exp(
                    self, std=self._reward_cfg.get("tracking_sigma", 0.25)
                )
            return fn
        if name == "air_time_variance":
            def fn():
                return mdp_rewards.air_time_variance_penalty(self)
            return fn
        if name == "feet_slide":
            def fn():
                return mdp_rewards.feet_slide(self)
            return fn
        if name == "feet_air_time":
            def fn():
                return mdp_rewards.feet_air_time(self)
            return fn
        if name == "feet_impact_vel":
            def fn():
                return mdp_rewards.feet_impact_vel(self)
            return fn
        if name == "feet_clearance":
            def fn():
                return mdp_rewards.feet_clearance(self)
            return fn
        if name == "joint_position_penalty":
            def fn():
                return mdp_rewards.joint_position_penalty(self)
            return fn
        # Default: look up by name in mdp.rewards and bind ``self`` so the
        # function can be called with zero arguments in ``compute_reward``.
        mdp_fn = getattr(mdp_rewards, name, None)
        if mdp_fn is None:
            raise AttributeError(f"Reward function '{name}' not found in env or mdp.rewards")

        def fn():
            return mdp_fn(self)

        return fn

    # ------------------------------------------------------------------
    # Buffers
    # ------------------------------------------------------------------

    def _init_buffers(self):
        # ----- base state -----
        self.base_euler = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor(np.array([0.0, 0.0, -1.0]), device=self.device, dtype=gs.tc_float)
        self.forward_vec = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.forward_vec[:, 0] = 1.0

        # ----- observation buffers -----
        self.obs_buf = torch.zeros((self.num_envs, self.num_single_obs), device=self.device, dtype=gs.tc_float)
        self.final_obs_history_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.obs_history_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.obs_noise = torch.zeros((self.num_envs, self.num_single_obs), device=self.device, dtype=gs.tc_float)
        self._prepare_obs_noise()
        self.privileged_obs_buf = (
            None
            if self.num_privileged_obs is None
            else torch.zeros((self.num_envs, self.num_privileged_obs), device=self.device, dtype=gs.tc_float)
        )
        self.final_privileged_obs_buf = (
            None
            if self.num_privileged_obs is None
            else torch.zeros((self.num_envs, self.num_privileged_obs), device=self.device, dtype=gs.tc_float)
        )

        # ----- reward / reset -----
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.rew_buf_pos = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.rew_buf_neg = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.time_out_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)

        # ----- commands -----
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)
        self.commands_body = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            device=self.device, dtype=gs.tc_float,
        )
        self.stand_still = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)

        # ----- dof / link indices -----
        # ``dofs_idx_local`` (the canonical replacement for the deprecated
        # ``dof_idx_local``) returns a *list* of int because a single
        # joint can in principle own multiple DoFs. Go2 joints happen to
        # each own exactly one DoF, so we get a list-of-lists. Flatten
        # it to a flat list of ints, otherwise downstream methods like
        # ``set_dofs_kp`` raise "can only concatenate list (not "int")
        # to list" because they iterate ``idx + offset`` assuming ints.
        raw = (self.robot.get_joint(name).dofs_idx_local for name in self._env_cfg["dof_names"])
        self.motor_dofs = [idx for sub in raw for idx in sub]

        def find_link_indices(names):
            link_indices = list()
            for link in self.robot.links:
                flag = False
                for name in names:
                    if name in link.name:
                        flag = True
                if flag:
                    link_indices.append(link.idx - self.robot.link_start)
            return link_indices

        self.termination_contact_link_indices = find_link_indices(self._env_cfg["termination_contact_link_names"])
        self.penalized_contact_link_indices = find_link_indices(self._env_cfg["penalized_contact_link_names"])
        self.feet_link_indices = find_link_indices(self._env_cfg["feet_link_names"])
        assert len(self.termination_contact_link_indices) > 0
        assert len(self.penalized_contact_link_indices) > 0
        assert len(self.feet_link_indices) > 0
        self.feet_link_indices_world_frame = [i + 1 for i in self.feet_link_indices]

        # ----- action / dof state -----
        self.actions = torch.zeros((self.num_envs, self.num_dof), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros((self.num_envs, self.num_dof), device=self.device, dtype=gs.tc_float)
        self.last_last_actions = torch.zeros((self.num_envs, self.num_dof), device=self.device, dtype=gs.tc_float)
        self.dof_pos = torch.zeros((self.num_envs, self.num_dof), device=self.device, dtype=gs.tc_float)
        self.dof_vel = torch.zeros((self.num_envs, self.num_dof), device=self.device, dtype=gs.tc_float)
        self.last_dof_vel = torch.zeros((self.num_envs, self.num_dof), device=self.device, dtype=gs.tc_float)
        self.root_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.last_root_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.last_base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.torques = torch.zeros((self.num_envs, self.num_dof), device=self.device, dtype=gs.tc_float)
        self.link_contact_forces = torch.zeros(
            (self.num_envs, self.robot.n_links, 3), device=self.device, dtype=gs.tc_float
        )

        # ----- foot state -----
        self.feet_air_time = torch.zeros(
            (self.num_envs, len(self.feet_link_indices)), device=self.device, dtype=gs.tc_float
        )
        self.feet_max_height = torch.zeros(
            (self.num_envs, len(self.feet_link_indices)), device=self.device, dtype=gs.tc_float
        )
        self.last_contacts = torch.zeros(
            (self.num_envs, len(self.feet_link_indices)), device=self.device, dtype=gs.tc_int
        )
        self.prev_foot_velocities = torch.zeros(
            (self.num_envs, len(self.feet_link_indices), 3), device=self.device, dtype=gs.tc_float
        )

        # ----- height scan buffer (filled in compute_observations) -----
        if self._use_height_scan:
            self.height_scan_buf = torch.zeros(
                (self.num_envs, self._height_scan_dim), device=self.device, dtype=gs.tc_float
            )
        else:
            self.height_scan_buf = None

        # ----- misc -----
        self.continuous_push = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.env_identities = torch.arange(self.num_envs, device=self.device, dtype=gs.tc_int)
        self.common_step_counter = 0
        self.extras: dict[str, Any] = {}
        self.terrain_heights = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

        # ----- PD gains -----
        stiffness = self._env_cfg["PD_stiffness"]
        damping = self._env_cfg["PD_damping"]
        self.p_gains, self.d_gains = [], []
        for dof_name in self._env_cfg["dof_names"]:
            for key in stiffness.keys():
                if key in dof_name:
                    self.p_gains.append(stiffness[key])
                    self.d_gains.append(damping[key])
        self.p_gains = torch.tensor(self.p_gains, device=self.device)
        self.d_gains = torch.tensor(self.d_gains, device=self.device)
        self.batched_p_gains = self.p_gains[None, :].repeat(self.num_envs, 1)
        self.batched_d_gains = self.d_gains[None, :].repeat(self.num_envs, 1)

        self.robot.set_dofs_kp(self.p_gains, self.motor_dofs)
        self.robot.set_dofs_kv(self.d_gains, self.motor_dofs)

        default_joint_angles = self._env_cfg["default_joint_angles"]
        self.default_dof_pos = torch.tensor(
            [default_joint_angles[name] for name in self._env_cfg["dof_names"]], device=self.device
        )

        # ----- dof limits + velocity limits -----
        self.dof_pos_limits = torch.stack(self.robot.get_dofs_limit(self.motor_dofs), dim=1)
        self.torque_limits = self.robot.get_dofs_force_range(self.motor_dofs)[1]
        urdf_path = self._env_cfg["urdf_path"]
        if not os.path.isabs(urdf_path):
            urdf_path = os.path.join(gs.__path__[0], "assets", urdf_path)
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        vel_limit_map = {}
        for joint in root.findall("joint"):
            jname = joint.get("name")
            limit_elem = joint.find("limit")
            if limit_elem is not None:
                vel = limit_elem.get("velocity")
                if vel is not None:
                    vel_limit_map[jname] = float(vel)
        self.dof_vel_limits = torch.tensor(
            [vel_limit_map.get(name, 30.0) for name in self._env_cfg["dof_names"]],
            device=self.device, dtype=torch.float,
        )
        if "soft_dof_pos_limit" in self._reward_cfg:
            for i in range(self.dof_pos_limits.shape[0]):
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self._reward_cfg["soft_dof_pos_limit"]
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self._reward_cfg["soft_dof_pos_limit"]

        self.motor_strengths = gs.ones((self.num_envs, self.num_dof), dtype=float)
        self.motor_offsets = gs.zeros((self.num_envs, self.num_dof), dtype=float)

        self.foot_positions = torch.ones(
            self.num_envs, len(self.feet_link_indices), 3, device=self.device, dtype=gs.tc_float
        )
        self.foot_quaternions = torch.ones(
            self.num_envs, len(self.feet_link_indices), 4, device=self.device, dtype=gs.tc_float
        )
        self.foot_velocities = torch.ones(
            self.num_envs, len(self.feet_link_indices), 3, device=self.device, dtype=gs.tc_float
        )
        self.base_link_index = 1
        self.com = torch.zeros(self.num_envs, 3, device=self.device, dtype=gs.tc_float)

    # ------------------------------------------------------------------
    # VecEnv interface
    # ------------------------------------------------------------------

    def _get_obs_dict(self) -> dict:
        obs: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {"state": self.obs_history_buf}
        if self._use_vision and hasattr(self, "_front_camera"):
            obs["images"] = self.get_camera_observation()
        return obs

    def get_observations(self) -> dict:
        return self._get_obs_dict()

    def get_camera_observation(self) -> torch.Tensor:
        """Render the front camera and return a normalized image tensor."""
        if not self._use_vision or not hasattr(self, "_front_camera"):
            return None
        # Update camera pose to follow the first env
        base_pos = self.base_pos[0].cpu().numpy()
        base_quat = self.base_quat[0].cpu().numpy()  # w, x, y, z

        def _rotate(v):
            w, x, y, z = base_quat
            tx = 2.0 * (y * v[2] - z * v[1])
            ty = 2.0 * (z * v[0] - x * v[2])
            tz = 2.0 * (x * v[1] - y * v[0])
            return v + w * np.array([tx, ty, tz]) + np.array([
                y * tz - z * ty,
                z * tx - x * tz,
                x * ty - y * tx,
            ])

        front_pos = base_pos + _rotate(np.array(self._vision_offset))
        front_lookat = front_pos + _rotate(np.array(self._vision_lookat_offset))
        self._front_camera.set_pose(pos=front_pos, lookat=front_lookat)
        frame, _, _, _ = self._front_camera.render()
        frame = np.array(frame)
        if frame.shape[-1] == 4:
            frame = frame[..., :3]
        image = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        if image.ndim == 3 and image.shape[1:] != self._vision_res:
            image = F.interpolate(
                image.unsqueeze(0), size=self._vision_res, mode="bilinear", align_corners=False
            ).squeeze(0)
        return image.unsqueeze(0).repeat(self.num_envs, 1, 1, 1).to(self.device)

    def get_rewards(self) -> torch.Tensor:
        return self.rew_buf

    def reset(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> tuple[VecEnvObs, dict]:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.reset_buf[env_ids] = True
        self.reset_idx(env_ids)
        self.compute_observations()
        return self._get_obs_dict(), self.extras

    def step(self, actions: torch.Tensor):
        action_range = self._env_cfg["action_range"]
        self.actions = torch.clip(actions, -action_range, action_range)
        exec_actions = self.last_actions if self.action_latency > 0 else self.actions

        target_dof_pos = self._compute_target_dof_pos(exec_actions)
        for _ in range(self._env_cfg["decimation"]):
            self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)
            self.scene.step()

        self.post_physics_step()

        self.extras["privileged_observations"] = self.privileged_obs_buf
        self.extras["final_observations"] = self.final_obs_history_buf
        self.extras["final_privileged_observations"] = self.final_privileged_obs_buf

        self.done_buf = self.reset_buf.clone()
        done = self.done_buf.to(dtype=gs.tc_float)

        return self._get_obs_dict(), self.rew_buf, done, self.extras

    def seed(self, seed: int = -1) -> int:
        if seed != -1:
            torch.manual_seed(seed)
            np.random.seed(seed)
        return seed

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Internal: torques / state updates
    # ------------------------------------------------------------------

    def _compute_torques(self, actions=None):
        if actions is None:
            actions = self.last_actions if self.action_latency > 0 else self.actions
        actions_scaled = actions * self._env_cfg["action_scale"]
        torques = (
            self.batched_p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos + self.motor_offsets)
            - self.batched_d_gains * self.dof_vel
        )
        return torques * self.motor_strengths

    def _compute_target_dof_pos(self, actions):
        actions_scaled = actions * self._env_cfg["action_scale"]
        return actions_scaled + self.default_dof_pos

    def _sample_base_pos_difficulty_biased(self, envs_idx):
        """Sample a base_pos for each env that biases toward hard cells.

        Returns a tensor of shape (len(envs_idx), 3) with positions on
        the heightfield.  For each env we:

        1. Build a per-cell probability by raising the difficulty
           rating to a power ``alpha`` (``difficulty_exponent`` in the
           config, default 1.5) and clipping cells below
           ``difficulty_threshold`` (default 1.5) to zero.  This
           flattens the easy cells to 0 and emphasises the hard ones
           (stairs, pits, stepping stones).
        2. Renormalise the remaining weights and sample a cell index
           per env via ``torch.multinomial``.
        3. Sample a uniform point inside the chosen cell, with a
           ``spawn_inner_margin`` of the cell size subtracted from
           each side so the robot never spawns flush against a cell
           boundary.
        """
        n = len(envs_idx)
        cfg = self._env_cfg
        # ``difficulty_exponent`` defaults to 0.5 (square-root) rather
        # than the previous 1.5.  This softens the bias: the
        # centre-flat cell (diff=1.0) still gets a tiny weight from
        # the 5% jitter mean, while hard cells (diff=4.5) get
        # 4.5^0.5 = 2.12× relative weight.  A 1.5 exponent was so
        # aggressive that ~70% of resets landed on the 4.5-rated
        # stair cells, which on the new terrain (which the policy
        # had never seen) caused it to fall on the very first step
        # and learn "I cannot recover from a hard-cell spawn".
        alpha = float(cfg.get("difficulty_exponent", 0.5))
        # Threshold raised from 1.5 to 2.0 so the centre-flat cell
        # (diff=1.0) and the discrete-obstacles cell (diff=2.5)
        # still get some spawn probability in early training.  This
        # gives the policy a mix of ~50% easy / ~50% hard spawns at
        # the start, which we can later tighten by raising the
        # exponent in a curriculum.
        threshold = float(cfg.get("difficulty_threshold", 2.0))
        margin = float(cfg.get("spawn_inner_margin", 0.6))

        # Per-cell weights: (difficulty - threshold).clamp(0) ^ alpha.
        diff = (self.difficulty_map - threshold).clamp(min=0.0) ** alpha
        flat_w = diff.flatten()
        # Avoid degenerate zero-probability (e.g. all flat) by adding
        # a tiny uniform component (5% of the mean weight).
        if flat_w.sum() <= 0.0:
            flat_w = torch.ones_like(flat_w)
        else:
            flat_w = flat_w + 0.05 * flat_w.mean()
        flat_w = flat_w / flat_w.sum()

        # Sample a cell per env.
        sampled = torch.multinomial(flat_w, n, replacement=True)
        cell_x = (sampled // self.difficulty_map.shape[1]).to(self.device)
        cell_y = (sampled % self.difficulty_map.shape[1]).to(self.device)

        # Cell origin in world coordinates (cell 0 at min corner).
        size_x = self.terrain_cfg["subterrain_size"][0]
        size_y = self.terrain_cfg["subterrain_size"][1]
        # Uniform point inside cell with margin.
        cell_origin_x = cell_x.to(self.device, dtype=gs.tc_float) * size_x
        cell_origin_y = cell_y.to(self.device, dtype=gs.tc_float) * size_y
        # Sample the inner region of the cell.
        u = gs_rand_float(0.0, 1.0, (n, 1), self.device)
        v = gs_rand_float(0.0, 1.0, (n, 1), self.device)
        # Inset by ``margin`` on all 4 sides.
        span = max(0.0, 1.0 - 2.0 * margin / min(size_x, size_y))
        u = u * span + margin / min(size_x, size_y)
        v = v * span + margin / min(size_x, size_y)
        x = (cell_origin_x + u.squeeze(-1) * size_x).unsqueeze(-1)
        y = (cell_origin_y + v.squeeze(-1) * size_y).unsqueeze(-1)
        z = torch.full_like(x, self.base_init_pos[2].item())
        return torch.cat([x, y, z], dim=-1)

    def _update_buffers(self):
        self.last_base_pos = self.base_pos.clone()
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        base_quat_rel = gs_quat_mul(
            self.base_quat, gs_inv_quat(self.base_init_quat.reshape(1, -1).repeat(self.num_envs, 1))
        )
        self.base_euler = gs_quat2euler(base_quat_rel)

        inv_quat_yaw = gs_quat_from_angle_axis(
            -self.base_euler[:, 2], torch.tensor([0, 0, 1], device=self.device, dtype=torch.float)
        )
        inv_base_quat = gs_inv_quat(self.base_quat)
        self.base_lin_vel[:] = gs_transform_by_quat(self.robot.get_vel(), inv_quat_yaw)
        self.base_ang_vel[:] = gs_transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = gs_transform_by_quat(self.global_gravity, inv_base_quat)

        self.last_dof_vel[:] = self.dof_vel[:]
        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)
        self.link_contact_forces[:] = torch.tensor(
            self.robot.get_links_net_contact_force(), device=self.device, dtype=gs.tc_float
        )
        self.com[:] = self.rigid_solver.get_links_root_COM([self.base_link_index]).squeeze(dim=1)

        self.foot_positions[:] = self.rigid_solver.get_links_pos(self.feet_link_indices_world_frame)
        self.foot_quaternions[:] = self.rigid_solver.get_links_quat(self.feet_link_indices_world_frame)
        self.prev_foot_velocities[:] = self.foot_velocities[:]
        self.foot_velocities[:] = self.rigid_solver.get_links_vel(self.feet_link_indices_world_frame)

        if self._env_cfg.get("use_terrain", False) and self.height_field is not None:
            clipped_base_pos = self.base_pos[:, :2].clamp(
                min=torch.zeros(2, device=self.device), max=self.terrain_margin
            )
            height_field_ids = (clipped_base_pos / self.terrain_cfg["horizontal_scale"] - 0.5).floor().int()
            height_field_ids.clamp(min=0)
            self.terrain_heights = self.height_field[height_field_ids[:, 0], height_field_ids[:, 1]]

    # ------------------------------------------------------------------
    # Termination / reward
    # ------------------------------------------------------------------

    def check_termination(self):
        # Episode length timeout
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.time_out_buf = self.reset_buf.clone()

        # Pitch / roll angle limits
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self._env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self._env_cfg["termination_if_roll_greater_than"]

        # Illegal contact termination (opt-in)
        if self._env_cfg.get("use_contact_termination", False):
            self.reset_buf |= torch.any(
                torch.norm(self.link_contact_forces[:, self.termination_contact_link_indices, :], dim=-1) > 1.0,
                dim=1,
            )

        # Terrain boundaries
        if self._env_cfg.get("use_terrain", False):
            self.reset_buf |= torch.logical_or(
                self.base_pos[:, 0] > self.terrain_margin[0],
                self.base_pos[:, 1] > self.terrain_margin[1],
            )
            self.reset_buf |= torch.logical_or(self.base_pos[:, 0] < -1, self.base_pos[:, 1] < -1)

        # Base height termination (opt-in, e.g. for backflip)
        if "termination_if_height_lower_than" in self._env_cfg:
            self.reset_buf |= self.base_pos[:, 2] < self._env_cfg["termination_if_height_lower_than"]

    def compute_reward(self):
        self.rew_buf[:] = 0.0
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
        if self._env_cfg.get("only_positive_rewards", False):
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.0)

    # ------------------------------------------------------------------
    # Per-step orchestration
    # ------------------------------------------------------------------

    def post_physics_step(self):
        self.episode_length_buf += 1
        self.common_step_counter += 1
        self._update_buffers()
        self.torques = self._compute_torques()

        # Command resampling
        resampling_time_s = self._command_cfg.get("resampling_time_s", 10.0)
        envs_idx = (self.episode_length_buf % int(resampling_time_s / self.dt) == 0).nonzero(as_tuple=False).flatten()
        self._resample_commands(envs_idx)
        self._randomize_rigids(envs_idx)
        self._randomize_controls(envs_idx)

        # Heading command: convert target heading → angular vel
        if self.command_type == "heading":
            forward = gs_transform_by_quat(self.forward_vec, self.base_quat)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(
                wrap_to_pi(self.commands[:, 3] - heading), -1.0, 1.0
            )

        # Interval push event (mimics isaac-lab's push_by_setting_velocity)
        push_interval_s = self._env_cfg.get("push_interval_s", -1)
        if push_interval_s > 0 and not (self.debug or self.eval):
            self._apply_push_robot(push_interval_s)

        # Body-frame command for consistent reward/obs
        if self.command_type == "ang_vel_yaw":
            yaw = self.base_euler[:, 2]
            cos_yaw = torch.cos(yaw)
            sin_yaw = torch.sin(yaw)
            self.commands_body[:, 0] = cos_yaw * self.commands[:, 0] + sin_yaw * self.commands[:, 1]
            self.commands_body[:, 1] = -sin_yaw * self.commands[:, 0] + cos_yaw * self.commands[:, 1]
            self.commands_body[:, 2] = self.commands[:, 2]
        else:
            self.commands_body[:] = self.commands[:]

        # Curriculum step (after resampling, before termination)
        self._apply_curriculums()

        self.check_termination()
        self.compute_reward()

        if torch.any(self.reset_buf):
            self.extras["episode_length"] = (
                (self.episode_length_buf * self.reset_buf).sum() / self.reset_buf.sum()
            ).item()
        envs_idx = self.reset_buf.nonzero(as_tuple=False).flatten()
        if self.num_build_envs > 0:
            self.compute_observations()
            self.final_obs_history_buf = self.obs_history_buf.detach().clone()
            self.final_privileged_obs_buf = (
                self.privileged_obs_buf.detach().clone()
                if self.privileged_obs_buf is not None
                else None
            )
            self.reset_idx(envs_idx)

        self.compute_observations()

        if not self.headless and self.debug:
            self._draw_debug_vis()

        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_root_vel[:] = self.robot.get_vel()

    def compute_observations(self):
        # Core state obs (proprioception)
        if self._use_height_scan:
            from envs.genesis.mdp import observations as mdp_obs
            scan = mdp_obs.height_scan(
                self,
                resolution=self._height_resolution,
                size_x=self._height_size_x,
                size_y=self._height_size_y,
            )
            self.height_scan_buf[:] = scan
            scan_scaled = scan * self.obs_scales.get("height_scan", 1.0)
            self.obs_buf = torch.cat(
                [
                    self.base_lin_vel * self.obs_scales["lin_vel"],
                    self.base_ang_vel * self.obs_scales["ang_vel"],
                    self.projected_gravity,
                    self.commands_body[:, :3] * self.commands_scale,
                    (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],
                    self.dof_vel * self.obs_scales["dof_vel"],
                    self.actions,
                    scan_scaled,
                ],
                axis=-1,
            )
        else:
            self.obs_buf = torch.cat(
                [
                    self.base_lin_vel * self.obs_scales["lin_vel"],
                    self.base_ang_vel * self.obs_scales["ang_vel"],
                    self.projected_gravity,
                    self.commands_body[:, :3] * self.commands_scale,
                    (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],
                    self.dof_vel * self.obs_scales["dof_vel"],
                    self.actions,
                ],
                axis=-1,
            )
        if not self.eval:
            self.obs_buf += gs_rand_float(-1.0, 1.0, (self.num_single_obs,), self.device) * self.obs_noise
        clip_obs = 100.0
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        self.obs_history_buf = torch.cat(
            [self.obs_history_buf[:, self.num_single_obs:], self.obs_buf.detach()], dim=1
        )
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.cat(
                [
                    self.obs_buf,
                    self.base_lin_vel * self.obs_scales["lin_vel"],
                    self.last_actions,
                ],
                axis=-1,
            )
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
            # ``num_privileged_obs`` is sized to the *initial* buffer; it
            # must equal the dimension the components above actually
            # produce.  Mismatch means a subclass declared the wrong
            # ``num_priv_obs`` in its config — fail loudly here so the
            # training loop never silently consumes a wrongly-shaped
            # buffer.
            actual = self.privileged_obs_buf.shape[-1]
            if actual != self.num_privileged_obs:
                raise RuntimeError(
                    f"{type(self).__name__}: privileged_obs_buf dim "
                    f"({actual}) != num_privileged_obs "
                    f"({self.num_privileged_obs}). Update "
                    f"obs_cfg['num_priv_obs'] to {actual} (the env "
                    f"concatenates [obs_buf, base_lin_vel*scale, "
                    f"last_actions])."
                )

    # ------------------------------------------------------------------
    # Curriculum / push / obs noise
    # ------------------------------------------------------------------

    def _apply_curriculums(self) -> None:
        """Run any configured curriculum functions (lin_vel, terrain, etc.)."""
        for name, fn in self._env_cfg.get("curriculum_terms", {}).items():
            try:
                fn(self)
            except Exception as exc:  # noqa: BLE001
                if self.debug:
                    print(f"[curriculum:{name}] failed: {exc}")

    def _apply_push_robot(self, push_interval_s: float) -> None:
        """Periodically apply a base-velocity push to non-eval envs."""
        from envs.genesis.mdp import events as mdp_events
        push_range = self._env_cfg.get("push_velocity_range", {"x": (-0.5, 0.5), "y": (-0.5, 0.5)})
        period = int(push_interval_s / self.dt)
        if period <= 0:
            return
        env_ids = ((self.common_step_counter + self.env_identities) % period == 0).nonzero(as_tuple=False).flatten()
        if len(env_ids) == 0:
            return
        # Apply via the per-env link velocity API: this gives an instantaneous
        # nudge to the base, equivalent to isaac-lab's push_by_setting_velocity.
        try:
            mdp_events.push_by_setting_velocity(self, env_ids, push_range)
        except Exception:
            # Fall back to no-op (e.g. for very old genesis API) so the env
            # still runs.
            pass

    def _prepare_obs_noise(self):
        # Obs layout in the canonical Go2Walk policy:
        #   0:3   base lin vel
        #   3:6   base ang vel
        #   6:9   projected gravity
        #   9:12  commands (lin_vel_x/y, ang_vel_z)
        #   12:24 joint pos rel (12 dofs)
        #   24:36 joint vel rel (12 dofs)
        #   36:48 last actions (12)
        #   48:   height scan / vision
        # The noise channels must line up with these slices — an
        # off-by-12 here would put dof_pos noise on the joint_vel
        # channel and dof_vel noise on the last_action channel, which
        # silently destroys the signal.
        n = self.num_dof
        obs_noise = self._obs_cfg["obs_noise"]
        self.obs_noise[:, 0:3] = obs_noise.get("lin_vel", 0.0)
        self.obs_noise[:, 3:6] = obs_noise["ang_vel"]
        self.obs_noise[:, 6:9] = obs_noise["gravity"]
        self.obs_noise[:, 12:12 + n] = obs_noise["dof_pos"]
        self.obs_noise[:, 12 + n:12 + 2 * n] = obs_noise["dof_vel"]

    def _resample_commands(self, envs_idx):
        if len(envs_idx) == 0:
            return
        self.commands[envs_idx, 0] = gs_rand_float(*self._command_cfg["lin_vel_x_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self._command_cfg["lin_vel_y_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, :2] *= (torch.norm(self.commands[envs_idx, :2], dim=1) > 0.2).unsqueeze(1)
        if self.command_type == "heading":
            self.commands[envs_idx, 3] = gs_rand_float(-3.14, 3.14, (len(envs_idx),), self.device)
        elif self.command_type == "ang_vel_yaw":
            self.commands[envs_idx, 2] = gs_rand_float(*self._command_cfg["ang_vel_range"], (len(envs_idx),), self.device)
            self.commands[envs_idx, 2] *= torch.abs(self.commands[envs_idx, 2]) > 0.2

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        self.dof_pos[envs_idx] = self.default_dof_pos + gs_rand_float(-0.3, 0.3, (len(envs_idx), self.num_dof), self.device)
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx], dofs_idx_local=self.motor_dofs, zero_velocity=True, envs_idx=envs_idx
        )

        # ---- Spawn position: difficulty-biased ----
        #
        # Instead of a uniform ±range around the centre, we sample a
        # random cell from the difficulty map (probability ∝
        # max(difficulty - threshold, 0)) and then sample a uniform
        # point inside that cell.  This guarantees that a significant
        # fraction of resets place the robot directly on a hard
        # obstacle (stair / pit / stepping stones) so the policy
        # *must* learn to handle those terrains from the very first
        # step of every new episode.  Without this bias the robot
        # spends most of its time on the centre flat cell (low
        # difficulty) and never sees enough hard examples to make
        # meaningful progress on them.
        if (
            self.difficulty_map is not None
            and self._env_cfg.get("difficulty_biased_spawn", True)
        ):
            self.base_pos[envs_idx] = self._sample_base_pos_difficulty_biased(envs_idx)
        else:
            spawn_range = self._env_cfg.get("base_init_pos_sampling_range", [-1.0, 1.0])
            self.base_pos[envs_idx] = self.base_init_pos
            self.base_pos[envs_idx, :2] += gs_rand_float(
                spawn_range[0], spawn_range[1], (len(envs_idx), 2), self.device
            )
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        # In eval/play mode we want the robot to spawn with a
        # *known* orientation so the user can drive it predictably
        # (e.g. always face +x at the start).  In train mode we
        # still randomise roll / pitch / yaw for domain
        # randomisation.
        if getattr(self, "eval_mode", False):
            # Identity base quat + small roll/pitch jitter only;
            # **no** random yaw so the robot always starts facing
            # the +x axis (the canonical "forward" direction).
            base_euler = gs_rand_float(-0.1, 0.1, (len(envs_idx), 3), self.device)
            base_euler[:, 2] = 0.0
        else:
            # Train mode: full random orientation for domain randomisation.
            base_euler = gs_rand_float(-0.1, 0.1, (len(envs_idx), 3), self.device)
            base_euler[:, 2] = gs_rand_float(0.0, 3.14, (len(envs_idx),), self.device)
        self.base_quat[envs_idx] = gs_quat_mul(gs_euler2quat(base_euler), self.base_quat[envs_idx])
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.zero_all_dofs_velocity(envs_idx)

        inv_base_quat = gs_inv_quat(self.base_quat)
        self.projected_gravity = gs_transform_by_quat(self.global_gravity, inv_base_quat)

        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0.0
        # Random initial base velocity (TienKung-Lab-style: small perturbation
        # at reset so the policy learns to recover from a non-stationary
        # starting state). Disabled by default — only kick in for envs
        # that opt in via ``randomize_reset_velocity=True``.
        if self._env_cfg.get("randomize_reset_velocity", False):
            vel_range = self._env_cfg.get("reset_velocity_range", 0.5)
            self.base_lin_vel[envs_idx, 0] = gs_rand_float(
                -vel_range, vel_range, (len(envs_idx),), self.device
            )
            self.base_lin_vel[envs_idx, 1] = gs_rand_float(
                -vel_range, vel_range, (len(envs_idx),), self.device
            )
        base_vel = torch.concat([self.base_lin_vel[envs_idx], self.base_ang_vel[envs_idx]], dim=1)
        self.robot.set_dofs_velocity(velocity=base_vel, dofs_idx_local=[0, 1, 2, 3, 4, 5], envs_idx=envs_idx)

        self._resample_commands(envs_idx)

        self.obs_history_buf[envs_idx] = 0.0
        self.actions[envs_idx] = 0.0
        self.last_actions[envs_idx] = 0.0
        self.last_last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.feet_air_time[envs_idx] = 0.0
        self.feet_max_height[envs_idx] = 0.0
        if hasattr(self, "feet_contact_time"):
            self.feet_contact_time[envs_idx] = 0.0
        self.last_contacts[envs_idx] = 0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = 1

        if self.height_scan_buf is not None:
            self.height_scan_buf[envs_idx] = 0.0

        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.max_episode_length_s
            )
            self.episode_sums[key][envs_idx] = 0.0

        if self._env_cfg.get("send_timeouts", True):
            self.extras["time_outs"] = self.time_out_buf
        self.time_out_buf[envs_idx] = 0

    # ------------------------------------------------------------------
    # Domain randomization
    # ------------------------------------------------------------------

    def _randomize_rigids(self, env_ids=None):
        if self.eval:
            return
        if env_ids is None:
            env_ids = torch.arange(0, self.num_envs)
        elif len(env_ids) == 0:
            return
        if self._env_cfg.get("randomize_friction", True):
            self._randomize_link_friction(env_ids)
        if self._env_cfg.get("randomize_base_mass", True):
            self._randomize_base_mass(env_ids)
        if self._env_cfg.get("randomize_com_displacement", True):
            self._randomize_com_displacement(env_ids)

    def _randomize_controls(self, env_ids=None):
        if self.eval:
            return
        if env_ids is None:
            env_ids = torch.arange(0, self.num_envs)
        elif len(env_ids) == 0:
            return
        if self._env_cfg.get("randomize_motor_strength", False):
            self._randomize_motor_strength(env_ids)
        if self._env_cfg.get("randomize_motor_offset", False):
            self._randomize_motor_offset(env_ids)
        if self._env_cfg.get("randomize_kp_scale", False):
            self._randomize_kp(env_ids)
        if self._env_cfg.get("randomize_kd_scale", False):
            self._randomize_kd(env_ids)

    def _randomize_link_friction(self, env_ids):
        min_friction, max_friction = self._env_cfg["friction_range"]
        solver = self.rigid_solver
        ratios = gs.rand((len(env_ids), 1), dtype=float).repeat(1, solver.n_geoms) * (max_friction - min_friction) + min_friction
        solver.set_geoms_friction_ratio(ratios, torch.arange(0, solver.n_geoms), env_ids)

    def _randomize_base_mass(self, env_ids):
        min_mass, max_mass = self._env_cfg["added_mass_range"]
        base_link_id = 1
        added_mass = gs.rand((len(env_ids), 1), dtype=float) * (max_mass - min_mass) + min_mass
        self.rigid_solver.set_links_mass_shift(added_mass, [base_link_id], env_ids)

    def _randomize_com_displacement(self, env_ids):
        min_displacement, max_displacement = self._env_cfg["com_displacement_range"]
        base_link_id = 1
        com_displacement = (
            gs.rand((len(env_ids), 1, 3), dtype=float) * (max_displacement - min_displacement) + min_displacement
        )
        self.rigid_solver.set_links_COM_shift(com_displacement, [base_link_id], env_ids)

    def _randomize_motor_strength(self, env_ids):
        min_strength, max_strength = self._env_cfg["motor_strength_range"]
        self.motor_strengths[env_ids, :] = (
            gs.rand((len(env_ids), 1), dtype=float) * (max_strength - min_strength) + min_strength
        )

    def _randomize_motor_offset(self, env_ids):
        min_offset, max_offset = self._env_cfg["motor_offset_range"]
        self.motor_offsets[env_ids, :] = (
            gs.rand((len(env_ids), self.num_dof), dtype=float) * (max_offset - min_offset) + min_offset
        )

    def _randomize_kp(self, env_ids):
        min_scale, max_scale = self._env_cfg["kp_scale_range"]
        kp_scales = gs.rand((len(env_ids), self.num_dof), dtype=float) * (max_scale - min_scale) + min_scale
        self.batched_p_gains[env_ids, :] = kp_scales * self.p_gains[None, :]

    def _randomize_kd(self, env_ids):
        min_scale, max_scale = self._env_cfg["kd_scale_range"]
        kd_scales = gs.rand((len(env_ids), self.num_dof), dtype=float) * (max_scale - min_scale) + min_scale
        self.batched_d_gains[env_ids, :] = kd_scales * self.d_gains[None, :]

    # ------------------------------------------------------------------
    # Default reward fallbacks (used when subclass doesn't override)
    # ------------------------------------------------------------------

    def _reward_tracking_lin_vel(self):
        lin_vel_error = torch.sum(torch.square(self.commands_body[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self._reward_cfg.get("tracking_sigma", 0.25))

    def _reward_tracking_ang_vel(self):
        ang_vel_error = torch.square(self.commands_body[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self._reward_cfg.get("tracking_sigma", 0.25))

    def _reward_upright(self):
        return torch.exp(-torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1) / 0.2)

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_termination(self):
        return self.reset_buf * ~self.time_out_buf

    def _reward_feet_slip(self):
        in_contact = self.link_contact_forces[:, self.feet_link_indices, 2] > 1.0
        foot_vel_xy = self.foot_velocities[:, :, :2]
        slip = torch.sum(torch.square(torch.norm(foot_vel_xy, dim=-1)) * in_contact.float(), dim=1)
        cmd_norm = torch.norm(self.commands[:, :2], dim=1)
        return slip * (cmd_norm > 0.1).float()

    def _reward_feet_air_time(self):
        contact = self.link_contact_forces[:, self.feet_link_indices, 2] > 1.0
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.0) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1)
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1
        self.feet_air_time *= ~contact_filt
        return rew_airTime

    # ------------------------------------------------------------------
    # Debug visualization
    # ------------------------------------------------------------------

    def _draw_debug_vis(self):
        self.scene.clear_debug_objects()
        foot_poss = self.foot_positions[0].reshape(-1, 3)
        foot_poss = foot_poss.cpu()
        if len(foot_poss) >= 4:
            self.scene.draw_debug_line(foot_poss[0], foot_poss[3], radius=0.002, color=(1, 0, 0, 0.7))
            self.scene.draw_debug_line(foot_poss[1], foot_poss[2], radius=0.002, color=(1, 0, 0, 0.7))
        com = self.com[0]
        com[2] = 0.02 + self.terrain_heights[0]
        self.scene.draw_debug_sphere(pos=com, radius=0.02, color=(0, 0, 1, 0.7))

    def _set_camera(self):
        self._floating_camera = self.scene.add_camera(
            pos=np.array([0, -1, 1]), lookat=np.array([0, 0, 0]), fov=40, GUI=False
        )
        self._recording = False
        self._recorded_frames = []

    def render(self):
        robot_pos = np.array(self.base_pos[0].cpu())
        self._floating_camera.set_pose(
            pos=robot_pos + np.array([-1, -1, 0.5]), lookat=robot_pos + np.array([0, 0, -0.1])
        )
        frame, _, _, _ = self._floating_camera.render()
        return frame
