# Copyright (c) 2025, Your Name
# All rights reserved.

"""Canonical Go2 all-in-one environment (state + height scan + full MDP).

This is the *single* Go2 environment the user is expected to use by default.
It unifies the mjlab-style VecEnv contract with the isaac-lab-style MDP
vocabulary from ``envs.genesis.mdp`` and gives the policy **all** the
observations it might need:

* Proprioception (base linear / angular velocity, projected gravity,
  velocity commands, joint positions / velocities, last actions).
* **Height scan** (ray-cast grid over the terrain) — relative heights
  around the robot, so the policy can decide whether to walk on the
  flat ground or lift its legs over an obstacle.
* Optional *vision* off (kept simple by default; the dedicated
  :class:`envs.genesis.go2_vision.Go2VisionEnv` enables it).

It is also wired up with the standard suite of MDP components:

* **Commands**: ``UniformLevelVelocityCommand``-style resampling of
  ``(lin_vel_x, lin_vel_y, ang_vel_z)`` with the standard
  "norm > 0.2" filter and the optional "stand-still" environment
  fraction. Curriculum support is included via
  ``lin_vel_cmd_levels`` / ``ang_vel_cmd_levels``.
* **Events**: full domain randomization — friction, base mass, COM
  displacement, motor strength / offset, PD gain scaling.
* **Terminations**: time-out, base contact, bad orientation,
  optional terrain-out-of-bounds.
* **Rewards**: built dynamically from
  :py:mod:`envs.genesis.mdp.rewards` so the reward formulation stays
  declarative and can be tweaked per env.

Subclasses (e.g. :class:`envs.genesis.go2_rough.Go2RoughEnv`,
:class:`envs.genesis.go2_walk_stairs.Go2WalkStairsEnv`, …) only override
:py:meth:`_default_configs` to specialize the env for their terrain /
task; they do **not** redefine the obs / reward plumbing.
"""

from __future__ import annotations

import torch

from envs.genesis.go2_base import Go2BaseEnv, wrap_to_pi


class Go2WalkEnv(Go2BaseEnv):
    """Canonical all-in-one Go2 environment.

    State observations:
        * base lin/ang vel (6,)
        * projected gravity (3,)
        * velocity commands body frame (3,)
        * joint pos relative (12,)
        * joint vel (12,)
        * last actions (12,)

    Optional height scan (11x11 by default, 1.0m × 0.8m grid in front
    of the robot) — relative heights around the base, so the policy
    can lift its legs to walk over uneven ground.

    Commands are sampled uniformly; ``lin_vel`` and ``ang_vel`` ranges
    are expanded by the curriculum term when the policy achieves >80%
    of the configured reward weight.
    """

    def __init__(self, num_envs: int, show_viewer: bool = False, eval_mode: bool = False) -> None:
        super().__init__(num_envs=num_envs, show_viewer=show_viewer, eval_mode=eval_mode)
        self.name = "go2_walk"

    @classmethod
    def _default_configs(cls):
        # ----------------- env / robot -----------------
        env_cfg = {
            "urdf_path": "urdf/go2/urdf/go2.urdf",
            "links_to_keep": ["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
            "num_actions": 12,
            "num_dofs": 12,
            "default_joint_angles": {
                "FL_hip_joint": 0.1, "FR_hip_joint": -0.1, "RL_hip_joint": 0.1, "RR_hip_joint": -0.1,
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
            "PD_stiffness": {"joint": 25.0},
            "PD_damping": {"joint": 0.5},
            "use_implicit_controller": False,
            "termination_if_roll_greater_than": 0.8,
            "termination_if_pitch_greater_than": 0.8,
            "use_contact_termination": True,
            "base_init_pos": [0.0, 0.0, 0.4],
            "base_init_quat": [1.0, 0.0, 0.0, 0.0],
            "episode_length_s": 20.0,
            "resampling_time_s": 10.0,
            "command_type": "ang_vel_yaw",
            "action_scale": 0.25,
            "action_latency": 0.02,
            "action_range": 3.0,
            "send_timeouts": True,
            "control_freq": 50,
            "decimation": 4,
            "feet_geom_offset": 1,
            "coupling": False,
            # ----------------- terrain -----------------
            "use_terrain": False,
            "terrain_cfg": None,
            # ----------------- obs extras -----------------
            "use_height_scan": True,
            "height_scan_cfg": {
                "resolution": 0.1,
                "size_x": 1.0,
                "size_y": 0.8,
            },
            "use_vision": False,
            "vision_cfg": {},
            # ----------------- randomization -----------------
            "randomize_friction": True,
            "friction_range": [0.3, 1.2],
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
            # ----------------- events (interval) -----------------
            "push_interval_s": 8.0,
            "push_velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
            # ----------------- curriculum -----------------
            "curriculum_terms": {
                # Optional curriculum that expands the linear-velocity
                # command range as the policy improves. Matches the
                # isaac-lab default (only ``lin_vel_cmd_levels``; no
                # ang_vel curriculum so the yaw tracking stays sharp).
                "lin_vel_cmd_levels": cls._lin_vel_cmd_levels,
            },
        }
        # ----------------- obs -----------------
        # 3 (base lin vel) + 3 (ang vel) + 3 (gravity) + 3 (commands) +
        # 12 (joint pos) + 12 (joint vel) + 12 (last_actions)
        # = 48 base obs; + 11 * 11 = 121 if height scan enabled.
        scan_cfg = env_cfg["height_scan_cfg"]
        n_x = int(round(scan_cfg["size_x"] / scan_cfg["resolution"])) + 1
        n_y = int(round(scan_cfg["size_y"] / scan_cfg["resolution"])) + 1
        scan_dim = n_x * n_y
        base_obs = 12 + 3 * env_cfg["num_dofs"]
        total_obs = base_obs + scan_dim
        obs_cfg = {
            "num_obs": total_obs,
            "num_history_obs": 5,
            "obs_noise": {"lin_vel": 0.0, "ang_vel": 0.2, "gravity": 0.05, "dof_pos": 0.01, "dof_vel": 1.5},
            "obs_scales": {
                "lin_vel": 2.0,
                "ang_vel": 0.2,
                "dof_pos": 1.0,
                "dof_vel": 0.05,
                "height_scan": 1.0,
            },
            # Match the actual ``privileged_obs_buf`` produced in
            # ``Go2BaseEnv.compute_observations``:
            # ``[obs_buf (num_obs), base_lin_vel (3), last_actions (num_dofs)]``.
            "num_priv_obs": base_obs + scan_dim + 3 + env_cfg["num_dofs"],
        }
        # ----------------- rewards -----------------
        # The reward ids are looked up dynamically in envs.genesis.mdp.rewards
        # by Go2BaseEnv._dispatch_mdp_reward. The vocabulary is therefore
        # the mdp.rewards module's exports (track_lin_vel_xy, lin_vel_z_l2,
        # joint_torques_l2, …). The custom *_velocity / facing_direction /
        # feet_* terms that aren't in mdp.rewards are implemented as
        # methods on the env below.
        reward_cfg = {
            "tracking_sigma": 0.25,
            "soft_dof_pos_limit": 0.9,
            "reward_scales": {
                # -- task
                "track_lin_vel_xy": 1.5,
                "track_ang_vel_z": 0.75,
                "facing_direction": 1.0,
                # -- base penalties
                "lin_vel_z_l2": -2.0,
                "ang_vel_xy_l2": -0.05,
                "joint_vel_l2": -0.001,
                "joint_acc_l2": -2.5e-7,
                "joint_torques_l2": -2.0e-4,
                "action_rate_l2": -0.1,
                "joint_pos_limits": -10.0,
                "energy": -2.0e-5,
                "flat_orientation_l2": -2.5,
                "joint_position_penalty": -0.7,
                # -- feet
                "feet_air_time": 0.1,
                "air_time_variance": -1.0,
                "feet_slide": -0.1,
                # -- other
                "undesired_contacts": -1.0,
                # -- fall
                "termination": -2.0,
            },
        }
        # ----------------- commands -----------------
        command_cfg = {
            "num_commands": 3,
            "lin_vel_x_range": [-0.1, 0.1],
            "lin_vel_y_range": [-0.1, 0.1],
            "ang_vel_range": [-1.0, 1.0],
            # Curriculum upper-bounds (the curriculum term widens the
            # sampling ranges when the policy is succeeding).
            "lin_vel_x_limit": [-1.0, 1.0],
            "lin_vel_y_limit": [-0.4, 0.4],
            "ang_vel_limit": [-1.0, 1.0],
            # rel_standing_envs: a fraction of envs are forced to stand
            # still on each resample. Mirrors isaac-lab's default (0.1).
            "rel_standing_envs": 0.1,
        }
        return env_cfg, obs_cfg, reward_cfg, command_cfg

    # ------------------------------------------------------------------
    # Custom rewards (looked up before the mdp dispatch in
    # Go2BaseEnv._prepare_reward_function).
    # ------------------------------------------------------------------

    def _reward_facing_direction(self):
        """Reward body x-axis aligning with world-frame command direction."""
        cmd_dir = torch.atan2(self.commands[:, 1], self.commands[:, 0])
        body_dir = self.base_euler[:, 2]
        dir_error = torch.abs(wrap_to_pi(cmd_dir - body_dir))
        mask = (torch.norm(self.commands[:, :2], dim=1) > 0.1).float()
        return torch.exp(-dir_error / self._reward_cfg["tracking_sigma"]) * mask

    def _reward_upright(self):
        """Exponential reward for staying upright (mjlab style)."""
        return torch.exp(-torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1) / 0.2)

    def _reward_feet_slip(self):
        """Penalize foot sliding (xy velocity while in contact)."""
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

    def _reward_termination(self):
        return self.reset_buf * ~self.time_out_buf

    # ------------------------------------------------------------------
    # Curriculum helpers (called from Go2BaseEnv._apply_curriculums).
    # Bound as plain methods so they can be referenced from env_cfg
    # without becoming reward functions.
    # ------------------------------------------------------------------

    @staticmethod
    def _lin_vel_cmd_levels(env) -> None:
        """Linearly expand ``lin_vel_x`` / ``lin_vel_y`` ranges when the
        ``track_lin_vel_xy`` reward is performing above 80% of its target."""
        from envs.genesis.mdp import curriculums as mdp_curr

        env_ids = torch.arange(env.num_envs, device=env.device)
        mdp_curr.lin_vel_cmd_levels(
            env,
            env_ids,
            reward_term_name="track_lin_vel_xy",
            delta_range=(0.1, 0.1),
        )

    # ------------------------------------------------------------------
    # Optional command resampling override (stand-still fraction).
    # ------------------------------------------------------------------

    def _resample_commands(self, envs_idx):
        """Override the default resample so we can also enforce the
        ``rel_standing_envs`` fraction on the resampled envs (matching
        isaac-lab's behavior)."""
        from envs.genesis.mdp import commands as mdp_cmd
        from envs.genesis.mdp.math_utils import rand_float

        if len(envs_idx) == 0:
            return
        rel_standing = float(self._command_cfg.get("rel_standing_envs", 0.0))
        if rel_standing > 0.0:
            mdp_cmd.sample_commands(
                self,
                env_ids=envs_idx,
                rel_standing_envs=rel_standing,
            )
        else:
            # Fall back to the original resample (so the parent class'
            # command_type "heading" override still works for subclasses).
            super()._resample_commands(envs_idx)


def get_env(num_envs: int, eval_mode: bool = False) -> Go2WalkEnv:
    try:
        import genesis as gs
        gs.init(logging_level="warning")
    except Exception:
        pass
    return Go2WalkEnv(num_envs=num_envs, eval_mode=eval_mode)
