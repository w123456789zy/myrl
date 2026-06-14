# Copyright (c) 2025, Your Name
# All rights reserved.

"""Go2 walk on rough (multi-terrain) terrain.

Subclass of the canonical :class:`Go2WalkEnv`. Inherits the full state
+ height scan + MDP machinery and only overrides the env config to
swap the flat plane for a procedurally generated terrain.

The terrain layout is a **5x5 grid** (6 m x 6 m per cell, 30 m x 30 m
total) of sub-terrains.  Every cardinal direction from the centre
leads to a different obstacle family, so the dog always hits
something interesting no matter which way it walks.

Layout::

    row 0  f   f   f   f   f    <- outermost flat margin
    row 1  f  rU  oS  sT  gP    <- NORTH perimeter (random bumps,
    row 2  f  oS   f  pT   f       discrete obs, stones, gap)
    row 3  f  gP  sU  sT  oS    <- CENTRE (flat + box bumps/pit)
    row 4  f   f  sD   f   f    <- SOUTH perimeter (gap, stairs-up+plateau,
                                     stones, discrete obs)
                                     <- stairs-down + flat run-out)

Legend: ``f``=flat, ``rU``=random_uniform, ``oS``=discrete_obstacles,
        ``sT``=stepping_stones, ``gP``=gap, ``pT``=pit,
        ``sU``=stairs_up (+y, 15 steps → 1.5m + 3m plateau),
        ``sD``=stairs_down (+y, 15 steps → 0m + 3m flat run-out).

Key design decisions:

* **Centre cell (2,2) is flat** — the dog always has stable ground
  to stand on.  Cells (2,1) and (2,3) add box-shaped bumps and a
  pit — "中间是平地+长方体的凹凸".
* **Staircase on the SOUTH column** (row 3+4, col 2), forming a
  complete **up → plateau → down → flat** sequence:
  - (3,2): 15 steps up (0→1.5m) via ``stairs_terrain_y``, then 3m
    flat plateau at 1.5m.
  - (4,2): 15 steps down (1.5m→0m) via ``down_stairs_terrain_y``,
    then 3m flat run-out at 0m so Go2 never "falls off" the stairs.
  Go2 walks +y from centre, enters stairs at h=0 (no wall), climbs,
  crosses the plateau, descends, and has a generous flat run-out.
* **North, east and west perimeters** each carry a distinct obstacle
  family, so the dog encounters variety in every direction.
* **Spawn at the grid centre** (15 m, 15 m) with moderate jitter,
  on the flat cell (2,2).  All four cardinal directions are
  reachable within a few seconds of walking at 1-2 m/s.

The heightfield is built locally by
:func:`envs.genesis.terrain.build_go2_rough_layout` and passed
directly to ``gs.morphs.Terrain(height_field=hf)`` — **no Genesis
source code is patched**.
"""

from __future__ import annotations

import numpy as np
import torch

from envs.genesis.go2_walk import Go2WalkEnv
from envs.genesis.mdp.math_utils import wrap_to_pi
from envs.genesis.terrain import build_go2_rough_layout


class Go2RoughEnv(Go2WalkEnv):
    """Go2 walk on procedurally generated rough terrain."""

    def __init__(self, num_envs: int, show_viewer: bool = False, eval_mode: bool = False) -> None:
        super().__init__(num_envs=num_envs, show_viewer=show_viewer, eval_mode=eval_mode)
        self.name = "go2_rough"

    @classmethod
    def _default_configs(cls):
        env_cfg, obs_cfg, reward_cfg, command_cfg = super()._default_configs()

        # ----------------- terrain -----------------
        env_cfg["use_terrain"] = True

        # 5x5 subterrains (6 m each -> 30 m x 30 m total).
        #
        # Layout (see module docstring for the full table):
        #   row 0  f  f   f   f   f     outermost flat margin
        #   row 1  f  rU  oS  sT  gP    NORTH perimeter
        #   row 2  f  oS  f   pT  f     CENTRE (flat + box bumps/pit)
        #   row 3  f  gP  sU  sT  oS    SOUTH (stairs-up+plateau, 1.5m)
        #   row 4  f  f   sD  f   f     SOUTH (stairs-down+flat run-out)
        #
        # Staircase cells (3,2)+(4,2) form up→plateau→down→flat:
        #   15 steps × 0.20m wide × 0.10m high → 3m ramp, 1.5m rise.
        #   stairs_up: first 3m = stairs (0→1.5m), last 3m = plateau (1.5m).
        #   stairs_down: first 3m = stairs (1.5m→0m), last 3m = flat (0m).
        _STEP_HEIGHT = 0.10
        _STEP_WIDTH = 0.20
        _NUM_STEPS_TRAIN = 15
        _TOP_HEIGHT = _STEP_HEIGHT * _NUM_STEPS_TRAIN  # = 1.50 m
        _N_SUBTERRAINS = (5, 5)
        _SUBTERRAIN_SIZE = (6.0, 6.0)
        _HORIZONTAL_SCALE = 0.1
        _VERTICAL_SCALE = 0.005

        # Build the 5x5 heightfield locally — no Genesis source
        # patches needed (see envs/genesis/terrain/).
        heightfield, subterrain_parameters, difficulty_map = build_go2_rough_layout(
            n_subterrains=_N_SUBTERRAINS,
            subterrain_size=_SUBTERRAIN_SIZE,
            horizontal_scale=_HORIZONTAL_SCALE,
            vertical_scale=_VERTICAL_SCALE,
            step_width=_STEP_WIDTH,
            step_height=_STEP_HEIGHT,
            num_steps=_NUM_STEPS_TRAIN,
        )

        env_cfg["terrain_cfg"] = {
            "n_subterrains": _N_SUBTERRAINS,
            "horizontal_scale": _HORIZONTAL_SCALE,
            "vertical_scale": _VERTICAL_SCALE,
            "subterrain_size": _SUBTERRAIN_SIZE,
            # Pre-computed heightfield — Genesis skips its own
            # subterrain_types parser and rasterises this directly.
            "height_field": heightfield,
            # Per-cell difficulty rating in [1.0, 5.0].  Stored as
            # a torch tensor on the env so the forward-progress
            # reward can weight each env's displacement by the
            # difficulty of the cell it is currently walking on.
            "difficulty_map": difficulty_map,
            # ``subterrain_parameters`` is only used by Genesis' built-in
            # subterrain_types parser, which we bypass with the
            # pre-computed heightfield above.  We keep the dict around
            # for debugging / future reference, but it's harmless to
            # leave it in the config — go2_base only forwards it
            # when no pre-computed heightfield is provided.
            "subterrain_parameters": subterrain_parameters,
        }

        # Spawn at the centre of the 30m x 30m grid (15, 15) — this
        # is the flat cell (2,2).  With +-3 m jitter Go2 stays in
        # the centre area and can walk any cardinal direction to
        # encounter a different obstacle family.
        env_cfg["base_init_pos"] = [15.0, 15.0, 0.42]
        # Spawn jitter limited to **the centre flat cell** (cell
        # (2, 2) is at world [12, 18] × [12, 18]).  We restrict the
        # sampling to ±3 m around (15, 15) so the policy is always
        # born on the centre flat cell and must *walk outward* to
        # reach a perimeter obstacle.  This avoids the
        # ``difficulty_biased_spawn`` failure mode where the policy
        # kept falling on the very first step of every episode (it
        # was being teleported straight onto the stairs without ever
        # learning to walk on flat ground first).
        env_cfg["base_init_pos_sampling_range"] = [-3.0, 3.0]
        # ``difficulty_biased_spawn`` is explicitly **disabled** —
        # every reset goes back to the centre cell.  The
        # ``forward_progress`` reward (with the soft 2.0× difficulty
        # cap) still provides the "go to the hard cells" incentive;
        # we just don't force the policy to start *on* a hard cell.
        env_cfg["difficulty_biased_spawn"] = False
        env_cfg["base_init_quat"] = [1.0, 0.0, 0.0, 0.0]
        # Match leggedskill's 20 s episode horizon. Long episodes (40 s)
        # dilute per-step reward signal and slow early PPO learning when
        # the policy is still essentially random.
        env_cfg["episode_length_s"] = 20.0
        # Heading mode: the robot is given a *target heading angle*
        # (commands[:, 3]) and must turn to face it. The angular-velocity
        # command (commands[:, 2]) is computed automatically in
        # ``post_physics_step`` as the heading-tracking error.
        # Example: commands = [1.0, 0.0, ?, 0.0]  ->  walk forward at 1 m/s
        #          while facing the +x axis (heading = 0).
        env_cfg["command_type"] = "heading"
        # Heading mode needs 4 command slots: [vx, vy, omega_z, target_heading]
        command_cfg["num_commands"] = 4
        # 0% standing envs — competition style forces every env to walk.
        # Standing envs on rough terrain become a local minimum where
        # the policy learns "don't move = don't fall = high reward".
        command_cfg["rel_standing_envs"] = 0.0

        # ----------------- obs extras -----------------
        # Bigger, denser height scan for rough terrain — matches the
        # unitree_rl_lab defaults (1.6m x 1.0m x 0.1m grid = 17x11 = 187
        # points). The 1.0m x 0.8m scan inherited from go2_walk is fine
        # for flat ground but doesn't see far enough ahead for stairs /
        # obstacles. With 0.1m resolution we still detect 10cm steps.
        env_cfg["use_height_scan"] = True
        env_cfg["height_scan_cfg"] = {
            "resolution": 0.1,
            "size_x": 1.6,
            "size_y": 1.0,
        }
        # CRITICAL: ``obs_cfg`` is computed from the *parent*'s
        # ``height_scan_cfg`` (1.0m x 0.8m -> 11x9 = 99 points). Since we
        # just overrode the scan to 1.6m x 1.0m (17x11 = 187 points),
        # we must also recompute ``num_obs`` / ``num_priv_obs`` so the
        # obs buffer matches the actual scan dimension. Otherwise
        # ``compute_observations`` either overwrites scan values with
        # proprioception (if scan dim > obs dim) or crashes on the
        # concatenation (if obs dim > scan dim).
        scan_cfg = env_cfg["height_scan_cfg"]
        n_x = int(round(scan_cfg["size_x"] / scan_cfg["resolution"])) + 1
        n_y = int(round(scan_cfg["size_y"] / scan_cfg["resolution"])) + 1
        scan_dim = n_x * n_y
        base_obs = 12 + 3 * env_cfg["num_dofs"]   # 48
        obs_cfg["num_obs"] = base_obs + scan_dim  # 48 + 187 = 235
        # CRITICAL: ``compute_observations`` actually concatenates
        # ``[obs_buf, base_lin_vel*scale, last_actions]`` into
        # ``privileged_obs_buf``, so its true dim is
        # ``num_obs + 3 (lin_vel) + num_dofs (last_actions)`` = 250.
        # If we set ``num_priv_obs`` smaller the buffer is allocated to
        # the wrong size and downstream code (MoE-PPO critic / rollout
        # storage) sees a shape mismatch.
        obs_cfg["num_priv_obs"] = obs_cfg["num_obs"] + 3 + env_cfg["num_dofs"]

        # Termination: only base contact and boundary checks; angle-based
        # termination is DISABLED (competition style).  Climbing stairs
        # inevitably tilts the body beyond 45°, and terminating on angle
        # teaches the policy to avoid the tilt → avoid stairs.
        env_cfg["termination_if_roll_greater_than"] = 999.0
        env_cfg["termination_if_pitch_greater_than"] = 999.0
        env_cfg["use_contact_termination"] = False

        # Core task rewards
        # track_lin_vel_xy: raised from 1.5 to 2.0 to give a stronger
        # "go forward" gradient.  The previous weight on rough terrain
        # was just barely out-weighing the posture penalties, so the
        # policy converged to "stand still to avoid falling" on the
        # uneven cells.  With 2.0 the velocity term clearly dominates
        # and the policy has a real incentive to keep moving.
        reward_cfg["reward_scales"]["track_lin_vel_xy"] = 2.0
        # feet_air_time: match competition-style 1.0 to strongly
        # encourage high leg lift over obstacles.
        reward_cfg["reward_scales"]["feet_air_time"] = 1.0
        # Termination penalty set to 0 (competition style).  A negative
        # death penalty made the policy afraid to explore hard terrain
        # because the risk/reward ratio was too unfavourable.
        reward_cfg["reward_scales"]["termination"] = 0.0
        # Flat orientation penalty reduced to -0.1 (competition style).
        # Climbing stairs and stepping over obstacles requires body tilt;
        # a heavy penalty directly conflicts with rough-terrain traversal.
        reward_cfg["reward_scales"]["flat_orientation_l2"] = -0.1
        # Feet stumble penalty disabled (competition style: -0.0).
        # Feet inevitably bump stair risers; penalising it teaches the
        # policy to keep feet away from obstacles entirely.
        reward_cfg["reward_scales"]["feet_stumble"] = 0.0

        # DreamWaQ-style smoothness penalties
        # base_height disabled (competition style: -0.0).  Penalising a
        # low base on stairs directly conflicts with the natural crouch
        # posture during ascent.
        reward_cfg["reward_scales"]["base_height"] = 0.0
        reward_cfg["reward_scales"]["joint_acc_l2"] = -2.5e-7
        reward_cfg["reward_scales"]["action_rate_l2"] = -0.01
        reward_cfg["reward_scales"]["energy"] = -2e-5
        reward_cfg["reward_scales"]["smoothness"] = -0.01

        # Competition-style stand-still penalty: when the robot is
        # commanded to move but stays nearly stationary, apply a
        # penalty.  This prevents the policy from falling into a
        # "stay still to avoid obstacles" local minimum.
        reward_cfg["reward_scales"]["stand_still"] = -0.1

        # Height-aware reward: penalize the robot's body being too low
        # (encourages lifting legs over obstacles). Implemented in
        # ``Go2RoughEnv._reward_terrain_base_height``; activated by
        # adding it to ``reward_scales``. Halved from 1.0 to 0.5 so it
        # supports (rather than dominates) the rest of the reward mix.
        reward_cfg["reward_scales"]["terrain_base_height"] = 0.5

        # Forward-progress reward: linear m/s bonus for moving in the
        # commanded forward direction.  Complements the exponential
        # track_lin_vel_xy reward by providing a strictly-monotonic
        # signal — the policy cannot get stuck in a "drift at 0.1 m/s"
        # local minimum because each metre of progress adds the same
        # bonus.
        #
        # Weight reduced from 0.3 → 0.10: with the per-cell difficulty
        # multiplier on top (1.0–4.5×), the *effective* weight on
        # stairs reached 0.3 × 4.5 = 1.35 per metre, which
        # over-amplified negative feedback on early-episode failures
        # and pushed the policy into a "tuck-and-fall" local minimum
        # (value loss 0.1 → 5.0, reward 40 → -7).  0.10 keeps the
        # "go forward" gradient clear without dominating the suite of
        # other terms.
        reward_cfg["reward_scales"]["forward_progress"] = 0.10

        # Competition-style: clip negative per-step rewards to zero so the
        # policy is NOT penalised for imperfect posture on hard terrain.
        # This eliminates the "avoid obstacles to skip posture penalties"
        # gradient that was causing the policy to hesitate.
        env_cfg["only_positive_rewards"] = True

        # NOTE: terrain curriculum is disabled for now because the current
        # mylab curriculum API expects callable functions, not dict configs.
        # Re-enable once a Genesis-compatible terrain curriculum is implemented.
        env_cfg["curriculum_terms"] = {}

        # ---- domain randomisation (competition-style: minimal) ----
        # Competition disables all domain randomisation — it slows early
        # convergence and adds noise that confuses the policy on hard
        # terrain.  We'll re-enable selective items once the policy
        # can reliably traverse stairs (~500-1000 iterations).
        env_cfg["friction_range"] = [0.5, 1.0]
        env_cfg["randomize_friction"] = False
        env_cfg["com_displacement_range"] = [-0.1, 0.1]
        env_cfg["randomize_com_displacement"] = False
        env_cfg["randomize_motor_strength"] = False
        env_cfg["motor_strength_range"] = [0.9, 1.1]
        env_cfg["randomize_base_mass"] = False
        env_cfg["added_mass_range"] = [-1.0, 2.0]
        env_cfg["randomize_kp_scale"] = False
        env_cfg["kp_scale_range"] = [0.9, 1.1]
        env_cfg["randomize_kd_scale"] = False
        env_cfg["kd_scale_range"] = [0.9, 1.1]
        # Disable push and reset-velocity perturbations.
        env_cfg["push_velocity_range"] = None
        env_cfg["randomize_reset_velocity"] = False
        env_cfg["reset_velocity_range"] = 0.5

        # ---- vision ----
        # The rough terrain is exactly where vision is most useful
        # (stair detection, obstacle avoidance, …). Enabling it by
        # default means ``--env go2-rough`` works with **both** ``ppo``
        # and ``vision_ppo``: the former ignores the image and trains
        # state-only, the latter uses it. Camera render cost is
        # negligible (64x64 RGB, GPU-accelerated).
        env_cfg["use_vision"] = True
        env_cfg["vision_cfg"] = {
            "res": (96, 96),               # 必须与训练时一致 (CNN 输入)
            "fov": 100.0,                  # 训练时一致 — 改 FOV 会让 policy 困惑
            "offset": (0.3, 0.0, 0.25),    # 相机高度，俯瞰更多地面
            # 1m 远、向下 0.5m —— 相机焦点落在 Go2 即将踩到的位置，
            # 而不是远处的 cell 边界。这样当 Go2 在中心平地时画面
            # 是均一的灰色，**走出中心进入障碍格后**障碍物才会
            # 逐渐出现在画面里，便于观察深度区分。
            #
            # 注意：lookat_offset 训练时是 (2.0, 0.0, -0.4)，现在
            # 改成 (1.0, 0.0, -0.5) 会让画面视角与训练时略有差异，
            # policy 输出会轻微偏离训练分布。如果想让 policy 行为
            # 与训练完全一致，**需要用新的 lookat_offset 重新训练**。
            "lookat_offset": (1.0, 0.0, -0.5),
        }
        # The camera provides rich temporal info; reinforce with 5-step
        # history stacking so the MoE encoder can pick up velocity
        # transitions and contact sequences that single-step obs misses.
        # 5 steps is the leggedskill himloco default — enough for the
        # gait period of a trot (~0.4s x 4 steps ~ 5 frames at 50 Hz).
        obs_cfg["num_history_obs"] = 5

        return env_cfg, obs_cfg, reward_cfg, command_cfg

    def _resample_commands(self, envs_idx):
        """Heading-mode commands for rough terrain.

        The robot is asked to walk forward (vx element [0.3, 0.8] m/s) while
        facing a *random target heading* (commands[:, 3] element [-pi, pi]).
        The angular-velocity command (commands[:, 2]) is computed
        automatically in ``post_physics_step`` as the heading-tracking
        error, so we only need to sample vx and target_heading here.

        Note: the **lower bound is 0.3 m/s** (not 0.0) — a vx of zero
        would be a degenerate "stand still" command that the policy
        could satisfy by simply not moving, which on rough terrain
        becomes a local minimum.  Forcing every active env to walk
        forward at 0.3+ m/s eliminates that failure mode.
        """
        if len(envs_idx) == 0:
            return
        # Speed range 0.5 – 1.0 m/s (competition style: 0.5–2.0).
        # A minimum of 0.5 m/s ensures the robot is always commanded
        # to walk forward and cannot "stand still" to avoid obstacles.
        # Cap at 1.0 m/s instead of 2.0 because Go2's max trot speed
        # on rough terrain is lower than the competition benchmark.
        self.commands[envs_idx, 0] = torch.empty(len(envs_idx), device=self.device).uniform_(0.5, 1.0)
        self.commands[envs_idx, 1] = 0.0
        # Target heading angle in radians.  0 = +x axis,  pi/2 = +y axis,
        # pi = -x axis,  -pi/2 = -y axis.  Uniform over the full circle so
        # the policy learns to turn in *any* direction.
        self.commands[envs_idx, 3] = torch.empty(len(envs_idx), device=self.device).uniform_(-3.14, 3.14)

    def _reward_facing_direction(self):
        """Reward body yaw aligning with ``commands[:, 3]`` (target heading).

        The parent class's version uses ``atan2(commands[:, 1], commands[:, 0])``
        which in heading-mode always computes 0 (because ``commands[:, 1]`` is
        always 0). This version reads the *actual* target heading from
        ``commands[:, 3]``.
        """
        target_heading = self.commands[:, 3]
        body_yaw = self.base_euler[:, 2]
        heading_error = torch.abs(wrap_to_pi(target_heading - body_yaw))
        mask = (self.commands[:, 0] > 0.05).float()  # only when speed > 0
        return torch.exp(-heading_error / self._reward_cfg["tracking_sigma"]) * mask

    def post_physics_step(self):
        """Hook: update the front-camera pose every step so it tracks the base."""
        super().post_physics_step()
        if self._use_vision and hasattr(self, "_front_camera"):
            base_pos_world = self.base_pos[0].cpu().numpy()
            base_quat_world = self.base_quat[0].cpu().numpy()

            def _rotate(v):
                w, x, y, z = base_quat_world
                tx = 2.0 * (y * v[2] - z * v[1])
                ty = 2.0 * (z * v[0] - x * v[2])
                tz = 2.0 * (x * v[1] - y * v[0])
                return v + w * np.array([tx, ty, tz]) + np.array(
                    [y * tz - z * ty, z * tx - x * tz, x * ty - y * tx]
                )

            # Both camera position and lookat are body-frame offsets
            # rotated into world frame so the camera stays fixed on
            # the robot's head and always looks forward.
            front_pos = base_pos_world + _rotate(np.asarray(self._vision_offset, dtype=np.float32))
            front_lookat = front_pos + _rotate(np.asarray(self._vision_lookat_offset, dtype=np.float32))
            self._front_camera.set_pose(pos=front_pos, lookat=front_lookat)

    def _reward_terrain_base_height(self):
        """Reward the base being at the right height for the terrain.

        For a Go2 (base_link z ~ 0.32m on flat ground), a sensible target
        is the *standing* height. On rough terrain (stairs / obstacles /
        ramps) the base should be **higher** so the legs can reach the
        uneven ground beneath; we therefore make the target height
        adaptive by adding a fraction of the maximum height observed in
        the height scan in front of the robot (so a step in front lifts
        the target height, which in turn encourages the controller to
        lift the legs / stand taller).

        Without the height-scan component the reward would be the same
        flat 0.32m target everywhere — which is exactly the failure
        mode the user observed: the robot learns to "stand low" on flat
        ground and then can't recover when stepping on an obstacle.
        """
        # Per-env max scan height in front of the robot. ``height_scan_buf``
        # stores the *relative* heights (scan_z - base_z - height_offset),
        # so to recover the absolute obstacle height we add the base z
        # back. We then convert to a target base height of
        # ``standing + 0.6 * max_obstacle_offset``.
        if not hasattr(self, "height_scan_buf") or self.height_scan_buf is None:
            scan_abs = torch.zeros(self.num_envs, device=self.device)
        else:
            scan_abs = self.height_scan_buf + self.base_pos[:, 2:3]
        obstacle_offset = scan_abs.max(dim=-1).values.clamp(min=0.0, max=0.25)
        target = 0.32 + 0.6 * obstacle_offset
        return torch.exp(-((self.base_pos[:, 2] - target) ** 2) / 0.02)

    # ------------------------------------------------------------------
    # Competition-style stand-still penalty
    # ------------------------------------------------------------------
    # When the robot is commanded to move at > 0.1 m/s but its actual
    # speed is < 0.05 m/s, this penalty fires.  It prevents the policy
    # from discovering a local minimum where "stay still" avoids both
    # posture penalties and obstacle collisions.
    # ------------------------------------------------------------------
    def _reward_stand_still(self):
        cmd_norm = torch.norm(self.commands[:, :2], dim=1)
        vel_norm = torch.norm(self.base_lin_vel[:, :2], dim=1)
        return (cmd_norm > 0.1).float() * (vel_norm < 0.05).float()

    # ------------------------------------------------------------------
    # Per-step forward-progress reward (NEW)
    # ------------------------------------------------------------------
    # Tracks the *signed* distance traveled in the commanded forward
    # direction since the last step, and rewards it linearly.  The
    # ``track_lin_vel_xy`` reward is exponential in the velocity error
    # and is therefore happy with any non-zero velocity (e.g. a slow
    # drift).  A linear "displacement" reward provides a strictly
    # *monotonic* signal: each meter of forward progress adds the same
    # reward, so the policy cannot get into a "creep along at 0.1 m/s"
    # local minimum.  Combined with the velocity reward, this drives
    # the policy to both *walk fast* and *walk far* every episode.
    #
    # Implementation: the projected displacement is computed as
    #   delta = (base_pos - last_base_pos) . forward_unit_vector
    # where forward_unit_vector = [cos(yaw), sin(yaw), 0].  This is the
    # ground-frame forward direction the robot is currently facing.
    # We then dot it with the *commanded* forward direction
    # (cos(target_heading), sin(target_heading), 0) to get a signed
    # scalar in [-dt, +dt] roughly.  The dt is removed by giving the
    # raw metres (so the reward is m/s × dt independent of dt).
    # ------------------------------------------------------------------
    def _reward_forward_progress(self):
        # ``last_base_pos`` is updated each step in _update_buffers.
        if not hasattr(self, "last_base_pos") or self.last_base_pos is None:
            return torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float)
        # Current body forward (x,y) in world frame from yaw.
        yaw = self.base_euler[:, 2]
        forward = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)
        # Commanded forward direction.
        target_heading = self.commands[:, 3]
        target_forward = torch.stack([torch.cos(target_heading), torch.sin(target_heading)], dim=-1)
        # Per-step displacement projected on body forward.
        displacement = self.base_pos[:, :2] - self.last_base_pos[:, :2]
        forward_disp = (displacement * forward).sum(dim=-1)
        # Reward only the component along the *commanded* direction
        # (positive when walking toward target_heading, negative when
        # backing away / circling).
        target_alignment = (forward * target_forward).sum(dim=-1)
        # Mask: skip envs that are commanded to stand still.
        moving = (self.commands[:, 0] > 0.05).float()

        # ---- Difficulty weighting (CAPPED at 3.0) ----
        # Look up the difficulty rating of the cell the robot is
        # currently standing in.  A robot on the centre flat cell
        # (difficulty=1.0) gets a 1.0× multiplier — *no extra* credit
        # for cruising on flat ground.  A robot climbing the stairs
        # (difficulty=4.5) would normally get 4.5× credit per metre,
        # but we **soft-cap at 3.0** to balance two competing goals:
        #  (a) enough reward differential between centre flat (1.0×)
        #      and stairs (3.0×) to actually pull the policy
        #      *toward* the hard cells;
        #  (b) not so much amplification that early-episode falls on
        #      a hard cell catastrophically destabilise PPO.
        # 3.0× gives a clear "harder is more rewarding" signal
        # without the 4.5× over-amplification that caused the
        # earlier "tuck-and-fall" collapse.
        if getattr(self, "difficulty_map", None) is not None:
            dm = self.difficulty_map
            n_grid_x, n_grid_y = dm.shape
            # World position -> cell index.  The terrain grid is
            # placed with cell (0, 0) at world (0, 0); the heightfield
            # has subterrain_size m per cell.  Clip to valid range so
            # envs that wander off the grid still get a reasonable
            # weight.
            sx = self.terrain_cfg["subterrain_size"][0]
            sy = self.terrain_cfg["subterrain_size"][1]
            cell_x = (self.base_pos[:, 0] / sx).clamp(0, n_grid_x - 1).long()
            cell_y = (self.base_pos[:, 1] / sy).clamp(0, n_grid_y - 1).long()
            cell_difficulty = dm[cell_x, cell_y].clamp(min=1.0, max=3.0)
        else:
            cell_difficulty = torch.ones(self.num_envs, device=self.device)

        return forward_disp * target_alignment * moving * cell_difficulty
