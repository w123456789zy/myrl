# Copyright (c) 2025, Your Name
# All rights reserved.

"""Go2 vision-guided environment (state + height scan + front camera).

This is the *vision* variant of the canonical :class:`Go2WalkEnv`. It
inherits the full state + height-scan + MDP machinery from
:py:meth:`Go2BaseEnv` and adds a forward-facing camera that renders an
RGB image used as the policy's "images" observation.

Trainable with ``VisionPPO`` (CNN encoder + MLP actor/critic). The
returned observations are a ``dict`` with two keys:

* ``"state"`` — proprioception + height scan (same as the canonical
  :class:`Go2WalkEnv`).
* ``"images"`` — front-camera RGB ``(num_envs, 3, H, W)`` (with
  ``H = W = 64`` by default).

The camera is added during scene construction
(:py:meth:`Go2BaseEnv._build_scene`) by setting ``use_vision=True`` and
``vision_cfg`` in the env config, and its pose is updated every
control step in :py:meth:`post_physics_step` so the camera always
faces the robot's heading.
"""

from __future__ import annotations

import torch

from envs.genesis.go2_walk import Go2WalkEnv


class Go2VisionEnv(Go2WalkEnv):
    """Go2 walk with a forward-facing RGB camera.

    The camera is positioned slightly in front of and above the base
    and looks along the robot's heading. The rendered image is part
    of the observation dict (under the ``"images"`` key).

    Defaults:

    * Image resolution: 64x64 (small enough for fast GPU rendering,
      large enough to capture nearby obstacles).
    * Field of view: 90 degrees (wide enough to keep stairs / boxes
      visible from a few metres away).
    * Camera offset: ``(0.3, 0.0, 0.15)`` m in front of / above the base.
    """

    def __init__(self, num_envs: int, show_viewer: bool = False, eval_mode: bool = False) -> None:
        super().__init__(num_envs=num_envs, show_viewer=show_viewer, eval_mode=eval_mode)
        self.name = "go2_vision"

    @classmethod
    def _default_configs(cls):
        env_cfg, obs_cfg, reward_cfg, command_cfg = super()._default_configs()

        # ---- enable vision ----
        env_cfg["use_vision"] = True
        env_cfg["vision_cfg"] = {
            "res": (64, 64),
            "fov": 90.0,
            # Camera offset relative to the base position (m).
            "offset": (0.3, 0.0, 0.15),
            # Where the camera looks at, relative to its own position.
            "lookat_offset": (1.0, 0.0, -0.15),
        }

        # Use a richer height scan in the vision env so the policy can
        # cross-check what the camera sees with what is under its feet.
        env_cfg["height_scan_cfg"] = {
            "resolution": 0.1,
            "size_x": 1.6,  # wider FOV
            "size_y": 1.0,
        }

        # Tighten push interval slightly (vision policies are easier to
        # destabilize when perturbed).
        env_cfg["push_interval_s"] = 6.0

        # ----------------- obs (state only — vision is appended by base) -----------------
        scan_cfg = env_cfg["height_scan_cfg"]
        n_x = int(round(scan_cfg["size_x"] / scan_cfg["resolution"])) + 1
        n_y = int(round(scan_cfg["size_y"] / scan_cfg["resolution"])) + 1
        scan_dim = n_x * n_y
        base_obs = 12 + 3 * env_cfg["num_dofs"]
        obs_cfg["num_obs"] = base_obs + scan_dim
        obs_cfg["num_history_obs"] = 1  # the camera replaces temporal history

        # Slightly larger privileged obs to keep the critic informative.
        # Match the actual ``privileged_obs_buf`` produced in
        # ``Go2BaseEnv.compute_observations``:
        # ``[obs_buf (num_obs), base_lin_vel (3), last_actions (num_dofs)]``.
        obs_cfg["num_priv_obs"] = base_obs + scan_dim + 3 + env_cfg["num_dofs"]

        return env_cfg, obs_cfg, reward_cfg, command_cfg

    def _resample_commands(self, envs_idx):
        """Vision env: gentle forward commands (we want the camera to see things)."""
        if len(envs_idx) == 0:
            return
        # Bias toward forward motion so the front camera captures motion.
        self.commands[envs_idx, 0] = torch.empty(len(envs_idx), device=self.device).uniform_(-0.3, 0.6)
        self.commands[envs_idx, 1] = 0.0
        self.commands[envs_idx, 2] = torch.empty(len(envs_idx), device=self.device).uniform_(-0.5, 0.5)

    def post_physics_step(self):
        """Hook: update the front-camera pose every step so it tracks the base."""
        super().post_physics_step()
        if self._use_vision and hasattr(self, "_front_camera"):
            front_pos = self.base_pos[0].cpu().numpy() + self._vision_offset
            front_lookat = front_pos + self._vision_lookat_offset
            self._front_camera.set_pose(pos=front_pos, lookat=front_lookat)


def get_env(num_envs: int, eval_mode: bool = False) -> Go2VisionEnv:
    try:
        import genesis as gs
        gs.init(logging_level="warning")
    except Exception:
        pass
    return Go2VisionEnv(num_envs=num_envs, eval_mode=eval_mode)
