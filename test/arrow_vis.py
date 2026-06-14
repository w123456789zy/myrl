# Copyright (c) 2025, Your Name
# All rights reserved.

"""Helper for drawing debug arrows in play scripts.

This is *only* intended for inference/visualization scripts (play.py and the
test/play_*.py variants). It must NOT be called from the env's training code
path, otherwise it will incur per-step debug mesh upload cost.

Usage from a play script::

    from test.arrow_vis import draw_command_arrows

    for ...:
        obs, rewards, dones, extras = env.step(actions)
        draw_command_arrows(env)  # call AFTER env.step(), BEFORE env.render()
        env.render()
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from envs.genesis.go2_base import Go2BaseEnv


# ---------------------------------------------------------------------------
# Color palette (RGBA, 0..1)
# ---------------------------------------------------------------------------
_COLOR_ACTUAL = (0.95, 0.25, 0.25, 0.95)   # red  — actual body velocity
_COLOR_TARGET = (0.25, 0.95, 0.45, 0.95)   # green — commanded target
_COLOR_FRAME_X = (0.95, 0.30, 0.30, 0.85)  # dim red   — body +X axis
_COLOR_FRAME_Y = (0.30, 0.95, 0.30, 0.85)  # dim green — body +Y axis

_ARROW_BODY_RADIUS = 0.012
_ARROW_MIN_LEN = 0.05
_ARROW_SPEED_SCALE = 1.0   # arrow length per (m/s)
_ARROW_HEAD_LEN = 0.08     # arrow head length (head ≈ cone height)
_ARROW_HEAD_RADIUS = 0.02


def _wrap_to_pi(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def _yaw_to_vec(yaw: float, length: float = 1.0) -> np.ndarray:
    return np.array([math.cos(yaw) * length, math.sin(yaw) * length, 0.0], dtype=np.float32)


def draw_command_arrows(env: "Go2BaseEnv") -> None:
    """Draw two arrows on the go2 base:

    * red   — actual body-forward velocity in world frame
    * green — target direction (heading-mode: target_heading; else: command vec)

    Plus a small body-aligned XYZ frame for reference.
    Safe to call when ``show_viewer=False`` — it will silently no-op if the
    scene has no visualizer (e.g. headless training).
    """
    scene = getattr(env, "scene", None)
    if scene is None or not hasattr(scene, "draw_debug_arrow"):
        return
    vis = getattr(scene, "_visualizer", None)
    if vis is None or vis.viewer is None:
        return

    # Clear previous arrows (cheap; per-frame)
    if hasattr(scene, "clear_debug_objects"):
        scene.clear_debug_objects()

    # Get base position (env 0 — play runs with num_envs=1)
    base_pos = np.array(env.base_pos[0].cpu(), dtype=np.float32)
    # Lift slightly above the base to avoid z-fighting with the robot mesh
    arrow_root = base_pos.copy()
    arrow_root[2] = max(arrow_root[2], base_pos[2] + 0.30)

    # --- actual body velocity (world frame) ---
    vel_world = np.array(env.base_lin_vel[0].cpu(), dtype=np.float32)
    speed = float(np.linalg.norm(vel_world[:2]))
    if speed > 1e-3:
        v_dir = vel_world[:2] / max(speed, 1e-6)
        actual_len = max(_ARROW_MIN_LEN, speed * _ARROW_SPEED_SCALE)
        vec = np.array([v_dir[0] * actual_len, v_dir[1] * actual_len, 0.0], dtype=np.float32)
        scene.draw_debug_arrow(
            pos=arrow_root,
            vec=vec,
            radius=_ARROW_BODY_RADIUS,
            color=_COLOR_ACTUAL,
        )

    # --- target direction ---
    cmd = np.array(env.commands[0].cpu(), dtype=np.float32)
    if env.command_type == "heading":
        target_heading = float(cmd[3])
        speed_cmd = float(cmd[0])
    else:
        # ang_vel_yaw mode: derive target heading from vx, vy
        if abs(cmd[0]) + abs(cmd[1]) > 1e-3:
            target_heading = math.atan2(float(cmd[1]), float(cmd[0]))
        else:
            target_heading = None  # type: ignore[assignment]
        speed_cmd = math.sqrt(float(cmd[0]) ** 2 + float(cmd[1]) ** 2)

    if target_heading is not None and speed_cmd > 1e-3:
        target_len = max(_ARROW_MIN_LEN, speed_cmd * _ARROW_SPEED_SCALE)
        vec = _yaw_to_vec(target_heading, target_len)
        scene.draw_debug_arrow(
            pos=arrow_root,
            vec=vec,
            radius=_ARROW_BODY_RADIUS,
            color=_COLOR_TARGET,
        )

    # --- small body-frame (X = red, Y = green) for orientation reference ---
    body_yaw = float(env.base_euler[0, 2].cpu())
    frame_len = 0.20
    body_x = arrow_root + _yaw_to_vec(body_yaw, frame_len)
    body_y = arrow_root + _yaw_to_vec(body_yaw + math.pi / 2.0, frame_len)
    scene.draw_debug_line(arrow_root, body_x, radius=0.006, color=_COLOR_FRAME_X)
    scene.draw_debug_line(arrow_root, body_y, radius=0.006, color=_COLOR_FRAME_Y)


# Re-export for callers that prefer ``from test.arrow_vis import draw_command_arrows``
__all__ = ["draw_command_arrows"]
