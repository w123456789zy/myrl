# Copyright (c) 2025, Your Name
# All rights reserved.

"""Real-time Go2 front-camera + 3D viewer visualisation for go2-rough.

Combines:
  * The keyboard-controlled play loop of ``test/play_go2_rough.py``
    (loads a trained checkpoint, lets you drive the robot with
    ``vx vy`` typed in the terminal).
  * A **second window** showing the live front-camera image from
    Go2's body-mounted camera (cv2.imshow).

So you can:
  * Type ``1.0 0.0`` and press Enter  →  Go2 walks forward
  * Type ``0.0 1.0`` and press Enter  →  Go2 strafes left
  * Watch both the 3D viewer AND the first-person camera at the
    same time, to verify the camera can distinguish flat / bumpy /
    stair / pit cells.

Press ``q`` (or ``Esc``) in the **cv2 camera window** to quit.  If
the cv2 window doesn't have keyboard focus, click on it first.

Usage::

    python test/test_go2_rough_camera.py --checkpoint logs/go2_rough/model_3500.pt
    python test/test_go2_rough_camera.py --checkpoint logs/go2_rough/model_3500.pt --res 256 256
"""

from __future__ import annotations

import argparse
import math
import os
import queue
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
import torch

import genesis as gs
from envs.genesis import get_genesis_env

# ---------------------------------------------------------------------------
# Algorithm registry (shared with train.py and play_go2_rough.py)
# ---------------------------------------------------------------------------
_ALGORITHM_REGISTRY: dict[str, str] = {
    "ppo": "mylab.rl.alg.ppo.PPO",
    "vision_ppo": "mylab.rl.alg.vision_ppo.VisionPPO",
    "moe_vision_ppo": "mylab.rl.alg.moe_ppo.MoEPPO",
    "flashsac": "mylab.rl.alg.flashsac.FlashSAC",
    "vision_flashsac": "mylab.rl.alg.vision_flashsac.VisionFlashSAC",
}


def _resolve_algorithm(alg_name: str):
    import_path = _ALGORITHM_REGISTRY.get(alg_name)
    if import_path is None:
        raise ValueError(
            f"Unknown algorithm: '{alg_name}'. "
            f"Available: {list(_ALGORITHM_REGISTRY.keys())}."
        )
    parts = import_path.rsplit(".", 1)
    mod = __import__(parts[0], fromlist=[parts[1]])
    return getattr(mod, parts[1])


def _input_thread(input_queue: queue.Queue):
    """Daemon thread reading lines from stdin (for keyboard control)."""
    while True:
        try:
            line = input()
            input_queue.put(line)
        except EOFError:
            break


def _update_env_commands(env, vx: float, vy: float) -> None:
    """Apply a (vx, vy) world-frame command in heading mode.

    go2-rough uses *heading mode*::

        commands[:, 0] = forward speed (m/s)
        commands[:, 1] = 0
        commands[:, 3] = target heading (radians, world frame)

    We convert (vx, vy) → (speed, heading) and write it to the
    command buffer for every env.
    """
    if env.command_type == "heading":
        speed = math.sqrt(vx * vx + vy * vy)
        speed = min(speed, 0.6)  # cap at the configured lin_vel_x range
        target_heading = math.atan2(vy, vx)
        env.commands[:, 0] = speed
        env.commands[:, 1] = 0.0
        env.commands[:, 3] = target_heading
    else:
        env.commands[:, 0] = vx
        env.commands[:, 1] = vy


WINDOW_NAME = "Go2 Front Camera (press q or Esc to quit)"


def _rotate_vec_by_quat_np(q, v):
    """Rotate a vector v by a quaternion q = (w, x, y, z)."""
    w, x, y, z = q
    tx = 2.0 * (y * v[2] - z * v[1])
    ty = 2.0 * (z * v[0] - x * v[2])
    tz = 2.0 * (x * v[1] - y * v[0])
    return v + w * np.array([tx, ty, tz]) + np.array([
        y * tz - z * ty,
        z * tx - x * tz,
        x * ty - y * tx,
    ])


def _render_camera_frame(env) -> np.ndarray:
    """Render the front camera and return a BGR uint8 image for cv2.

    Bypasses :meth:`Go2BaseEnv.get_camera_observation` so we can read
    the raw frame as it actually appears on screen.  The pose is set
    here using the same offsets stored on the env.
    """
    base_pos = env.base_pos[0].cpu().numpy()
    base_quat = env.base_quat[0].cpu().numpy()  # (w, x, y, z)
    vision_offset = np.array(env._vision_offset, dtype=np.float64)
    lookat_offset = np.array(env._vision_lookat_offset, dtype=np.float64)
    front_pos = base_pos + _rotate_vec_by_quat_np(base_quat, vision_offset)
    front_lookat = front_pos + _rotate_vec_by_quat_np(base_quat, lookat_offset)
    env._front_camera.set_pose(pos=front_pos, lookat=front_lookat)
    frame, _, _, _ = env._front_camera.render()
    frame = np.array(frame)
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    if frame.dtype != np.uint8:
        if frame.max() <= 1.0:
            frame = (frame * 255.0).clip(0, 255).astype(np.uint8)
        else:
            frame = frame.clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def _overlay_hud(image_bgr: np.ndarray, env, step_count: int, fps: float,
                 vx: float, vy: float) -> np.ndarray:
    """Draw a small HUD on the camera frame so you can read state."""
    h, w = image_bgr.shape[:2]
    lines = [
        f"step={step_count}",
        f"fps={fps:5.1f}",
        f"base_z={env.base_pos[0, 2].item():.2f}m",
        f"yaw={math.degrees(env.base_euler[0, 2].item()):6.1f} deg",
        f"cmd vx={vx:+.2f}  vy={vy:+.2f}",
        f"cmd heading={math.degrees(math.atan2(vy, vx)):6.1f} deg",
    ]
    # Cell index (informational)
    sx = env.terrain_cfg["subterrain_size"][0]
    sy = env.terrain_cfg["subterrain_size"][1]
    cx = int(np.clip(env.base_pos[0, 0].item() / sx, 0, 4))
    cy = int(np.clip(env.base_pos[0, 1].item() / sy, 0, 4))
    lines.append(f"cell=({cx},{cy})")
    if hasattr(env, "difficulty_map") and env.difficulty_map is not None:
        try:
            diff = env.difficulty_map[cx, cy].item()
            lines.append(f"cell_difficulty={diff:.1f}")
        except Exception:
            pass

    out = image_bgr.copy()
    y = 18
    for line in lines:
        cv2.putText(out, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (0, 0, 0), 2, cv2.LINE_AA)  # black outline
        cv2.putText(out, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (255, 255, 255), 1, cv2.LINE_AA)  # white text
        y += 16
    # Bottom-right banner
    banner = "go2-rough front camera"
    (tw, th), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    cv2.putText(out, banner, (w - tw - 8, h - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(out, banner, (w - tw - 8, h - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Play go2-rough with checkpoint + live front camera.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env", type=str, default="go2-rough",
                        help="Environment name.")
    parser.add_argument("--alg", type=str, default="moe_vision_ppo",
                        help="Algorithm name.  go2-rough was trained with "
                             "moe_vision_ppo — do NOT use plain 'ppo' here "
                             "or the CNN+MoE weights won't match.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the trained model checkpoint.")
    parser.add_argument("--num-envs", type=int, default=1,
                        help="Number of parallel environments (1 for play).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--steps", type=int, default=0,
                        help="Max steps (0 = indefinite).")
    # NOTE: no --res flag.  The camera resolution is part of the
    # trained model — it must match the resolution the policy was
    # trained on (96x96 for go2-rough moe_vision_ppo).  Changing
    # it on the fly would break the CNN input dimensions.  If you
    # need a different resolution, re-train the policy first.
    parser.add_argument("--fov", type=float, default=-1.0,
                        help="Override camera FOV in degrees (-1 = use env default). "
                             "FOV does not affect CNN input size, only the rendered "
                             "image content.")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, falling back to CPU.")
        args.device = "cpu"

    torch.manual_seed(args.seed)

    backend = gs.constants.backend.gpu if args.device == "cuda" else gs.constants.backend.cpu
    gs.init(backend=backend, precision="32", seed=args.seed, logging_level="warning")

    # ---- Build the env ----
    print(f"Creating environment '{args.env}' with {args.num_envs} envs (3D viewer ON) ...")
    env = get_genesis_env(args.env, num_envs=args.num_envs, show_viewer=True)
    env.seed(args.seed)
    print(f"  num_actions: {env.num_actions}, device: {env.device}")
    print(f"  command_type: {env.command_type}")
    print(f"  camera res: {env._vision_res}, fov: {env._vision_fov}, "
          f"offset: {env._vision_offset}, lookat_offset: {env._vision_lookat_offset}")

    # Camera resolution is FIXED (96x96 for moe_vision_ppo trained
    # on go2-rough).  We do not allow runtime override — the CNN
    # weights encode a specific (C, H, W) input shape and changing
    # the rendered image size would silently desync the model.
    if args.fov > 0:
        try:
            env._front_camera.set_params(fov=args.fov)
            env._vision_fov = args.fov
        except Exception as e:
            print(f"[WARN] could not set fov on existing camera: {e}")

    # ---- Load the policy ----
    from alg_config.genesis import get_alg_config
    train_cfg = get_alg_config(args.env, args.alg)
    alg_name = train_cfg["algorithm"]["name"]
    alg_cls = _resolve_algorithm(alg_name)
    print(f"Algorithm: {alg_name}  ({alg_cls.__module__}.{alg_cls.__name__})")

    obs, extras = env.reset()
    state = obs["state"]
    image = obs.get("images")
    alg = alg_cls.construct_algorithm(state, env, train_cfg, args.device)

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    alg.load(ckpt)
    alg.eval_mode()
    print("Checkpoint loaded.")

    # ---- Reset env, then start keyboard thread ----
    obs, extras = env.reset()
    state = obs["state"].to(args.device)
    image = obs.get("images")
    if image is not None:
        image = image.to(args.device)

    input_queue: queue.Queue[str] = queue.Queue()
    threading.Thread(target=_input_thread, args=(input_queue,), daemon=True).start()

    # Default command: walk forward (+x)
    current_vx, current_vy = 0.5, 0.0
    _update_env_commands(env, current_vx, current_vy)

    print("\n" + "=" * 60)
    print("Camera + viewer running.  Two windows open:")
    print("  1) Genesis 3D viewer (3D scene, drag to orbit)")
    print("  2) OpenCV window 'Go2 Front Camera' (1st-person view)")
    print("=" * 60)
    print("Keyboard control (type in this terminal, press Enter):")
    print("  vx vy  →  e.g.  '1.0 0.0' (forward)")
    print("              '0.0 1.0' (left)")
    print("              '-1.0 0.0' (backward)")
    print("              '0.0 0.0' (stop)")
    print("Press 'q' or 'Esc' in the **cv2 window** to quit.")
    print("=" * 60 + "\n")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    max_steps = args.steps if args.steps > 0 else float("inf")
    step_count = 0
    total_reward = 0.0
    episode_count = 0
    last_frame_time = time.time()
    fps_smoothed = 0.0
    quit_requested = False

    try:
        while step_count < max_steps and not quit_requested:
            # ---- Read pending keyboard commands ----
            while not input_queue.empty():
                line = input_queue.get_nowait().strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        current_vx = float(parts[0])
                        current_vy = float(parts[1])
                        _update_env_commands(env, current_vx, current_vy)
                        print(f"  [CMD] vx={current_vx:+.2f}, vy={current_vy:+.2f}, "
                              f"heading={math.degrees(math.atan2(current_vy, current_vx)):+.1f}deg")
                    except ValueError:
                        print(f"  [WARN] invalid: '{line}'. Use 'vx vy'.")
                else:
                    print(f"  [WARN] invalid: '{line}'. Use 'vx vy'.")

            # Re-apply command every step (resample_commands in
            # post_physics_step may overwrite it on reset).
            _update_env_commands(env, current_vx, current_vy)

            # ---- Policy forward pass + env step ----
            with torch.inference_mode():
                actions = alg.act(state, image)
            obs, rewards, dones, extras = env.step(actions)
            state = obs["state"].to(args.device)
            image = obs.get("images")
            if image is not None:
                image = image.to(args.device)
            rewards = rewards.to(args.device)
            dones = dones.to(args.device)

            step_count += 1
            total_reward += rewards.mean().item()

            if hasattr(env, "render"):
                env.render()

            if dones.any():
                episode_count += dones.sum().item()
                # Re-apply command after reset
                _update_env_commands(env, current_vx, current_vy)

            # ---- Render the front camera into the cv2 window ----
            try:
                frame_bgr = _render_camera_frame(env)
            except Exception as e:
                # On the first few frames the camera may not be ready
                # (no depth buffer etc.).  Just show a placeholder.
                frame_bgr = np.zeros((env._vision_res[0], env._vision_res[1], 3), dtype=np.uint8)
                cv2.putText(frame_bgr, f"camera init... ({e})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            now = time.time()
            dt = max(now - last_frame_time, 1e-6)
            inst_fps = 1.0 / dt
            fps_smoothed = 0.9 * fps_smoothed + 0.1 * inst_fps if fps_smoothed > 0 else inst_fps
            last_frame_time = now

            hud = _overlay_hud(frame_bgr, env, step_count, fps_smoothed,
                               current_vx, current_vy)
            cv2.imshow(WINDOW_NAME, hud)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # 'q' or Esc
                print("Quitting (cv2 window).")
                quit_requested = True

            # Small sleep so the input thread gets CPU
            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C).")

    cv2.destroyAllWindows()
    print(f"\nDone: ran {step_count} steps, {episode_count} episodes, "
          f"avg reward: {total_reward / max(step_count, 1):.4f}")
    env.close()


if __name__ == "__main__":
    main()
