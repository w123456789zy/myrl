# Copyright (c) 2025, Your Name
# All rights reserved.

"""Play go2-rough with real-time keyboard command input.

Loads a trained checkpoint and lets you control the robot's walking direction
via console input on rough terrain.  go2-rough uses **heading mode**:

    commands[0] = forward speed
    commands[3] = target heading angle (radians)

The user inputs a 2-D velocity vector [vx, vy] which is converted to
speed = sqrt(vx^2 + vy^2) and heading = atan2(vy, vx).

Usage::

    python test/play_go2_rough.py --checkpoint logs/go2_rough/model_2000.pt

Interactive commands (same format as go2-walk):
    Enter two numbers separated by space to set [vx, vy], e.g.:
        1.0 0.0   -> forward (+x)
        0.0 1.0   -> left (+y)
       -1.0 0.0   -> backward (-x)
        0.5 0.5   -> diagonal
    Press Enter without input to keep the last command.
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

import torch

import genesis as gs
from envs.genesis import get_genesis_env
from arrow_vis import draw_command_arrows

# ---------------------------------------------------------------------------
# Algorithm registry (shared with train.py)
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
    """Daemon thread that reads lines from stdin."""
    while True:
        try:
            line = input()
            input_queue.put(line)
        except EOFError:
            break


def _update_env_commands(env, vx: float, vy: float):
    """Set environment commands based on user input.

    heading mode (go2-rough):
        commands[0] = speed (body-forward)
        commands[1] = 0
        commands[3] = target heading = atan2(vy, vx)

    ang_vel_yaw mode (fallback):
        commands[0] = vx (world-x)
        commands[1] = vy (world-y)
        commands[2] = desired yaw rate computed from heading error
    """
    if env.command_type == "heading":
        speed = math.sqrt(vx * vx + vy * vy)
        target_heading = math.atan2(vy, vx)
        env.commands[:, 0] = speed
        env.commands[:, 1] = 0.0
        env.commands[:, 3] = target_heading
    else:
        env.commands[:, 0] = vx
        env.commands[:, 1] = vy
        # Compute desired yaw rate from heading error to help the policy learn turning
        if hasattr(env, "base_euler") and env.base_euler is not None:
            current_heading = env.base_euler[0, 2].item()
            target_heading = math.atan2(vy, vx)
            heading_error = math.atan2(
                math.sin(target_heading - current_heading),
                math.cos(target_heading - current_heading),
            )
            env.commands[:, 2] = max(-1.0, min(1.0, heading_error))
        else:
            env.commands[:, 2] = 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Play go2-rough with real-time command input.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env", type=str, default="go2-rough",
                        help="Environment name (default: go2-rough).")
    parser.add_argument("--alg", type=str, default=None,
                        help="Algorithm name. Overrides config if set.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the trained model checkpoint.")
    parser.add_argument("--num-envs", type=int, default=1,
                        help="Number of parallel environments (usually 1 for play).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--steps", type=int, default=0,
                        help="Max steps to run (0 = indefinite).")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, falling back to CPU.")
        args.device = "cpu"

    torch.manual_seed(args.seed)

    backend = gs.constants.backend.gpu if args.device == "cuda" else gs.constants.backend.cpu
    gs.init(backend=backend, precision="32", seed=args.seed, logging_level="warning")

    print(f"Creating environment '{args.env}' with {args.num_envs} envs (render ON) ...")
    env = get_genesis_env(args.env, args.num_envs, show_viewer=True)
    env.seed(args.seed)
    print(f"  num_actions: {env.num_actions}, device: {env.device}")
    print(f"  command_type: {env.command_type}")

    from alg_config.genesis import get_alg_config
    train_cfg = get_alg_config(args.env, args.alg)
    alg_name = train_cfg["algorithm"]["name"]
    alg_cls = _resolve_algorithm(alg_name)

    obs = env.get_observations()
    state = obs["state"]
    image = obs.get("images")
    alg = alg_cls.construct_algorithm(state, env, train_cfg, args.device)

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    alg.load(ckpt)
    alg.eval_mode()

    obs, extras = env.reset()
    state = obs["state"].to(args.device)
    image = obs.get("images")
    if image is not None:
        image = image.to(args.device)

    # ----- Start input thread -----
    input_queue: queue.Queue[str] = queue.Queue()
    threading.Thread(target=_input_thread, args=(input_queue,), daemon=True).start()

    # Default command: move along +x
    current_vx = 1.0
    current_vy = 0.0
    _update_env_commands(env, current_vx, current_vy)

    print("\n" + "=" * 60)
    print("Interactive control started!")
    print("Default command: +x direction (1.0 0.0)")
    print("Type 'vx vy' and press Enter to change direction.")
    print("Examples:  '1.0 0.0' (forward) | '0.0 1.0' (left) | '-1.0 0.0' (backward)")
    print("Press Ctrl+C to stop.")
    print("=" * 60 + "\n")

    max_steps = args.steps if args.steps > 0 else float("inf")
    step_count = 0
    total_reward = 0.0
    episode_count = 0
    last_print_time = time.time()

    try:
        while step_count < max_steps:
            # Check for new user input
            while not input_queue.empty():
                line = input_queue.get_nowait().strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        new_vx = float(parts[0])
                        new_vy = float(parts[1])
                        current_vx, current_vy = new_vx, new_vy
                        _update_env_commands(env, current_vx, current_vy)
                        print(f"  [CMD] vx={current_vx:.2f}, vy={current_vy:.2f}, "
                              f"heading={math.degrees(math.atan2(current_vy, current_vx)):.1f}deg")
                    except ValueError:
                        print(f"  [WARN] Invalid input: '{line}'. Use format: 'vx vy'")
                else:
                    print(f"  [WARN] Invalid input: '{line}'. Use format: 'vx vy'")

            # Re-apply command every step (resample_commands may overwrite it)
            _update_env_commands(env, current_vx, current_vy)

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
                draw_command_arrows(env)
                env.render()

            if dones.any():
                episode_count += dones.sum().item()
                # After reset, re-apply current command
                _update_env_commands(env, current_vx, current_vy)

            # Periodic status print
            now = time.time()
            if now - last_print_time >= 2.0:
                print(f"  step={step_count}, episodes={episode_count}, "
                      f"reward={total_reward / max(step_count, 1):.4f}, "
                      f"cmd=({current_vx:.2f}, {current_vy:.2f})")
                last_print_time = now

            # Small sleep to prevent 100% CPU and allow input thread to run
            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    print(f"\nPlay finished: {step_count} steps, {episode_count} episodes, "
          f"avg reward: {total_reward / max(step_count, 1):.4f}")

    env.close()


if __name__ == "__main__":
    main()
