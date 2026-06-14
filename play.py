# Copyright (c) 2025, Your Name
# All rights reserved.

"""Play / inference entry point for mylab RL framework.

Loads a trained checkpoint and runs the policy with rendering enabled.

Usage::

    python play.py --env go2-walk --checkpoint logs/model_1000.pt
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import torch

import genesis as gs
from envs.genesis import get_genesis_env
from alg_config.genesis import get_alg_config
from mylab.rl.alg import resolve_algorithm
from mylab.training import apply_training_seed

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "test"))
try:
    from arrow_vis import draw_command_arrows
except ImportError:  # fallback for non-Go2 play scripts
    draw_command_arrows = None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a trained RL policy with rendering.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env", type=str, required=True,
                        help="Environment name (e.g. go2-walk, go2-walk_easy, panda-grasp).")
    parser.add_argument("--alg", type=str, default=None,
                        help="Algorithm name (same as used during training). Overrides config if set.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to the trained model checkpoint.")
    parser.add_argument("--load-run", type=str, default=None,
                        help="Resume from logs/<env>/<load-run>/<latest>.pt. Use -1 for the latest run.")
    parser.add_argument("--num-envs", type=int, default=1,
                        help="Number of parallel environments (usually 1 for play).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: 'cuda', 'cpu', or 'cuda:N'.")
    parser.add_argument("--steps", type=int, default=0,
                        help="Max steps to run (0 = run indefinitely).")
    parser.add_argument("--episodes", type=int, default=0,
                        help="Max episodes to run (0 = ignore).")
    parser.add_argument("--eval-mode", dest="eval_mode", action="store_true", default=True,
                        help="Disable domain randomization in the env at play time (default ON).")
    parser.add_argument("--no-eval-mode", dest="eval_mode", action="store_false",
                        help="Re-enable domain randomization in the env at play time.")
    parser.add_argument("--warmup-history", action="store_true", default=True,
                        help="Step the env with zero actions N times to fill obs history before policy acts.")
    parser.add_argument("--no-warmup-history", dest="warmup_history", action="store_false",
                        help="Skip the obs-history warmup step.")
    parser.add_argument("--deterministic", action="store_true", default=False,
                        help="Use deterministic action (mean) for the policy.")
    parser.add_argument("--debug-obs", action="store_true", default=False,
                        help="Print a one-shot sanity check of state / image / actor normalizer stats before the loop.")

    args = parser.parse_args()

    # ----- Resolve device -----
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, falling back to CPU.")
        args.device = "cpu"

    # ----- Set seed (Python / NumPy / Torch / CUDA) -----
    apply_training_seed(args.seed, torch_deterministic=False, cuda=(args.device == "cuda"))

    # ----- Initialize genesis -----
    backend = gs.constants.backend.gpu if args.device == "cuda" else gs.constants.backend.cpu
    gs.init(backend=backend, precision="32", seed=args.seed, logging_level="warning")

    # ----- Create environment (rendering enabled by default) -----
    print(
        f"Creating environment '{args.env}' with {args.num_envs} envs "
        f"(render ON, eval_mode={args.eval_mode}) ..."
    )
    env = get_genesis_env(args.env, args.num_envs, show_viewer=True, eval_mode=args.eval_mode)
    env.seed(args.seed)
    print(f"  num_actions: {env.num_actions}, device: {env.device}")

    # ----- Load algorithm configuration -----
    train_cfg = get_alg_config(args.env, args.alg)
    alg_name = train_cfg["algorithm"]["name"]

    # ----- Resolve and construct algorithm (via ALGORITHM_REGISTRY) -----
    print(f"Resolving algorithm '{alg_name}' ...")
    alg_cls = resolve_algorithm(alg_name)

    # ----- Resolve checkpoint (--checkpoint or --load-run) -----
    from mylab.training import resolve_load_path
    checkpoint_path = resolve_load_path(
        log_root="logs",
        env_name=env.name,
        checkpoint_arg=args.checkpoint,
        load_run_arg=args.load_run,
    )
    if checkpoint_path is None:
        raise FileNotFoundError(
            "No checkpoint found. Pass --checkpoint <path> or --load-run -1 (or --load-run <name>)."
        )

    print("Constructing algorithm for inference ...")
    obs = env.get_observations()
    state = obs["state"]
    image = obs.get("images")
    # Inference doesn't need replay buffer — shrink it to save ~5GB VRAM
    if "algorithm" in train_cfg and "buffer" in train_cfg["algorithm"]:
        train_cfg["algorithm"]["buffer"]["capacity"] = 1
    alg = alg_cls.construct_algorithm(state, env, train_cfg, args.device)

    # ----- Load checkpoint -----
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    alg.load(ckpt)

    # ----- Switch to eval mode -----
    alg.eval_mode()

    # ----- Reset environment to get initial observations -----
    obs, extras = env.reset()
    state = obs["state"].to(args.device)
    image = obs.get("images")
    if image is not None:
        image = image.to(args.device)

    # ----- Optional: warm up obs history with zero actions -----
    # PPO-style policies with num_history_obs > 1 expect the previous N-1
    # history slots to be non-zero. After env.reset() the first N-1 slots are
    # zero, which is a state the policy has never seen during training. We
    # therefore step the env N-1 times with zero actions so the history buffer
    # gets fully populated by the actual physics evolution.
    num_history = 1
    try:
        num_history = int(env._obs_cfg.get("num_history_obs", 1))
    except (AttributeError, TypeError):
        num_history = 1
    warmup_steps = max(0, num_history - 1)
    if args.warmup_history and warmup_steps > 0:
        zero_action = torch.zeros(args.num_envs, env.num_actions, device=args.device)
        for _ in range(warmup_steps):
            obs, rewards, dones, extras = env.step(zero_action)
            state = obs["state"].to(args.device)
            image = obs.get("images")
            if image is not None:
                image = image.to(args.device)
        print(
            f"  Warmed up obs history with {warmup_steps} zero-action steps "
            f"(num_history_obs={num_history})."
        )

    # ----- Optional: debug sanity check of observations vs. checkpoint stats -----
    if args.debug_obs:
        with torch.no_grad():
            print("  [debug] state mean/std:", state.mean().item(), state.std().item())
            if image is not None:
                print(
                    "  [debug] image shape/min/max/mean:",
                    tuple(image.shape),
                    image.min().item(),
                    image.max().item(),
                    image.mean().item(),
                )
            policy = alg.get_policy()
            obs_normalizer = getattr(policy, "obs_normalizer", None)
            if obs_normalizer is not None and not isinstance(obs_normalizer, torch.nn.Identity):
                mean = obs_normalizer.mean.detach()
                var = obs_normalizer.var.detach()
                print(
                    "  [debug] obs_normalizer mean abs mean:",
                    mean.abs().mean().item(),
                    "var abs mean:",
                    var.abs().mean().item(),
                )
            if hasattr(policy, "distribution"):
                try:
                    print("  [debug] actor std mean:", policy.distribution.std.mean().item())
                except AttributeError:
                    pass

    # ----- Inference loop -----
    print("\nRunning policy (Ctrl+C to stop)...")
    max_steps = args.steps if args.steps > 0 else float("inf")
    step_count = 0
    total_reward = 0.0
    episode_count = 0
    episode_sums: list[dict[str, float]] = []
    last_extras: dict = extras

    def _set_actor_stochastic(stochastic: bool) -> None:
        """Override actor forward stochastic flag for deterministic play."""
        policy = alg.get_policy()
        if hasattr(policy, "stochastic"):
            policy.stochastic = stochastic

    if args.deterministic:
        _set_actor_stochastic(False)
        print("  [play] deterministic mode: actor will output mean action.")

    try:
        while step_count < max_steps and (args.episodes == 0 or episode_count < args.episodes):
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
            last_extras = extras

            # Render
            if hasattr(env, "render"):
                if draw_command_arrows is not None and args.num_envs == 1:
                    draw_command_arrows(env)
                env.render()

            # Handle episode termination
            if dones.any():
                ep_count_this_step = int(dones.sum().item())
                episode_count += ep_count_this_step
                ep_rewards = extras.get("episode", {})
                if ep_rewards:
                    for k, v in ep_rewards.items():
                        if isinstance(v, torch.Tensor):
                            episode_sums.append({"key": k, "value": v.float().mean().item()})
                print(
                    f"  Episodes finished: {episode_count}, "
                    f"avg reward: {total_reward / max(step_count, 1):.4f}, "
                    f"steps: {step_count}"
                )

            if step_count % 200 == 0:
                print(
                    f"  Step {step_count}, "
                    f"avg reward: {total_reward / max(step_count, 1):.4f}"
                )

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    print(
        f"\nPlay finished: {step_count} steps, "
        f"{episode_count} episodes, "
        f"avg reward: {total_reward / max(step_count, 1):.4f}"
    )
    if episode_sums:
        # Aggregate per-key episode reward components
        agg: dict[str, list[float]] = {}
        for entry in episode_sums:
            agg.setdefault(entry["key"], []).append(entry["value"])
        print("  Episode reward components (mean over episodes):")
        for k, vs in agg.items():
            print(f"    {k}: {sum(vs) / len(vs):+.4f}")

    env.close()


if __name__ == "__main__":
    main()