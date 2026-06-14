# Copyright (c) 2025, Your Name
# All rights reserved.

"""Training entry point for mylab RL framework.

Usage::

    python train.py --env go2-walk --alg ppo --num-envs 4096 --iters 1000
"""

from __future__ import annotations

import argparse
import os
import time

import torch

import genesis as gs
from envs.genesis import get_genesis_env
from mylab.utils.logger import Logger
from alg_config.genesis import get_alg_config
from mylab.rl.alg import resolve_algorithm
from mylab.training import apply_training_seed, make_run_dir, resolve_load_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an RL policy using the mylab framework.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env", type=str, required=True,
                        help="Environment name (e.g. go2-walk, go2-walk_easy, panda-grasp).")
    parser.add_argument("--alg", type=str, default=None,
                        help="Algorithm name (registered in ALGORITHM_REGISTRY). Overrides config if set.")
    parser.add_argument("--num-envs", type=int, default=4096,
                        help="Number of parallel environments.")
    parser.add_argument("--iters", type=int, default=1000,
                        help="Number of learning iterations.")
    parser.add_argument("--max-iters", type=int, default=None,
                        help="Alias for --iters, takes precedence if set.")
    parser.add_argument("--num-steps", type=int, default=24,
                        help="Number of environment steps per iteration.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    parser.add_argument("--log-dir", type=str, default="logs",
                        help="Root directory for log output.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a checkpoint to resume from.")
    parser.add_argument("--load-run", type=str, default=None,
                        help="Resume from logs/<env>/<load-run>/<latest>.pt. "
                             "Use -1 for the most recent run.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: 'cuda', 'cpu', or 'cuda:N'.")
    parser.add_argument("--log-type", type=str, default="console", choices=["console", "tensorboard"],
                        help="Logger type: 'console' or 'tensorboard'.")
    parser.add_argument("--init-random-ep-len", action="store_true", default=True,
                        help="Randomize episode lengths at initialization for better exploration.")
    parser.add_argument("--run-suffix", type=str, default=None,
                        help="Optional suffix appended to the run directory name.")

    args = parser.parse_args()
    if args.max_iters is not None:
        args.iters = args.max_iters

    # ----- Resolve device -----
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, falling back to CPU.")
        args.device = "cpu"

    # ----- Set seed (Python / NumPy / Torch / CUDA) -----
    apply_training_seed(args.seed, torch_deterministic=False, cuda=(args.device == "cuda"))

    # ----- Initialize genesis -----
    backend = gs.constants.backend.gpu if args.device == "cuda" else gs.constants.backend.cpu
    gs.init(backend=backend, precision="32", seed=args.seed, performance_mode=True, logging_level="warning")

    # ----- Create environment -----
    print(f"Creating environment '{args.env}' with {args.num_envs} envs ...")
    env = get_genesis_env(args.env, args.num_envs, show_viewer=False)
    env.seed(args.seed)
    print(f"  num_actions: {env.num_actions}, device: {env.device}")

    # ----- Load algorithm configuration -----
    train_cfg = get_alg_config(args.env, args.alg)
    alg_name = train_cfg["algorithm"]["name"]

    # ----- Resolve algorithm (via mylab.rl.alg.ALGORITHM_REGISTRY) -----
    print(f"Resolving algorithm '{alg_name}' ...")
    alg_cls = resolve_algorithm(alg_name)

    # ----- Resolve checkpoint (if any) -----
    checkpoint_path = resolve_load_path(
        log_root=args.log_dir,
        env_name=env.name,
        checkpoint_arg=args.checkpoint,
        load_run_arg=args.load_run,
    )

    # ----- Create log directory (timestamped) -----
    suffix = args.run_suffix or alg_name
    log_dir = make_run_dir(args.log_dir, env.name, suffix=suffix)
    print(f"Logging to: {log_dir}")

    # ----- Create algorithm -----
    print("Constructing algorithm ...")
    obs = env.get_observations()
    state = obs["state"]
    image = obs.get("images")
    alg = alg_cls.construct_algorithm(state, env, train_cfg, args.device)
    alg.train_mode()

    # ----- Create logger -----
    logger = Logger(
        log_dir=log_dir if args.log_type == "tensorboard" else None,
        cfg=train_cfg,
        num_envs=args.num_envs,
        device=args.device,
        logger_type=args.log_type,
    )

    # ----- Load checkpoint if requested -----
    current_it = 0
    if args.checkpoint is not None:
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        alg.load(ckpt)
        current_it = ckpt.get("iter", 0)
        print(f"  Resuming from iteration {current_it}")

    # If the checkpoint is already at or past the target, extend
    if current_it >= args.iters:
        args.iters = current_it + args.iters
        print(f"  Checkpoint >= target → extending to {args.iters} iterations.")

    # ----- Randomize episode lengths at start (improves exploration) -----
    if args.init_random_ep_len and current_it == 0:
        max_ep_len = getattr(env, "max_episode_length", None)
        if max_ep_len is not None:
            rand_ep_len = torch.randint_like(env.episode_length_buf, high=max_ep_len)
            env.episode_length_buf[:] = rand_ep_len
            print(f"  Randomized episode lengths: {rand_ep_len[:4].tolist()} ...")

    # ----- Save run configuration (for reproducibility) -----
    run_config = {
        "env": env.name,
        "alg": alg_name,
        "num_envs": args.num_envs,
        "num_steps": train_cfg["num_steps_per_env"],
        "iters": args.iters,
        "seed": args.seed,
        "device": args.device,
        "log_dir": log_dir,
        "checkpoint_resumed": checkpoint_path,
        "config": train_cfg,
    }
    import json
    with open(os.path.join(log_dir, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2, default=str)
    print(f"  Wrote run_config.json to {log_dir}")

    # ----- Training loop -----
    print(f"\nStarting training: {current_it} -> {args.iters} iterations.\n")
    total_start = time.time()

    for it in range(current_it, args.iters):
        start = time.time()

        # --- Rollout ---
        alg.train_mode()  # needed so actor obs_normalizer can update running stats
        with torch.inference_mode():
            for _ in range(train_cfg["num_steps_per_env"]):
                actions = alg.act(state, image)
                obs, rewards, dones, extras = env.step(actions)
                state = obs["state"].to(args.device)
                image = obs.get("images")
                if image is not None:
                    image = image.to(args.device)
                rewards = rewards.to(args.device)
                dones = dones.to(args.device)
                alg.process_env_step(state, rewards, dones, extras, image)
                logger.process_env_step(rewards, dones, extras)

            collect_time = time.time() - start
            start = time.time()

            # --- Compute returns ---
            alg.compute_returns(state, image)

        # --- Update ---
        loss_dict = alg.update()
        learn_time = time.time() - start

        # --- Log ---
        with torch.no_grad():
            if hasattr(alg.actor, "distribution"):
                action_std = alg.actor.distribution.std.mean()
            else:
                kwargs = {"training": False}
                if image is not None:
                    kwargs["image"] = image
                _, action_std = alg.actor.get_mean_and_std(state, **kwargs)
                action_std = action_std.mean()
        learning_rate = train_cfg["algorithm"]["learning_rate"]
        if hasattr(alg, "learning_rate"):
            learning_rate = alg.learning_rate

        logger.log(
            it=it,
            start_it=current_it,
            total_it=args.iters,
            collect_time=collect_time,
            learn_time=learn_time,
            loss_dict=loss_dict,
            learning_rate=learning_rate,
            action_std=action_std,
            num_steps_per_env=train_cfg["num_steps_per_env"],
        )

        # --- Save checkpoint ---
        if it % train_cfg["save_interval"] == 0:
            save_path = os.path.join(log_dir, f"model_{it}.pt")
            saved = alg.save()
            saved["iter"] = it + 1
            torch.save(saved, save_path)
            print(f"  Saved checkpoint: {save_path}")

    # ----- Final save -----
    final_path = os.path.join(log_dir, f"model_{args.iters}.pt")
    saved = alg.save()
    saved["iter"] = args.iters
    torch.save(saved, final_path)
    print(f"\nFinal model saved: {final_path}")

    total_time = time.time() - total_start
    print(f"Training finished in {total_time:.1f}s ({total_time / 3600:.2f}h).")

    env.close()
    logger.close()


if __name__ == "__main__":
    main()