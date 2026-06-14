# Copyright (c) 2025, Your Name
# All rights reserved.

"""Capture a single front-camera image from go2-walk-hard environment.

Usage::

    python test/capture_front_camera.py --output front_camera.png
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

import genesis as gs
from envs.genesis.go2_walk_hard import Go2WalkHardEnv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture front camera image from go2-walk-hard.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output", type=str, default="test/front_camera.png",
                        help="Output PNG file path.")
    parser.add_argument("--steps", type=int, default=30,
                        help="Number of env steps to run before capturing.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: 'cuda' or 'cpu'.")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, falling back to CPU.")
        args.device = "cpu"

    backend = gs.constants.backend.gpu if args.device == "cuda" else gs.constants.backend.cpu
    gs.init(backend=backend, precision="32", logging_level="warning")

    print("Creating go2-walk-hard environment (show_viewer=False) ...")
    env = Go2WalkHardEnv(num_envs=1, show_viewer=True)
    env.seed(42)
    print(f"  num_actions: {env.num_actions}, device: {env.device}")

    # Reset and run a few steps so the robot stabilizes and camera pose updates
    obs, extras = env.reset()
    for _ in range(args.steps):
        # Use default standing pose so the robot doesn't collapse
        actions = env.default_dof_pos.unsqueeze(0)
        obs, rewards, dones, extras = env.step(actions)

    # Retrieve front camera image directly to inspect raw frame
    frame, _, _, _ = env._front_camera.render()
    frame = np.array(frame)
    print(f"Raw frame shape={frame.shape}, dtype={frame.dtype}, min={frame.min():.4f}, max={frame.max():.4f}")

    if frame.shape[-1] == 4:
        frame = frame[..., :3]

    image = torch.from_numpy(frame)
    if image.ndim == 3:
        image = image.permute(2, 0, 1)
    image = image.float()
    # Genesis may return float [0,1] or uint8 [0,255]; normalize accordingly
    if image.max() <= 1.0:
        image_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    else:
        image_np = image.permute(1, 2, 0).numpy().astype(np.uint8)

    # Save using PIL if available, otherwise fallback to cv2
    try:
        from PIL import Image
        img = Image.fromarray(image_np)
        img.save(args.output)
    except ImportError:
        import cv2
        cv2.imwrite(args.output, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

    print(f"Saved front camera image: {args.output}  shape={image_np.shape}")
    env.close()


if __name__ == "__main__":
    main()
