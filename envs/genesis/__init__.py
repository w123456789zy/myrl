# Copyright (c) 2025, Your Name
# All rights reserved.

import warnings
from typing import Any

from envs.genesis.go2_backflip import Go2BackflipEnv
from envs.genesis.go2_footstand import Go2FootStandEnv
from envs.genesis.go2_handstand import Go2HandStandEnv
from envs.genesis.go2_rough import Go2RoughEnv
from envs.genesis.go2_rough_vision import Go2RoughVisionEnv
from envs.genesis.go2_vision import Go2VisionEnv
from envs.genesis.go2_walk import Go2WalkEnv
from envs.genesis.go2_walk_easy import Go2WalkEasyEnv
from envs.genesis.go2_walk_hard import Go2WalkHardEnv
from envs.genesis.go2_test import Go2TestEnv
from envs.genesis.go2_walk_stairs import Go2WalkStairsEnv
from envs.genesis.panda_grasp import FrankaPandaGraspEnv

# Suppress Genesis 0.4.7 internal deprecation warnings (not from our code)
warnings.filterwarnings(
    "ignore",
    message="This property is deprecated and will be removed in future release.*dofs_idx_local",
    category=UserWarning,
)

__all__ = [
    "Go2WalkEnv",
    "Go2VisionEnv",
    "Go2BackflipEnv",
    "Go2WalkEasyEnv",
    "Go2WalkHardEnv",
    "Go2WalkStairsEnv",
    "Go2TestEnv",
    "Go2RoughEnv",
    "Go2RoughVisionEnv",
    "Go2FootStandEnv",
    "Go2HandStandEnv",
    "FrankaPandaGraspEnv",
    "get_genesis_env",
]


def get_genesis_env(
    env_name: str,
    num_envs: int,
    eval_mode: bool = False,
    show_viewer: bool = False,
) -> Any:
    """Factory function to create genesis environments.

    Args:
        env_name: Name of the environment. Options:
            ``"go2-walk"``, ``"go2-walk-stairs"``, ``"go2-backflip"``,
            ``"panda-grasp"``, ``"go2-walk_easy"``, ``"go2-walk-hard"``,
            ``"go2-rough"``, ``"go2-rough-vision"`` (rough terrain +
            front camera), ``"go2-footstand"``, ``"go2-handstand"``,
            ``"go2-vision"`` (flat walk + front camera),
            ``"go2-test"`` (zero-shot eval track: 13-tile long
            corridor with stairs up/down, obstacles, stepping
            stones, gap, and pit in sequence).
        num_envs: Number of parallel environments.
        eval_mode: Whether to run in evaluation mode (no noise, no randomization).
        show_viewer: Whether to open the GUI viewer (for play/inference).

    Returns:
        An environment instance conforming to the VecEnv interface.
    """
    registry = {
        "go2-walk": lambda: Go2WalkEnv(num_envs=num_envs, eval_mode=eval_mode, show_viewer=show_viewer),
        "go2-vision": lambda: Go2VisionEnv(num_envs=num_envs, eval_mode=eval_mode, show_viewer=show_viewer),
        "go2-walk-stairs": lambda: Go2WalkStairsEnv(num_envs=num_envs, eval_mode=eval_mode, show_viewer=show_viewer),
        "go2-test": lambda: Go2TestEnv(num_envs=num_envs, eval_mode=eval_mode, show_viewer=show_viewer),
        "go2-backflip": lambda: Go2BackflipEnv(num_envs=num_envs, eval_mode=eval_mode, show_viewer=show_viewer),
        "panda-grasp": lambda: FrankaPandaGraspEnv(num_envs=num_envs, show_viewer=show_viewer),
        "go2-walk_easy": lambda: Go2WalkEasyEnv(num_envs=num_envs, show_viewer=show_viewer),
        "go2-walk-hard": lambda: Go2WalkHardEnv(num_envs=num_envs, eval_mode=eval_mode, show_viewer=show_viewer),
        "go2-rough": lambda: Go2RoughEnv(num_envs=num_envs, eval_mode=eval_mode, show_viewer=show_viewer),
        # ``Go2RoughEnv`` now has ``use_vision=True`` by default, so this
        # alias shares the same env class. The name is preserved for
        # backward compatibility with the original go2-rough-vision
        # env (which used to be a separate class). The only effective
        # difference is the env name string — both PPO and vision_ppo
        # algs work with both names.
        "go2-rough-vision": lambda: Go2RoughEnv(num_envs=num_envs, eval_mode=eval_mode, show_viewer=show_viewer),
        "go2-footstand": lambda: Go2FootStandEnv(num_envs=num_envs, eval_mode=eval_mode, show_viewer=show_viewer),
        "go2-handstand": lambda: Go2HandStandEnv(num_envs=num_envs, eval_mode=eval_mode, show_viewer=show_viewer),
    }
    if env_name not in registry:
        raise ValueError(
            f"Unknown env_name: {env_name}. Available: {sorted(registry.keys())}"
        )
    return registry[env_name]()
