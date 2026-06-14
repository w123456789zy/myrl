import json
import os

_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))

_DEFAULT_CONFIGS = {
    "go2-walk": {
        "ppo": "go2_walk.json",
        "flashsac": "go2_walk_flashsac.json",
    },
    "go2-vision": {
        "vision_ppo": "go2_vision.json",
    },
    "go2-walk-hard": {
        "vision_ppo": "go2_walk_hard.json",
        "vision_flashsac": "go2_walk_hard_flashsac.json",
    },
    "go2-walk-stairs": {
        "ppo": "go2_walk.json",
        "vision_ppo": "go2_rough_vision.json",
        "moe_vision_ppo": "go2_rough_moe_vision.json",
    },
    "go2-backflip": {
        "ppo": "go2_walk.json",
    },
    "go2-rough": {
        "ppo": "go2_rough.json",
        "vision_ppo": "go2_rough_vision.json",
        "moe_vision_ppo": "go2_rough_moe_vision.json",
    },
    "go2-test": {
        "vision_ppo": "go2_rough_vision.json",
        "moe_vision_ppo": "go2_rough_moe_vision.json",
    },
    "go2-footstand": {
        "ppo": "go2_footstand.json",
    },
    "go2-handstand": {
        "ppo": "go2_handstand.json",
    },
}


def get_alg_config(env_name: str, alg_name: str | None = None) -> dict:
    """Load algorithm config for the given environment and algorithm.

    Args:
        env_name: Environment name (e.g. "go2-walk").
        alg_name: Algorithm name (e.g. "ppo", "flashsac"). If None, uses the
            first available algorithm for this environment.

    Returns:
        Parsed JSON configuration dict.
    """
    env_configs = _DEFAULT_CONFIGS.get(env_name)
    if env_configs is None:
        raise ValueError(f"No algorithm config found for env: {env_name}")

    if alg_name is None:
        alg_name = list(env_configs.keys())[0]
        print(f"  [INFO] No --alg specified, using default '{alg_name}' for env '{env_name}'")

    filename = env_configs.get(alg_name)
    if filename is None:
        available = list(env_configs.keys())
        raise ValueError(
            f"No config for algorithm '{alg_name}' in env '{env_name}'. "
            f"Available algorithms: {available}"
        )

    path = os.path.join(_CONFIG_DIR, filename)
    with open(path, "r") as f:
        return json.load(f)
