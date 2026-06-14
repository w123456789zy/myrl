"""Algorithm package.

Importing this package (or any submodule of it) populates
:data:`mylab.rl.alg.base_alg.ALGORITHM_REGISTRY` with every concrete
algorithm class. To add a new algorithm, simply create a new module in
this package, define a class that inherits from :class:`BaseAlgorithm`
(or :class:`OnPolicyBase` / :class:`OffPolicyBase` plus
:class:`BaseAlgorithm`), and set ``algorithm_name = "..."``. The class
will be registered automatically on import.
"""

from mylab.rl.alg.base_alg import (
    ALGORITHM_REGISTRY,
    BaseAlgorithm,
    OnPolicyBase,
    OffPolicyBase,
    register_algorithm,
    resolve_algorithm,
)
from mylab.rl.alg.ppo import PPO
from mylab.rl.alg.vision_ppo import VisionPPO
from mylab.rl.alg.moe_ppo import MoEPPO
from mylab.rl.alg.flashsac import FlashSAC
from mylab.rl.alg.vision_flashsac import VisionFlashSAC

__all__ = [
    "ALGORITHM_REGISTRY",
    "BaseAlgorithm",
    "OnPolicyBase",
    "OffPolicyBase",
    "PPO",
    "VisionPPO",
    "MoEPPO",
    "FlashSAC",
    "VisionFlashSAC",
    "register_algorithm",
    "resolve_algorithm",
]
