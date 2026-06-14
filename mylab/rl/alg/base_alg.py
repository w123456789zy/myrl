from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Type

import torch
import torch.nn as nn

from mylab.env.vec_env import VecEnv


# ---------------------------------------------------------------------------
# Algorithm registry
# ---------------------------------------------------------------------------
#
# Algorithms are registered through two equivalent paths:
#
#   1. Explicit registration via :func:`register_algorithm`:
#
#          register_algorithm("my_algo", "my.module.MyAlgo")
#
#   2. Class decorator via :func:`register_algorithm` (used as a decorator):
#
#          @register_algorithm("my_algo")
#          class MyAlgo(BaseAlgorithm):
#              ...
#
#   3. Auto-registration via :meth:`BaseAlgorithm.__init_subclass__`: any
#      subclass that defines ``algorithm_name = "..."`` is registered under
#      that name automatically.
#
# The training runner imports :data:`ALGORITHM_REGISTRY` and looks up the
# algorithm class by name; it does not need to know the module path.

ALGORITHM_REGISTRY: dict[str, type["BaseAlgorithm"]] = {}


def register_algorithm(
    name: str | type["BaseAlgorithm"],
    cls_or_path: str | type["BaseAlgorithm"] | None = None,
) -> type["BaseAlgorithm"]:
    """Register an algorithm class under a string name.

    Can be used in three ways::

        @register_algorithm("ppo")
        class PPO(BaseAlgorithm): ...

        @register_algorithm             # uses cls.algorithm_name
        class PPO(BaseAlgorithm):
            algorithm_name = "ppo"

        register_algorithm("ppo", "mylab.rl.alg.ppo.PPO")   # deferred import
    """
    # Form 1: used as @register_algorithm("name") → returns decorator
    if isinstance(name, type) and cls_or_path is None:
        cls = name
        cls_name = getattr(cls, "algorithm_name", cls.__name__.lower())
        ALGORITHM_REGISTRY[cls_name] = cls
        return cls

    # Form 1b: used as @register_algorithm (no args) → decorator factory
    if isinstance(name, str) and cls_or_path is None:
        def _decorator(cls: type["BaseAlgorithm"]) -> type["BaseAlgorithm"]:
            ALGORITHM_REGISTRY[name] = cls
            return cls
        return _decorator

    assert isinstance(name, str)
    # Form 2: explicit class
    if isinstance(cls_or_path, type):
        ALGORITHM_REGISTRY[name] = cls_or_path
        return cls_or_path
    # Form 3: deferred import path
    assert isinstance(cls_or_path, str)
    mod_path, _, attr = cls_or_path.rpartition(".")
    mod = __import__(mod_path, fromlist=[attr])
    cls = getattr(mod, attr)
    ALGORITHM_REGISTRY[name] = cls
    return cls


def resolve_algorithm(name: str) -> type["BaseAlgorithm"]:
    """Look up an algorithm class by name from :data:`ALGORITHM_REGISTRY`."""
    cls = ALGORITHM_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown algorithm: '{name}'. "
            f"Available: {sorted(ALGORITHM_REGISTRY.keys())}. "
            f"Either implement it in mylab.rl.alg.<name> and let it inherit "
            f"BaseAlgorithm, or call register_algorithm('{name}', '<module.path>')."
        )
    return cls


# ---------------------------------------------------------------------------
# Mixins
# ---------------------------------------------------------------------------
#
# OnPolicyBase / OffPolicyBase are *mixins* that bundle the most common
# pieces of state and methods shared between the on-policy and off-policy
# algorithm families. They DO NOT replace BaseAlgorithm's ABC interface —
# they sit next to it in the MRO and add reusable defaults.
#
# The expected MRO is::
#
#     class PPO(OnPolicyBase, BaseAlgorithm):
#         ...
#
# The mixin MUST come before BaseAlgorithm so that its methods override
# the abstract ones — but every abstract method declared in BaseAlgorithm
# still has to be implemented in the leaf class.

class OnPolicyBase:
    """Mixin shared by on-policy algorithms (PPO family).

    Provides:
    * a default :meth:`_on_policy_post_step` (timeout bootstrap),
    * a default :meth:`_maybe_adapt_lr` (KL-adaptive schedule).
    """

    schedule: str = "adaptive"
    desired_kl: float | None = 0.01
    learning_rate: float = 1e-3
    optimizer: torch.optim.Optimizer
    gamma: float = 0.99

    def _on_policy_post_step(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict,
        last_values: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the GAE-style timeout bootstrap to the current step's rewards.

        Returns the modified reward tensor (in-place on a copy).
        """
        rewards = rewards.clone()
        if "time_outs" in extras:
            rewards = rewards + self.gamma * torch.squeeze(
                last_values * extras["time_outs"].unsqueeze(-1).to(last_values.device),
                -1,
            )
        return rewards

    def _maybe_adapt_lr(self, kl_mean: torch.Tensor) -> None:
        """KL-adaptive learning rate schedule (rsl_rl style)."""
        if self.desired_kl is None or self.schedule != "adaptive":
            return
        if kl_mean > self.desired_kl * 2.0:
            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.learning_rate


class OffPolicyBase:
    """Mixin shared by off-policy algorithms (FlashSAC family).

    Provides:
    * a default :meth:`_compute_terminated_truncated` helper,
    * a default :meth:`_tau_soft_update` for target networks.
    """

    gamma: float = 0.99
    n_step: int = 1
    target_tau: float = 0.005
    critic: nn.Module
    target_critic: nn.Module

    @staticmethod
    def _compute_terminated_truncated(
        dones: torch.Tensor,
        extras: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split ``dones`` into ``terminated`` and ``truncated`` flags.

        ``terminated = dones - time_outs`` (clipped to [0, 1]).
        ``truncated = time_outs``.
        """
        truncated = extras.get("time_outs", torch.zeros_like(dones)).float()
        terminated = (dones.float() - truncated).clamp(0.0, 1.0)
        return terminated, truncated

    def _tau_soft_update(self) -> None:
        """Polyak-average the target critic towards the online critic."""
        tau = self.target_tau
        with torch.no_grad():
            for target_param, param in zip(
                self.target_critic.parameters(), self.critic.parameters()
            ):
                target_param.data.lerp_(param.data, tau)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BaseAlgorithm(ABC):
    """Abstract base class for all reinforcement learning algorithms.

    Every RL algorithm (PPO, SAC, TD3, etc.) must inherit from this class
    and implement all abstract methods. This enforces a consistent interface
    that the training runner can depend on.

    The lifecycle of a training iteration is::

        # 1. Collect rollout
        obs = env.get_observations()
        for _ in range(num_steps):
            actions = alg.act(obs)
            obs, rewards, dones, extras = env.step(actions)
            alg.process_env_step(obs, rewards, dones, extras)

        # 2. Compute targets
        last_obs = env.get_observations()
        alg.compute_returns(last_obs)

        # 3. Update policy
        loss_dict = alg.update()
    """

    # ------------------------------------------------------------------
    # Class-level metadata (used by the registry)
    # ------------------------------------------------------------------

    #: Optional human-readable name. If set, the class auto-registers into
    #: :data:`ALGORITHM_REGISTRY` on import.
    algorithm_name: str | None = None

    # ------------------------------------------------------------------
    # Shared attributes (present in every algorithm)
    # ------------------------------------------------------------------

    device: str
    """Device for computation ('cpu' or 'cuda')."""

    is_train_mode: bool = True
    """Whether the algorithm is in training mode."""

    learning_rate: float = 0.001
    """Current learning rate (may be adapted during training)."""

    # ------------------------------------------------------------------
    # Auto-registration
    # ------------------------------------------------------------------

    def __init_subclass__(cls, **kwargs) -> None:  # noqa: D401
        super().__init_subclass__(**kwargs)
        name = getattr(cls, "algorithm_name", None)
        if name:
            ALGORITHM_REGISTRY.setdefault(name, cls)

    # ------------------------------------------------------------------
    # Abstract methods — must be implemented by every algorithm
    # ------------------------------------------------------------------

    @abstractmethod
    def act(self, obs: torch.Tensor, image: torch.Tensor | None = None) -> torch.Tensor:
        """Sample actions given the current observations.

        This method should also record any intermediate data needed for
        later training (e.g. hidden states, values, log-probs).

        Args:
            obs: Current observations of shape ``(num_envs, num_obs)``.
            image: Visual observations ``(num_envs, C, H, W)`` (optional).

        Returns:
            Sampled actions of shape ``(num_envs, num_actions)``.
        """
        raise NotImplementedError

    @abstractmethod
    def process_env_step(
        self,
        obs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict,
        image: torch.Tensor | None = None,
    ) -> None:
        """Record one environment step into the rollout storage.

        Called after each ``env.step()`` during data collection.

        Args:
            obs: Post-step observations.
            rewards: Rewards ``(num_envs,)``.
            dones: Done flags ``(num_envs,)``, combining termination + truncation.
            extras: Extra info dict (may contain ``"time_outs"``).
            image: Visual observations for this step (optional).
        """
        raise NotImplementedError

    @abstractmethod
    def compute_returns(self, obs: torch.Tensor) -> None:
        """Compute return and advantage targets from the stored rollout.

        Args:
            obs: Observations from the *last* step (used for bootstrapping).
        """
        raise NotImplementedError

    @abstractmethod
    def update(self) -> dict[str, float]:
        """Run one or more optimization epochs over the stored rollout.

        Returns:
            Dictionary mapping loss names to scalar values (averaged over
            all mini-batch updates in this iteration).
        """
        raise NotImplementedError

    @abstractmethod
    def train_mode(self) -> None:
        """Switch all learnable models to training mode."""
        raise NotImplementedError

    @abstractmethod
    def eval_mode(self) -> None:
        """Switch all learnable models to evaluation mode."""
        raise NotImplementedError

    @abstractmethod
    def save(self) -> dict:
        """Return a dictionary of all learnable state to be checkpointed.

        Typical keys: ``"actor_state_dict"``, ``"critic_state_dict"``,
        ``"optimizer_state_dict"``.
        """
        raise NotImplementedError

    @abstractmethod
    def load(
        self,
        loaded_dict: dict,
        load_cfg: dict | None = None,
        strict: bool = True,
    ) -> bool:
        """Restore learnable state from a checkpoint dictionary.

        Args:
            loaded_dict: The checkpoint dictionary (output of :meth:`save`).
            load_cfg: Per-key flags controlling what to load. ``None`` means
                load everything.
            strict: Whether to enforce strict key matching in ``load_state_dict``.

        Returns:
            Whether the iteration counter should also be restored.
        """
        raise NotImplementedError

    @abstractmethod
    def get_policy(self) -> nn.Module:
        """Return the policy model (for export / inference)."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Static factory
    # ------------------------------------------------------------------

    @staticmethod
    @abstractmethod
    def construct_algorithm(
        obs: torch.Tensor,
        env: VecEnv,
        cfg: dict,
        device: str,
    ) -> BaseAlgorithm:
        """Factory method: build the full algorithm from configuration.

        This is the entry point used by the training runner. It should:
        1. Resolve model classes from ``cfg``.
        2. Build actor & critic models.
        3. Build rollout storage.
        4. Instantiate and return the algorithm.

        Args:
            obs: A sample observation tensor from the environment (used to
                infer observation shapes).
            env: The vectorized environment.
            cfg: Algorithm/hyperparameter configuration dictionary.
            device: Target device string.

        Returns:
            A fully constructed algorithm instance.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Optional convenience methods (may be overridden)
    # ------------------------------------------------------------------

    def compile(self, mode: str | None = None) -> None:
        """Compile models with ``torch.compile`` (optional, override if needed)."""
        pass
