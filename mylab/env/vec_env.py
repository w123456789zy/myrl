from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch

VecEnvObs = dict[str, torch.Tensor | dict[str, torch.Tensor]]
"""Observation type returned by :meth:`VecEnv.get_observations`.

Typically contains:
    - ``"state"``: Proprioception / state observations ``(num_envs, num_obs)``.
    - ``"images"``: Visual observations ``(num_envs, C, H, W)`` (optional).
"""

VecEnvStepReturn = tuple[VecEnvObs, torch.Tensor, torch.Tensor, dict]
"""Return type of :meth:`VecEnv.step`.

``(observations, rewards, dones, extras)``
"""


@dataclass(kw_only=True)
class EnvConfig:
    """Base configuration for a vectorized environment.

    All environment-related configuration lives here, including
    simulation parameters, episode settings, and reset behavior.
    Subclasses should add environment-specific fields.
    """

    num_envs: int
    """Number of parallel environments."""

    episode_length_s: float = 0.0
    """Maximum episode duration in seconds. 0 means no limit."""

    seed: int | None = None
    """Random seed for reproducibility."""

    auto_reset: bool = True
    """Whether to automatically reset terminated environments in step()."""

    is_finite_horizon: bool = False
    """Whether the task has a finite horizon (True) or is infinite (False).

    Finite horizon: time limit defines task boundary, done is terminal.
    Infinite horizon: time limit is an artificial cutoff, done is truncated.
    """

    scale_rewards_by_dt: bool = True
    """Whether to multiply rewards by dt for frequency-invariant returns."""

    viewer: dict[str, Any] = field(default_factory=dict)
    """Viewer/rendering configuration."""


class VecEnv(ABC):
    """Abstract base class for a vectorized environment.

    Compared to rsl_rl's VecEnv, this adds ``reset()`` and ``get_rewards()``
    methods, and embeds environment configuration in the :class:`EnvConfig`
    subclass so that each environment carries its own self-contained config.

    Subclasses must implement all abstract methods.
    """

    num_envs: int
    """Number of parallel environments."""

    num_actions: int
    """Dimensionality of the action space."""

    max_episode_length: int | torch.Tensor
    """Maximum episode length in steps (scalar or per-env tensor)."""

    episode_length_buf: torch.Tensor
    """Buffer tracking current episode lengths per environment."""

    device: torch.device | str
    """Device (cpu / cuda) for tensor operations."""

    cfg: EnvConfig
    """Environment configuration dataclass."""

    name: str
    """Human-readable name of the environment (used for logging)."""

    # ------------------------------------------------------------------
    # Abstract methods
    # ------------------------------------------------------------------

    @abstractmethod
    def get_observations(self) -> VecEnvObs:
        """Return the most recent observations.

        Returns:
            Dictionary of observations, typically containing ``"state"`` and
            optionally ``"images"`` keys.
        """
        raise NotImplementedError

    @abstractmethod
    def get_rewards(self) -> torch.Tensor:
        """Return the most recent rewards for all environments.

        Returns:
            Tensor of shape ``(num_envs,)`` with scalar reward per env.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> tuple[VecEnvObs, dict]:
        """Reset the specified environments (or all if None).

        This corresponds to what mjlab calls "reset events" -- entity state
        resets, domain randomization re-sampling, etc. are all triggered
        inside this method.

        Args:
            env_ids: Indices of environments to reset. If ``None``, reset all.

        Returns:
            observations: Post-reset observations (dict with ``"state"`` / ``"images"``).
            extras: Dictionary of extra information (episode logs, etc.).
        """
        raise NotImplementedError

    @abstractmethod
    def step(
        self, actions: torch.Tensor
    ) -> VecEnvStepReturn:
        """Step all environments forward by one control step.

        Args:
            actions: Tensor of shape ``(num_envs, num_actions)``.

        Returns:
            observations: Post-step observations.
            rewards: Scalar rewards ``(num_envs,)``.
            dones: Done flags ``(num_envs,)`` combining both terminated and
                truncated signals into a single boolean per env.
            extras: Dictionary of extra information (``"time_outs"``, ``"log"``, etc.).
        """
        raise NotImplementedError

    @abstractmethod
    def seed(self, seed: int = -1) -> int:
        """Set the random seed and return the effective seed."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Clean up resources (renderer, simulation, etc.)."""
        raise NotImplementedError

    @property
    def unwrapped(self) -> VecEnv:
        """Return the innermost environment (for wrapper chains)."""
        return self