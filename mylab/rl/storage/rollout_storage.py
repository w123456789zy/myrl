# Copyright (c) 2025, Your Name
# All rights reserved.

"""Rollout storage for PPO-style on-policy training.

Simplified from rsl_rl's RolloutStorage — feedforward only, no distillation,
no recurrent support, raw tensors instead of TensorDict.
"""

from __future__ import annotations

import torch


class RolloutStorage:
    """Storage for data collected during a rollout phase.

    Populated by adding transitions during rollout. Provides a mini-batch
    generator for PPO-style updates.
    """

    class Transition:
        """A single state transition, filled incrementally during rollout."""

        def __init__(self) -> None:
            self.observations: torch.Tensor | None = None
            self.image: torch.Tensor | None = None
            self.priv_observations: torch.Tensor | None = None
            self.actions: torch.Tensor | None = None
            self.rewards: torch.Tensor | None = None
            self.dones: torch.Tensor | None = None
            self.values: torch.Tensor | None = None
            self.actions_log_prob: torch.Tensor | None = None
            self.distribution_params: tuple[torch.Tensor, ...] | None = None

        def clear(self) -> None:
            self.__init__()

    class Batch:
        """Mini-batch yielded by :meth:`mini_batch_generator`."""

        def __init__(
            self,
            observations: torch.Tensor,
            observations_image: torch.Tensor | None,
            actions: torch.Tensor,
            values: torch.Tensor,
            advantages: torch.Tensor,
            returns: torch.Tensor,
            old_actions_log_prob: torch.Tensor,
            old_distribution_params: tuple[torch.Tensor, ...],
            observations_priv: torch.Tensor | None = None,
        ) -> None:
            self.observations = observations
            self.observations_image = observations_image
            self.observations_priv = observations_priv
            self.actions = actions
            self.values = values
            self.advantages = advantages
            self.returns = returns
            self.old_actions_log_prob = old_actions_log_prob
            self.old_distribution_params = old_distribution_params

    def __init__(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        obs_shape: tuple[int, ...] | list[int],
        actions_shape: tuple[int, ...] | list[int],
        device: str = "cpu",
        image_shape: tuple[int, ...] | list[int] | None = None,
        priv_obs_shape: tuple[int, ...] | list[int] | None = None,
    ) -> None:
        self.device = device
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=device)
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=device).byte()
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.distribution_params: tuple[torch.Tensor, ...] | None = None
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)

        self.image_shape = image_shape
        if image_shape is not None:
            self.images = torch.zeros(num_transitions_per_env, num_envs, *image_shape, device=device)
        else:
            self.images = None

        # Optional privileged observation buffer (e.g. for MoE-PPO critic).
        self.priv_obs_shape = priv_obs_shape
        if priv_obs_shape is not None:
            self.priv_observations = torch.zeros(
                num_transitions_per_env, num_envs, *priv_obs_shape, device=device
            )
        else:
            self.priv_observations = None

        self.step = 0

    def add_transition(self, transition: Transition) -> None:
        """Record one transition into the storage."""
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow — call clear() before new rollout.")

        # Handle multi-env observations: shape (num_envs, ...) → stored per step
        # If you need per-env shapes that differ, you'd handle it there.
        # We assume obs of shape (num_envs, *obs_shape) for now
        obs = transition.observations  # (num_envs, ...)

        # Pad to match initialized buffer row shape
        self.observations[self.step] = obs  # overwrites per-step slice
        if self.images is not None and transition.image is not None:
            self.images[self.step] = transition.image
        if self.priv_observations is not None and transition.priv_observations is not None:
            self.priv_observations[self.step] = transition.priv_observations
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))

        # Lazy-init distribution params buffer
        if self.distribution_params is None:
            self.distribution_params = tuple(
                torch.zeros(self.num_transitions_per_env, *p.shape, device=self.device)
                for p in transition.distribution_params
            )
        for i, p in enumerate(transition.distribution_params):
            self.distribution_params[i][self.step].copy_(p)

        self.step += 1

    def clear(self) -> None:
        """Reset the write cursor for the next rollout."""
        self.step = 0

    def mini_batch_generator(
        self, num_mini_batches: int, num_epochs: int = 5
    ):
        """Yield shuffled flat mini-batches for feedforward RL updates."""
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        # Flatten (T, E, ...) → (T*E, ...)
        observations = self.observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_distribution_params = tuple(p.flatten(0, 1) for p in self.distribution_params) if self.distribution_params else ()

        observations_image = self.images.flatten(0, 1) if self.images is not None else None
        observations_priv = self.priv_observations.flatten(0, 1) if self.priv_observations is not None else None

        for _epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size
                batch_idx = indices[start:stop]

                yield RolloutStorage.Batch(
                    observations=observations[batch_idx],
                    observations_image=observations_image[batch_idx] if observations_image is not None else None,
                    observations_priv=observations_priv[batch_idx] if observations_priv is not None else None,
                    actions=actions[batch_idx],
                    values=values[batch_idx],
                    advantages=advantages[batch_idx],
                    returns=returns[batch_idx],
                    old_actions_log_prob=old_actions_log_prob[batch_idx],
                    old_distribution_params=tuple(p[batch_idx] for p in old_distribution_params),
                )