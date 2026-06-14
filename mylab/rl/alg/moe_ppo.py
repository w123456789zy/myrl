# Copyright (c) 2025, Your Name
# All rights reserved.

"""MoE-PPO: PPO with a Mixture-of-Experts encoder for proprioception.

Architecture
------------
1. **Vision branch**: front-camera image (3, H, W) → CNNEncoder → vision_feat
   (cnn_feature_dim, default 64).  Mirrors :class:`ActorVision`.
2. **MoE proprio encoder**: the history-stacked proprio vector
   (num_history * num_single_obs) is reshaped into a sequence of
   ``num_history`` tokens of size ``num_single_obs`` and passed through a
   :class:`MoEBlock` to produce a 128-dim latent.  Soft-gated mixture of
   ``num_experts`` (default 4) experts with a single hidden layer.
3. **Actor head**: [vision_feat, MoE_latent, current_proprio] → MLP → action.
4. **Critic head**: [vision_feat, MoE_latent, current_proprio, privileged] →
   MLP → value.

Additional loss
---------------
A load-balance loss is added to the actor loss to discourage expert
collapse (all inputs routed to the same expert):

    L_balance = MSE(mean(gate_weights, dim=batch), 1/num_experts)
    L_total = L_ppo + balance_loss_coef * L_balance

Reference
---------
go2_rl_gym (vbot_rl_gym) MoE-CTS. Simplified: we omit the Teacher-Student
distillation since our Critic already uses privileged observations, and we
collapse the multi-expert gating to a single shared MoE block.
"""

from __future__ import annotations

from itertools import chain
from typing import Tuple

import torch
import torch.nn as nn

from mylab.rl.alg.vision_ppo import VisionPPO
from mylab.rl.modules.mlp import MLP
from mylab.rl.modules.cnn.cnn_encoder import CNNEncoder
from mylab.rl.modules.moe import MoEBlock
from mylab.rl.modules.normalization import EmpiricalNormalization
from mylab.rl.modules.distribution import GaussianDistribution
from mylab.rl.storage.rollout_storage import RolloutStorage
from mylab.env.vec_env import VecEnv


# ---------------------------------------------------------------------------
# Actor
# ---------------------------------------------------------------------------


class ActorMoEVision(nn.Module):
    """Actor with CNN vision + MoE proprio encoder.

    The input ``obs`` has shape ``(B, num_history * num_single_obs)``; the
    last ``num_single_obs`` columns are the current proprioceptive step.
    Earlier columns are previous steps (0 = oldest, -1 = newest).
    """

    def __init__(
        self,
        num_single_obs: int,
        num_actions: int,
        num_history: int,
        image_shape: tuple[int, ...],
        cnn_feature_dim: int = 64,
        moe_hidden_dim: int = 512,
        moe_out_dim: int = 128,
        num_experts: int = 4,
        actor_hidden_dims: tuple[int, ...] = (512, 256, 128),
        activation: str = "elu",
        obs_normalization: bool = True,
        init_std: float = 0.5,
        std_type: str = "scalar",
        learn_std: bool = True,
    ) -> None:
        super().__init__()
        self.num_single_obs = num_single_obs
        self.num_history = num_history
        self.num_actions = num_actions
        self.cnn_feature_dim = cnn_feature_dim
        self.moe_out_dim = moe_out_dim

        # Observation normalization is applied on each single-step slice.
        self.obs_normalizer: nn.Module = (
            EmpiricalNormalization(num_single_obs) if obs_normalization else nn.Identity()
        )

        # Vision branch.
        self.cnn = CNNEncoder(
            image_shape[0], cnn_feature_dim, input_size=(image_shape[1], image_shape[2])
        )

        # MoE encoder over the full history (flattened).
        self.moe = MoEBlock(
            in_dim=num_single_obs * num_history,
            hidden_dim=moe_hidden_dim,
            out_dim=moe_out_dim,
            num_experts=num_experts,
        )

        # Action distribution head.
        self.distribution: GaussianDistribution = GaussianDistribution(
            num_actions,
            init_std=init_std,
            std_type=std_type,
            learn_std=learn_std,
        )

        # Final MLP: [vis_feat, moe_feat, current_proprio] → action_mean.
        mlp_in = cnn_feature_dim + moe_out_dim + num_single_obs
        self.mlp = MLP(mlp_in, self.distribution.input_dim, actor_hidden_dims, activation)
        self.distribution.init_mlp_weights(self.mlp)

    # ------------------------------------------------------------------
    def _split_history(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split history-stacked obs into (current_step, all_steps)."""
        # ``obs`` shape: (B, num_history * num_single_obs)
        b = obs.shape[0]
        obs_view = obs.view(b, self.num_history, self.num_single_obs)
        current = obs_view[:, -1, :]  # (B, num_single_obs)
        return current, obs

    def forward(
        self,
        obs: torch.Tensor,
        image: torch.Tensor,
        stochastic: bool = True,
    ) -> torch.Tensor:
        if image is None:
            raise ValueError("ActorMoEVision requires image input but got None")

        current_obs, history_obs = self._split_history(obs)

        # Normalize the current step (the MoE sees unnormalized history so
        # the gate learns to compare across the same scale as the network
        # was trained on; both are kept consistent via the same normalizer
        # that is updated on every step).
        current_norm = self.obs_normalizer(current_obs)

        # MoE: takes the full history (raw scale; consistent with what
        # _prepare_obs feeds into the obs_history_buf).
        moe_feat = self.moe(history_obs)

        # Vision.
        image_features = self.cnn(image)

        # Concatenate and run the actor head.
        combined = torch.cat([image_features, moe_feat, current_norm], dim=-1)
        mlp_out = self.mlp(combined)

        if stochastic:
            self.distribution.update(mlp_out)
            return self.distribution.sample()
        return self.distribution.deterministic_output(mlp_out)

    def update_normalization(self, obs: torch.Tensor) -> None:
        if isinstance(self.obs_normalizer, nn.Identity):
            return
        # Update normalizer with the *current* step only.
        current = obs.view(obs.shape[0], self.num_history, self.num_single_obs)[:, -1, :]
        self.obs_normalizer.update(current)

    @property
    def output_entropy(self) -> torch.Tensor:
        return self.distribution.entropy

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        return self.distribution.params

    def get_output_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions)

    def get_kl_divergence(
        self,
        old_params: tuple[torch.Tensor, ...],
        new_params: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        return self.distribution.kl_divergence(old_params, new_params)

    # ------------------------------------------------------------------
    def compute_balance_loss(self) -> torch.Tensor:
        """Load-balance loss: delegate to :py:meth:`MoEBlock.balance_loss`.

        See :class:`mylab.rl.modules.moe.MoEBlock` for details. We keep this
        thin method on the actor for backward compatibility with existing
        training loops, but the actual computation lives in the MoE module
        so that any algorithm (not just PPO) can reuse the same loss.
        """
        return self.moe.balance_loss()


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------


class CriticMoEVision(nn.Module):
    """Critic with CNN vision + MoE proprio + privileged observations.

    Critic uses the same MoE encoder as the actor (independent weights) so
    that the value function can also benefit from multi-expert structure.
    """

    def __init__(
        self,
        num_single_obs: int,
        num_priv_obs: int,
        num_history: int,
        image_shape: tuple[int, ...],
        cnn_feature_dim: int = 64,
        moe_hidden_dim: int = 512,
        moe_out_dim: int = 128,
        num_experts: int = 4,
        critic_hidden_dims: tuple[int, ...] = (512, 256, 128),
        activation: str = "elu",
    ) -> None:
        super().__init__()
        self.num_single_obs = num_single_obs
        self.num_history = num_history

        self.cnn = CNNEncoder(
            image_shape[0], cnn_feature_dim, input_size=(image_shape[1], image_shape[2])
        )
        self.moe = MoEBlock(
            in_dim=num_single_obs * num_history,
            hidden_dim=moe_hidden_dim,
            out_dim=moe_out_dim,
            num_experts=num_experts,
        )

        # MLP input: [vis, moe, current_proprio, privileged]
        mlp_in = cnn_feature_dim + moe_out_dim + num_single_obs + num_priv_obs
        self.mlp = MLP(mlp_in, 1, critic_hidden_dims, activation)

    def _split_history(self, obs: torch.Tensor) -> torch.Tensor:
        b = obs.shape[0]
        return obs.view(b, self.num_history, self.num_single_obs)[:, -1, :]

    def forward(
        self,
        obs: torch.Tensor,
        priv_obs: torch.Tensor,
        image: torch.Tensor,
    ) -> torch.Tensor:
        if image is None:
            raise ValueError("CriticMoEVision requires image input but got None")
        current_obs = self._split_history(obs)
        moe_feat = self.moe(obs)
        image_features = self.cnn(image)
        combined = torch.cat([image_features, moe_feat, current_obs, priv_obs], dim=-1)
        return self.mlp(combined)

    def compute_balance_loss(self) -> torch.Tensor:
        """Load-balance loss: delegate to :py:meth:`MoEBlock.balance_loss`.

        See :class:`mylab.rl.modules.moe.MoEBlock` for details. The actual
        computation lives in the MoE module so that any algorithm (not
        just PPO) can reuse the same loss.
        """
        return self.moe.balance_loss()


# ---------------------------------------------------------------------------
# MoE-PPO algorithm
# ---------------------------------------------------------------------------


class MoEPPO(VisionPPO):
    """PPO with a Mixture-of-Experts proprio encoder (and optional vision).

    Inherits :class:`VisionPPO` for the training loop / PPO machinery; only
    the actor and critic are replaced with the MoE variants, and the
    ``update`` step adds a load-balance loss.
    """

    algorithm_name: str = "moe_vision_ppo"

    # ------------------------------------------------------------------
    def act(self, obs: torch.Tensor, image: torch.Tensor | None = None) -> torch.Tensor:
        actions = self.actor(obs, image, stochastic=True).detach()
        # Critic uses privileged obs if available, else falls back to obs.
        priv = self._get_priv_obs(obs)
        values = self.critic(obs, priv, image).detach()

        self.transition.observations = obs
        self.transition.image = image
        self.transition.actions = actions
        self.transition.values = values
        self.transition.actions_log_prob = self.actor.get_output_log_prob(actions).detach().unsqueeze(-1)
        self.transition.distribution_params = tuple(p.detach() for p in self.actor.output_distribution_params)
        return actions

    def process_env_step(
        self,
        obs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict,
        image: torch.Tensor | None = None,
    ) -> None:
        """Record one env step into the storage.

        Overrides :py:meth:`PPO.process_env_step` to additionally stash the
        privileged observation (used by the critic) on the transition so it
        flows into the rollout storage.
        """
        self.actor.update_normalization(obs)
        # Capture privileged obs for the critic.  The env's buffer last
        # dim must equal ``env.num_privileged_obs``, which the rollout
        # storage was sized to.  Any mismatch is a configuration bug in
        # the env and must be fixed there — we deliberately do NOT
        # silently slice/pad here.
        env = getattr(self, "env_ref", None)
        if env is not None and hasattr(env, "privileged_obs_buf") and env.privileged_obs_buf is not None:
            self.transition.priv_observations = env.privileged_obs_buf.clone()
        # Reward normalization (mirrors PPO.process_env_step).
        if getattr(self, "reward_normalizer", None) is not None:
            time_outs = extras.get("time_outs", torch.zeros_like(dones))
            dones_b = dones.bool() if dones.dtype != torch.bool else dones
            time_outs_b = time_outs.bool() if time_outs.dtype != torch.bool else time_outs
            terminated = dones_b & ~time_outs_b
            agg_reward = rewards.mean()
            self.reward_normalizer.update_reward_stats(
                reward=agg_reward,
                terminated=terminated.max().unsqueeze(0),
                truncated=time_outs_b.max().unsqueeze(0),
            )
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(-1).to(self.device),
                -1,
            )
        self.storage.add_transition(self.transition)
        self.transition.clear()

    def compute_returns(self, obs: torch.Tensor, image: torch.Tensor | None = None) -> None:
        st = self.storage
        priv = self._get_priv_obs(obs)
        last_values = self.critic(obs, priv, image).detach()
        advantage = 0.0
        for step in reversed(range(st.num_transitions_per_env)):
            if step == st.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = st.values[step + 1]
            next_is_not_terminal = 1.0 - st.dones[step].float()
            delta = st.rewards[step] + next_is_not_terminal * self.gamma * next_values - st.values[step]
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            st.returns[step] = advantage + st.values[step]
        st.advantages = st.returns - st.values
        if not self.normalize_advantage_per_mini_batch:
            st.advantages = (st.advantages - st.advantages.mean()) / (st.advantages.std() + 1e-8)

    # ------------------------------------------------------------------
    def _get_priv_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Return the privileged obs buffer aligned to ``obs``'s env ordering.

        The ``env_ref`` is stashed during :py:meth:`construct_algorithm`. If
        unavailable (e.g. during inference without an env attached), we
        return a zero tensor of the correct shape so the critic's MLP can
        still forward.
        """
        env = getattr(self, "env_ref", None)
        if env is None or not hasattr(env, "privileged_obs_buf") or env.privileged_obs_buf is None:
            priv_dim = getattr(self, "critic_num_priv_obs", 0)
            if priv_dim <= 0:
                priv_dim = obs.shape[-1]  # safe fallback: same shape as obs
            return torch.zeros(obs.shape[0], priv_dim, device=obs.device)
        return env.privileged_obs_buf

    # ------------------------------------------------------------------
    def update(self) -> dict[str, float]:
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_balance_loss = 0.0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for batch in generator:
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (
                        batch.advantages.std() + 1e-8
                    )

            # Forward through actor.
            self.actor(batch.observations, batch.observations_image, stochastic=True)
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)

            # Forward through critic with privileged obs.
            # The storage has separate field ``observations_priv`` if priv
            # is stored; otherwise the critic receives zeros (handled by
            # construct_algorithm wiring below).
            priv_obs = getattr(batch, "observations_priv", None)
            if priv_obs is None:
                priv_obs = torch.zeros(
                    batch.observations.shape[0], self.critic_num_priv_obs,
                    device=batch.observations.device,
                )
            values = self.critic(batch.observations, priv_obs, batch.observations_image)

            entropy = self.actor.output_entropy
            distribution_params = self.actor.output_distribution_params

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)
                    kl_mean = torch.mean(kl)
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob, -1))
            advantage = torch.squeeze(batch.advantages, -1)
            surrogate = -advantage * ratio
            surrogate_clipped = -advantage * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                values_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (values_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            # Load-balance loss (NEW for MoE).
            balance_loss = (
                self.actor.compute_balance_loss() + self.critic.compute_balance_loss()
            ) * 0.5

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy.mean()
                + self.balance_loss_coef * balance_loss
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                chain(self.actor.parameters(), self.critic.parameters()), self.max_grad_norm
            )
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            mean_balance_loss += balance_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_balance_loss /= num_updates

        self.storage.clear()

        return {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "balance": mean_balance_loss,
            "learning_rate": self.learning_rate,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def construct_algorithm(
        obs: torch.Tensor,
        env: VecEnv,
        cfg: dict,
        device: str,
    ) -> MoEPPO:
        obs_dim = obs.shape[-1]
        num_single_obs = getattr(env, "num_single_obs", obs_dim)
        # ``num_history`` is the number of stacked obs steps. Prefer the
        # explicit attribute, otherwise derive from the env's full/stacked
        # obs dims (which are the canonical sources in Go2BaseEnv).
        num_history = getattr(env, "num_history", None)
        if num_history is None or num_history <= 0:
            num_history = max(1, obs_dim // max(num_single_obs, 1))
        num_priv_obs = getattr(env, "num_privileged_obs", None) or 0

        algo_cfg = cfg.get("algorithm", {})
        actor_hidden_dims = tuple(algo_cfg.get("actor_hidden_dims", [512, 256, 128]))
        critic_hidden_dims = tuple(algo_cfg.get("critic_hidden_dims", [512, 256, 128]))
        activation = algo_cfg.get("activation", "elu")

        image_shape = algo_cfg.get("image_shape")
        cnn_feature_dim = algo_cfg.get("cnn_feature_dim", 64)
        moe_hidden_dim = algo_cfg.get("moe_hidden_dim", 512)
        moe_out_dim = algo_cfg.get("moe_out_dim", 128)
        num_experts = algo_cfg.get("num_experts", 4)
        balance_loss_coef = algo_cfg.get("balance_loss_coef", 0.01)

        actor = ActorMoEVision(
            num_single_obs=num_single_obs,
            num_actions=env.num_actions,
            num_history=num_history,
            image_shape=image_shape,
            cnn_feature_dim=cnn_feature_dim,
            moe_hidden_dim=moe_hidden_dim,
            moe_out_dim=moe_out_dim,
            num_experts=num_experts,
            actor_hidden_dims=actor_hidden_dims,
            activation=activation,
            init_std=algo_cfg.get("init_std", 0.5),
            std_type=algo_cfg.get("std_type", "scalar"),
            learn_std=algo_cfg.get("learn_std", True),
        )
        print(f"ActorMoEVision Model:\n{actor}")

        critic = CriticMoEVision(
            num_single_obs=num_single_obs,
            num_priv_obs=num_priv_obs,
            num_history=num_history,
            image_shape=image_shape,
            cnn_feature_dim=cnn_feature_dim,
            moe_hidden_dim=moe_hidden_dim,
            moe_out_dim=moe_out_dim,
            num_experts=num_experts,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
        )
        print(f"CriticMoEVision Model:\n{critic}")

        priv_dim = num_priv_obs if num_priv_obs and num_priv_obs > 0 else 0
        storage = RolloutStorage(
            num_envs=env.num_envs,
            num_transitions_per_env=cfg["num_steps_per_env"],
            obs_shape=[obs_dim],
            actions_shape=[env.num_actions],
            device=device,
            image_shape=image_shape,
            priv_obs_shape=[priv_dim] if priv_dim > 0 else None,
        )

        moe_ppo = MoEPPO(
            actor=actor,
            critic=critic,
            storage=storage,
            device=device,
            num_learning_epochs=algo_cfg.get("num_learning_epochs", 5),
            num_mini_batches=algo_cfg.get("num_mini_batches", 4),
            clip_param=algo_cfg.get("clip_param", 0.2),
            gamma=algo_cfg.get("gamma", 0.99),
            lam=algo_cfg.get("lam", 0.95),
            value_loss_coef=algo_cfg.get("value_loss_coef", 0.5),
            entropy_coef=algo_cfg.get("entropy_coef", 0.005),
            learning_rate=algo_cfg.get("learning_rate", 0.001),
            max_grad_norm=algo_cfg.get("max_grad_norm", 1.0),
            use_clipped_value_loss=algo_cfg.get("use_clipped_value_loss", True),
            schedule=algo_cfg.get("schedule", "adaptive"),
            desired_kl=algo_cfg.get("desired_kl", 0.01),
            normalize_advantage_per_mini_batch=algo_cfg.get("normalize_advantage_per_mini_batch", False),
            normalize_reward=algo_cfg.get("normalize_reward", True),
            reward_G_max=algo_cfg.get("reward_G_max", 10.0),
        )

        # Stash additional state.
        moe_ppo.balance_loss_coef = balance_loss_coef
        moe_ppo.critic_num_priv_obs = priv_dim
        moe_ppo.env_ref = env

        return moe_ppo
