from __future__ import annotations

from itertools import chain

import torch
import torch.nn as nn

from mylab.rl.alg.ppo import PPO
from mylab.rl.modules.mlp import MLP
from mylab.rl.modules.cnn.cnn_encoder import CNNEncoder
from mylab.rl.modules.normalization import EmpiricalNormalization
from mylab.rl.modules.distribution import GaussianDistribution
from mylab.rl.storage.rollout_storage import RolloutStorage
from mylab.env.vec_env import VecEnv


class ActorVision(nn.Module):
    """Actor with CNN vision encoder."""

    def __init__(
        self,
        num_obs: int,
        num_actions: int,
        image_shape: tuple[int, ...],
        cnn_feature_dim: int,
        hidden_dims: tuple[int, ...] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = True,
        init_std: float = 1.0,
        std_type: str = "scalar",
        learn_std: bool = True,
    ) -> None:
        super().__init__()
        self.obs_normalizer = EmpiricalNormalization(num_obs) if obs_normalization else nn.Identity()
        self.cnn = CNNEncoder(image_shape[0], cnn_feature_dim, input_size=(image_shape[1], image_shape[2]))
        self.distribution = GaussianDistribution(
            num_actions,
            init_std=init_std,
            std_type=std_type,
            learn_std=learn_std,
        )
        self.mlp = MLP(num_obs + cnn_feature_dim, self.distribution.input_dim, hidden_dims, activation)
        self.distribution.init_mlp_weights(self.mlp)

    def forward(self, obs: torch.Tensor, image: torch.Tensor, stochastic: bool = True) -> torch.Tensor:
        if image is None:
            raise ValueError("ActorVision requires image input but got None")
        obs_norm = self.obs_normalizer(obs)
        image_features = self.cnn(image)
        combined = torch.cat([obs_norm, image_features], dim=-1)
        mlp_out = self.mlp(combined)
        if stochastic:
            self.distribution.update(mlp_out)
            return self.distribution.sample()
        return self.distribution.deterministic_output(mlp_out)

    def update_normalization(self, obs: torch.Tensor) -> None:
        if not isinstance(self.obs_normalizer, nn.Identity):
            self.obs_normalizer.update(obs)

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


class VisionCritic(nn.Module):
    """Critic with CNN vision encoder."""

    def __init__(
        self,
        num_obs: int,
        image_shape: tuple[int, ...],
        cnn_feature_dim: int,
        hidden_dims: tuple[int, ...] = (256, 256, 256),
        activation: str = "elu",
    ) -> None:
        super().__init__()
        self.cnn = CNNEncoder(image_shape[0], cnn_feature_dim, input_size=(image_shape[1], image_shape[2]))
        self.mlp = MLP(num_obs + cnn_feature_dim, 1, hidden_dims, activation)

    def forward(self, obs: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        if image is None:
            raise ValueError("VisionCritic requires image input but got None")
        image_features = self.cnn(image)
        combined = torch.cat([obs, image_features], dim=-1)
        return self.mlp(combined)


class VisionPPO(PPO):
    """PPO with vision (CNN + MLP)."""

    #: Auto-registered name (consumed by :data:`ALGORITHM_REGISTRY`).
    algorithm_name: str = "vision_ppo"

    def act(self, obs: torch.Tensor, image: torch.Tensor | None = None) -> torch.Tensor:
        actions = self.actor(obs, image, stochastic=True).detach()
        values = self.critic(obs, image).detach()
        self.transition.observations = obs
        self.transition.image = image
        self.transition.actions = actions
        self.transition.values = values
        self.transition.actions_log_prob = self.actor.get_output_log_prob(actions).detach().unsqueeze(-1)
        self.transition.distribution_params = tuple(p.detach() for p in self.actor.output_distribution_params)
        return actions

    def compute_returns(self, obs: torch.Tensor, image: torch.Tensor | None = None) -> None:
        st = self.storage
        last_values = self.critic(obs, image).detach()
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

    def update(self) -> dict[str, float]:
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for batch in generator:
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (
                        batch.advantages.std() + 1e-8
                    )

            self.actor(batch.observations, batch.observations_image, stochastic=True)
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(batch.observations, batch.observations_image)
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

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(chain(self.actor.parameters(), self.critic.parameters()), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates

        self.storage.clear()

        return {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "learning_rate": self.learning_rate,
        }

    @staticmethod
    def construct_algorithm(
        obs: torch.Tensor,
        env: VecEnv,
        cfg: dict,
        device: str,
    ) -> VisionPPO:
        obs_dim = obs.shape[-1]

        algo_cfg = cfg.get("algorithm", {})
        actor_hidden_dims = tuple(algo_cfg.get("actor_hidden_dims", [512, 256, 128]))
        critic_hidden_dims = tuple(algo_cfg.get("critic_hidden_dims", [512, 256, 128]))
        activation = algo_cfg.get("activation", "elu")

        image_shape = algo_cfg.get("image_shape")
        cnn_feature_dim = algo_cfg.get("cnn_feature_dim", 32)

        actor = ActorVision(
            num_obs=obs_dim,
            num_actions=env.num_actions,
            image_shape=image_shape,
            cnn_feature_dim=cnn_feature_dim,
            hidden_dims=actor_hidden_dims,
            activation=activation,
            obs_normalization=True,
            init_std=algo_cfg.get("init_std", 1.0),
            std_type=algo_cfg.get("std_type", "scalar"),
            learn_std=algo_cfg.get("learn_std", True),
        )
        print(f"ActorVision Model:\n{actor}")

        critic = VisionCritic(
            num_obs=obs_dim,
            image_shape=image_shape,
            cnn_feature_dim=cnn_feature_dim,
            hidden_dims=critic_hidden_dims,
            activation=activation,
        )
        print(f"VisionCritic Model:\n{critic}")

        storage = RolloutStorage(
            num_envs=env.num_envs,
            num_transitions_per_env=cfg["num_steps_per_env"],
            obs_shape=[obs_dim],
            actions_shape=[env.num_actions],
            device=device,
            image_shape=image_shape,
        )

        ppo = VisionPPO(
            actor=actor,
            critic=critic,
            storage=storage,
            device=device,
            num_learning_epochs=algo_cfg.get("num_learning_epochs", 5),
            num_mini_batches=algo_cfg.get("num_mini_batches", 4),
            clip_param=algo_cfg.get("clip_param", 0.2),
            gamma=algo_cfg.get("gamma", 0.99),
            lam=algo_cfg.get("lam", 0.95),
            value_loss_coef=algo_cfg.get("value_loss_coef", 1.0),
            entropy_coef=algo_cfg.get("entropy_coef", 0.01),
            learning_rate=algo_cfg.get("learning_rate", 0.001),
            max_grad_norm=algo_cfg.get("max_grad_norm", 1.0),
            use_clipped_value_loss=algo_cfg.get("use_clipped_value_loss", True),
            schedule=algo_cfg.get("schedule", "adaptive"),
            desired_kl=algo_cfg.get("desired_kl", 0.01),
            normalize_advantage_per_mini_batch=algo_cfg.get("normalize_advantage_per_mini_batch", False),
        )

        return ppo
