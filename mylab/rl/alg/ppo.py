# Copyright (c) 2025, Your Name
# All rights reserved.

"""PPO (Proximal Policy Optimization) algorithm.

Simplified from rsl_rl — feedforward only, no RND, no symmetry, no multi-GPU.
"""

from __future__ import annotations

from itertools import chain

import torch
import torch.nn as nn

from mylab.rl.alg.base_alg import BaseAlgorithm, OnPolicyBase
from mylab.rl.modules.mlp import MLP
from mylab.rl.modules.normalization import EmpiricalNormalization
from mylab.rl.modules.distribution import GaussianDistribution, Distribution
from mylab.rl.modules.flashsac.reward_normalization import RewardNormalizer
from mylab.rl.storage.rollout_storage import RolloutStorage
from mylab.env.vec_env import VecEnv


# ---------------------------------------------------------------------------
# Actor — wraps MLP + observation normalization + output distribution.
# This is PPO-specific (not a general reusable module), hence kept here.
# ---------------------------------------------------------------------------


class Actor(nn.Module):
    """Actor model: MLP → GaussianDistribution (stochastic output).

    Matches rsl_rl's MLPModel design:
    - Distribution is a sub-module → ``actor.parameters()`` includes ``std_param``
    - Observation normalizer built-in
    - No explicit weight init — relies on PyTorch's default Kaiming Uniform
    """

    def __init__(
        self,
        num_obs: int,
        num_actions: int,
        hidden_dims: tuple[int, ...] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = True,
        init_std: float = 1.0,
        std_type: str = "scalar",
        learn_std: bool = True,
    ) -> None:
        super().__init__()

        # Observation normalizer
        self.obs_normalizer: nn.Module = EmpiricalNormalization(num_obs) if obs_normalization else nn.Identity()

        # Distribution (sub-module → optimizer sees std_param via actor.parameters())
        self.distribution: Distribution = GaussianDistribution(
            num_actions,
            init_std=init_std,
            std_type=std_type,
            learn_std=learn_std,
        )

        # MLP: input_dim = num_obs, output_dim = distribution.input_dim (= num_actions)
        self.mlp = MLP(num_obs, self.distribution.input_dim, hidden_dims, activation)

        # Distribution-specific weight init (no-op for standard GaussianDistribution)
        self.distribution.init_mlp_weights(self.mlp)

    def forward(self, obs: torch.Tensor, stochastic: bool = True) -> torch.Tensor:
        obs_norm = self.obs_normalizer(obs)
        mlp_out = self.mlp(obs_norm)
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


# ---------------------------------------------------------------------------
# PPO
# ---------------------------------------------------------------------------


class PPO(OnPolicyBase, BaseAlgorithm):
    """Proximal Policy Optimization for continuous control.

    Reference:
        Schulman et al. "Proximal policy optimization algorithms." arXiv:1707.06347 (2017).
    """

    #: Auto-registered name (consumed by :data:`ALGORITHM_REGISTRY`).
    algorithm_name: str = "ppo"

    def __init__(
        self,
        actor: Actor,
        critic: nn.Module,
        storage: RolloutStorage,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 0.001,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        normalize_reward: bool = True,
        reward_G_max: float = 10.0,
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.is_train_mode = True

        # Models — actor wraps MLP + distribution => actor.parameters() has std_param
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)

        # Optimizer — only actor + critic (std_param is inside actor)
        self.optimizer = torch.optim.Adam(
            chain(self.actor.parameters(), self.critic.parameters()),
            lr=learning_rate,
        )

        # Storage
        self.storage = storage
        self.transition = RolloutStorage.Transition()

        # PPO hyper-params
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

        # Reward normalizer — scales per-step reward to ~unit variance using
        self.normalize_reward = normalize_reward
        if self.normalize_reward:
            self.reward_normalizer = RewardNormalizer(
                gamma=gamma,
                G_max=reward_G_max,
                device=torch.device(self.device),
            )
        else:
            self.reward_normalizer = None

    # ------------------------------------------------------------------
    # BaseAlgorithm interface
    # ------------------------------------------------------------------

    def act(self, obs: torch.Tensor, image=None) -> torch.Tensor:
        """Sample actions and store transition data."""
        actions = self.actor(obs, stochastic=True).detach()
        values = self.critic(obs).detach()
        self.transition.observations = obs
        self.transition.image = image
        self.transition.actions = actions
        self.transition.values = values
        self.transition.actions_log_prob = self.actor.get_output_log_prob(actions).detach().unsqueeze(-1)
        self.transition.distribution_params = tuple(p.detach() for p in self.actor.output_distribution_params)
        return actions

    def process_env_step(
        self, obs: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor, extras: dict, image: torch.Tensor | None = None
    ) -> None:
        self.actor.update_normalization(obs)
        # Update reward-normalizer stats *before* applying bootstrapping
        # for time-outs. The normalizer only uses the raw reward signal —
        # the time-out bootstrapped reward is what we store, but the
        # statistic the normalizer tracks should reflect the per-step
        # environment reward.
        #
        # The ``RewardNormalizer`` was designed for off-policy SAC where
        # one sample is processed at a time. For PPO we have a vector
        # of per-env rewards, so we summarize with the mean (or a max
        # if you want conservative behavior). Using the mean makes the
        # running statistic reflect the average per-step reward across
        # the env batch.
        if self.reward_normalizer is not None:
            time_outs = extras.get("time_outs", torch.zeros_like(dones))
            # ``dones`` is a *float* tensor — see ``go2_base.py:636``
            # where env.step() explicitly casts ``done`` to
            # ``gs.tc_float``. ``time_outs`` is an integer tensor
            # (0/1). Neither is directly bitwise-safe. Convert both
            # to bool first so ``&`` and ``~`` work on CUDA.
            dones_b = dones.bool() if dones.dtype != torch.bool else dones
            time_outs_b = time_outs.bool() if time_outs.dtype != torch.bool else time_outs
            terminated = dones_b & ~time_outs_b
            # Aggregate per-env signals down to the shape the normalizer
            # expects (scalar or shape (1,)). The ``G_r`` running estimate
            # is shape (1,), so we feed it a scalar reward.
            agg_reward = rewards.mean()
            agg_terminated = terminated.max().unsqueeze(0)
            agg_truncated = time_outs_b.max().unsqueeze(0)
            self.reward_normalizer.update_reward_stats(
                reward=agg_reward,
                terminated=agg_terminated,
                truncated=agg_truncated,
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

    def compute_returns(self, obs: torch.Tensor, image=None) -> None:
        st = self.storage
        last_values = self.critic(obs).detach()
        # Normalize rewards *before* computing GAE so the value function
        # targets are in the same scale as the value network output.
        if self.reward_normalizer is not None:
            with torch.no_grad():
                st.rewards = self.reward_normalizer.normalize_rewards(st.rewards)
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

            # Recompute distribution and values with current params
            self.actor(batch.observations, stochastic=True)
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(batch.observations)
            entropy = self.actor.output_entropy
            distribution_params = self.actor.output_distribution_params

            # Adaptive KL schedule
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

            # Surrogate loss
            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob, -1))
            advantage = torch.squeeze(batch.advantages, -1)
            surrogate = -advantage * ratio
            surrogate_clipped = -advantage * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value loss
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

    def train_mode(self) -> None:
        self.actor.train()
        self.critic.train()
        self.is_train_mode = True

    def eval_mode(self) -> None:
        self.actor.eval()
        self.critic.eval()
        self.is_train_mode = False

    def save(self) -> dict:
        return {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

    def load(self, loaded_dict: dict, load_cfg: dict | None = None, strict: bool = True) -> bool:
        if load_cfg is None:
            load_cfg = {"actor": True, "critic": True, "optimizer": True, "iteration": True}
        if load_cfg.get("actor"):
            state = loaded_dict["actor_state_dict"]
            # Remap distribution param keys between scalar (std_param) and log (log_std_param)
            current_keys = {k for k, _ in self.actor.named_parameters()}
            ckpt_keys = set(state.keys())
            if "distribution.std_param" in ckpt_keys and "distribution.log_std_param" in current_keys:
                # Old checkpoint (scalar) → new code (log)
                state["distribution.log_std_param"] = state.pop("distribution.std_param").log()
            elif "distribution.log_std_param" in ckpt_keys and "distribution.std_param" in current_keys:
                # New checkpoint (log) → old code (scalar)
                state["distribution.std_param"] = state.pop("distribution.log_std_param").exp()
            self.actor.load_state_dict(state, strict=strict)
        if load_cfg.get("critic"):
            self.critic.load_state_dict(loaded_dict["critic_state_dict"], strict=strict)
        if load_cfg.get("optimizer"):
            try:
                self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            except (ValueError, KeyError):
                pass
        return load_cfg.get("iteration", False)

    def get_policy(self) -> nn.Module:
        return self.actor

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @staticmethod
    def construct_algorithm(
        obs: torch.Tensor,
        env: VecEnv,
        cfg: dict,
        device: str,
    ) -> PPO:
        obs_dim = obs.shape[-1]

        algo_cfg = cfg.get("algorithm", {})
        actor_hidden_dims = tuple(algo_cfg.get("actor_hidden_dims", cfg.get("actor_hidden_dims", [512, 256, 128])))
        critic_hidden_dims = tuple(algo_cfg.get("critic_hidden_dims", cfg.get("critic_hidden_dims", [512, 256, 128])))
        activation = algo_cfg.get("activation", cfg.get("activation", "elu"))

        actor = Actor(
            num_obs=obs_dim,
            num_actions=env.num_actions,
            hidden_dims=actor_hidden_dims,
            activation=activation,
            obs_normalization=True,
            init_std=algo_cfg.get("init_std", 1.0),
            std_type=algo_cfg.get("std_type", "scalar"),
            learn_std=algo_cfg.get("learn_std", True),
        )
        print(f"Actor Model:\n{actor}")

        critic = MLP(
            input_dim=obs_dim,
            output_dim=1,
            hidden_dims=critic_hidden_dims,
            activation=activation,
        )
        print(f"Critic Model:\n{critic}")

        image_shape = algo_cfg.get("image_shape") if algo_cfg.get("use_vision") else None
        storage = RolloutStorage(
            num_envs=env.num_envs,
            num_transitions_per_env=cfg["num_steps_per_env"],
            obs_shape=[obs_dim],
            actions_shape=[env.num_actions],
            device=device,
            image_shape=image_shape,
        )

        ppo = PPO(
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
            # Reward normalization is critical when the env returns
            # large-magnitude rewards (e.g. go2-rough with rough-terrain
            # penalties summing to -300/step). Without it, the value
            # function tries to fit returns of -50k and the value-loss
            # term dominates the policy gradient.
            normalize_reward=algo_cfg.get("normalize_reward", True),
            reward_G_max=algo_cfg.get("reward_G_max", 10.0),
        )

        return ppo
