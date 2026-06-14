from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim

from mylab.rl.alg.base_alg import BaseAlgorithm
from mylab.rl.modules.flashsac.network import (
    FlashSACTemperature,
    VisionFlashSACActor,
    VisionFlashSACDoubleCritic,
)
from mylab.rl.modules.flashsac.scheduler import warmup_cosine_decay_scheduler
from mylab.rl.modules.flashsac.reward_normalization import RewardNormalizer
from mylab.env.vec_env import VecEnv


# ---------------------------------------------------------------------------
# Replay Buffer with image support
# ---------------------------------------------------------------------------

class VisionFlashSACReplayBuffer:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        image_shape: tuple[int, ...],
        capacity: int,
        batch_size: int,
        device: torch.device,
    ):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.ptr = 0
        self.size = 0

        self.obs = torch.zeros(capacity, obs_dim, device=device)
        self.next_obs = torch.zeros(capacity, obs_dim, device=device)
        self.actions = torch.zeros(capacity, action_dim, device=device)
        self.rewards = torch.zeros(capacity, device=device)
        self.terminated = torch.zeros(capacity, device=device)
        self.truncated = torch.zeros(capacity, device=device)

        # Keep images on CPU to avoid GPU OOM (image replay buffers are huge)
        self.obs_images = torch.zeros(capacity, *image_shape, device=device)
        self.next_obs_images = torch.zeros(capacity, *image_shape, device=device)

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        obs_image: torch.Tensor,
        next_obs_image: torch.Tensor,
    ) -> None:
        num = obs.shape[0]
        for i in range(num):
            idx = (self.ptr + i) % self.capacity
            self.obs[idx] = obs[i]
            self.actions[idx] = action[i]
            self.rewards[idx] = reward[i]
            self.next_obs[idx] = next_obs[i]
            self.terminated[idx] = terminated[i]
            self.truncated[idx] = truncated[i]
            self.obs_images[idx] = obs_image[i]
            self.next_obs_images[idx] = next_obs_image[i]

        self.ptr = (self.ptr + num) % self.capacity
        self.size = min(self.size + num, self.capacity)

    def can_sample(self) -> bool:
        return self.size >= self.batch_size

    def sample(self) -> dict[str, torch.Tensor]:
        idxs = torch.randint(0, self.size, (self.batch_size,), device=self.device)
        return {
            "observation": self.obs[idxs],
            "observation_image": self.obs_images[idxs].to(self.device),
            "action": self.actions[idxs],
            "reward": self.rewards[idxs],
            "next_observation": self.next_obs[idxs],
            "next_observation_image": self.next_obs_images[idxs].to(self.device),
            "terminated": self.terminated[idxs],
            "truncated": self.truncated[idxs],
        }


# ---------------------------------------------------------------------------
# VisionFlashSAC Algorithm
# ---------------------------------------------------------------------------

class VisionFlashSAC(BaseAlgorithm):
    """FlashSAC with CNN vision encoder for visual observations.

    Supports environments that provide ``{"state": tensor, "images": tensor}``.
    """

    #: Auto-registered name (consumed by :data:`ALGORITHM_REGISTRY`).
    algorithm_name: str = "vision_flashsac"

    def __init__(
        self,
        actor: VisionFlashSACActor,
        critic: VisionFlashSACDoubleCritic,
        target_critic: VisionFlashSACDoubleCritic,
        temperature: FlashSACTemperature,
        actor_optimizer: optim.Optimizer,
        critic_optimizer: optim.Optimizer,
        temperature_optimizer: optim.Optimizer,
        actor_scheduler: optim.lr_scheduler._LRScheduler | None,
        critic_scheduler: optim.lr_scheduler._LRScheduler | None,
        temperature_scheduler: optim.lr_scheduler._LRScheduler | None,
        replay_buffer: VisionFlashSACReplayBuffer,
        reward_normalizer: RewardNormalizer | None,
        actor_update_period: int,
        bc_alpha: float,
        gamma: float,
        n_step: int,
        num_bins: int,
        min_v: float,
        max_v: float,
        target_tau: float,
        target_entropy: float,
        updates_per_iter: int,
        device: str,
    ):
        self.actor = actor
        self.critic = critic
        self.target_critic = target_critic
        self.temperature = temperature

        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.temperature_optimizer = temperature_optimizer
        self.actor_scheduler = actor_scheduler
        self.critic_scheduler = critic_scheduler
        self.temperature_scheduler = temperature_scheduler

        self.replay_buffer = replay_buffer
        self.reward_normalizer = reward_normalizer

        self.actor_update_period = actor_update_period
        self.bc_alpha = bc_alpha
        self.gamma = gamma
        self.n_step = n_step
        self.num_bins = num_bins
        self.min_v = min_v
        self.max_v = max_v
        self.target_tau = target_tau
        self.target_entropy = target_entropy
        self.updates_per_iter = updates_per_iter
        self.device = device

        self._update_step = 0
        self.is_train_mode = True
        self.learning_rate = actor_optimizer.defaults["lr"]

        self.transition = SimpleNamespace()
        self._prev_obs: torch.Tensor | None = None
        self._prev_image: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # BaseAlgorithm interface
    # ------------------------------------------------------------------

    def act(self, obs: torch.Tensor, image: torch.Tensor | None = None) -> torch.Tensor:
        self._prev_obs = obs
        self._prev_image = image
        with torch.no_grad():
            with torch.no_grad():
                actions, _ = self.actor(obs, image, training=self.is_train_mode)
        self.transition.actions = actions
        return actions

    def process_env_step(
        self,
        obs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict,
        image: torch.Tensor | None = None,
    ) -> None:
        if self._prev_obs is None or self._prev_image is None:
            return

        truncated = extras.get("time_outs", torch.zeros_like(dones)).float()
        terminated = (dones.float() - truncated).clamp(0.0, 1.0)

        self.replay_buffer.add(
            self._prev_obs,
            self.transition.actions,
            rewards,
            obs,
            terminated,
            truncated,
            self._prev_image,
            image,
        )

        if self.reward_normalizer is not None:
            self.reward_normalizer.update_reward_stats(rewards, terminated, truncated)

    def compute_returns(self, obs: torch.Tensor, image: torch.Tensor | None = None) -> None:
        pass

    def update(self) -> dict[str, float]:
        if not self.replay_buffer.can_sample():
            return {}

        info_acc: dict[str, float] = {}
        for _ in range(self.updates_per_iter):
            batch = self.replay_buffer.sample()

            if self.reward_normalizer is not None:
                batch["reward"] = self.reward_normalizer.normalize_rewards(batch["reward"])

            do_actor = (self._update_step % self.actor_update_period == 0)
            step_info = self._update_step_once(batch, do_actor)
            self._update_step += 1

            for k, v in step_info.items():
                if k not in info_acc:
                    info_acc[k] = 0.0
                info_acc[k] += v

        for k in info_acc:
            info_acc[k] /= self.updates_per_iter

        self.learning_rate = self.actor_optimizer.param_groups[0]["lr"]
        return info_acc

    def train_mode(self) -> None:
        self.is_train_mode = True
        self.actor.train()
        self.critic.train()
        self.target_critic.train()
        self.temperature.train()

    def eval_mode(self) -> None:
        self.is_train_mode = False
        self.actor.eval()
        self.critic.eval()
        self.target_critic.eval()
        self.temperature.eval()

    def save(self) -> dict:
        return {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "target_critic_state_dict": self.target_critic.state_dict(),
            "temperature_state_dict": self.temperature.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "temperature_optimizer_state_dict": self.temperature_optimizer.state_dict(),
            "update_step": self._update_step,
        }

    def load(
        self,
        loaded_dict: dict,
        load_cfg: dict | None = None,
        strict: bool = True,
    ) -> bool:
        self.actor.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
        self.critic.load_state_dict(loaded_dict["critic_state_dict"], strict=strict)
        self.target_critic.load_state_dict(loaded_dict["target_critic_state_dict"], strict=strict)
        self.temperature.load_state_dict(loaded_dict["temperature_state_dict"], strict=strict)

        if "actor_optimizer_state_dict" in loaded_dict:
            self.actor_optimizer.load_state_dict(loaded_dict["actor_optimizer_state_dict"])
        if "critic_optimizer_state_dict" in loaded_dict:
            self.critic_optimizer.load_state_dict(loaded_dict["critic_optimizer_state_dict"])
        if "temperature_optimizer_state_dict" in loaded_dict:
            self.temperature_optimizer.load_state_dict(loaded_dict["temperature_optimizer_state_dict"])

        self._update_step = loaded_dict.get("update_step", 0)
        return False

    def get_policy(self) -> nn.Module:
        return self.actor

    # ------------------------------------------------------------------
    # Internal update logic
    # ------------------------------------------------------------------

    def _update_step_once(
        self,
        batch: dict[str, torch.Tensor],
        do_actor: bool,
    ) -> dict[str, float]:
        info: dict[str, torch.Tensor] = {}

        if do_actor:
            actor_info = self._update_actor(batch)
            temp_info = self._update_temperature(actor_info["actor/entropy"])
            info.update(actor_info)
            info.update(temp_info)

        critic_info = self._update_critic(batch)
        info.update(critic_info)

        self._update_target_network()

        return {k: v.item() if isinstance(v, torch.Tensor) else float(v) for k, v in info.items()}

    def _update_actor(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        actor_obs_all = torch.cat([batch["observation"], batch["next_observation"]], dim=0)
        actor_img_all = torch.cat([batch["observation_image"], batch["next_observation_image"]], dim=0)
        actions_all, act_info = self.actor(actor_obs_all, actor_img_all, training=True)
        log_probs_all = act_info["log_prob"]

        actions = torch.chunk(actions_all, 2, dim=0)[0]
        log_probs = torch.chunk(log_probs_all, 2, dim=0)[0]

        for p in self.critic.parameters():
            p.requires_grad = False
        qs, _ = self.critic(batch["observation"], batch["observation_image"], actions, training=False)
        q = torch.minimum(qs[0], qs[1])
        for p in self.critic.parameters():
            p.requires_grad = True

        temp_value = self.temperature().detach()
        actor_loss = (log_probs * temp_value - q).mean()

        if self.bc_alpha > 0:
            q_abs = torch.abs(q).mean().detach()
            bc_loss = ((actions - batch["action"]) ** 2).mean()
            actor_loss = actor_loss + self.bc_alpha * q_abs * bc_loss

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()
        if self.actor_scheduler is not None:
            self.actor_scheduler.step()

        for m in self.actor.modules():
            if hasattr(m, "normalize_parameters"):
                m.normalize_parameters()

        return {
            "actor/loss": actor_loss,
            "actor/entropy": -log_probs.mean(),
            "actor/mean_action": actions.mean(),
        }

    def _update_critic(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            next_actions, info = self.actor(
                batch["next_observation"], batch["next_observation_image"], training=False
            )
            next_actions = next_actions.clone()
            next_actor_log_probs = info["log_prob"].clone()

            temp_value = self.temperature()
            next_actor_entropy = temp_value * next_actor_log_probs

            obs_all = torch.cat([batch["observation"], batch["next_observation"]], dim=0)
            img_all = torch.cat([batch["observation_image"], batch["next_observation_image"]], dim=0)
            act_all = torch.cat([batch["action"], next_actions], dim=0)

            qs_all, q_infos_all = self.target_critic(obs_all, img_all, act_all, training=True)
            next_qs = qs_all.chunk(2, dim=1)[1]
            next_q_log_probs = q_infos_all["log_prob"].chunk(2, dim=1)[1]
            next_q_log_probs = self._select_min_q_log_probs(next_qs, next_q_log_probs)

            target_probs = self._compute_categorical_td_target(
                target_log_probs=next_q_log_probs,
                reward=batch["reward"],
                done=batch["terminated"],
                actor_entropy=next_actor_entropy,
                gamma=self.gamma ** self.n_step,
                num_bins=self.num_bins,
                min_v=self.min_v,
                max_v=self.max_v,
            )
            max_entropy_bonus = next_actor_entropy.max()

        pred_qs_all, pred_q_infos = self.critic(obs_all, img_all, act_all, training=True)
        pred_log_probs = torch.chunk(pred_q_infos["log_prob"], 2, dim=1)[0]

        ce_loss = -(target_probs.unsqueeze(0) * pred_log_probs).sum(dim=-1)
        critic_loss = ce_loss.mean()

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()
        if self.critic_scheduler is not None:
            self.critic_scheduler.step()

        for m in self.critic.modules():
            if hasattr(m, "normalize_parameters"):
                m.normalize_parameters()

        return {
            "critic/loss": critic_loss,
            "critic/max_entropy_bonus": max_entropy_bonus,
        }

    def _update_target_network(self) -> None:
        tau = self.target_tau
        with torch.no_grad():
            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.data.lerp_(param.data, tau)

    def _update_temperature(self, entropy: torch.Tensor) -> dict[str, torch.Tensor]:
        temperature_value = self.temperature().clone()
        temperature_loss = temperature_value * (entropy.detach() - self.target_entropy).mean()

        self.temperature_optimizer.zero_grad(set_to_none=True)
        temperature_loss.backward()
        self.temperature_optimizer.step()
        if self.temperature_scheduler is not None:
            self.temperature_scheduler.step()

        return {
            "temperature/value": temperature_value,
            "temperature/loss": temperature_loss,
        }

    # ------------------------------------------------------------------
    # Static helpers (copied from FlashSAC)
    # ------------------------------------------------------------------

    @staticmethod
    def _select_min_q_log_probs(
        next_qs: torch.Tensor,
        next_q_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        num_bins = next_q_log_probs.shape[-1]
        min_indices = next_qs.argmin(dim=0)
        selected = torch.gather(
            next_q_log_probs,
            dim=0,
            index=min_indices[None, :, None].expand(1, -1, num_bins),
        )[0]
        return selected

    @staticmethod
    def _compute_categorical_td_target(
        target_log_probs: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        actor_entropy: torch.Tensor,
        gamma: float,
        num_bins: int,
        min_v: float,
        max_v: float,
    ) -> torch.Tensor:
        batch_size = reward.shape[0]

        reward = reward.reshape(-1, 1)
        done = done.reshape(-1, 1)
        actor_entropy = actor_entropy.reshape(-1, 1)

        bin_width = (max_v - min_v) / (num_bins - 1)
        bin_values = torch.linspace(
            min_v, max_v, num_bins, device=target_log_probs.device, dtype=target_log_probs.dtype
        ).view(1, -1)

        target_bin_values = reward + gamma * (bin_values - actor_entropy) * (1.0 - done)
        target_bin_values = torch.clamp(target_bin_values, min_v, max_v)

        b = (target_bin_values - min_v) / bin_width
        lower = torch.floor(b).long()
        upper = torch.clamp(lower + 1, 0, num_bins - 1)
        frac = b - lower.float()

        target_probs_exp = target_log_probs.exp()
        m_l = target_probs_exp * (1.0 - frac)
        m_u = target_probs_exp * frac

        target_probs = torch.zeros(batch_size, num_bins, dtype=target_probs_exp.dtype, device=target_probs_exp.device)
        target_probs.scatter_add_(1, lower, m_l)
        target_probs.scatter_add_(1, upper, m_u)

        return target_probs

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @staticmethod
    def construct_algorithm(
        obs: torch.Tensor,
        env: VecEnv,
        cfg: dict,
        device: str,
    ) -> VisionFlashSAC:
        algo_cfg = cfg.get("algorithm", {})
        obs_dim = obs.shape[-1]
        action_dim = env.num_actions

        image_shape = tuple(algo_cfg["image_shape"])
        cnn_feature_dim = algo_cfg["cnn_feature_dim"]

        actor_num_blocks = algo_cfg.get("actor_num_blocks", 2)
        actor_hidden_dim = algo_cfg.get("actor_hidden_dim", 256)
        critic_num_blocks = algo_cfg.get("critic_num_blocks", 2)
        critic_hidden_dim = algo_cfg.get("critic_hidden_dim", 256)
        num_bins = algo_cfg.get("critic_num_bins", 101)
        min_v = algo_cfg.get("critic_min_v", -10.0)
        max_v = algo_cfg.get("critic_max_v", 10.0)
        temp_initial = algo_cfg.get("temp_initial_value", 0.01)
        target_entropy = algo_cfg.get("temp_target_entropy", None)
        if target_entropy is None:
            target_entropy = 0.5 * action_dim * math.log(2 * math.pi * math.e * 0.5**2)

        actor = VisionFlashSACActor(
            num_blocks=actor_num_blocks,
            state_dim=obs_dim,
            hidden_dim=actor_hidden_dim,
            action_dim=action_dim,
            image_shape=image_shape,
            cnn_feature_dim=cnn_feature_dim,
        ).to(device)

        critic = VisionFlashSACDoubleCritic(
            num_blocks=critic_num_blocks,
            state_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=critic_hidden_dim,
            num_bins=num_bins,
            min_v=min_v,
            max_v=max_v,
            image_shape=image_shape,
            cnn_feature_dim=cnn_feature_dim,
        ).to(device)

        target_critic = VisionFlashSACDoubleCritic(
            num_blocks=critic_num_blocks,
            state_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=critic_hidden_dim,
            num_bins=num_bins,
            min_v=min_v,
            max_v=max_v,
            image_shape=image_shape,
            cnn_feature_dim=cnn_feature_dim,
        ).to(device)
        target_critic.load_state_dict(critic.state_dict())

        temperature = FlashSACTemperature(temp_initial).to(device)

        lr = algo_cfg.get("learning_rate", 3e-4)
        actor_optimizer = optim.Adam(actor.parameters(), lr=lr)
        critic_optimizer = optim.Adam(critic.parameters(), lr=lr)
        temperature_optimizer = optim.Adam(temperature.parameters(), lr=lr)

        actor_scheduler = None
        critic_scheduler = None
        temperature_scheduler = None
        if algo_cfg.get("use_lr_schedule", False):
            warmup_steps = algo_cfg.get("lr_warmup_steps", 0)
            decay_steps = algo_cfg.get("lr_decay_steps", 1000000)
            peak_lr = algo_cfg.get("learning_rate_peak", lr)
            end_lr = algo_cfg.get("learning_rate_end", 1e-6)
            init_lr = algo_cfg.get("learning_rate_init", 0.0)
            schedule_fn = warmup_cosine_decay_scheduler(
                init_value=init_lr,
                peak_value=peak_lr,
                end_value=end_lr,
                warmup_steps=warmup_steps,
                decay_steps=decay_steps,
            )
            actor_scheduler = optim.lr_scheduler.LambdaLR(actor_optimizer, lr_lambda=schedule_fn)
            critic_scheduler = optim.lr_scheduler.LambdaLR(critic_optimizer, lr_lambda=schedule_fn)
            temperature_scheduler = optim.lr_scheduler.LambdaLR(temperature_optimizer, lr_lambda=schedule_fn)

        buffer_cfg = algo_cfg.get("buffer", {})
        buffer_capacity = buffer_cfg.get("capacity", 1000000)
        buffer_batch_size = buffer_cfg.get("batch_size", 256)
        replay_buffer = VisionFlashSACReplayBuffer(
            obs_dim=obs_dim,
            action_dim=action_dim,
            image_shape=image_shape,
            capacity=buffer_capacity,
            batch_size=buffer_batch_size,
            device=torch.device(device),
        )

        reward_normalizer = None
        if algo_cfg.get("normalize_reward", False):
            reward_normalizer = RewardNormalizer(
                gamma=algo_cfg.get("gamma", 0.99),
                G_max=algo_cfg.get("normalized_G_max", 10.0),
                device=torch.device(device),
            )

        for m in actor.modules():
            if hasattr(m, "normalize_parameters"):
                m.normalize_parameters()
        for m in critic.modules():
            if hasattr(m, "normalize_parameters"):
                m.normalize_parameters()
        for m in target_critic.modules():
            if hasattr(m, "normalize_parameters"):
                m.normalize_parameters()

        print(f"VisionFlashSAC Actor:\n{actor}")
        print(f"VisionFlashSAC Critic:\n{critic}")

        return VisionFlashSAC(
            actor=actor,
            critic=critic,
            target_critic=target_critic,
            temperature=temperature,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            temperature_optimizer=temperature_optimizer,
            actor_scheduler=actor_scheduler,
            critic_scheduler=critic_scheduler,
            temperature_scheduler=temperature_scheduler,
            replay_buffer=replay_buffer,
            reward_normalizer=reward_normalizer,
            actor_update_period=algo_cfg.get("actor_update_period", 2),
            bc_alpha=algo_cfg.get("actor_bc_alpha", 0.0),
            gamma=algo_cfg.get("gamma", 0.99),
            n_step=algo_cfg.get("n_step", 1),
            num_bins=num_bins,
            min_v=min_v,
            max_v=max_v,
            target_tau=algo_cfg.get("critic_target_update_tau", 0.005),
            target_entropy=target_entropy,
            updates_per_iter=algo_cfg.get("updates_per_iter", 1),
            device=device,
        )
