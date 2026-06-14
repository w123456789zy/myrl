from __future__ import annotations

import torch
import torch.nn as nn

from mylab.rl.modules.mlp import MLP
from mylab.rl.modules.cnn.cnn_encoder import CNNEncoder
from mylab.rl.modules.normalization import EmpiricalNormalization
from mylab.rl.modules.distribution import GaussianDistribution


class ActorVision(nn.Module):
    """Actor with CNN vision encoder.

    CNN encodes image → concat with proprioception obs → MLP → distribution.
    """

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
    """Critic with CNN vision encoder.

    CNN encodes image → concat with proprioception obs → MLP → value.
    """

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
