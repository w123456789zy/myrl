import math

import torch
import torch.nn as nn

from mylab.rl.modules.flashsac.layer import (
    EnsembleCategoricalValue,
    EnsembleFlashSACBlock,
    EnsembleFlashSACEmbedder,
    EnsembleUnitRMSNorm,
    FlashSACBlock,
    FlashSACEmbedder,
    NormalTanhPolicy,
    UnitRMSNorm,
)


class FlashSACActor(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        input_dim: int,
        hidden_dim: int,
        action_dim: int,
    ):
        super().__init__()
        self.embedder = FlashSACEmbedder(input_dim=input_dim, hidden_dim=hidden_dim)
        self.encoder = nn.ModuleList([FlashSACBlock(hidden_dim) for _ in range(num_blocks)])
        self.post_norm = UnitRMSNorm(hidden_dim)
        self.predictor = NormalTanhPolicy(hidden_dim=hidden_dim, action_dim=action_dim)

    def get_mean_and_std(
        self,
        observations: torch.Tensor,
        training: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = observations
        x = self.embedder(x, training)
        for block in self.encoder:
            x = block(x, training)
        x = self.post_norm(x)
        mean, std = self.predictor.get_mean_and_std(x, training)
        return mean, std

    def forward(
        self,
        observations: torch.Tensor,
        training: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = observations
        x = self.embedder(x, training)
        for block in self.encoder:
            x = block(x, training)
        x = self.post_norm(x)
        actions, info = self.predictor(x, training)
        return actions, info


class FlashSACDoubleCritic(nn.Module):
    """Double-Q for Clipped Double Q-learning.

    Fuses N parallel critic networks into single batched operations.
    All internal computation uses (N, batch, dim) tensor layout.
    """

    def __init__(
        self,
        num_blocks: int,
        input_dim: int,
        hidden_dim: int,
        num_bins: int,
        min_v: float,
        max_v: float,
        num_qs: int = 2,
    ):
        super().__init__()
        self.num_qs = num_qs

        self.embedder = EnsembleFlashSACEmbedder(num_qs, input_dim, hidden_dim)
        self.encoder = nn.ModuleList([EnsembleFlashSACBlock(num_qs, hidden_dim) for _ in range(num_blocks)])
        self.post_norm = EnsembleUnitRMSNorm(num_qs, hidden_dim)
        self.predictor = EnsembleCategoricalValue(
            num_ensemble=num_qs,
            hidden_dim=hidden_dim,
            num_bins=num_bins,
            min_v=min_v,
            max_v=max_v,
        )

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        training: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = torch.cat((observations, actions), dim=-1)  # [B, in_dim]
        x = x.unsqueeze(0).expand(self.num_qs, -1, -1)  # [num_qs, B, in_dim]
        x = self.embedder(x, training)
        for block in self.encoder:
            x = block(x, training)
        x = self.post_norm(x)
        qs, infos = self.predictor(x, training)
        return qs, infos


class FlashSACTemperature(nn.Module):
    def __init__(self, initial_value: float = 0.01):
        super().__init__()
        self.log_temp = nn.Parameter(torch.tensor([math.log(initial_value)], dtype=torch.float32))

    def forward(self) -> torch.Tensor:
        # Clamp log_temp to prevent temperature explosion: exp(1.5) ≈ 4.5, exp(-5) ≈ 0.007
        return torch.exp(self.log_temp.clamp(-5.0, 1.5))


# ---------------------------------------------------------------------------
# Vision variants (CNN + FlashSAC)
# ---------------------------------------------------------------------------

class VisionFlashSACActor(nn.Module):
    """FlashSAC Actor with CNN vision encoder.

    CNN encodes image → concat with state → FlashSACActor body.
    """

    def __init__(
        self,
        num_blocks: int,
        state_dim: int,
        hidden_dim: int,
        action_dim: int,
        image_shape: tuple[int, ...],
        cnn_feature_dim: int,
    ):
        super().__init__()
        from mylab.rl.modules.cnn.cnn_encoder import CNNEncoder
        self.cnn = CNNEncoder(
            image_shape[0], cnn_feature_dim, input_size=(image_shape[1], image_shape[2])
        )
        self.actor = FlashSACActor(
            num_blocks=num_blocks,
            input_dim=state_dim + cnn_feature_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
        )

    def get_mean_and_std(
        self,
        state: torch.Tensor,
        image: torch.Tensor,
        training: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        img_features = self.cnn(image)
        obs = torch.cat([state, img_features], dim=-1)
        return self.actor.get_mean_and_std(obs, training)

    def forward(
        self,
        state: torch.Tensor,
        image: torch.Tensor,
        training: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        img_features = self.cnn(image)
        obs = torch.cat([state, img_features], dim=-1)
        return self.actor(obs, training)


class VisionFlashSACDoubleCritic(nn.Module):
    """FlashSAC Double Critic with CNN vision encoder.

    CNN encodes image → concat with state → FlashSACDoubleCritic body
    (action is concatenated internally by the critic).
    """

    def __init__(
        self,
        num_blocks: int,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        num_bins: int,
        min_v: float,
        max_v: float,
        image_shape: tuple[int, ...],
        cnn_feature_dim: int,
        num_qs: int = 2,
    ):
        super().__init__()
        from mylab.rl.modules.cnn.cnn_encoder import CNNEncoder
        self.cnn = CNNEncoder(
            image_shape[0], cnn_feature_dim, input_size=(image_shape[1], image_shape[2])
        )
        self.critic = FlashSACDoubleCritic(
            num_blocks=num_blocks,
            input_dim=state_dim + cnn_feature_dim + action_dim,
            hidden_dim=hidden_dim,
            num_bins=num_bins,
            min_v=min_v,
            max_v=max_v,
            num_qs=num_qs,
        )

    def forward(
        self,
        state: torch.Tensor,
        image: torch.Tensor,
        action: torch.Tensor,
        training: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        img_features = self.cnn(image)
        obs = torch.cat([state, img_features], dim=-1)
        return self.critic(obs, action, training)
