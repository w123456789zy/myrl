from __future__ import annotations

import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    """Lightweight CNN encoder for visual observations.

    Input: (B, C, H, W) → Output: (B, feature_dim)
    Default: 3×64×64 → 32 features.
    """

    def __init__(
        self,
        input_channels: int = 3,
        feature_dim: int = 32,
        input_size: tuple[int, int] = (64, 64),
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        # Dynamically compute flattened size based on input_size
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, input_size[0], input_size[1])
            conv_out = self.conv(dummy)
            flat_size = int(conv_out.reshape(1, -1).shape[1])
        self.fc = nn.Linear(flat_size, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)
