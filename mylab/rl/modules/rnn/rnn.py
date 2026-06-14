from __future__ import annotations

import torch
import torch.nn as nn
from typing import Union


HiddenState = Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor], None]
"""Type alias for RNN hidden states (GRU/LSTM)."""


def _unpad_trajectories(trajectories: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Remove padding from trajectories, returning a flat tensor of valid steps only."""
    batch_size, steps, hidden_dim = trajectories.shape
    flat = trajectories.view(-1, hidden_dim)
    keep = masks.view(-1).nonzero(as_tuple=True)[0]
    return flat[keep]


class RNN(nn.Module):
    """Recurrent Neural Network (GRU or LSTM).

    Maintains an internal hidden state that is updated across steps during
    rollout and can be reset per environment when episodes end.
    """

    def __init__(
        self,
        input_size: int,
        hidden_dim: int = 256,
        num_layers: int = 1,
        rnn_type: str = "lstm",
    ) -> None:
        super().__init__()
        rnn_cls = nn.GRU if rnn_type.lower() == "gru" else nn.LSTM
        self.rnn = rnn_cls(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers)
        self.hidden_state: HiddenState = None

    def forward(
        self,
        input: torch.Tensor,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        """Run recurrent inference.

        When ``masks`` is provided (batch mode), uses the external
        ``hidden_state`` and removes padding. Otherwise, updates the
        internal ``self.hidden_state`` (rollout mode).
        """
        if masks is not None:
            if hidden_state is None:
                raise ValueError("Hidden state required in batch mode.")
            out, _ = self.rnn(input, hidden_state)
            out = _unpad_trajectories(out, masks)
        else:
            out, self.hidden_state = self.rnn(input.unsqueeze(0), self.hidden_state)
        return out

    def reset(
        self,
        dones: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> None:
        """Reset hidden states.

        If ``dones`` is None, replace the entire hidden state. Otherwise,
        zero out hidden states for done environments.
        """
        if dones is None:
            self.hidden_state = hidden_state
        elif self.hidden_state is not None:
            if hidden_state is not None:
                raise NotImplementedError(
                    "Partial reset with custom hidden state is not supported."
                )
            if isinstance(self.hidden_state, tuple):
                for hs in self.hidden_state:
                    hs[..., dones == 1, :] = 0.0
            else:
                self.hidden_state[..., dones == 1, :] = 0.0

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        """Detach hidden states from the computation graph."""
        if self.hidden_state is not None:
            if dones is None:
                if isinstance(self.hidden_state, tuple):
                    self.hidden_state = tuple(hs.detach() for hs in self.hidden_state)
                else:
                    self.hidden_state = self.hidden_state.detach()
            else:
                if isinstance(self.hidden_state, tuple):
                    for hs in self.hidden_state:
                        hs[..., dones == 1, :] = hs[..., dones == 1, :].detach()
                else:
                    self.hidden_state[..., dones == 1, :] = self.hidden_state[..., dones == 1, :].detach()