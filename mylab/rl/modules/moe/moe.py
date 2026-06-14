from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoEBlock(nn.Module):
    """Sparsely-gated Mixture of Experts (soft-gated variant).

    Pipeline: in → shared_backbone(hidden) → N experts + gate → out
    Output: weighted sum of expert outputs, with weights from a softmax gate.

    Stores:
        - self.gate_weights: latest gate weights, shape (B, num_experts).
          Consumed externally for the load-balance loss.

    Reference: go2_rl_gym (vbot_rl_gym) MoE-CTS. Uses soft (dense) gating
    rather than top-k sparse routing to keep the implementation simple —
    every expert contributes to every output, but the gate weight controls
    how much.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_experts: int = 4,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_experts = num_experts

        # Shared backbone (Linear + ELU)
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ELU(),
        )

        # N expert networks (one Linear layer each — matches vbot_rl_gym's
        # lightweight design that adjusts "how features are reused" via 1x1
        # linear projections rather than deep per-expert MLPs)
        self.experts = nn.ModuleList(
            [nn.Linear(hidden_dim, out_dim) for _ in range(num_experts)]
        )

        # Gate: hidden → N logits
        self.gate = nn.Linear(hidden_dim, num_experts)

        # Latest gate weights (B, N); updated on each forward
        self.gate_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)  # (B, hidden_dim)
        # Stack expert outputs: (B, N, out_dim)
        expert_outs = torch.stack([e(h) for e in self.experts], dim=1)
        # Gate logits → softmax weights
        gate_logits = self.gate(h)  # (B, N)
        gate_weights = F.softmax(gate_logits, dim=-1)  # (B, N)
        self.gate_weights = gate_weights
        # Soft combination: (B, 1, N) × (B, N, out) → (B, out)
        out = (gate_weights.unsqueeze(-1) * expert_outs).sum(dim=1)
        return out

    # ------------------------------------------------------------------
    def balance_loss(self) -> torch.Tensor:
        """Load-balance loss for uniform expert usage.

        Computed as MSE between the mean gate weights across the batch and
        the uniform target ``1 / num_experts``.  Returns zero if the gate
        weights haven't been computed yet (e.g. before the first forward).
        This is a *module-level* loss — it does NOT depend on any specific
        algorithm (PPO/SAC/...). Any algorithm that trains a model
        containing an :class:`MoEBlock` can call this to get the auxiliary
        term for its total loss.

        Usage::

            loss = task_loss + balance_coef * moe_block.balance_loss()
        """
        if self.gate_weights is None:
            # No forward has happened yet; return a zero loss on the same
            # device as the parameters.
            ref = next(self.parameters(), None)
            device = ref.device if ref is not None else torch.device("cpu")
            return torch.zeros((), device=device)
        mean_gate = self.gate_weights.mean(dim=0)
        target = torch.full_like(mean_gate, 1.0 / self.num_experts)
        return F.mse_loss(mean_gate, target)
