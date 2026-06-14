from __future__ import annotations

import torch
import torch.nn as nn


class EmpiricalNormalization(nn.Module):
    """Online mean/variance normalization using Welford-style updates.

    Tracks running mean and std of input values during training and
    normalizes at forward time.
    """

    def __init__(
        self,
        shape: int | tuple[int, ...] | list[int],
        eps: float = 1e-2,
        until: int | None = None,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.until = until
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0))
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    @property
    def mean(self) -> torch.Tensor:
        return self._mean.squeeze(0).clone()

    @property
    def std(self) -> torch.Tensor:
        return self._std.squeeze(0).clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._mean) / (self._std + self.eps)

    @torch.jit.unused
    def update(self, x: torch.Tensor) -> None:
        """Update running statistics with a new batch of data."""
        if not self.training:
            return
        if self.until is not None and self.count >= self.until:
            return

        count_x = x.shape[0]
        self.count += count_x
        rate = count_x / self.count
        var_x = torch.var(x, dim=0, unbiased=False, keepdim=True)
        mean_x = torch.mean(x, dim=0, keepdim=True)
        delta_mean = mean_x - self._mean
        self._mean += rate * delta_mean
        self._var += rate * (var_x - self._var + delta_mean * (mean_x - self._mean))
        self._std = torch.sqrt(self._var)

    @torch.jit.unused
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """De-normalize: x = y * std + mean."""
        return y * (self._std + self.eps) + self._mean


class _DiscountedAverage:
    """Exponentially discounted running average: avg = gamma * avg + rew."""

    def __init__(self, gamma: float) -> None:
        self.gamma = gamma
        self.avg: torch.Tensor | None = None

    def update(self, rew: torch.Tensor) -> torch.Tensor:
        if self.avg is None:
            self.avg = rew
        else:
            self.avg = self.avg * self.gamma + rew
        return self.avg


class EmpiricalDiscountedVariationNormalization(nn.Module):
    """Reward normalization via running std of discounted returns.

    Divides rewards by the running std of discounted returns so that the
    value function sees a more stationary signal.
    """

    def __init__(
        self,
        shape: int | tuple[int, ...] | list[int],
        eps: float = 1e-2,
        gamma: float = 0.99,
        until: int | None = None,
    ) -> None:
        super().__init__()
        self.emp_norm = EmpiricalNormalization(shape, eps, until)
        self.disc_avg = _DiscountedAverage(gamma)

    def forward(self, rew: torch.Tensor) -> torch.Tensor:
        if self.training:
            avg = self.disc_avg.update(rew)
            self.emp_norm.update(avg)
        if self.emp_norm._std > 0:
            return rew / self.emp_norm._std
        return rew