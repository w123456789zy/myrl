from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Beta, Normal


class Distribution(nn.Module):
    """Abstract base class for action distributions.

    Subclasses define how raw MLP output is turned into a parameterized
    distribution, and provide sampling, log-prob, entropy, and KL methods.
    """

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.output_dim = output_dim

    def update(self, mlp_output: torch.Tensor) -> None:
        """Update distribution parameters from MLP output."""
        raise NotImplementedError

    def sample(self) -> torch.Tensor:
        """Sample from the current distribution."""
        raise NotImplementedError

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        """Return the deterministic (mean) output from raw MLP output."""
        raise NotImplementedError

    def as_deterministic_output_module(self) -> nn.Module:
        """Return an export-friendly module for deterministic inference."""
        raise NotImplementedError

    @property
    def input_dim(self) -> int | list[int]:
        """Required input dimensionality from the MLP."""
        raise NotImplementedError

    @property
    def mean(self) -> torch.Tensor:
        """Mean of the distribution."""
        raise NotImplementedError

    @property
    def std(self) -> torch.Tensor:
        """Standard deviation (or spread) of the distribution."""
        raise NotImplementedError

    @property
    def entropy(self) -> torch.Tensor:
        """Entropy summed over the last dimension."""
        raise NotImplementedError

    @property
    def params(self) -> tuple[torch.Tensor, ...]:
        """Distribution parameters (for KL computation)."""
        raise NotImplementedError

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        """Log-probability of ``outputs``, summed over last dim."""
        raise NotImplementedError

    def kl_divergence(
        self,
        old_params: tuple[torch.Tensor, ...],
        new_params: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """KL(old || new) summed over last dim."""
        raise NotImplementedError

    def init_mlp_weights(self, mlp: nn.Module) -> None:
        """Optional weight init hook called after MLP creation."""
        pass


# ---------------------------------------------------------------------------
# Gaussian
# ---------------------------------------------------------------------------


class GaussianDistribution(Distribution):
    """Multivariate Gaussian with state-independent diagonal std."""

    def __init__(
        self,
        output_dim: int,
        init_std: float = 1.0,
        std_range: tuple[float, float] = (1e-6, 1e6),
        std_type: str = "scalar",
        learn_std: bool = True,
    ) -> None:
        super().__init__(output_dim)
        self.std_type = std_type

        if std_type == "scalar":
            self.std_param = nn.Parameter(init_std * torch.ones(output_dim), requires_grad=learn_std)
        elif std_type == "log":
            self.log_std_param = nn.Parameter(torch.log(init_std * torch.ones(output_dim) + 1e-7), requires_grad=learn_std)
        else:
            raise ValueError(f"Unknown std_type: {std_type}")

        self.std_range = [max(std_range[0], 1e-6), std_range[1]]
        self.log_std_range = [float(np.log(self.std_range[0])), float(np.log(self.std_range[1]))]
        self._distribution: Normal | None = None
        Normal.set_default_validate_args(False)

    def update(self, mlp_output: torch.Tensor) -> None:
        mean = mlp_output
        if self.std_type == "scalar":
            std = self.std_param.clamp(*self.std_range)
        else:
            log_std = self.log_std_param.clamp(*self.log_std_range)
            std = torch.exp(log_std)
        self._distribution = Normal(mean, std)

    def sample(self) -> torch.Tensor:
        return self._distribution.sample()

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        return mlp_output

    def as_deterministic_output_module(self) -> nn.Module:
        return _IdentityOutput()

    @property
    def input_dim(self) -> int:
        return self.output_dim

    @property
    def mean(self) -> torch.Tensor:
        return self._distribution.mean

    @property
    def std(self) -> torch.Tensor:
        return self._distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self._distribution.entropy().sum(dim=-1)

    @property
    def params(self) -> tuple[torch.Tensor, ...]:
        return (self.mean, self.std)

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        return self._distribution.log_prob(outputs).sum(dim=-1)

    def kl_divergence(
        self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        old_mean, old_std = old_params
        new_mean, new_std = new_params
        return torch.distributions.kl_divergence(Normal(old_mean, old_std), Normal(new_mean, new_std)).sum(dim=-1)


# ---------------------------------------------------------------------------
# Heteroscedastic Gaussian
# ---------------------------------------------------------------------------


class HeteroscedasticGaussianDistribution(Distribution):
    """Multivariate Gaussian with state-dependent diagonal std (from MLP)."""

    def __init__(
        self,
        output_dim: int,
        init_std: float = 1.0,
        std_range: tuple[float, float] = (1e-6, 1e6),
        std_type: str = "scalar",
    ) -> None:
        super().__init__(output_dim)
        self.std_type = std_type
        self.init_std = init_std

        if std_type not in ("scalar", "log"):
            raise ValueError(f"Unknown std_type: {std_type}")

        self.std_range = [max(std_range[0], 1e-6), std_range[1]]
        self.log_std_range = [float(np.log(self.std_range[0])), float(np.log(self.std_range[1]))]
        self._distribution: Normal | None = None
        Normal.set_default_validate_args(False)

    def update(self, mlp_output: torch.Tensor) -> None:
        mean, std_raw = torch.unbind(mlp_output, dim=-2)
        if self.std_type == "scalar":
            std = torch.clamp(std_raw, *self.std_range)
        else:
            log_std = torch.clamp(std_raw, *self.log_std_range)
            std = torch.exp(log_std)
        self._distribution = Normal(mean, std)

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        return mlp_output[..., 0, :]

    def as_deterministic_output_module(self) -> nn.Module:
        return _FirstSliceOutput()

    @property
    def input_dim(self) -> list[int]:
        return [2, self.output_dim]

    @property
    def mean(self) -> torch.Tensor:
        return self._distribution.mean

    @property
    def std(self) -> torch.Tensor:
        return self._distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self._distribution.entropy().sum(dim=-1)

    @property
    def params(self) -> tuple[torch.Tensor, ...]:
        return (self.mean, self.std)

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        return self._distribution.log_prob(outputs).sum(dim=-1)

    def kl_divergence(
        self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        old_mean, old_std = old_params
        new_mean, new_std = new_params
        return torch.distributions.kl_divergence(Normal(old_mean, old_std), Normal(new_mean, new_std)).sum(dim=-1)

    def init_mlp_weights(self, mlp: nn.Module) -> None:
        torch.nn.init.zeros_(mlp[-2].weight[self.output_dim :])
        if self.std_type == "scalar":
            torch.nn.init.constant_(mlp[-2].bias[self.output_dim :], self.init_std)
        else:
            torch.nn.init.constant_(mlp[-2].bias[self.output_dim :], np.log(self.init_std + 1e-7))


# ---------------------------------------------------------------------------
# Beta
# ---------------------------------------------------------------------------


class BetaDistribution(Distribution):
    """Beta distribution for bounded action spaces in [action_range]."""

    def __init__(
        self,
        output_dim: int,
        action_range: tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        super().__init__(output_dim)
        self.action_range = action_range
        self._range_scale = action_range[1] - action_range[0]
        self._range_offset = action_range[0]
        self._log_range_scale = np.log(self._range_scale)
        self._distribution: Beta | None = None
        self._alpha: torch.Tensor | None = None
        self._beta: torch.Tensor | None = None
        Beta.set_default_validate_args(False)

    def update(self, mlp_output: torch.Tensor) -> None:
        alpha_raw, beta_raw = torch.unbind(mlp_output, dim=-2)
        self._alpha = torch.nn.functional.softplus(alpha_raw) + 1.0
        self._beta = torch.nn.functional.softplus(beta_raw) + 1.0
        self._distribution = Beta(self._alpha, self._beta)

    def sample(self) -> torch.Tensor:
        return self._distribution.sample() * self._range_scale + self._range_offset

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        alpha_raw, beta_raw = torch.unbind(mlp_output, dim=-2)
        alpha = torch.nn.functional.softplus(alpha_raw) + 1.0
        beta = torch.nn.functional.softplus(beta_raw) + 1.0
        return (alpha / (alpha + beta)) * self._range_scale + self._range_offset

    def as_deterministic_output_module(self) -> nn.Module:
        return _BetaDeterministicOutput(self._range_scale, self._range_offset)

    @property
    def input_dim(self) -> list[int]:
        return [2, self.output_dim]

    @property
    def mean(self) -> torch.Tensor:
        return (self._alpha / (self._alpha + self._beta)) * self._range_scale + self._range_offset

    @property
    def std(self) -> torch.Tensor:
        return self._distribution.stddev * self._range_scale

    @property
    def entropy(self) -> torch.Tensor:
        return self._distribution.entropy().sum(dim=-1)

    @property
    def params(self) -> tuple[torch.Tensor, ...]:
        return (self._alpha, self._beta)

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        unscaled = (outputs - self._range_offset) / self._range_scale
        unscaled = unscaled.clamp(1e-6, 1.0 - 1e-6)
        return (self._distribution.log_prob(unscaled) - self._log_range_scale).sum(dim=-1)

    def kl_divergence(
        self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        old_alpha, old_beta = old_params
        new_alpha, new_beta = new_params
        return torch.distributions.kl_divergence(Beta(old_alpha, old_beta), Beta(new_alpha, new_beta)).sum(dim=-1)

    def init_mlp_weights(self, mlp: nn.Module) -> None:
        torch.nn.init.zeros_(mlp[-2].weight[self.output_dim :])
        torch.nn.init.zeros_(mlp[-2].bias[self.output_dim :])


# ---------------------------------------------------------------------------
# Deterministic output helpers (for ONNX export)
# ---------------------------------------------------------------------------


class _IdentityOutput(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _FirstSliceOutput(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., 0, :]


class _BetaDeterministicOutput(nn.Module):
    def __init__(self, range_scale: float, range_offset: float) -> None:
        super().__init__()
        self.range_scale = range_scale
        self.range_offset = range_offset

    def forward(self, mlp_output: torch.Tensor) -> torch.Tensor:
        alpha_raw, beta_raw = torch.unbind(mlp_output, dim=-2)
        alpha = torch.nn.functional.softplus(alpha_raw) + 1.0
        beta = torch.nn.functional.softplus(beta_raw) + 1.0
        return (alpha / (alpha + beta)) * self.range_scale + self.range_offset