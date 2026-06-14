from __future__ import annotations

import torch
import torch.nn as nn
from functools import reduce


def _resolve_activation(name: str) -> nn.Module:
    """Resolve an activation function by name."""
    mapping = {
        "elu": nn.ELU(),
        "relu": nn.ReLU(),
        "leaky_relu": nn.LeakyReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "selu": nn.SELU(),
        "gelu": nn.GELU(),
        "softsign": nn.Softsign(),
        "identity": nn.Identity(),
    }
    if name.lower() not in mapping:
        raise ValueError(f"Unknown activation: {name}. Options: {list(mapping.keys())}")
    return mapping[name.lower()]


class MLP(nn.Sequential):
    """Multi-Layer Perceptron.

    A sequence of linear layers and activations. The last layer is linear
    unless ``last_activation`` is specified. If ``hidden_dims[i] == -1``,
    the dimension is inferred from the input dimension. If ``output_dim`` is
    a tuple, the output is reshaped accordingly.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int | tuple[int, ...] | list[int],
        hidden_dims: tuple[int, ...] | list[int],
        activation: str = "elu",
        last_activation: str | None = None,
    ) -> None:
        super().__init__()

        activation_mod = _resolve_activation(activation)
        last_activation_mod = _resolve_activation(last_activation) if last_activation else None

        processed = [input_dim if d == -1 else d for d in hidden_dims]

        layers: list[nn.Module] = []
        layers.append(nn.Linear(input_dim, processed[0]))
        layers.append(activation_mod)

        for i in range(len(processed) - 1):
            layers.append(nn.Linear(processed[i], processed[i + 1]))
            layers.append(activation_mod)

        if isinstance(output_dim, int):
            layers.append(nn.Linear(processed[-1], output_dim))
        else:
            total = reduce(lambda x, y: x * y, output_dim)
            layers.append(nn.Linear(processed[-1], total))
            layers.append(nn.Unflatten(dim=-1, unflattened_size=output_dim))

        if last_activation_mod is not None:
            layers.append(last_activation_mod)

        for idx, layer in enumerate(layers):
            self.add_module(str(idx), layer)

    def init_weights(self, scales: float | tuple[float, ...]) -> None:
        """Orthogonal init with per-layer gain."""
        for idx, module in enumerate(self):
            if isinstance(module, nn.Linear):
                gain = scales if isinstance(scales, float) else scales[idx // 2]
                nn.init.orthogonal_(module.weight, gain=gain)
                nn.init.zeros_(module.bias)