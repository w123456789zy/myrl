from __future__ import annotations

import torch
import torch.nn as nn


def _get_param(value, idx):
    """Return ``value[idx]`` if value is a sequence, else ``value``."""
    if isinstance(value, (list, tuple)):
        return value[idx]
    return value


def _compute_padding(input_dim, kernel_size, stride, dilation):
    """Compute symmetric padding for 'same' behaviour."""
    effective_kernel = (kernel_size - 1) * dilation + 1
    pad_total = max(0, effective_kernel - stride)
    pad_before = pad_total // 2
    pad_after = pad_total - pad_before
    return (pad_before, pad_after)


def _compute_output_dim(input_dim, kernel_size, stride, dilation, padding):
    """Compute spatial output dimension after a Conv2d layer."""
    h, w = (input_dim[0], input_dim[1])
    h_out = (h + 2 * padding[0] - dilation * (kernel_size - 1) - 1) // stride + 1
    w_out = (w + 2 * padding[1] - dilation * (kernel_size - 1) - 1) // stride + 1
    return (h_out, w_out)


def _resolve_activation(name: str) -> nn.Module:
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
        raise ValueError(f"Unknown activation: {name}")
    return mapping[name.lower()]


class CNN(nn.Sequential):
    """Convolutional Neural Network.

    A sequence of Conv2d layers with optional normalization, activation,
    max-pooling, and global pooling. The final output can be flattened.
    """

    def __init__(
        self,
        input_dim: tuple[int, int],
        input_channels: int,
        output_channels: tuple[int, ...] | list[int],
        kernel_size: int | tuple[int, ...] | list[int],
        stride: int | tuple[int, ...] | list[int] = 1,
        dilation: int | tuple[int, ...] | list[int] = 1,
        padding: str = "none",
        norm: str | tuple[str, ...] | list[str] = "none",
        activation: str = "elu",
        max_pool: bool | tuple[bool, ...] | list[bool] = False,
        global_pool: str = "none",
        flatten: bool = True,
    ) -> None:
        super().__init__()

        activation_fn = _resolve_activation(activation)
        layers: list[nn.Module] = []
        last_channels = input_channels
        last_dim = input_dim

        for idx in range(len(output_channels)):
            k = _get_param(kernel_size, idx)
            s = _get_param(stride, idx)
            d = _get_param(dilation, idx)
            p = (
                _compute_padding(last_dim, k, s, d)
                if padding in ("zeros", "reflect", "replicate", "circular")
                else (0, 0)
            )

            pad_mode = padding if padding in ("zeros", "reflect", "replicate", "circular") else "zeros"
            layers.append(
                nn.Conv2d(
                    in_channels=last_channels,
                    out_channels=output_channels[idx],
                    kernel_size=k,
                    stride=s,
                    padding=p,
                    dilation=d,
                    padding_mode=pad_mode,
                )
            )

            n = _get_param(norm, idx)
            if n == "batch":
                layers.append(nn.BatchNorm2d(output_channels[idx]))
            elif n == "layer":
                norm_dim = _compute_output_dim(last_dim, k, s, d, p)
                layers.append(nn.LayerNorm([output_channels[idx], *norm_dim]))

            layers.append(activation_fn)

            if _get_param(max_pool, idx):
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                last_dim = (last_dim[0] // 2, last_dim[1] // 2)
            else:
                last_dim = _compute_output_dim(last_dim, k, s, d, p)

            last_channels = output_channels[idx]

        if global_pool == "max":
            layers.append(nn.AdaptiveMaxPool2d((1, 1)))
        elif global_pool == "avg":
            layers.append(nn.AdaptiveAvgPool2d((1, 1)))

        if flatten:
            layers.append(nn.Flatten())

        for idx, layer in enumerate(layers):
            self.add_module(str(idx), layer)