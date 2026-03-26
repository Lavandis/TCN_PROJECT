from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return inputs.contiguous()
        return inputs[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = weight_norm(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        self.output_relu = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs if self.downsample is None else self.downsample(inputs)
        outputs = self.net(inputs)
        return self.output_relu(outputs + residual)


class TCN(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_steps: int,
        channels: Sequence[int],
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if not channels:
            raise ValueError("channels must contain at least one layer width.")

        levels: list[nn.Module] = []
        for index, out_channels in enumerate(channels):
            dilation = 2**index
            in_channels = input_channels if index == 0 else channels[index - 1]
            levels.append(
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )

        self.network = nn.Sequential(*levels)
        self.projection = nn.Linear(channels[-1], output_steps)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.network(inputs)
        last_step = features[:, :, -1]
        return self.projection(last_step)


def build_tcn_from_config(model_config: dict[str, Any], output_steps: int) -> TCN:
    channels = model_config.get("channels")
    if channels is None:
        channels = [int(model_config["channel_width"])] * int(model_config["num_layers"])

    return TCN(
        input_channels=int(model_config.get("input_channels", 1)),
        output_steps=output_steps,
        channels=[int(value) for value in channels],
        kernel_size=int(model_config["kernel_size"]),
        dropout=float(model_config.get("dropout", 0.0)),
    )
