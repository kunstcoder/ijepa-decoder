from __future__ import annotations

import math

import torch
from torch import nn


class ConvDecoder(nn.Module):
    """Progressive decoder that reconstructs the full image resolution from token grids."""

    def __init__(
        self,
        token_dim: int,
        output_channels: int = 3,
        hidden_dim: int = 256,
        upsample_factor: int = 16,
    ) -> None:
        super().__init__()
        if upsample_factor < 1 or upsample_factor & (upsample_factor - 1) != 0:
            raise ValueError("upsample_factor must be a positive power of two")

        num_stages = int(math.log2(upsample_factor))
        channels = hidden_dim
        layers: list[nn.Module] = [
            nn.Conv2d(token_dim, channels, kernel_size=3, padding=1),
            nn.GELU(),
        ]

        for _ in range(num_stages):
            next_channels = max(channels // 2, output_channels)
            layers.extend(
                [
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(channels, next_channels, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(next_channels, next_channels, kernel_size=3, padding=1),
                    nn.GELU(),
                ]
            )
            channels = next_channels

        layers.append(nn.Conv2d(channels, output_channels, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, token_grid: torch.Tensor) -> torch.Tensor:
        return self.net(token_grid)
