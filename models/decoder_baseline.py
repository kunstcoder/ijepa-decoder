from __future__ import annotations

import torch
from torch import nn


class ConvDecoder(nn.Module):
    """Minimal deterministic decoder for semantic token grids."""

    def __init__(self, token_dim: int, output_channels: int = 3, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(token_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden_dim // 2, output_channels, kernel_size=3, padding=1),
        )

    def forward(self, token_grid: torch.Tensor) -> torch.Tensor:
        return self.net(token_grid)
