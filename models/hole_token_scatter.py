from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TokenCanvas:
    """Container for a spatially aligned token grid."""

    tokens: torch.Tensor
    hole_mask: torch.Tensor


def scatter_hole_tokens(
    visible_tokens: torch.Tensor,
    predicted_hole_tokens: torch.Tensor,
    visible_indices: torch.Tensor,
    hole_indices: torch.Tensor,
    num_patches: int,
) -> TokenCanvas:
    """Rebuild a full token canvas from visible and predicted hole tokens.

    Args:
        visible_tokens: Tensor shaped [B, N_visible, D].
        predicted_hole_tokens: Tensor shaped [B, N_hole, D].
        visible_indices: Long tensor shaped [B, N_visible].
        hole_indices: Long tensor shaped [B, N_hole].
        num_patches: Total number of patches in the image grid.
    """
    if visible_tokens.ndim != 3 or predicted_hole_tokens.ndim != 3:
        raise ValueError("visible_tokens and predicted_hole_tokens must be rank-3 tensors")

    batch_size, _, dim = visible_tokens.shape
    canvas = visible_tokens.new_zeros(batch_size, num_patches, dim)
    hole_mask = torch.zeros(batch_size, num_patches, dtype=torch.bool, device=visible_tokens.device)

    visible_index_expanded = visible_indices.unsqueeze(-1).expand(-1, -1, dim)
    hole_index_expanded = hole_indices.unsqueeze(-1).expand(-1, -1, dim)

    canvas.scatter_(1, visible_index_expanded, visible_tokens)
    canvas.scatter_(1, hole_index_expanded, predicted_hole_tokens)
    hole_mask.scatter_(1, hole_indices, True)

    return TokenCanvas(tokens=canvas, hole_mask=hole_mask)


def reshape_token_canvas(tokens: torch.Tensor, grid_size: tuple[int, int]) -> torch.Tensor:
    """Convert [B, N, D] canvas into [B, D, H, W]."""
    batch_size, num_patches, dim = tokens.shape
    height, width = grid_size
    if num_patches != height * width:
        raise ValueError("grid_size does not match the number of patches")
    return tokens.transpose(1, 2).reshape(batch_size, dim, height, width)
