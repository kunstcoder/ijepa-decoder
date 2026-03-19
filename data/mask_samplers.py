from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class PatchMask:
    visible_indices: torch.Tensor
    hole_indices: torch.Tensor
    patch_mask: torch.Tensor


class PatchMaskConverter:
    """Convert binary pixel masks into patch-aligned visible/hole indices."""

    def __init__(self, patch_size: int, mask_threshold: float = 0.5) -> None:
        self.patch_size = patch_size
        self.mask_threshold = mask_threshold

    def __call__(self, binary_mask: torch.Tensor) -> PatchMask:
        if binary_mask.ndim != 4:
            raise ValueError("binary_mask must have shape [B, 1, H, W]")
        pooled = F.avg_pool2d(binary_mask.float(), kernel_size=self.patch_size, stride=self.patch_size)
        patch_mask = pooled >= self.mask_threshold

        batch_size = patch_mask.shape[0]
        flat_mask = patch_mask.flatten(1)
        visible = []
        holes = []
        for batch_idx in range(batch_size):
            holes.append(torch.nonzero(flat_mask[batch_idx], as_tuple=False).squeeze(-1))
            visible.append(torch.nonzero(~flat_mask[batch_idx], as_tuple=False).squeeze(-1))

        visible_indices = torch.stack(_pad_indices(visible), dim=0)
        hole_indices = torch.stack(_pad_indices(holes), dim=0)
        return PatchMask(visible_indices=visible_indices, hole_indices=hole_indices, patch_mask=flat_mask)


def apply_pixel_mask(image: torch.Tensor, binary_mask: torch.Tensor) -> torch.Tensor:
    if image.shape[-2:] != binary_mask.shape[-2:]:
        raise ValueError("image and binary_mask spatial sizes must match")
    return image * (1.0 - binary_mask)


def _pad_indices(index_list: list[torch.Tensor]) -> list[torch.Tensor]:
    max_len = max((tensor.numel() for tensor in index_list), default=0)
    padded = []
    for indices in index_list:
        if indices.numel() == max_len:
            padded.append(indices)
            continue
        pad_value = -1 if max_len > 0 else 0
        pad = torch.full((max_len - indices.numel(),), pad_value, dtype=indices.dtype, device=indices.device)
        padded.append(torch.cat([indices, pad], dim=0))
    return padded
