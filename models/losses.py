from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class LossOutput:
    total: torch.Tensor
    semantic: torch.Tensor
    reconstruction: torch.Tensor


class SemanticReconstructionLoss:
    def __init__(self, lambda_jepa: float = 1.0, lambda_rgb: float = 0.5) -> None:
        self.lambda_jepa = lambda_jepa
        self.lambda_rgb = lambda_rgb

    def __call__(
        self,
        predicted_hole_tokens: torch.Tensor,
        teacher_hole_tokens: torch.Tensor,
        reconstructed_image: torch.Tensor,
        target_image: torch.Tensor,
        binary_mask: torch.Tensor,
    ) -> LossOutput:
        semantic_loss = F.smooth_l1_loss(predicted_hole_tokens, teacher_hole_tokens)
        reconstruction_loss = ((reconstructed_image - target_image).abs() * binary_mask).sum()
        reconstruction_loss = reconstruction_loss / binary_mask.sum().clamp_min(1.0)
        total = self.lambda_jepa * semantic_loss + self.lambda_rgb * reconstruction_loss
        return LossOutput(total=total, semantic=semantic_loss, reconstruction=reconstruction_loss)
