from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from data.mask_samplers import PatchMaskConverter, apply_pixel_mask
from models.decoder_baseline import ConvDecoder
from models.hole_token_scatter import reshape_token_canvas, scatter_hole_tokens
from models.losses import SemanticReconstructionLoss


@dataclass
class TrainConfig:
    image_size: int = 32
    patch_size: int = 8
    token_dim: int = 16
    batch_size: int = 2
    lambda_jepa: float = 1.0
    lambda_rgb: float = 0.5


def run_dry_training_step(config: TrainConfig) -> dict[str, float]:
    grid_size = config.image_size // config.patch_size
    num_patches = grid_size * grid_size
    num_holes = num_patches // 4
    num_visible = num_patches - num_holes

    images = torch.randn(config.batch_size, 3, config.image_size, config.image_size)
    binary_mask = torch.zeros(config.batch_size, 1, config.image_size, config.image_size)
    binary_mask[:, :, : config.image_size // 2, : config.image_size // 2] = 1.0
    context_images = apply_pixel_mask(images, binary_mask)

    converter = PatchMaskConverter(patch_size=config.patch_size)
    patch_mask = converter(binary_mask)

    visible_indices = patch_mask.visible_indices[:, :num_visible].clamp_min(0)
    hole_indices = patch_mask.hole_indices[:, :num_holes].clamp_min(0)

    visible_tokens = torch.randn(config.batch_size, num_visible, config.token_dim)
    predicted_hole_tokens = torch.randn(config.batch_size, num_holes, config.token_dim)
    teacher_hole_tokens = torch.randn(config.batch_size, num_holes, config.token_dim)

    canvas = scatter_hole_tokens(
        visible_tokens=visible_tokens,
        predicted_hole_tokens=predicted_hole_tokens,
        visible_indices=visible_indices,
        hole_indices=hole_indices,
        num_patches=num_patches,
    )
    token_grid = reshape_token_canvas(canvas.tokens, (grid_size, grid_size))
    decoder = ConvDecoder(token_dim=config.token_dim)
    reconstructed = decoder(token_grid)
    reconstructed = torch.nn.functional.interpolate(reconstructed, size=images.shape[-2:], mode="bilinear", align_corners=False)

    criterion = SemanticReconstructionLoss(config.lambda_jepa, config.lambda_rgb)
    loss = criterion(predicted_hole_tokens, teacher_hole_tokens, reconstructed, images, binary_mask)
    return {
        **asdict(config),
        "context_mean": float(context_images.mean()),
        "semantic_loss": float(loss.semantic),
        "reconstruction_loss": float(loss.reconstruction),
        "total_loss": float(loss.total),
    }


if __name__ == "__main__":
    metrics = run_dry_training_step(TrainConfig())
    for key, value in metrics.items():
        print(f"{key}: {value}")
