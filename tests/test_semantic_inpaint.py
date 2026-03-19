from __future__ import annotations

import torch

from data.mask_samplers import PatchMaskConverter, apply_pixel_mask
from models.hole_token_scatter import reshape_token_canvas, scatter_hole_tokens
from train.train_semantic_inpaint import TrainConfig, run_dry_training_step


def test_patch_mask_converter_marks_holes() -> None:
    mask = torch.zeros(1, 1, 8, 8)
    mask[:, :, :4, :4] = 1.0
    converter = PatchMaskConverter(patch_size=4)
    patch_mask = converter(mask)
    assert patch_mask.patch_mask.shape == (1, 4)
    assert int(patch_mask.patch_mask.sum()) == 1


def test_apply_pixel_mask_zeros_hole_pixels() -> None:
    image = torch.ones(1, 3, 4, 4)
    mask = torch.zeros(1, 1, 4, 4)
    mask[:, :, 0, 0] = 1.0
    masked = apply_pixel_mask(image, mask)
    assert masked[:, :, 0, 0].sum() == 0


def test_scatter_hole_tokens_restores_canvas() -> None:
    visible_tokens = torch.tensor([[[1.0], [2.0]]])
    hole_tokens = torch.tensor([[[9.0], [8.0]]])
    visible_indices = torch.tensor([[0, 3]])
    hole_indices = torch.tensor([[1, 2]])
    canvas = scatter_hole_tokens(visible_tokens, hole_tokens, visible_indices, hole_indices, num_patches=4)
    reshaped = reshape_token_canvas(canvas.tokens, (2, 2))
    assert reshaped.shape == (1, 1, 2, 2)
    assert torch.equal(canvas.tokens.flatten(), torch.tensor([1.0, 9.0, 8.0, 2.0]))


def test_dry_training_step_runs() -> None:
    metrics = run_dry_training_step(TrainConfig())
    assert metrics["semantic_loss"] >= 0.0
    assert metrics["reconstruction_loss"] >= 0.0
