from __future__ import annotations

import torch
from torch import nn

from data.mask_samplers import PatchMaskConverter, apply_pixel_mask
from models.hole_token_scatter import reshape_token_canvas, scatter_hole_tokens
from models.ijepa_wrapper import IJEPAComponents, IJEPAWrapper, LinearProjection
from train.train_semantic_inpaint import (
    TRAINING_STAGES,
    TrainConfig,
    build_optimizer,
    configure_training_stage,
    resolve_stage_config,
    run_dry_training_step,
)


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


def test_checkpoint_loader_extracts_flat_state_dict(tmp_path) -> None:
    checkpoint_path = tmp_path / "ijepa.pt"
    checkpoint = {
        "state_dict": {
            "encoder.proj.weight": torch.eye(2),
            "encoder.proj.bias": torch.tensor([0.5, -0.5]),
            "predictor.proj.weight": torch.eye(2),
            "predictor.proj.bias": torch.tensor([1.0, 1.0]),
            "target_encoder.proj.weight": torch.eye(2),
            "target_encoder.proj.bias": torch.tensor([0.0, 0.0]),
        }
    }
    torch.save(checkpoint, checkpoint_path)

    wrapper = IJEPAWrapper.from_checkpoint(checkpoint_path)
    sample = torch.tensor([[1.0, 2.0]])

    assert isinstance(wrapper.encoder, LinearProjection)
    assert torch.allclose(wrapper.encode_context(sample), torch.tensor([[1.5, 1.5]]))
    assert torch.allclose(wrapper.predict_holes(sample), torch.tensor([[2.0, 3.0]]))


def test_wrapper_trainable_flags_and_ema_update() -> None:
    encoder = nn.Linear(2, 2, bias=False)
    predictor = nn.Linear(2, 2, bias=False)
    target_encoder = nn.Linear(2, 2, bias=False)

    with torch.no_grad():
        encoder.weight.copy_(torch.tensor([[2.0, 0.0], [0.0, 2.0]]))
        predictor.weight.copy_(torch.eye(2))
        target_encoder.weight.copy_(torch.tensor([[0.0, 0.0], [0.0, 0.0]]))

    wrapper = IJEPAWrapper(
        IJEPAComponents(
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
        )
    )
    wrapper.set_trainable(encoder=False, predictor=True, target_encoder=False)

    assert all(not parameter.requires_grad for parameter in wrapper.encoder.parameters())
    assert all(parameter.requires_grad for parameter in wrapper.predictor.parameters())
    assert all(not parameter.requires_grad for parameter in wrapper.target_encoder.parameters())

    wrapper.update_target_encoder(momentum=0.5)
    assert torch.allclose(wrapper.target_encoder.weight, torch.tensor([[1.0, 0.0], [0.0, 1.0]]))


def test_dry_training_step_runs() -> None:
    metrics = run_dry_training_step(TrainConfig())
    assert metrics["semantic_loss"] >= 0.0
    assert metrics["reconstruction_loss"] >= 0.0
    assert metrics["encoder_trainable"] == 0.0
    assert metrics["predictor_trainable"] == 0.0


def test_stage_configuration_for_predictor_finetune() -> None:
    encoder = nn.Linear(2, 2, bias=False)
    predictor = nn.Linear(2, 2, bias=False)
    target_encoder = nn.Linear(2, 2, bias=False)
    wrapper = IJEPAWrapper(
        IJEPAComponents(
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
        )
    )

    stage = configure_training_stage(wrapper, TrainConfig(stage="predictor_finetune"))

    assert stage == TRAINING_STAGES["predictor_finetune"]
    assert all(not parameter.requires_grad for parameter in wrapper.encoder.parameters())
    assert all(parameter.requires_grad for parameter in wrapper.predictor.parameters())
    assert all(not parameter.requires_grad for parameter in wrapper.target_encoder.parameters())


def test_resolve_stage_config_rejects_unknown_stage() -> None:
    try:
        resolve_stage_config(TrainConfig(stage="unknown"))
    except ValueError as error:
        assert "Unknown training stage" in str(error)
    else:
        raise AssertionError("Expected unknown stage to raise ValueError")


def test_build_optimizer_only_includes_trainable_modules() -> None:
    encoder = nn.Linear(2, 2)
    predictor = nn.Linear(2, 2)
    target_encoder = nn.Linear(2, 2)
    decoder = nn.Linear(2, 2)
    wrapper = IJEPAWrapper(
        IJEPAComponents(
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
        )
    )
    configure_training_stage(wrapper, TrainConfig(stage="decoder_warmup"))

    optimizer = build_optimizer(wrapper, decoder, TrainConfig().optimizer)

    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["name"] == "decoder"


def test_dry_training_step_predictor_stage_reports_optimizer_groups() -> None:
    metrics = run_dry_training_step(TrainConfig(stage="predictor_finetune"))
    assert metrics["stage_name"] == "predictor_finetune"
    assert metrics["predictor_trainable"] == 1.0
    assert metrics["encoder_trainable"] == 0.0
    assert metrics["ema_update_applied"] == 1.0
    assert metrics["optimizer_param_groups"] == 2.0
