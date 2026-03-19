from __future__ import annotations

import torch

from data.mask_samplers import PatchMaskConverter, apply_pixel_mask
from models.hole_token_scatter import reshape_token_canvas, scatter_hole_tokens
from models.ijepa_wrapper import IJEPAWrapper
from train.train_semantic_inpaint import (
    TrainConfig,
    collect_config_metrics,
    inspect_checkpoint,
    log_metrics_to_tensorboard,
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
            "predictor.proj.weight": torch.eye(2),
            "target_encoder.proj.weight": torch.eye(2),
        }
    }
    torch.save(checkpoint, checkpoint_path)

    loaded = IJEPAWrapper.load(checkpoint_path)

    assert loaded.encoder_state["proj.weight"].shape == (2, 2)
    assert loaded.predictor_state["proj.weight"].shape == (2, 2)
    assert loaded.target_encoder_state["proj.weight"].shape == (2, 2)


def test_checkpoint_loader_reads_nested_component_dicts(tmp_path) -> None:
    checkpoint_path = tmp_path / "ijepa_nested.pt"
    checkpoint = {
        "encoder": {"proj.weight": torch.eye(2)},
        "predictor": {"proj.weight": torch.eye(2)},
        "teacher_encoder": {"proj.weight": torch.eye(2)},
    }
    torch.save(checkpoint, checkpoint_path)

    loaded = IJEPAWrapper.load(checkpoint_path)

    assert loaded.summary()["encoder_loaded"] == 1.0
    assert loaded.summary()["predictor_loaded"] == 1.0
    assert loaded.summary()["target_encoder_loaded"] == 1.0


def test_inspect_checkpoint_returns_summary_metrics(tmp_path) -> None:
    checkpoint_path = tmp_path / "ijepa.pt"
    torch.save({"state_dict": {"encoder.proj.weight": torch.ones(2, 2)}}, checkpoint_path)

    metrics = inspect_checkpoint(TrainConfig(checkpoint_path=str(checkpoint_path)))

    assert metrics["checkpoint_path"] == str(checkpoint_path)
    assert metrics["encoder_loaded"] == 1.0
    assert metrics["predictor_loaded"] == 0.0
    assert metrics["target_encoder_loaded"] == 0.0
    assert metrics["encoder_parameter_count"] == 4.0


def test_collect_config_metrics_flattens_nested_dataclasses() -> None:
    metrics = collect_config_metrics(TrainConfig(checkpoint_path="/tmp/model.pt"))
    assert metrics["checkpoint_path"] == "/tmp/model.pt"
    assert metrics["tensorboard_log_dir"] == "runs/ijepa_checkpoint"


def test_log_metrics_to_tensorboard_records_scalars_and_text() -> None:
    class DummyWriter:
        def __init__(self) -> None:
            self.scalars: list[tuple[str, float, int]] = []
            self.texts: list[tuple[str, str, int]] = []

        def add_scalar(self, key: str, value: float, step: int) -> None:
            self.scalars.append((key, float(value), step))

        def add_text(self, key: str, value: str, step: int) -> None:
            self.texts.append((key, value, step))

    writer = DummyWriter()
    log_metrics_to_tensorboard(writer, {"loaded": 1.0, "path": "checkpoint.pt"}, global_step=7)

    assert writer.scalars == [("loaded", 1.0, 7)]
    assert writer.texts == [("path", "checkpoint.pt", 7)]
