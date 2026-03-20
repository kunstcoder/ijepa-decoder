from __future__ import annotations

import numpy as np
from PIL import Image
import torch

from data.mask_samplers import PatchMaskConverter, apply_pixel_mask
from models.hole_token_scatter import reshape_token_canvas, scatter_hole_tokens
from models.ijepa_wrapper import IJEPAWrapper
from train.train_semantic_inpaint import (
    ProgressConfig,
    RandomBlockMaskSampler,
    SemanticInpaintingModel,
    TrainConfig,
    build_dataloader,
    collect_config_metrics,
    inspect_checkpoint,
    load_config,
    log_metrics_to_tensorboard,
    log_training_previews_to_tensorboard,
    run_training,
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
    assert metrics["tensorboard_log_dir"] == "runs/semantic_inpaint"


def test_load_config_reads_yaml_and_normalizes_image_dir(tmp_path) -> None:
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        "\n".join(
            [
                "image_dir: /tmp/images",
                "epochs: 3",
                "data:",
                "  image_size: 32",
                "  batch_size: 2",
                "tensorboard:",
                "  enabled: false",
            ]
        )
    )

    config = load_config(config_path)

    assert config.image_dir == "/tmp/images"
    assert config.data.image_dir == "/tmp/images"
    assert config.epochs == 3
    assert config.data.image_size == 32
    assert config.tensorboard.enabled is False
    assert isinstance(config.progress, ProgressConfig)


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




def test_log_training_previews_to_tensorboard_records_image_batches() -> None:
    class DummyWriter:
        def __init__(self) -> None:
            self.images: list[tuple[str, tuple[int, ...], int]] = []

        def add_images(self, key: str, value: torch.Tensor, step: int) -> None:
            self.images.append((key, tuple(value.shape), step))

    writer = DummyWriter()
    image = torch.rand(2, 3, 8, 8)
    mask = torch.zeros(2, 1, 8, 8)
    mask[:, :, :4, :4] = 1.0
    reconstructed = torch.rand(2, 3, 8, 8)

    log_training_previews_to_tensorboard(
        writer,
        input_image=image,
        binary_mask=mask,
        reconstructed_image=reconstructed,
        global_step=5,
        max_images=1,
    )

    assert writer.images == [
        ("train/images/input", (1, 3, 8, 8), 5),
        ("train/images/mask", (1, 3, 8, 8), 5),
        ("train/images/masked_input", (1, 3, 8, 8), 5),
        ("train/images/reconstruction", (1, 3, 8, 8), 5),
        ("train/images/comparison", (1, 3, 8, 32), 5),
    ]

def test_random_block_mask_sampler_marks_square_region() -> None:
    sampler = RandomBlockMaskSampler(image_size=16, patch_size=4, min_hole_patches=1, max_hole_patches=2)
    mask = sampler.sample(2, device=torch.device("cpu"))
    assert mask.shape == (2, 1, 16, 16)
    assert float(mask.max()) == 1.0
    assert float(mask.min()) == 0.0


def test_semantic_inpainting_model_forward_shapes() -> None:
    model = SemanticInpaintingModel(image_size=16, patch_size=4, token_dim=8, decoder_hidden_dim=16)
    image = torch.rand(2, 3, 16, 16)
    mask = torch.zeros(2, 1, 16, 16)
    mask[:, :, :4, :4] = 1.0
    outputs = model(image, mask)
    assert outputs["predicted_hole_tokens"].shape == (2, 1, 8)
    assert outputs["teacher_hole_tokens"].shape == (2, 1, 8)
    assert outputs["reconstructed_image"].shape == (2, 3, 16, 16)


def test_build_dataloader_loads_symlinked_images_in_class_folders(tmp_path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    image = Image.fromarray(np.full((16, 16, 3), 200, dtype=np.uint8))
    source_path = source_dir / "sample.png"
    image.save(source_path)

    train_dir = tmp_path / "train" / "apple_pie"
    train_dir.mkdir(parents=True)
    (train_dir / "linked_sample.png").symlink_to(source_path)

    config = TrainConfig(image_dir=str(tmp_path / "train")).normalize()
    config.data.image_size = 16
    config.data.batch_size = 1
    dataloader = build_dataloader(config)

    batch = next(iter(dataloader))
    assert batch.shape == (1, 3, 16, 16)


def test_build_dataloader_rejects_broken_symlinks(tmp_path) -> None:
    train_dir = tmp_path / "train" / "bibimbap"
    train_dir.mkdir(parents=True)
    (train_dir / "broken.png").symlink_to(tmp_path / "missing.png")

    config = TrainConfig(image_dir=str(tmp_path / "train")).normalize()
    config.data.image_size = 16

    try:
        build_dataloader(config)
    except ValueError as exc:
        assert "broken image symlinks" in str(exc)
    else:
        raise AssertionError("Expected broken symlink validation to fail")


def test_build_dataloader_loads_images(tmp_path) -> None:
    image = Image.fromarray(np.full((16, 16, 3), 127, dtype=np.uint8))
    image.save(tmp_path / "sample.png")

    config = TrainConfig(image_dir=str(tmp_path)).normalize()
    config.data.image_size = 16
    config.data.batch_size = 1
    dataloader = build_dataloader(config)
    batch = next(iter(dataloader))
    assert batch.shape == (1, 3, 16, 16)


def test_run_training_executes_smoke_step(tmp_path) -> None:
    for index in range(2):
        image = Image.fromarray(np.full((16, 16, 3), 40 * (index + 1), dtype=np.uint8))
        image.save(tmp_path / f"sample_{index}.png")

    config = TrainConfig(image_dir=str(tmp_path), epochs=1, steps_per_epoch=1).normalize()
    config.data.image_size = 16
    config.data.batch_size = 1
    config.mask.patch_size = 4
    config.model.token_dim = 8
    config.model.decoder_hidden_dim = 16
    config.tensorboard.enabled = False

    metrics = run_training(config)

    assert metrics["epochs"] == 1.0
    assert metrics["dataset_size"] == 2.0
    assert "epoch_0_loss" in metrics
