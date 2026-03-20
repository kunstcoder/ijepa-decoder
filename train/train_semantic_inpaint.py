from __future__ import annotations

import argparse
import random
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:  # pragma: no cover - depends on optional dependency
    tqdm = None

from data.mask_samplers import PatchMaskConverter, apply_pixel_mask
from models.decoder_baseline import ConvDecoder
from models.hole_token_scatter import reshape_token_canvas, scatter_hole_tokens
from models.ijepa_wrapper import IJEPAWrapper, LoadedIJEPA
from models.losses import LossOutput, SemanticReconstructionLoss

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - depends on optional dependency
    yaml = None  # type: ignore[assignment]

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:  # pragma: no cover - depends on optional dependency
    SummaryWriter = None  # type: ignore[assignment]


@dataclass
class TensorBoardConfig:
    enabled: bool = True
    log_dir: str = "runs/semantic_inpaint"
    flush_secs: int = 10
    image_log_interval: int = 50
    max_images: int = 4


@dataclass
class ProgressConfig:
    enabled: bool = True
    refresh_rate: int = 1


@dataclass
class DataConfig:
    image_dir: str = ""
    image_size: int = 64
    batch_size: int = 8
    num_workers: int = 0
    shuffle: bool = True


@dataclass
class MaskConfig:
    patch_size: int = 16
    min_hole_patches: int = 1
    max_hole_patches: int = 4


@dataclass
class ModelConfig:
    token_dim: int = 128
    decoder_hidden_dim: int = 256


@dataclass
class OptimizerConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4


@dataclass
class LossConfig:
    lambda_jepa: float = 1.0
    lambda_rgb: float = 0.5


@dataclass
class TrainConfig:
    checkpoint_path: str = ""
    image_dir: str = ""
    epochs: int = 1
    steps_per_epoch: int | None = None
    seed: int = 0
    device: str = "cpu"
    tensorboard: TensorBoardConfig = field(default_factory=TensorBoardConfig)
    progress: ProgressConfig = field(default_factory=ProgressConfig)
    data: DataConfig = field(default_factory=DataConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    loss: LossConfig = field(default_factory=LossConfig)

    def normalize(self) -> TrainConfig:
        if not self.image_dir:
            self.image_dir = self.data.image_dir
        else:
            self.data.image_dir = self.image_dir
        return self


class ImageFolderDataset(Dataset[torch.Tensor]):
    IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}

    def __init__(self, image_dir: str | Path, image_size: int) -> None:
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.files = self._discover_files()
        if not self.files:
            raise ValueError(f"No image files were found under {self.image_dir}")

    def _discover_files(self) -> list[Path]:
        discovered: list[Path] = []
        broken_symlinks: list[Path] = []
        for path in sorted(self.image_dir.rglob("*")):
            if path.suffix.lower() not in self.IMAGE_SUFFIXES:
                continue
            if path.is_symlink() and not path.exists():
                broken_symlinks.append(path)
                continue
            if not path.is_file():
                continue
            discovered.append(path)
        if broken_symlinks:
            preview = ", ".join(str(path.relative_to(self.image_dir)) for path in broken_symlinks[:3])
            raise ValueError(
                f"Found {len(broken_symlinks)} broken image symlinks under {self.image_dir}. "
                f"Examples: {preview}"
            )
        return discovered

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> torch.Tensor:
        path = self.files[index]
        with Image.open(path) as image:
            image = image.convert("RGB").resize((self.image_size, self.image_size))
            array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)


class RandomBlockMaskSampler:
    def __init__(self, image_size: int, patch_size: int, min_hole_patches: int, max_hole_patches: int) -> None:
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.min_hole_patches = min_hole_patches
        self.max_hole_patches = max_hole_patches

    def sample(self, batch_size: int, *, device: torch.device) -> torch.Tensor:
        mask = torch.zeros(batch_size, 1, self.image_size, self.image_size, device=device)
        max_side = max(1, self.max_hole_patches)
        side = min(random.randint(self.min_hole_patches, max_side), self.grid_size)
        for batch_idx in range(batch_size):
            top = random.randint(0, self.grid_size - side)
            left = random.randint(0, self.grid_size - side)
            y0 = top * self.patch_size
            x0 = left * self.patch_size
            span = side * self.patch_size
            mask[batch_idx, :, y0 : y0 + span, x0 : x0 + span] = 1.0
        return mask


class PatchTokenizer(nn.Module):
    def __init__(self, patch_size: int, token_dim: int, channels: int = 3) -> None:
        super().__init__()
        self.patch_size = patch_size
        patch_dim = channels * patch_size * patch_size
        self.proj = nn.Linear(patch_dim, token_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = image.shape
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError("image height and width must be divisible by patch_size")
        h_patches = height // self.patch_size
        w_patches = width // self.patch_size
        patches = image.reshape(
            batch_size,
            channels,
            h_patches,
            self.patch_size,
            w_patches,
            self.patch_size,
        )
        patches = patches.permute(0, 2, 4, 1, 3, 5).reshape(batch_size, h_patches * w_patches, -1)
        return self.proj(patches)


class ContextPredictor(nn.Module):
    def __init__(self, token_dim: int, num_patches: int, num_heads: int = 4, depth: int = 2) -> None:
        super().__init__()
        if token_dim % num_heads != 0:
            raise ValueError("token_dim must be divisible by num_heads")
        self.position = nn.Embedding(num_patches, token_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=token_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.visible_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.output = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim * 2),
            nn.GELU(),
            nn.Linear(token_dim * 2, token_dim),
        )

    def forward(
        self,
        visible_tokens: torch.Tensor,
        visible_indices: torch.Tensor,
        hole_indices: torch.Tensor,
    ) -> torch.Tensor:
        visible_padding_mask = visible_indices < 0
        encoded_visible = visible_tokens + self.position(visible_indices.clamp_min(0))
        encoded_visible = self.visible_encoder(encoded_visible, src_key_padding_mask=visible_padding_mask)

        hole_queries = self.position(hole_indices.clamp_min(0))
        attended_holes, _ = self.cross_attention(
            query=hole_queries,
            key=encoded_visible,
            value=encoded_visible,
            key_padding_mask=visible_padding_mask,
            need_weights=False,
        )
        return self.output(attended_holes + hole_queries)


class SemanticInpaintingModel(nn.Module):
    def __init__(self, image_size: int, patch_size: int, token_dim: int, decoder_hidden_dim: int) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.tokenizer = PatchTokenizer(patch_size=patch_size, token_dim=token_dim)
        self.predictor = ContextPredictor(token_dim=token_dim, num_patches=self.num_patches)
        self.decoder = ConvDecoder(
            token_dim=token_dim,
            output_channels=3,
            hidden_dim=decoder_hidden_dim,
            upsample_factor=patch_size,
        )

    def forward(self, image: torch.Tensor, binary_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        patch_converter = PatchMaskConverter(patch_size=self.patch_size)
        patch_mask = patch_converter(binary_mask)

        visible_image = apply_pixel_mask(image, binary_mask)
        visible_tokens = self.tokenizer(visible_image)
        teacher_tokens = self.tokenizer(image).detach()

        visible_indices = patch_mask.visible_indices.clamp_min(0)
        gathered_visible_tokens = visible_tokens.gather(
            1,
            visible_indices.unsqueeze(-1).expand(-1, -1, visible_tokens.size(-1)),
        )
        predicted_hole_tokens = self.predictor(
            gathered_visible_tokens,
            patch_mask.visible_indices,
            patch_mask.hole_indices,
        )

        hole_indices = patch_mask.hole_indices.clamp_min(0)
        teacher_hole_tokens = teacher_tokens.gather(
            1,
            hole_indices.unsqueeze(-1).expand(-1, -1, teacher_tokens.size(-1)),
        )

        token_canvas = scatter_hole_tokens(
            visible_tokens=gathered_visible_tokens,
            predicted_hole_tokens=predicted_hole_tokens,
            visible_indices=visible_indices,
            hole_indices=hole_indices,
            num_patches=self.num_patches,
        )
        token_grid = reshape_token_canvas(token_canvas.tokens, (self.grid_size, self.grid_size))
        reconstructed = self.decoder(token_grid)

        return {
            "predicted_hole_tokens": predicted_hole_tokens,
            "teacher_hole_tokens": teacher_hole_tokens,
            "reconstructed_image": reconstructed,
            "binary_mask": binary_mask,
        }


def _flatten(prefix: str, value: Any) -> dict[str, float | str]:
    if hasattr(value, "__dataclass_fields__"):
        value = asdict(value)
    if isinstance(value, dict):
        flattened: dict[str, float | str] = {}
        for child_key, child_value in value.items():
            child_prefix = f"{prefix}_{child_key}" if prefix else str(child_key)
            flattened.update(_flatten(child_prefix, child_value))
        return flattened
    if isinstance(value, Path):
        return {prefix: str(value)}
    return {prefix: value}


def _merge_dataclass(instance: Any, values: dict[str, Any]) -> Any:
    field_map = {field_info.name: field_info for field_info in fields(instance)}
    for key, value in values.items():
        if key not in field_map:
            raise KeyError(f"Unknown config field: {key}")
        current_value = getattr(instance, key)
        if is_dataclass(current_value) and isinstance(value, dict):
            _merge_dataclass(current_value, value)
        else:
            setattr(instance, key, value)
    return instance


def load_config(config_path: str | Path) -> TrainConfig:
    if yaml is None:
        raise RuntimeError("YAML config loading requires PyYAML. Install it with `pip install PyYAML`.")

    path = Path(config_path)
    loaded = yaml.safe_load(path.read_text()) or {}
    if not isinstance(loaded, dict):
        raise ValueError("Top-level YAML config must be a mapping")

    config = TrainConfig()
    _merge_dataclass(config, loaded)
    return config.normalize()


def collect_config_metrics(config: TrainConfig) -> dict[str, float | str]:
    return _flatten("", asdict(config))


def create_tensorboard_writer(config: TensorBoardConfig) -> SummaryWriter | None:
    if not config.enabled:
        return None
    if SummaryWriter is None:
        raise RuntimeError(
            "TensorBoard logging requires tensorboard to be installed. Install it with `pip install tensorboard`."
        )
    return SummaryWriter(log_dir=config.log_dir, flush_secs=config.flush_secs)


def log_metrics_to_tensorboard(
    writer: SummaryWriter | None,
    metrics: dict[str, float | str],
    *,
    global_step: int,
) -> None:
    if writer is None:
        return

    for key, value in metrics.items():
        if isinstance(value, bool):
            writer.add_scalar(key, int(value), global_step)
        elif isinstance(value, (int, float)):
            writer.add_scalar(key, value, global_step)
        elif isinstance(value, str):
            writer.add_text(key, value, global_step)


def _prepare_image_preview_batch(images: torch.Tensor, max_images: int) -> torch.Tensor:
    preview = images[:max(1, max_images)].detach().cpu().float()
    return preview.clamp(0.0, 1.0)


def log_training_previews_to_tensorboard(
    writer: SummaryWriter | None,
    *,
    input_image: torch.Tensor,
    binary_mask: torch.Tensor,
    reconstructed_image: torch.Tensor,
    global_step: int,
    max_images: int,
) -> None:
    if writer is None:
        return

    input_preview = _prepare_image_preview_batch(input_image, max_images)
    reconstruction_preview = _prepare_image_preview_batch(reconstructed_image, max_images)
    mask_preview = _prepare_image_preview_batch(binary_mask.expand(-1, 3, -1, -1), max_images)
    masked_preview = _prepare_image_preview_batch(input_image * (1.0 - binary_mask), max_images)
    comparison_preview = torch.cat(
        [input_preview, mask_preview, masked_preview, reconstruction_preview],
        dim=-1,
    )

    writer.add_images("train/images/input", input_preview, global_step)
    writer.add_images("train/images/mask", mask_preview, global_step)
    writer.add_images("train/images/masked_input", masked_preview, global_step)
    writer.add_images("train/images/reconstruction", reconstruction_preview, global_step)
    writer.add_images("train/images/comparison", comparison_preview, global_step)


def inspect_checkpoint(
    config: TrainConfig,
    *,
    writer: SummaryWriter | None = None,
    global_step: int = 0,
) -> dict[str, float | str]:
    if not config.checkpoint_path:
        raise ValueError("checkpoint_path must be set to load a local I-JEPA checkpoint")

    loaded_checkpoint: LoadedIJEPA = IJEPAWrapper.load(config.checkpoint_path)
    metrics = {
        **collect_config_metrics(config),
        **loaded_checkpoint.summary(),
    }
    log_metrics_to_tensorboard(writer, metrics, global_step=global_step)
    return metrics


def build_dataloader(config: TrainConfig) -> DataLoader[torch.Tensor]:
    image_dir = config.image_dir or config.data.image_dir
    if not image_dir:
        raise ValueError("image_dir must be provided for training")
    dataset = ImageFolderDataset(image_dir=image_dir, image_size=config.data.image_size)
    return DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=config.data.shuffle,
        num_workers=config.data.num_workers,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one_epoch(
    model: SemanticInpaintingModel,
    dataloader: DataLoader[torch.Tensor],
    optimizer: torch.optim.Optimizer,
    loss_fn: SemanticReconstructionLoss,
    mask_sampler: RandomBlockMaskSampler,
    device: torch.device,
    *,
    epoch_index: int,
    writer: SummaryWriter | None,
    steps_per_epoch: int | None,
    progress_enabled: bool,
    progress_refresh_rate: int,
    image_log_interval: int,
    max_preview_images: int,
) -> dict[str, float]:
    model.train()
    running = {"loss": 0.0, "semantic_loss": 0.0, "reconstruction_loss": 0.0}
    total_steps = 0
    epoch_steps = steps_per_epoch or len(dataloader)
    iterator = enumerate(dataloader)
    progress_bar = None
    if progress_enabled and tqdm is not None:
        progress_bar = tqdm(total=epoch_steps, desc=f"Epoch {epoch_index + 1}", leave=True)

    for step_index, batch in iterator:
        if steps_per_epoch is not None and step_index >= steps_per_epoch:
            break

        image = batch.to(device)
        mask = mask_sampler.sample(image.size(0), device=device)
        outputs = model(image, mask)
        loss_output: LossOutput = loss_fn(
            predicted_hole_tokens=outputs["predicted_hole_tokens"],
            teacher_hole_tokens=outputs["teacher_hole_tokens"],
            reconstructed_image=outputs["reconstructed_image"],
            target_image=image,
            binary_mask=outputs["binary_mask"],
        )

        optimizer.zero_grad(set_to_none=True)
        loss_output.total.backward()
        optimizer.step()

        running["loss"] += float(loss_output.total.detach().cpu())
        running["semantic_loss"] += float(loss_output.semantic.detach().cpu())
        running["reconstruction_loss"] += float(loss_output.reconstruction.detach().cpu())
        global_step = epoch_index * max(1, steps_per_epoch or len(dataloader)) + step_index
        log_metrics_to_tensorboard(
            writer,
            {
                "train/loss": running["loss"] / (step_index + 1),
                "train/semantic_loss": running["semantic_loss"] / (step_index + 1),
                "train/reconstruction_loss": running["reconstruction_loss"] / (step_index + 1),
            },
            global_step=global_step,
        )
        total_steps += 1
        if writer is not None and step_index % max(1, image_log_interval) == 0:
            log_training_previews_to_tensorboard(
                writer,
                input_image=image,
                binary_mask=outputs["binary_mask"],
                reconstructed_image=outputs["reconstructed_image"],
                global_step=global_step,
                max_images=max(1, max_preview_images),
            )
        if progress_bar is not None:
            progress_bar.update(1)
            if step_index % max(1, progress_refresh_rate) == 0:
                progress_bar.set_postfix(
                    loss=f"{running['loss'] / total_steps:.4f}",
                    semantic=f"{running['semantic_loss'] / total_steps:.4f}",
                    reconstruction=f"{running['reconstruction_loss'] / total_steps:.4f}",
                )
        elif progress_enabled:
            print(
                f"[epoch {epoch_index + 1} step {step_index + 1}/{epoch_steps}] "
                f"loss={running['loss'] / total_steps:.4f} "
                f"semantic={running['semantic_loss'] / total_steps:.4f} "
                f"reconstruction={running['reconstruction_loss'] / total_steps:.4f}",
                flush=True,
            )

    if progress_bar is not None:
        progress_bar.close()

    if total_steps == 0:
        raise ValueError("No training steps were executed. Check steps_per_epoch and dataset size.")
    return {key: value / total_steps for key, value in running.items()}


def run_training(config: TrainConfig, *, writer: SummaryWriter | None = None) -> dict[str, float | str]:
    config.normalize()
    set_seed(config.seed)
    device = torch.device(config.device)
    dataloader = build_dataloader(config)
    print(
        f"Loaded {len(dataloader.dataset)} images from {config.image_dir or config.data.image_dir} "
        f"(batch_size={config.data.batch_size}, num_workers={config.data.num_workers})",
        flush=True,
    )
    model = SemanticInpaintingModel(
        image_size=config.data.image_size,
        patch_size=config.mask.patch_size,
        token_dim=config.model.token_dim,
        decoder_hidden_dim=config.model.decoder_hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimizer.lr,
        weight_decay=config.optimizer.weight_decay,
    )
    loss_fn = SemanticReconstructionLoss(
        lambda_jepa=config.loss.lambda_jepa,
        lambda_rgb=config.loss.lambda_rgb,
    )
    mask_sampler = RandomBlockMaskSampler(
        image_size=config.data.image_size,
        patch_size=config.mask.patch_size,
        min_hole_patches=config.mask.min_hole_patches,
        max_hole_patches=config.mask.max_hole_patches,
    )

    summary: dict[str, float | str] = {}
    if config.checkpoint_path:
        summary.update(inspect_checkpoint(config, writer=writer, global_step=0))

    for epoch_index in range(config.epochs):
        epoch_metrics = train_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            mask_sampler=mask_sampler,
            device=device,
            epoch_index=epoch_index,
            writer=writer,
            steps_per_epoch=config.steps_per_epoch,
            progress_enabled=config.progress.enabled,
            progress_refresh_rate=config.progress.refresh_rate,
            image_log_interval=config.tensorboard.image_log_interval,
            max_preview_images=config.tensorboard.max_images,
        )
        summary.update({f"epoch_{epoch_index}_{key}": value for key, value in epoch_metrics.items()})

    summary.update(
        {
            "epochs": float(config.epochs),
            "device": str(device),
            "dataset_size": float(len(dataloader.dataset)),
            "steps_per_epoch": float(config.steps_per_epoch or len(dataloader)),
        }
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight semantic inpainting baseline from a YAML config.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/downstream_semantic_inpaint.yaml",
        help="Path to the YAML config file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    config = load_config(arguments.config)

    writer = create_tensorboard_writer(config.tensorboard)
    try:
        metrics = run_training(config, writer=writer)
    finally:
        if writer is not None:
            writer.close()

    for key, value in metrics.items():
        print(f"{key}: {value}")
