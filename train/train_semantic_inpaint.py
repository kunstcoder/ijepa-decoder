from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch
from torch import nn

from data.mask_samplers import PatchMaskConverter, apply_pixel_mask
from models.decoder_baseline import ConvDecoder
from models.hole_token_scatter import reshape_token_canvas, scatter_hole_tokens
from models.ijepa_wrapper import IJEPAComponents, IJEPAWrapper
from models.losses import SemanticReconstructionLoss


@dataclass(frozen=True)
class StageConfig:
    name: str
    freeze_encoder: bool
    freeze_predictor: bool
    freeze_target_encoder: bool
    update_target_encoder: bool


@dataclass
class OptimizerConfig:
    encoder_lr: float = 1e-5
    predictor_lr: float = 5e-5
    decoder_lr: float = 1e-4
    weight_decay: float = 0.0


@dataclass
class TrainConfig:
    image_size: int = 32
    patch_size: int = 8
    token_dim: int = 16
    batch_size: int = 2
    lambda_jepa: float = 1.0
    lambda_rgb: float = 0.5
    freeze_encoder: bool = True
    freeze_predictor: bool = True
    freeze_target_encoder: bool = True
    ema_momentum: float = 0.996
    stage: str = "decoder_warmup"
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)


TRAINING_STAGES: dict[str, StageConfig] = {
    "decoder_warmup": StageConfig(
        name="decoder_warmup",
        freeze_encoder=True,
        freeze_predictor=True,
        freeze_target_encoder=True,
        update_target_encoder=False,
    ),
    "predictor_finetune": StageConfig(
        name="predictor_finetune",
        freeze_encoder=True,
        freeze_predictor=False,
        freeze_target_encoder=True,
        update_target_encoder=True,
    ),
    "end_to_end": StageConfig(
        name="end_to_end",
        freeze_encoder=False,
        freeze_predictor=False,
        freeze_target_encoder=True,
        update_target_encoder=True,
    ),
}


def resolve_stage_config(config: TrainConfig) -> StageConfig:
    try:
        return TRAINING_STAGES[config.stage]
    except KeyError as error:
        available = ", ".join(sorted(TRAINING_STAGES))
        raise ValueError(f"Unknown training stage '{config.stage}'. Available stages: {available}") from error


def configure_training_stage(wrapper: IJEPAWrapper, config: TrainConfig) -> StageConfig:
    stage = resolve_stage_config(config)
    wrapper.set_trainable(
        encoder=not stage.freeze_encoder,
        predictor=not stage.freeze_predictor,
        target_encoder=not stage.freeze_target_encoder,
    )
    return stage


def build_optimizer(
    wrapper: IJEPAWrapper,
    decoder: nn.Module,
    optimizer_config: OptimizerConfig,
) -> torch.optim.Optimizer:
    param_groups: list[dict[str, object]] = []

    group_specs = (
        ("encoder", wrapper.encoder, optimizer_config.encoder_lr),
        ("predictor", wrapper.predictor, optimizer_config.predictor_lr),
        ("decoder", decoder, optimizer_config.decoder_lr),
    )
    for group_name, module, learning_rate in group_specs:
        params = [parameter for parameter in module.parameters() if parameter.requires_grad]
        if not params:
            continue
        param_groups.append(
            {
                "name": group_name,
                "params": params,
                "lr": learning_rate,
                "weight_decay": optimizer_config.weight_decay,
            }
        )

    if not param_groups:
        raise ValueError("No trainable parameters found for optimizer construction")

    return torch.optim.AdamW(param_groups)


def summarize_optimizer(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    summary: dict[str, float] = {
        "optimizer_param_groups": float(len(optimizer.param_groups)),
        "optimizer_trainable_parameters": 0.0,
    }
    for group in optimizer.param_groups:
        group_name = str(group.get("name", f"group_{len(summary)}"))
        param_count = sum(parameter.numel() for parameter in group["params"])
        summary[f"optimizer_{group_name}_lr"] = float(group["lr"])
        summary[f"optimizer_{group_name}_params"] = float(param_count)
        summary["optimizer_trainable_parameters"] += float(param_count)
    return summary


def _build_dummy_wrapper(token_dim: int) -> IJEPAWrapper:
    encoder = nn.Linear(token_dim, token_dim)
    predictor = nn.Linear(token_dim, token_dim)
    target_encoder = nn.Linear(token_dim, token_dim)
    return IJEPAWrapper(
        IJEPAComponents(
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
        )
    )


def run_dry_training_step(config: TrainConfig) -> dict[str, float | str]:
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

    wrapper = _build_dummy_wrapper(config.token_dim)
    stage = configure_training_stage(wrapper, config)

    visible_tokens = wrapper.encode_context(torch.randn(config.batch_size, num_visible, config.token_dim))
    predicted_hole_tokens = wrapper.predict_holes(torch.randn(config.batch_size, num_holes, config.token_dim))
    teacher_hole_tokens = wrapper.encode_target(torch.randn(config.batch_size, num_holes, config.token_dim))

    canvas = scatter_hole_tokens(
        visible_tokens=visible_tokens,
        predicted_hole_tokens=predicted_hole_tokens,
        visible_indices=visible_indices,
        hole_indices=hole_indices,
        num_patches=num_patches,
    )
    token_grid = reshape_token_canvas(canvas.tokens, (grid_size, grid_size))
    decoder = ConvDecoder(token_dim=config.token_dim)
    optimizer = build_optimizer(wrapper, decoder, config.optimizer)

    reconstructed = decoder(token_grid)
    reconstructed = torch.nn.functional.interpolate(reconstructed, size=images.shape[-2:], mode="bilinear", align_corners=False)

    criterion = SemanticReconstructionLoss(config.lambda_jepa, config.lambda_rgb)
    loss = criterion(predicted_hole_tokens, teacher_hole_tokens, reconstructed, images, binary_mask)

    if stage.update_target_encoder:
        wrapper.update_target_encoder(momentum=config.ema_momentum)

    metrics: dict[str, float | str] = {
        **asdict(config),
        "stage_name": stage.name,
        "ema_update_applied": float(stage.update_target_encoder),
        "context_mean": float(context_images.mean()),
        "semantic_loss": float(loss.semantic),
        "reconstruction_loss": float(loss.reconstruction),
        "total_loss": float(loss.total),
        "encoder_trainable": float(any(parameter.requires_grad for parameter in wrapper.encoder.parameters())),
        "predictor_trainable": float(any(parameter.requires_grad for parameter in wrapper.predictor.parameters())),
        "target_encoder_trainable": float(any(parameter.requires_grad for parameter in wrapper.target_encoder.parameters())),
    }
    metrics.update(summarize_optimizer(optimizer))
    return metrics


if __name__ == "__main__":
    metrics = run_dry_training_step(TrainConfig())
    for key, value in metrics.items():
        print(f"{key}: {value}")
