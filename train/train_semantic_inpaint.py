from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from models.ijepa_wrapper import IJEPAWrapper, LoadedIJEPA

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:  # pragma: no cover - depends on optional dependency
    SummaryWriter = None  # type: ignore[assignment]


@dataclass
class TensorBoardConfig:
    enabled: bool = True
    log_dir: str = "runs/ijepa_checkpoint"
    flush_secs: int = 10


@dataclass
class TrainConfig:
    checkpoint_path: str = ""
    tensorboard: TensorBoardConfig = field(default_factory=TensorBoardConfig)


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


def collect_config_metrics(config: TrainConfig) -> dict[str, float | str]:
    return _flatten("", asdict(config))


def create_tensorboard_writer(config: TensorBoardConfig) -> SummaryWriter | None:
    if not config.enabled:
        return None
    if SummaryWriter is None:
        raise RuntimeError(
            "TensorBoard logging requires tensorboard to be installed. "
            "Install it with `pip install tensorboard`."
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load a local I-JEPA checkpoint and log summary metrics.")
    parser.add_argument("checkpoint_path", type=str, help="Local path to the I-JEPA checkpoint.")
    parser.add_argument("--global-step", type=int, default=0, help="Global step used for TensorBoard logging.")
    parser.add_argument(
        "--disable-tensorboard",
        action="store_true",
        help="Skip TensorBoard writer creation even when the config enables it.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Override the TensorBoard log directory for this run.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    config = TrainConfig(checkpoint_path=arguments.checkpoint_path)
    if arguments.disable_tensorboard:
        config.tensorboard.enabled = False
    if arguments.log_dir is not None:
        config.tensorboard.log_dir = arguments.log_dir

    writer = create_tensorboard_writer(config.tensorboard)
    try:
        metrics = inspect_checkpoint(config, writer=writer, global_step=arguments.global_step)
    finally:
        if writer is not None:
            writer.close()

    for key, value in metrics.items():
        print(f"{key}: {value}")
