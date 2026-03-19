from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class LoadedIJEPA:
    checkpoint_path: Path
    encoder_state: dict[str, torch.Tensor]
    predictor_state: dict[str, torch.Tensor]
    target_encoder_state: dict[str, torch.Tensor]

    def component_states(self) -> dict[str, dict[str, torch.Tensor]]:
        return {
            "encoder": self.encoder_state,
            "predictor": self.predictor_state,
            "target_encoder": self.target_encoder_state,
        }

    def summary(self) -> dict[str, float | str]:
        metrics: dict[str, float | str] = {
            "checkpoint_path": str(self.checkpoint_path),
        }
        for component_name, state_dict in self.component_states().items():
            tensor_count = len(state_dict)
            parameter_count = sum(tensor.numel() for tensor in state_dict.values())
            metrics[f"{component_name}_tensor_count"] = float(tensor_count)
            metrics[f"{component_name}_parameter_count"] = float(parameter_count)
            metrics[f"{component_name}_loaded"] = float(bool(state_dict))
        return metrics


class IJEPAWrapper:
    """Minimal checkpoint reader for local I-JEPA downstream experiments."""

    COMPONENT_PREFIXES: dict[str, tuple[str, ...]] = {
        "encoder": ("encoder", "context_encoder", "student_encoder", "online_encoder"),
        "predictor": ("predictor",),
        "target_encoder": ("target_encoder", "teacher_encoder", "ema_encoder"),
    }

    @classmethod
    def load(cls, checkpoint_path: str | Path) -> LoadedIJEPA:
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        return LoadedIJEPA(
            checkpoint_path=checkpoint_path,
            encoder_state=cls._extract_component_state_dict(checkpoint, "encoder"),
            predictor_state=cls._extract_component_state_dict(checkpoint, "predictor"),
            target_encoder_state=cls._extract_component_state_dict(checkpoint, "target_encoder"),
        )

    @classmethod
    def _extract_component_state_dict(
        cls,
        checkpoint: dict[str, Any],
        component_name: str,
    ) -> dict[str, torch.Tensor]:
        for prefix in cls.COMPONENT_PREFIXES[component_name]:
            nested_value = checkpoint.get(prefix)
            if isinstance(nested_value, dict):
                tensor_dict = {str(key): value for key, value in nested_value.items() if torch.is_tensor(value)}
                if tensor_dict:
                    return tensor_dict

        flat_state = checkpoint.get("state_dict") if isinstance(checkpoint.get("state_dict"), dict) else checkpoint
        if not isinstance(flat_state, dict):
            return {}

        extracted: dict[str, torch.Tensor] = {}
        valid_prefixes = tuple(f"{prefix}." for prefix in cls.COMPONENT_PREFIXES[component_name])
        for key, value in flat_state.items():
            if not torch.is_tensor(value) or not key.startswith(valid_prefixes):
                continue
            normalized_key = key.split(".", 1)[1]
            extracted[normalized_key] = value
        return extracted
