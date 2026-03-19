from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn


ModuleFactory = Callable[[dict[str, torch.Tensor]], nn.Module]


@dataclass
class IJEPAComponents:
    encoder: nn.Module
    predictor: nn.Module
    target_encoder: nn.Module


class IdentityModule(nn.Module):
    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        return x


class LinearProjection(nn.Module):
    """Fallback module for checkpoints that only expose a single linear projection."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.proj = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class SequentialMLP(nn.Module):
    """Fallback module for checkpoints that store stacked linear MLP weights."""

    def __init__(self, layer_dims: list[tuple[int, int]], bias_map: dict[int, bool]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for idx, (in_features, out_features) in enumerate(layer_dims):
            layers.append(nn.Linear(in_features, out_features, bias=bias_map.get(idx, False)))
            if idx < len(layer_dims) - 1:
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class IJEPAWrapper(nn.Module):
    """Wrapper around encoder/predictor/target modules for downstream prototyping."""

    CHECKPOINT_CANDIDATE_KEYS: dict[str, tuple[str, ...]] = {
        "encoder": (
            "encoder",
            "context_encoder",
            "student_encoder",
            "online_encoder",
            "module.encoder",
            "state_dict.encoder",
        ),
        "predictor": (
            "predictor",
            "module.predictor",
            "state_dict.predictor",
        ),
        "target_encoder": (
            "target_encoder",
            "teacher_encoder",
            "ema_encoder",
            "module.target_encoder",
            "state_dict.target_encoder",
        ),
    }

    def __init__(self, components: IJEPAComponents) -> None:
        super().__init__()
        self.encoder = components.encoder
        self.predictor = components.predictor
        self.target_encoder = components.target_encoder

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        module_factories: dict[str, ModuleFactory] | None = None,
        strict: bool = False,
    ) -> "IJEPAWrapper":
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        module_factories = module_factories or {}

        components: dict[str, nn.Module] = {}
        for component_name in ("encoder", "predictor", "target_encoder"):
            state_dict = cls._extract_component_state_dict(checkpoint, component_name)
            factory = module_factories.get(component_name)
            module = cls._build_component(state_dict, factory)
            if state_dict:
                module.load_state_dict(state_dict, strict=strict)
            components[component_name] = module

        return cls(IJEPAComponents(**components))

    @classmethod
    def _extract_component_state_dict(
        cls,
        checkpoint: dict[str, Any],
        component_name: str,
    ) -> dict[str, torch.Tensor]:
        for key in cls.CHECKPOINT_CANDIDATE_KEYS[component_name]:
            value = cls._lookup_key(checkpoint, key)
            if isinstance(value, dict):
                tensor_dict = {str(k): v for k, v in value.items() if torch.is_tensor(v)}
                if tensor_dict:
                    return tensor_dict

        flat_state = checkpoint.get("state_dict") if isinstance(checkpoint.get("state_dict"), dict) else checkpoint
        if isinstance(flat_state, dict):
            prefixes = tuple(candidate.split(".")[-1] for candidate in cls.CHECKPOINT_CANDIDATE_KEYS[component_name])
            extracted: dict[str, torch.Tensor] = {}
            for key, value in flat_state.items():
                if not torch.is_tensor(value):
                    continue
                if key.startswith(prefixes):
                    normalized = key.split(".", 1)[1] if "." in key else key
                    extracted[normalized] = value
            if extracted:
                return extracted
        return {}

    @staticmethod
    def _lookup_key(checkpoint: dict[str, Any], dotted_key: str) -> Any:
        current: Any = checkpoint
        for part in dotted_key.split("."):
            if not isinstance(current, dict) or part not in current:
                return None
            current = current[part]
        return current

    @classmethod
    def _build_component(
        cls,
        state_dict: dict[str, torch.Tensor],
        factory: ModuleFactory | None,
    ) -> nn.Module:
        if factory is not None:
            return factory(state_dict)
        inferred = cls._infer_module_from_state_dict(state_dict)
        return inferred if inferred is not None else IdentityModule()

    @staticmethod
    def _infer_module_from_state_dict(state_dict: dict[str, torch.Tensor]) -> nn.Module | None:
        if not state_dict:
            return None

        weight_items = sorted(
            (name, tensor) for name, tensor in state_dict.items() if name.endswith("weight") and tensor.ndim == 2
        )
        if not weight_items:
            return None

        if len(weight_items) == 1:
            weight_name, weight_tensor = weight_items[0]
            bias_name = weight_name[:-6] + "bias"
            return LinearProjection(
                in_features=weight_tensor.shape[1],
                out_features=weight_tensor.shape[0],
                bias=bias_name in state_dict,
            )

        layer_dims: list[tuple[int, int]] = []
        bias_map: dict[int, bool] = {}
        for idx, (weight_name, weight_tensor) in enumerate(weight_items):
            layer_dims.append((weight_tensor.shape[1], weight_tensor.shape[0]))
            bias_map[idx] = weight_name[:-6] + "bias" in state_dict
        return SequentialMLP(layer_dims=layer_dims, bias_map=bias_map)

    def set_trainable(
        self,
        encoder: bool | None = None,
        predictor: bool | None = None,
        target_encoder: bool | None = None,
    ) -> None:
        for module_name, flag in {
            "encoder": encoder,
            "predictor": predictor,
            "target_encoder": target_encoder,
        }.items():
            if flag is None:
                continue
            module = getattr(self, module_name)
            module.train(flag)
            for parameter in module.parameters():
                parameter.requires_grad = flag

    @torch.no_grad()
    def update_target_encoder(self, momentum: float = 0.996) -> None:
        if not 0.0 <= momentum <= 1.0:
            raise ValueError("momentum must be between 0 and 1")

        encoder_params = dict(self.encoder.named_parameters())
        target_params = dict(self.target_encoder.named_parameters())
        if encoder_params.keys() != target_params.keys():
            raise ValueError("encoder and target_encoder parameters must share the same structure for EMA updates")

        for name, target_param in target_params.items():
            source_param = encoder_params[name]
            target_param.data.mul_(momentum).add_(source_param.data, alpha=1.0 - momentum)

        encoder_buffers = dict(self.encoder.named_buffers())
        target_buffers = dict(self.target_encoder.named_buffers())
        if encoder_buffers.keys() != target_buffers.keys():
            raise ValueError("encoder and target_encoder buffers must share the same structure for EMA updates")

        for name, target_buffer in target_buffers.items():
            target_buffer.copy_(encoder_buffers[name])

    def encode_context(self, visible_tokens: torch.Tensor) -> torch.Tensor:
        return self.encoder(visible_tokens)

    def predict_holes(self, context_tokens: torch.Tensor) -> torch.Tensor:
        return self.predictor(context_tokens)

    def encode_target(self, target_tokens: torch.Tensor) -> torch.Tensor:
        return self.target_encoder(target_tokens)
