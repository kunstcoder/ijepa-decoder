from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


@dataclass
class IJEPAComponents:
    encoder: nn.Module
    predictor: nn.Module
    target_encoder: nn.Module


class IdentityModule(nn.Module):
    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        return x


class IJEPAWrapper(nn.Module):
    """Minimal wrapper around encoder/predictor/target modules for downstream prototyping."""

    def __init__(self, components: IJEPAComponents) -> None:
        super().__init__()
        self.encoder = components.encoder
        self.predictor = components.predictor
        self.target_encoder = components.target_encoder

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path) -> "IJEPAWrapper":
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        encoder = IdentityModule()
        predictor = IdentityModule()
        target_encoder = IdentityModule()

        for module_name, module in {
            "encoder": encoder,
            "predictor": predictor,
            "target_encoder": target_encoder,
        }.items():
            state_dict = checkpoint.get(module_name)
            if isinstance(state_dict, dict):
                module.load_state_dict(state_dict, strict=False)

        return cls(IJEPAComponents(encoder=encoder, predictor=predictor, target_encoder=target_encoder))

    def encode_context(self, visible_tokens: torch.Tensor) -> torch.Tensor:
        return self.encoder(visible_tokens)

    def predict_holes(self, context_tokens: torch.Tensor) -> torch.Tensor:
        return self.predictor(context_tokens)

    def encode_target(self, target_tokens: torch.Tensor) -> torch.Tensor:
        return self.target_encoder(target_tokens)
