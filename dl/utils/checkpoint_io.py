"""Loading a trained `DDGPredictor`'s weights from a Lightning checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


def require_existing_file(path_str: str, description: str) -> Path:
    path = Path(path_str)
    if not path.is_file():
        sys.exit(f"{description} not found: {path}")
    return path


def strip_model_submodule_prefix(state_dict: dict) -> dict:
    """`DDGLightningModule` stores the `DDGPredictor` as `self.model`, so its
    checkpointed `state_dict` keys are prefixed `model.` -- this rebuilds a
    plain `DDGPredictor` state dict from that, ignoring any other logged
    submodule (losses/metrics carry no learnable state of their own).
    """
    prefix = "model."
    return {key[len(prefix):]: value for key, value in state_dict.items() if key.startswith(prefix)}


def load_model_weights(model: torch.nn.Module, ckpt_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(strip_model_submodule_prefix(checkpoint["state_dict"]))
