"""Normalizes a per-sample scalar prediction to shape (B,), accepting either
(B,) or (B, 1) -- the model/training workstream's regression head may emit
either, per its `PredictionHeads` output convention.
"""

from __future__ import annotations

import torch


def flatten_last_singleton_dim(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() >= 2 and tensor.shape[-1] == 1:
        return tensor.squeeze(-1)
    return tensor
