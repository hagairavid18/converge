"""Per-sample metadata features concatenated onto the pooled representation
(see `dl.models.ddg_model.DDGPredictor`), gated by
`ModelConfig.extra_metadata_fields`.
"""

from __future__ import annotations

import torch

from dl.models.schemas import ModelInputs
from dl.utils.constants import METADATA_FIELD_NORMALIZATION_STATS


def z_score(value: torch.Tensor, field: str) -> torch.Tensor:
    mean, std = METADATA_FIELD_NORMALIZATION_STATS[field]
    return (value - mean) / std


def build_metadata_tensor(inputs: ModelInputs, fields: tuple[str, ...]) -> torch.Tensor:
    normalized = [z_score(getattr(inputs, field), field) for field in fields]
    return torch.stack(normalized, dim=-1)
