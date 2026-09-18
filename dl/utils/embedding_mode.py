"""Predicates over `shared.constants.EmbeddingSourceMode`.

Shared by the embedding-fusion layer (`dl.layers.embedding_fusion`) and the
model's branch-confidence selection logic (`dl.models.confidence_selection`)
so the two never disagree about which embedding source(s) are active.
"""

from __future__ import annotations

from typing import Optional

import torch

from shared.constants import EmbeddingSourceMode


def uses_structure_embedding(mode: EmbeddingSourceMode) -> bool:
    return mode in (EmbeddingSourceMode.STRUCTURE_ONLY, EmbeddingSourceMode.STRUCTURE_AND_SEQUENCE)


def uses_sequence_embedding(mode: EmbeddingSourceMode) -> bool:
    return mode in (EmbeddingSourceMode.SEQUENCE_ONLY, EmbeddingSourceMode.STRUCTURE_AND_SEQUENCE)


def require_present(tensor: Optional[torch.Tensor], field_name: str, mode: EmbeddingSourceMode) -> torch.Tensor:
    if tensor is None:
        raise ValueError(f"EmbeddingSourceMode.{mode.name} requires '{field_name}' but it is None")
    return tensor
