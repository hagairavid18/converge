"""Per-residue fusion of the active embedding source(s) (Implementation Spec §3
embedding ablation, driven by `shared.constants.EmbeddingSourceMode`).

Plain concatenation of whichever embedding source(s) are active -- no
learned transform. Per CLAUDE.md's simplicity rule, the classification head
is the model's only learned component.
"""

from __future__ import annotations

from typing import Optional

import torch

from dl.utils.embedding_mode import require_present, uses_sequence_embedding, uses_structure_embedding
from shared.constants import EmbeddingSourceMode


def fuse_embeddings(
    structure_embedding: Optional[torch.Tensor],
    sequence_embedding: Optional[torch.Tensor],
    mode: EmbeddingSourceMode,
) -> torch.Tensor:
    parts = active_embedding_parts(structure_embedding, sequence_embedding, mode)
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)


def active_embedding_parts(
    structure_embedding: Optional[torch.Tensor],
    sequence_embedding: Optional[torch.Tensor],
    mode: EmbeddingSourceMode,
) -> list[torch.Tensor]:
    parts = []
    if uses_structure_embedding(mode):
        parts.append(require_present(structure_embedding, "structure_embedding", mode))
    if uses_sequence_embedding(mode):
        parts.append(require_present(sequence_embedding, "sequence_embedding", mode))
    return parts
