"""Per-residue fusion of the active embedding source(s) (Implementation Spec §3
embedding ablation, driven by `shared.constants.EmbeddingSourceMode`).

Plain concatenation of whichever embedding source(s) are active -- no
learned transform. Per CLAUDE.md's simplicity rule, the classification head
is the model's only learned component.

`normalize_before_fusion` (`ModelConfig`) optionally standardizes
(parameter-free, per-residue z-score) each active source independently
before concatenating them -- only meaningful with more than one source
active, since two differently-scaled embedding spaces (e.g. SaProt's
structure embedding next to ESM-2's sequence embedding) concatenated
unnormalized let whichever has the larger raw scale dominate the fused
vector unfairly.
"""

from __future__ import annotations

from typing import Optional

import torch

from dl.utils.embedding_mode import require_present, uses_sequence_embedding, uses_structure_embedding
from shared.constants import EmbeddingSourceMode


def standardize_last_dim(embedding: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.layer_norm(embedding, embedding.shape[-1:])


def fuse_embeddings(
    structure_embedding: Optional[torch.Tensor],
    sequence_embedding: Optional[torch.Tensor],
    mode: EmbeddingSourceMode,
    normalize_before_fusion: bool = False,
) -> torch.Tensor:
    parts = active_embedding_parts(structure_embedding, sequence_embedding, mode)
    if normalize_before_fusion and len(parts) > 1:
        parts = [standardize_last_dim(part) for part in parts]
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
