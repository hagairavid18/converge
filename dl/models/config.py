"""Architecture configuration (Implementation Spec §3).

`structure_embed_dim` defaults to 1024 (ESMFold's per-residue single
representation width) and `sequence_embed_dim` defaults to 480 (the hidden
size of `ACTIVE_ESM2_CHECKPOINT`, esm2_t12_35M, in shared/constants.py).
Both are placeholders: confirm them against the real per-residue embedding
tensors pre-processing writes under `EMBEDDING_CACHE_DIR` before real
training and adjust here -- no other code changes are needed.

Kept intentionally minimal (CLAUDE.md's simplicity rule): every field here
maps directly onto a piece the Implementation Spec calls for, with nothing
extra bolted on. There is no learned per-residue projection (see
`dl.layers.embedding_fusion`), so the classification head is the model's
only learned component, and `fused_embedding_dim()` -- not a separate
`hidden_dim` knob -- is what sizes it.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from dl.utils.constants import DEFAULT_ACTIVE_HEADS, DEFAULT_GAUSSIAN_SIGMA_ANGSTROM, DEFAULT_SEQUENCE_EMBED_DIM, DEFAULT_STRUCTURE_EMBED_DIM
from dl.utils.embedding_mode import uses_sequence_embedding, uses_structure_embedding
from shared.constants import ACTIVE_EMBEDDING_SOURCE_MODE, DDG_NUM_BINS, EmbeddingSourceMode


class ModelConfig(BaseModel):
    structure_embed_dim: int = DEFAULT_STRUCTURE_EMBED_DIM
    sequence_embed_dim: int = DEFAULT_SEQUENCE_EMBED_DIM
    gaussian_sigma_angstrom: float = DEFAULT_GAUSSIAN_SIGMA_ANGSTROM
    num_ddg_bins: int = DDG_NUM_BINS
    embedding_source_mode: EmbeddingSourceMode = ACTIVE_EMBEDDING_SOURCE_MODE
    active_heads: tuple[str, ...] = Field(default=DEFAULT_ACTIVE_HEADS)

    def fused_embedding_dim(self) -> int:
        dim = 0
        if uses_structure_embedding(self.embedding_source_mode):
            dim += self.structure_embed_dim
        if uses_sequence_embedding(self.embedding_source_mode):
            dim += self.sequence_embed_dim
        return dim
