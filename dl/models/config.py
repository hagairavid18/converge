"""Architecture configuration (Implementation Spec §3).

`structure_embed_dim` defaults to 1024 (ESMFold's per-residue single
representation width -- fixed regardless of checkpoint choice).
`sequence_embed_dim` defaults to
`shared.constants.ACTIVE_ESM2_HIDDEN_DIM`, which tracks whichever ESM2
checkpoint `SKEMPI_USE_SMALL_CHECKPOINTS` actually activated (480 for the
CPU-dev esm2_t12_35M checkpoint, 1280 for the full esm2_t33_650M checkpoint)
-- so the two always agree with what pre-processing actually wrote under
`EMBEDDING_CACHE_DIR`, without a manual update here.

Kept intentionally minimal (CLAUDE.md's simplicity rule): every field here
maps directly onto a piece the Implementation Spec calls for, with nothing
extra bolted on. There is no learned per-residue projection (see
`dl.layers.embedding_fusion`), so the prediction head(s) are the model's
only learned component, and `fused_embedding_dim()` -- not a separate
`hidden_dim` knob -- is what sizes their input.

`regression_hidden_dims`/`regression_dropout` default to an empty tuple /
0.0, i.e. a plain `Linear(input_dim, 1)` regression head (see
`dl.layers.heads.RegressionHead`); a config sets non-empty
`regression_hidden_dims` to grow it into a small MLP instead, deliberately
per-experiment rather than a single global default, since the right head
size depends on `fused_embedding_dim()` (e.g. much smaller under
`sequence_only` than with structure embeddings included too).
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from dl.utils.constants import (
    DEFAULT_ACTIVE_HEADS,
    DEFAULT_EXTRA_METADATA_FIELDS,
    DEFAULT_GAUSSIAN_SIGMA_ANGSTROM,
    DEFAULT_NORMALIZE_BEFORE_FUSION,
    DEFAULT_NORMALIZE_POOLED_REPRESENTATION,
    DEFAULT_POOLING_STRATEGY,
    DEFAULT_POOLING_TEMPERATURE,
    DEFAULT_REGRESSION_DROPOUT,
    DEFAULT_REGRESSION_HIDDEN_DIMS,
    DEFAULT_SEQUENCE_EMBED_DIM,
    DEFAULT_STRUCTURE_EMBED_DIM,
    METADATA_FIELD_NORMALIZATION_STATS,
)
from dl.utils.embedding_mode import uses_sequence_embedding, uses_structure_embedding
from shared.constants import ACTIVE_EMBEDDING_SOURCE_MODE, DDG_NUM_BINS, EmbeddingSourceMode


class ModelConfig(BaseModel):
    structure_embed_dim: int = DEFAULT_STRUCTURE_EMBED_DIM
    sequence_embed_dim: int = DEFAULT_SEQUENCE_EMBED_DIM
    num_ddg_bins: int = DDG_NUM_BINS
    embedding_source_mode: EmbeddingSourceMode = ACTIVE_EMBEDDING_SOURCE_MODE
    active_heads: tuple[str, ...] = Field(default=DEFAULT_ACTIVE_HEADS)
    regression_hidden_dims: tuple[int, ...] = Field(default=DEFAULT_REGRESSION_HIDDEN_DIMS)
    regression_dropout: float = DEFAULT_REGRESSION_DROPOUT
    pooling_strategy: str = DEFAULT_POOLING_STRATEGY
    gaussian_sigma_angstrom: float = DEFAULT_GAUSSIAN_SIGMA_ANGSTROM
    pooling_temperature: float = DEFAULT_POOLING_TEMPERATURE
    normalize_pooled_representation: bool = DEFAULT_NORMALIZE_POOLED_REPRESENTATION
    normalize_before_fusion: bool = DEFAULT_NORMALIZE_BEFORE_FUSION
    extra_metadata_fields: tuple[str, ...] = Field(default=DEFAULT_EXTRA_METADATA_FIELDS)

    @field_validator("extra_metadata_fields")
    @classmethod
    def _check_extra_metadata_fields_allowlisted(cls, fields: tuple[str, ...]) -> tuple[str, ...]:
        allowed = METADATA_FIELD_NORMALIZATION_STATS.keys()
        unknown = [field for field in fields if field not in allowed]
        if unknown:
            raise ValueError(f"extra_metadata_fields contains unsupported field(s) {unknown} - allowed: {sorted(allowed)}")
        return fields

    def fused_embedding_dim(self) -> int:
        dim = 0
        if uses_structure_embedding(self.embedding_source_mode):
            dim += self.structure_embed_dim
        if uses_sequence_embedding(self.embedding_source_mode):
            dim += self.sequence_embed_dim
        return dim

    def metadata_feature_dim(self) -> int:
        return len(self.extra_metadata_fields)
