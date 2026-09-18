"""Branch confidence selection (Implementation Spec §3, weighting mechanism 1).

The mutant branch is weighted by structural confidence times sequence
confidence; the wild-type branch is weighted by sequence confidence only,
and only when sequence embeddings are part of the active
`EmbeddingSourceMode` -- it never receives a structural confidence score.
See the module docstring in `dl.models.ddg_model` for the open design
question this raises under `EmbeddingSourceMode.STRUCTURE_ONLY`.
"""

from __future__ import annotations

from typing import Optional

import torch

from dl.models.schemas import ModelInputs
from dl.utils.embedding_mode import uses_sequence_embedding, uses_structure_embedding
from shared.constants import EmbeddingSourceMode


def combine_confidences(*confidences: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    present = [confidence for confidence in confidences if confidence is not None]
    if not present:
        return None
    product = present[0]
    for confidence in present[1:]:
        product = product * confidence
    return product


def mutant_branch_confidence(inputs: ModelInputs, mode: EmbeddingSourceMode) -> Optional[torch.Tensor]:
    structural = inputs.mut_structure_confidence if uses_structure_embedding(mode) else None
    sequence = inputs.mut_sequence_confidence if uses_sequence_embedding(mode) else None
    return combine_confidences(structural, sequence)


def wildtype_branch_confidence(inputs: ModelInputs, mode: EmbeddingSourceMode) -> Optional[torch.Tensor]:
    return inputs.wt_sequence_confidence if uses_sequence_embedding(mode) else None
