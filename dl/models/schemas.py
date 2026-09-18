"""Input contract between pre-processing and the model (Implementation Spec §3).

Every per-residue tensor below is produced per mutation sample, batched and
padded to a common per-sample residue length `L` (whatever chains
pre-processing concatenates into one sequence per sample -- heavy/light/
antigen -- share the same `L` within a batch). `B` is the batch size.

Field-by-field contract:

- `wt_structure_embedding`, `mut_structure_embedding`: float32
  `[B, L, structure_embed_dim]`. Per-residue structure-backend embedding
  (ESMFold per-residue single representation, see
  `shared.constants.ACTIVE_STRUCTURE_BACKEND`) for the wild-type and mutant
  complex respectively -- the wild type is also run through ESMFold (same
  embedding space), just with no structural confidence reported for it.
  `None` when the active `EmbeddingSourceMode` does not use structure
  embeddings.
- `wt_sequence_embedding`, `mut_sequence_embedding`: float32
  `[B, L, sequence_embed_dim]`. Per-residue sequence-backend embedding
  (ESM-2, see `shared.constants.ACTIVE_SEQUENCE_BACKEND`). `None` when the
  active `EmbeddingSourceMode` does not use sequence embeddings.
- `wt_sequence_confidence`: float32 `[B, L]` in `[0, 1]`. Sequence-model
  per-residue confidence for the wild-type branch. The wild-type branch
  never receives a structural confidence score (it is the real PDB
  structure, not a prediction) -- there is deliberately no
  `wt_structure_confidence` field.
- `mut_structure_confidence`, `mut_sequence_confidence`: float32 `[B, L]` in
  `[0, 1]`. Structure- and sequence-model per-residue confidence for the
  mutant branch.
- `padding_mask`: bool `[B, L]`. `True` marks a real residue, `False` marks
  padding. Not part of a per-sample cache/record -- a single unbatched
  sample has no padding by definition, so this is always constructed at
  collation time from each sample's own residue count (see
  `dl.datasets.collation`), never supplied upstream.
- `mutation_distances`: float32 `[B, M, L]`. Precomputed Euclidean C-alpha
  distance (Angstroms), from each of the sample's point mutations to every
  residue in the concatenated complex -- computed once by pre-processing,
  not by this codebase (`M` = the largest mutation count of any sample in
  the batch; most samples are a single point mutation, but a real
  multi-point mutation has `M > 1`, see `dl.datasets.mutation_csv_dataset`).
  `dl.layers.pooling.MutationWindowPooling` applies the Gaussian falloff to
  these directly and takes the max over `M` per residue.
- `mutation_site_mask`: bool `[B, M]`. `True` marks a real point mutation,
  `False` marks padding out to `M`. Like `padding_mask`, this is always
  constructed at collation time from each sample's own mutation count,
  never supplied upstream.

`ddg_bin` classification targets are not part of this schema -- they travel
alongside a batch as a plain `long [B]` tensor (see `dl.models.synthetic`
and `dl.training.lightning_module`) since they are a label, not a model
input.
"""

from __future__ import annotations

from typing import Optional

import torch
from pydantic import BaseModel, ConfigDict, model_validator

PER_RESIDUE_FIELDS = frozenset(
    {
        "wt_structure_embedding",
        "wt_sequence_embedding",
        "mut_structure_embedding",
        "mut_sequence_embedding",
        "wt_sequence_confidence",
        "mut_structure_confidence",
        "mut_sequence_confidence",
    }
)


class ModelInputs(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    wt_structure_embedding: Optional[torch.Tensor] = None
    wt_sequence_embedding: Optional[torch.Tensor] = None
    mut_structure_embedding: Optional[torch.Tensor] = None
    mut_sequence_embedding: Optional[torch.Tensor] = None

    wt_sequence_confidence: Optional[torch.Tensor] = None
    mut_structure_confidence: Optional[torch.Tensor] = None
    mut_sequence_confidence: Optional[torch.Tensor] = None

    padding_mask: torch.Tensor

    mutation_distances: torch.Tensor
    mutation_site_mask: torch.Tensor

    @model_validator(mode="after")
    def _check_batch_size_consistency(self) -> "ModelInputs":
        batch_sizes = {tensor.shape[0] for tensor in self._present_tensors()}
        if len(batch_sizes) > 1:
            raise ValueError(f"Inconsistent batch size across ModelInputs fields: {batch_sizes}")
        return self

    def _present_tensors(self) -> list[torch.Tensor]:
        return [value for value in self.__dict__.values() if isinstance(value, torch.Tensor)]

    def batch_size(self) -> int:
        return self.padding_mask.shape[0]

    def sequence_length(self) -> int:
        return self.padding_mask.shape[1]

    def to(self, device: torch.device) -> "ModelInputs":
        moved = {
            name: value.to(device) if isinstance(value, torch.Tensor) else value
            for name, value in self.__dict__.items()
        }
        return self.model_copy(update=moved)
