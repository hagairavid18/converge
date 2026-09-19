"""Per-batch label bundle produced by the dataset alongside `ModelInputs`
(`dl.models.schemas`) and fed to `dl.losses.ddg_loss.DDGLoss` together with
the classification head's logits (Implementation Spec §4).

- `label_type_id`: long `[B]`, one of `dl.utils.label_codes.{BOUNDED_ID,INEQ_ID,NB_ID}`.
- `target_bin`: long `[B]`, valid (and only used) where `label_type_id ==
  BOUNDED_ID`, for the classification head (`dl.losses.ddg_loss.DDGLoss`).
- `target_ddg`: float `[B]`, valid (and only used) where `label_type_id ==
  BOUNDED_ID`, for the regression head
  (`dl.losses.ddg_regression_loss.DDGRegressionLoss`). The same raw ddG
  value `target_bin` was discretized from.
- `bound_kcal_mol`, `direction`: float `[B]`, required only once the hinge
  term is active (`shared.constants.LossIteration.ITERATION_2_WITH_HINGE`);
  see `dl.losses.hinge_loss` and `dl.utils.bound_resolution` for their
  contract.
- `complex_seen_in_train`: bool `[B]`, whether this sample's `complex_name`
  (the normalized antibody-antigen pair identity, per
  `data.homology_dedup.normalize_protein_name` -- not the literal `pdb_id`,
  since the same complex can be re-crystallized under a different PDB code)
  also appears among the training split's own records (always `False` on
  `held_out_pdb`, by construction; a real train/val mix on
  `same_pdb_allowed`). Used only by `dl.training.lightning_module`'s
  new-complex-vs-seen-complex validation breakout.
- `mutation_side_id`: long `[B]`, one of
  `dl.utils.label_codes.{MUTATION_SIDE_ANTIGEN_ONLY, MUTATION_SIDE_ANTIBODY_ONLY,
  MUTATION_SIDE_BOTH}`, classifying the record's `mutations` by
  `shared.constants.ChainRole`: `MUTATION_SIDE_ANTIGEN_ONLY` if every
  mutation is on the antigen chain, `MUTATION_SIDE_ANTIBODY_ONLY` if every
  mutation is on a heavy or light chain, `MUTATION_SIDE_BOTH` otherwise. Used
  only by `dl.training.lightning_module`'s antigen-only-vs-antibody-only
  validation breakout.
"""

from __future__ import annotations

from typing import Optional

import torch
from pydantic import BaseModel, ConfigDict


class DDGLabels(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    label_type_id: torch.Tensor
    target_bin: torch.Tensor
    complex_seen_in_train: torch.Tensor
    mutation_side_id: torch.Tensor
    target_ddg: Optional[torch.Tensor] = None
    bound_kcal_mol: Optional[torch.Tensor] = None
    direction: Optional[torch.Tensor] = None

    def to(self, device: torch.device) -> "DDGLabels":
        moved = {
            name: value.to(device) if isinstance(value, torch.Tensor) else value
            for name, value in self.__dict__.items()
        }
        return self.model_copy(update=moved)
