"""Per-batch label bundle produced by the dataset alongside `ModelInputs`
(`dl.models.schemas`) and fed to `dl.losses.ddg_loss.DDGLoss` together with
the classification head's logits (Implementation Spec §4).

- `label_type_id`: long `[B]`, one of `dl.utils.label_codes.{BOUNDED_ID,INEQ_ID,NB_ID}`.
- `target_bin`: long `[B]`, valid (and only used) where `label_type_id == BOUNDED_ID`.
- `bound_kcal_mol`, `direction`: float `[B]`, required only once the hinge
  term is active (`shared.constants.LossIteration.ITERATION_2_WITH_HINGE`);
  see `dl.losses.hinge_loss` and `dl.utils.bound_resolution` for their
  contract.
"""

from __future__ import annotations

from typing import Optional

import torch
from pydantic import BaseModel, ConfigDict


class DDGLabels(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    label_type_id: torch.Tensor
    target_bin: torch.Tensor
    bound_kcal_mol: Optional[torch.Tensor] = None
    direction: Optional[torch.Tensor] = None

    def to(self, device: torch.device) -> "DDGLabels":
        moved = {
            name: value.to(device) if isinstance(value, torch.Tensor) else value
            for name, value in self.__dict__.items()
        }
        return self.model_copy(update=moved)
