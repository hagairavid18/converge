"""Integer codes for `shared.constants.LabelType`, shared by `dl/losses` and
`dl/metrics` so tensor-based code never re-derives the mapping.
"""

from __future__ import annotations

import torch

from shared.constants import LabelType

from dl.utils.constants import BOUNDED_LABEL_ID, INEQ_LABEL_ID, NB_LABEL_ID

BOUNDED_ID = BOUNDED_LABEL_ID
INEQ_ID = INEQ_LABEL_ID
NB_ID = NB_LABEL_ID

LABEL_TYPE_TO_ID: dict[LabelType, int] = {
    LabelType.BOUNDED: BOUNDED_ID,
    LabelType.INEQ: INEQ_ID,
    LabelType.NB: NB_ID,
}
ID_TO_LABEL_TYPE: dict[int, LabelType] = {v: k for k, v in LABEL_TYPE_TO_ID.items()}


def label_types_to_tensor(label_types: list[LabelType]) -> torch.Tensor:
    return torch.tensor([LABEL_TYPE_TO_ID[lt] for lt in label_types], dtype=torch.long)
