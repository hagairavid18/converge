"""Integer codes for `shared.constants.InterfaceRegion`, mirroring
`dl.utils.label_codes` so region-keyed tensors have one canonical mapping.
"""

from __future__ import annotations

import torch

from shared.constants import InterfaceRegion

from dl.utils.constants import UNKNOWN_REGION_ID

REGION_TO_ID: dict[InterfaceRegion, int] = {region: i for i, region in enumerate(InterfaceRegion)}
ID_TO_REGION: dict[int, InterfaceRegion] = {v: k for k, v in REGION_TO_ID.items()}


def regions_to_tensor(regions: list[InterfaceRegion | None]) -> torch.Tensor:
    return torch.tensor(
        [REGION_TO_ID[r] if r is not None else UNKNOWN_REGION_ID for r in regions], dtype=torch.long
    )
