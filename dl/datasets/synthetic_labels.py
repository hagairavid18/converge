"""Synthetic `DDGLabels` generator, paired with `dl.models.synthetic` for the
overfitting sanity check and the synthetic data fallback.

`shared.constants.ACTIVE_LOSS_ITERATION` is `ITERATION_1_BOUNDED_ONLY`, so
every synthetic sample is labeled `BOUNDED_ID` and `bound_kcal_mol`/
`direction` are left `None` -- `dl.losses.ddg_loss.DDGLoss` never needs them
while the hinge term is inactive.
"""

from __future__ import annotations

import torch

from dl.datasets.schemas import DDGLabels
from dl.utils.label_codes import BOUNDED_ID
from shared.constants import RANDOM_SEED


def make_synthetic_labels(batch_size: int, num_bins: int, seed: int = RANDOM_SEED) -> DDGLabels:
    generator = torch.Generator().manual_seed(seed)
    return DDGLabels(
        label_type_id=torch.full((batch_size,), BOUNDED_ID, dtype=torch.long),
        target_bin=torch.randint(0, num_bins, (batch_size,), generator=generator),
    )
