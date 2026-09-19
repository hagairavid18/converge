"""Synthetic `DDGLabels` generator, paired with `dl.models.synthetic` for the
overfitting sanity check and the synthetic data fallback.

`shared.constants.ACTIVE_LOSS_ITERATION` is `ITERATION_1_BOUNDED_ONLY`, so
every synthetic sample is labeled `BOUNDED_ID` and `bound_kcal_mol`/
`direction` are left `None` -- `dl.losses.ddg_loss.DDGLoss` never needs them
while the hinge term is inactive. `target_ddg` is generated independently of
`target_bin` (not derived from it) since the two heads are never trained
together -- whichever loss/metric is inactive for a given run just ignores
its own target field. `complex_seen_in_train` has no synthetic equivalent (no
real complexes involved), so it is always `False`. `mutation_side_id`
likewise has no synthetic equivalent (no real mutations/chain roles
involved), so it is always `MUTATION_SIDE_ANTIBODY_ONLY`, the majority class
in real data.
"""

from __future__ import annotations

import torch

from dl.datasets.schemas import DDGLabels
from dl.utils.label_codes import BOUNDED_ID, MUTATION_SIDE_ANTIBODY_ONLY
from shared.constants import DDG_BIN_EDGES_KCAL_MOL, RANDOM_SEED


def make_synthetic_labels(batch_size: int, num_bins: int, seed: int = RANDOM_SEED) -> DDGLabels:
    generator = torch.Generator().manual_seed(seed)
    low, high = DDG_BIN_EDGES_KCAL_MOL[0], DDG_BIN_EDGES_KCAL_MOL[-1]
    return DDGLabels(
        label_type_id=torch.full((batch_size,), BOUNDED_ID, dtype=torch.long),
        target_bin=torch.randint(0, num_bins, (batch_size,), generator=generator),
        target_ddg=torch.empty(batch_size).uniform_(low, high, generator=generator),
        complex_seen_in_train=torch.zeros(batch_size, dtype=torch.bool),
        mutation_side_id=torch.full((batch_size,), MUTATION_SIDE_ANTIBODY_ONLY, dtype=torch.long),
    )
