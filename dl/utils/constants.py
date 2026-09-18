"""Constants internal to the modeling codebase only (models/losses/metrics/training).

Cross-workstream values (paths, schemas, split names, label types) live in
`shared/constants.py` instead. Promote a constant here to `shared/constants.py`
only once the data pipeline needs the same value.
"""

from __future__ import annotations

BOUNDED_LABEL_ID = 0
INEQ_LABEL_ID = 1
NB_LABEL_ID = 2

UNKNOWN_REGION_ID = -1

DEFAULT_STRUCTURE_EMBED_DIM = 1024
DEFAULT_SEQUENCE_EMBED_DIM = 480
DEFAULT_GAUSSIAN_SIGMA_ANGSTROM = 8.0
DEFAULT_ACTIVE_HEADS = ("classification",)

OVERFIT_LOSS_PASS_THRESHOLD = 0.1
OVERFIT_EXACT_ACCURACY_INFO_THRESHOLD = 0.5
OVERFIT_WITHIN_ONE_BIN_ACCURACY_PASS_THRESHOLD = 0.95
