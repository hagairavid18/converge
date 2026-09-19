"""Constants internal to the modeling codebase only (models/losses/metrics/training).

Cross-workstream values (paths, schemas, split names, label types) live in
`shared/constants.py` instead. Promote a constant here to `shared/constants.py`
only once the data pipeline needs the same value.
"""

from __future__ import annotations

from shared.constants import (
    ACTIVE_ESM2_HIDDEN_DIM,
    ACTIVE_SAPROT_HIDDEN_DIM,
    DDG_BIN_EDGES_KCAL_MOL,
    DDG_NUM_BINS,
    NOISE_FLOOR_KCAL_MOL,
    USE_SAPROT_STRUCTURE,
)

BOUNDED_LABEL_ID = 0
INEQ_LABEL_ID = 1
NB_LABEL_ID = 2

# Mutation-side breakout (dl.training.lightning_module): every mutation in
# the record is on the antigen chain, every mutation is on an antibody chain
# (heavy or light), or a mix of both.
MUTATION_SIDE_ANTIGEN_ONLY_ID = 0
MUTATION_SIDE_ANTIBODY_ONLY_ID = 1
MUTATION_SIDE_BOTH_ID = 2

UNKNOWN_REGION_ID = -1

# 1024 is IgFold's shared zero-padded common width (structure_backend.py's
# STRUCTURE_EMBEDDING_COMMON_DIM); SaProt's cached tensors are unpadded, at
# its own native checkpoint width -- this tracks USE_SAPROT_STRUCTURE the
# same way DEFAULT_SEQUENCE_EMBED_DIM tracks the active ESM2 checkpoint.
DEFAULT_STRUCTURE_EMBED_DIM = ACTIVE_SAPROT_HIDDEN_DIM if USE_SAPROT_STRUCTURE else 1024
DEFAULT_SEQUENCE_EMBED_DIM = ACTIVE_ESM2_HIDDEN_DIM
DEFAULT_GAUSSIAN_SIGMA_ANGSTROM = 8.0
DEFAULT_POOLING_STRATEGY = "norm_softmax"
# 1.0 = no softening (the original parameter-free softmax). norm_softmax's
# site-vs-elsewhere diff-norm gap can be large enough (confirmed empirically,
# see docs/future_work.md) that softmax(diff_norm) collapses to a near
# one-hot selection of a single residue rather than a real weighted window
# -- dividing by a temperature > 1 before softmax softens that back into an
# actual neighborhood average. A fixed hyperparameter (like
# DEFAULT_GAUSSIAN_SIGMA_ANGSTROM was for the Gaussian pooling it's an
# alternative to), not learned.
DEFAULT_POOLING_TEMPERATURE = 1.0
# Whether to standardize the pooled representation (z-score, per sample,
# parameter-free) before it reaches the prediction head.
DEFAULT_NORMALIZE_POOLED_REPRESENTATION = False
# Whether to standardize each active embedding source independently (z-score,
# per residue, parameter-free) before concatenating them -- only meaningful
# under structure_and_sequence, where two differently-scaled sources are
# fused into one vector.
DEFAULT_NORMALIZE_BEFORE_FUSION = False

# Empty tuple = plain Linear(input_dim, 1), matching the original head. A
# config sets these to grow the regression head into an MLP -- see
# dl/configs/sequence_only.yaml for the first experiment using this.
DEFAULT_REGRESSION_HIDDEN_DIMS: tuple[int, ...] = ()
DEFAULT_REGRESSION_DROPOUT = 0.0

# Switched back to the regression head (2026-09-19): classification stays
# fully implemented and selectable via config, but is no longer the default
# -- see dl.training.lightning_module for the head-dispatch this drives.
DEFAULT_ACTIVE_HEADS = ("regression",)

# Diagnostic-only threshold (dl.training.lightning_module): |ddG| at or
# above this is reported as a separate "large effect" val/train metric
# breakout, alongside the pooled bounded-entry metric -- not used by any
# loss term. Chosen as the empirical ~75th percentile of |ddG| across the
# full bounded dataset (see docs/future_work.md), i.e. roughly the top
# quartile of mutations by effect size.
HIGH_ABS_DDG_THRESHOLD_KCAL_MOL = 2.0

# Validation-only MAE breakout (dl.training.lightning_module, regression head
# only): 5 coarser, named magnitude buckets over the same continuous ddG axis
# the 10 DDG_NUM_BINS classification bins discretize, reusing
# HIGH_ABS_DDG_THRESHOLD_KCAL_MOL and NOISE_FLOOR_KCAL_MOL (the existing bin
# width) as the two cut points, mirrored to their negative counterparts.
MAGNITUDE_BUCKET_NAMES: tuple[str, ...] = (
    "large_destabilizing",
    "moderate_destabilizing",
    "near_zero",
    "moderate_stabilizing",
    "large_stabilizing",
)
MAGNITUDE_BUCKET_BOUNDARIES_KCAL_MOL: tuple[float, float, float, float] = (
    -HIGH_ABS_DDG_THRESHOLD_KCAL_MOL,
    -NOISE_FLOOR_KCAL_MOL,
    NOISE_FLOOR_KCAL_MOL,
    HIGH_ABS_DDG_THRESHOLD_KCAL_MOL,
)


def _magnitude_bucket_name_for_ddg(ddg: float) -> str:
    lower, moderate_lower, moderate_upper, upper = MAGNITUDE_BUCKET_BOUNDARIES_KCAL_MOL
    if ddg <= lower:
        return MAGNITUDE_BUCKET_NAMES[0]
    if ddg <= moderate_lower:
        return MAGNITUDE_BUCKET_NAMES[1]
    if ddg < moderate_upper:
        return MAGNITUDE_BUCKET_NAMES[2]
    if ddg < upper:
        return MAGNITUDE_BUCKET_NAMES[3]
    return MAGNITUDE_BUCKET_NAMES[4]


# Derived once from DDG_BIN_EDGES_KCAL_MOL (fine bin i spans
# [edges[i], edges[i + 1])) and MAGNITUDE_BUCKET_BOUNDARIES_KCAL_MOL above,
# evaluated at each bin's midpoint rather than an edge: two of the five
# bucket boundaries (-2.0, -1.0) coincide exactly with a bin edge, and
# evaluating there would put that single boundary point's bin in a different
# bucket than the (continuous-valued) rest of its span.
BIN_TO_MAGNITUDE_BUCKET: tuple[str, ...] = tuple(
    _magnitude_bucket_name_for_ddg((DDG_BIN_EDGES_KCAL_MOL[i] + DDG_BIN_EDGES_KCAL_MOL[i + 1]) / 2.0)
    for i in range(DDG_NUM_BINS)
)

# Empty tuple = no metadata concatenated onto the pooled representation
# (today's default, byte-identical behavior). A config sets this to grow
# ModelConfig.metadata_feature_dim() and widen the prediction head's input --
# see dl.layers.metadata_features.build_metadata_tensor.
DEFAULT_EXTRA_METADATA_FIELDS: tuple[str, ...] = ()

# Fixed (mean, std) z-score stats per allow-listed metadata field, measured
# from train-split stats -- not learned, matching the parameter-free
# standardization already used elsewhere in this codebase (e.g.
# standardize_last_dim).
METADATA_FIELD_NORMALIZATION_STATS: dict[str, tuple[float, float]] = {"temperature_kelvin": (298.0, 2.5)}
