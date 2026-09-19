"""Pydantic config for the ddG loss (CLAUDE.md: Pydantic for anything
crossing a module boundary).

Numeric defaults derive from `shared/constants.py`'s locked-in values via the
functions below rather than being re-guessed inline. Fields the Implementation
Spec left to "as you specify at implementation" are documented in
`adjacent_bin_cost_from_noise_floor` and `DDGLossConfig`'s field docstrings;
the rationale is also summarized in the final report.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from shared.constants import (
    ACTIVE_LOSS_ITERATION,
    BATCH_IMBALANCE_REWEIGHT,
    DDG_BIN_EDGES_KCAL_MOL,
    DDG_NUM_BINS,
    LossIteration,
    NOISE_FLOOR_AGREEMENT_FRACTION,
    NOISE_FLOOR_KCAL_MOL,
)

from dl.utils.constants import HIGH_ABS_DDG_THRESHOLD_KCAL_MOL


def outermost_bin_edge_kcal_mol() -> float:
    return DDG_BIN_EDGES_KCAL_MOL[-1]


def nb_anchor_kcal_mol_from_noise_floor() -> float:
    """Threshold used for "n.b." (no binding) entries: one noise-floor width
    beyond the outermost classification bin edge, so it is always strictly
    further out than any ordinary ineq bound (see `DDGLossConfig` docstring).
    """
    return outermost_bin_edge_kcal_mol() + NOISE_FLOOR_KCAL_MOL


class DDGLossConfig(BaseModel):
    """Configuration for `dl.losses.ddg_loss.DDGLoss`."""

    model_config = ConfigDict(frozen=True)

    num_bins: int = DDG_NUM_BINS
    bin_edges_kcal_mol: list[float] = Field(default_factory=lambda: list(DDG_BIN_EDGES_KCAL_MOL))

    noise_floor_agreement_fraction: float = NOISE_FLOOR_AGREEMENT_FRACTION
    distant_bin_cost_power: float = 2.0
    """Bin-distance-weighted classification cost (design choice, see
    `dl.losses.bounded_classification_loss.build_bin_distance_cost_matrix`):
    a 1-bin miss costs `1 - noise_floor_agreement_fraction` (~0.16, since 84%
    of independently-remeasured pairs already land within 1 bin purely from
    measurement noise); a k-bin miss (k >= 2) costs that plus
    `(k - 1) ** distant_bin_cost_power`.
    """

    hinge_penalty_growth: Literal["linear"] = "linear"
    hinge_scale: float = 0.25
    """Design choice ("as you specify at implementation"): plain linear
    hinge, zero penalty exactly on the correct side of the bound, penalty
    growing at `hinge_scale` kcal/mol^-1 of distance past the bound on the
    wrong side. `hinge_scale` < 1 keeps this term's typical magnitude
    comparable to the (small, noise-floor-scaled) bounded classification term.
    """

    regression_dead_zone_kcal_mol: float = NOISE_FLOOR_KCAL_MOL
    regression_penalty_growth: Literal["linear", "squared"] = "squared"
    regression_scale: float = 1.0
    """Design choice for the regression alternative to the bounded-ddG head
    (Implementation Spec sec. 4, "as you specify at implementation"): a
    dead-zone loss with zero penalty for predictions within
    `regression_dead_zone_kcal_mol` of the true value (sized to
    `NOISE_FLOOR_KCAL_MOL`, the same empirical noise floor used everywhere
    else), and penalty growing as the squared kcal/mol distance past the
    dead zone beyond that -- squared rather than linear so this term is
    directly aligned with RMSE, the primary regression evaluation metric.
    """

    ineq_default_bound_kcal_mol: float = Field(default_factory=outermost_bin_edge_kcal_mol)
    """Fallback bound for an "ineq" entry whose numeric bound is missing
    (MutationRecord.ddg_kcal_mol is only best-effort for ineq entries): the
    outermost bin edge, signed to match the entry's reported direction.
    """

    nb_anchor_kcal_mol: float = Field(default_factory=nb_anchor_kcal_mol_from_noise_floor)
    """Fixed threshold for "n.b." entries (no numeric bound exists at all):
    always further out than `ineq_default_bound_kcal_mol` and always in the
    destabilizing ("does not bind") direction, reflecting binding being
    effectively abolished.
    """

    iteration: LossIteration = ACTIVE_LOSS_ITERATION
    apply_batch_imbalance_reweight: bool = BATCH_IMBALANCE_REWEIGHT

    tail_reweight_enabled: bool = False
    tail_reweight_alpha: float = 1.0
    tail_reweight_reference_kcal_mol: float = HIGH_ABS_DDG_THRESHOLD_KCAL_MOL
    tail_reweight_cap: float = 2.0
    """Optional, off-by-default per-sample reweighting of the bounded loss
    term (`dl.losses.tail_weighting.tail_weight`), up-weighting large-
    magnitude ddG targets relative to near-zero ones: `weight = 1 +
    tail_reweight_alpha * min(|target_ddg| / tail_reweight_reference_kcal_mol,
    tail_reweight_cap)`, i.e. 1.0 (no-op) at `target_ddg == 0`, saturating at
    `1 + tail_reweight_alpha * tail_reweight_cap` beyond
    `tail_reweight_reference_kcal_mol * tail_reweight_cap`. Defaults to
    exactly today's behavior (disabled). `tail_reweight_reference_kcal_mol`
    reuses `HIGH_ABS_DDG_THRESHOLD_KCAL_MOL`, the same threshold already used
    to define a "large effect" for the val breakout.
    """

    @field_validator("bin_edges_kcal_mol")
    @classmethod
    def _validate_edges_sorted(cls, edges: list[float]) -> list[float]:
        if list(edges) != sorted(edges):
            raise ValueError("bin_edges_kcal_mol must be sorted ascending")
        return edges
