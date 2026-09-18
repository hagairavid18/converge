"""Regression counterpart to `dl.losses.ddg_loss.DDGLoss` (Implementation
Spec sec. 4): the bounded term uses `BoundedRegressionLoss` (a dead-zone
regression loss) instead of the bin-distance-weighted classifier, operating
directly on a single continuous ddG prediction per sample rather than
classification logits. The ineq/n.b. hinge term is unchanged -- it already
needs a continuous ddG estimate, which the regression head provides directly
(no need to derive one from classification logits, unlike `DDGLoss`).

Config-selectable alongside `DDGLoss` via `dl.utils.factory.build_object`:
`build_object(dl.losses, "DDGLoss", ...)` for the classification target,
`build_object(dl.losses, "DDGRegressionLoss", ...)` for this one. Both share
`DDGLossConfig` and the same label-type dispatch/combination logic
(`dl.losses.combination_utils`).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from shared.constants import LossIteration

from dl.losses.bounded_regression_loss import BoundedRegressionLoss
from dl.losses.combination_utils import (
    combine_bounded_and_hinge_terms,
    count_label_type,
    masked_mean,
)
from dl.losses.config import DDGLossConfig
from dl.losses.hinge_loss import HingeLoss
from dl.utils.bound_resolution import resolve_bounds as resolve_ddg_bounds
from dl.utils.label_codes import BOUNDED_ID, INEQ_ID, NB_ID
from dl.utils.shape_utils import flatten_last_singleton_dim


@dataclass
class DDGRegressionLossOutput:
    total_loss: torch.Tensor
    bounded_loss: torch.Tensor
    hinge_loss: torch.Tensor
    bounded_weight: float
    hinge_weight: float
    n_bounded: int
    n_ineq: int
    n_nb: int


class DDGRegressionLoss(nn.Module):
    """`config` may be omitted and/or partially overridden by flat kwargs,
    mirroring `dl.losses.ddg_loss.DDGLoss`.
    """

    def __init__(self, config: DDGLossConfig | None = None, **config_overrides):
        super().__init__()
        self.config = config or DDGLossConfig(**config_overrides)
        self.bounded_loss_fn = BoundedRegressionLoss(self.config)
        self.hinge_loss_fn = HingeLoss(self.config)

    def resolve_bounds(
        self,
        label_type_id: torch.Tensor,
        bound_kcal_mol: torch.Tensor,
        direction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return resolve_ddg_bounds(
            label_type_id,
            bound_kcal_mol,
            direction,
            self.config.ineq_default_bound_kcal_mol,
            self.config.nb_anchor_kcal_mol,
        )

    def compute_bounded_term(
        self, predicted_ddg: torch.Tensor, target_ddg: torch.Tensor, bounded_mask: torch.Tensor, n_bounded: int
    ) -> torch.Tensor:
        if n_bounded == 0:
            return torch.zeros((), device=predicted_ddg.device, dtype=predicted_ddg.dtype)
        safe_target_ddg = torch.where(bounded_mask, target_ddg, torch.zeros_like(target_ddg))
        per_sample_bounded = self.bounded_loss_fn(predicted_ddg, safe_target_ddg)
        return masked_mean(per_sample_bounded, bounded_mask, n_bounded)

    def compute_hinge_term(
        self,
        predicted_ddg: torch.Tensor,
        label_type_id: torch.Tensor,
        bound_kcal_mol: torch.Tensor | None,
        direction: torch.Tensor | None,
        hinge_mask: torch.Tensor,
        n_hinge: int,
    ) -> torch.Tensor:
        if n_hinge == 0:
            return torch.zeros((), device=predicted_ddg.device, dtype=predicted_ddg.dtype)
        if bound_kcal_mol is None or direction is None:
            raise ValueError("bound_kcal_mol and direction are required when the hinge term is active")
        resolved_bound, resolved_direction = self.resolve_bounds(label_type_id, bound_kcal_mol, direction)
        per_sample_hinge = self.hinge_loss_fn(predicted_ddg, resolved_bound, resolved_direction)
        return masked_mean(per_sample_hinge, hinge_mask, n_hinge)

    def combine_terms(
        self, bounded_loss: torch.Tensor, hinge_loss: torch.Tensor, n_bounded: int, n_hinge: int, hinge_active: bool
    ) -> tuple[torch.Tensor, float, float]:
        return combine_bounded_and_hinge_terms(
            bounded_loss, hinge_loss, n_bounded, n_hinge, hinge_active, self.config.apply_batch_imbalance_reweight
        )

    def forward(
        self,
        predicted_ddg: torch.Tensor,
        label_type_id: torch.Tensor,
        target_ddg: torch.Tensor,
        bound_kcal_mol: torch.Tensor | None = None,
        direction: torch.Tensor | None = None,
        iteration: LossIteration | None = None,
    ) -> DDGRegressionLossOutput:
        """
        predicted_ddg: (B,) or (B, 1) float, the regression head's single
            scalar ddG prediction per sample.
        label_type_id: (B,) long, values in {BOUNDED_ID, INEQ_ID, NB_ID}.
        target_ddg: (B,) float, the true ddG value; valid (and only used)
            where label_type_id == BOUNDED_ID.
        bound_kcal_mol, direction: (B,) float, required only when the hinge
            term is active; see `dl.losses.hinge_loss` and `resolve_bounds`.
        iteration: overrides `self.config.iteration` (mainly for tests).
        """
        predicted_ddg = flatten_last_singleton_dim(predicted_ddg)
        iteration = iteration or self.config.iteration

        bounded_mask = label_type_id == BOUNDED_ID
        hinge_mask = (label_type_id == INEQ_ID) | (label_type_id == NB_ID)
        n_bounded = count_label_type(label_type_id, BOUNDED_ID)
        n_ineq = count_label_type(label_type_id, INEQ_ID)
        n_nb = count_label_type(label_type_id, NB_ID)
        n_hinge = n_ineq + n_nb

        hinge_active = iteration == LossIteration.ITERATION_2_WITH_HINGE and n_hinge > 0

        bounded_loss = self.compute_bounded_term(predicted_ddg, target_ddg, bounded_mask, n_bounded)
        hinge_loss = (
            self.compute_hinge_term(predicted_ddg, label_type_id, bound_kcal_mol, direction, hinge_mask, n_hinge)
            if hinge_active
            else torch.zeros((), device=predicted_ddg.device, dtype=predicted_ddg.dtype)
        )
        total_loss, bounded_weight, hinge_weight = self.combine_terms(
            bounded_loss, hinge_loss, n_bounded, n_hinge, hinge_active
        )

        return DDGRegressionLossOutput(
            total_loss=total_loss,
            bounded_loss=bounded_loss.detach(),
            hinge_loss=hinge_loss.detach(),
            bounded_weight=bounded_weight,
            hinge_weight=hinge_weight,
            n_bounded=n_bounded,
            n_ineq=n_ineq,
            n_nb=n_nb,
        )
