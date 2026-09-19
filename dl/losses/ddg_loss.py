"""Combined, label-type-dispatched ddG loss (Implementation Spec sec. 4).

Dispatches each sample in a mixed batch to the bin-distance-weighted
classification term (bounded entries) and/or the hinge term (ineq/n.b.
entries, iteration 2 only), then combines the two batch-averaged terms with
optional inverse-frequency reweighting (`BATCH_IMBALANCE_REWEIGHT`).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from shared.constants import LossIteration

from dl.losses.bounded_classification_loss import BinDistanceWeightedLoss
from dl.losses.combination_utils import (
    combine_bounded_and_hinge_terms,
    count_label_type,
    masked_mean,
)
from dl.losses.config import DDGLossConfig
from dl.losses.hinge_loss import HingeLoss
from dl.losses.tail_weighting import tail_weight
from dl.utils.bin_utils import compute_bin_centers, expected_ddg_from_logits
from dl.utils.bound_resolution import resolve_bounds as resolve_ddg_bounds
from dl.utils.label_codes import BOUNDED_ID, INEQ_ID, NB_ID


@dataclass
class DDGLossOutput:
    """A plain dataclass, not a Pydantic model: `total_loss` must stay a live,
    autograd-attached tensor for `.backward()` to work, whereas Pydantic in
    this package is reserved for validated boundary data (config, metric
    reports) that never carries an in-flight autograd graph.
    """

    total_loss: torch.Tensor
    bounded_loss: torch.Tensor
    hinge_loss: torch.Tensor
    bounded_weight: float
    hinge_weight: float
    n_bounded: int
    n_ineq: int
    n_nb: int


class DDGLoss(nn.Module):
    """`config` may be omitted and/or partially overridden by flat kwargs, so
    `dl.utils.factory.build_object(dl.losses, "DDGLoss", **flat_params)` works
    directly from a plain config dict as well as from a pre-built `DDGLossConfig`.
    """

    def __init__(self, config: DDGLossConfig | None = None, **config_overrides):
        super().__init__()
        self.config = config or DDGLossConfig(**config_overrides)
        self.bounded_loss_fn = BinDistanceWeightedLoss(self.config)
        self.hinge_loss_fn = HingeLoss(self.config)
        self.register_buffer(
            "bin_centers", compute_bin_centers(self.config.bin_edges_kcal_mol), persistent=False
        )

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
        self,
        logits: torch.Tensor,
        target_bin: torch.Tensor,
        bounded_mask: torch.Tensor,
        n_bounded: int,
        target_ddg: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if n_bounded == 0:
            return torch.zeros((), device=logits.device, dtype=logits.dtype)
        safe_target_bin = target_bin.clone()
        safe_target_bin[~bounded_mask] = 0
        per_sample_bounded = self.bounded_loss_fn(logits, safe_target_bin)
        if self.config.tail_reweight_enabled:
            if target_ddg is None:
                raise ValueError("target_ddg is required when tail_reweight_enabled is set")
            safe_target_ddg = torch.where(bounded_mask, target_ddg, torch.zeros_like(target_ddg))
            per_sample_bounded = per_sample_bounded * tail_weight(
                safe_target_ddg,
                self.config.tail_reweight_alpha,
                self.config.tail_reweight_reference_kcal_mol,
                self.config.tail_reweight_cap,
            )
        return masked_mean(per_sample_bounded, bounded_mask, n_bounded)

    def compute_hinge_term(
        self,
        logits: torch.Tensor,
        label_type_id: torch.Tensor,
        bound_kcal_mol: torch.Tensor | None,
        direction: torch.Tensor | None,
        hinge_mask: torch.Tensor,
        n_hinge: int,
    ) -> torch.Tensor:
        if n_hinge == 0:
            return torch.zeros((), device=logits.device, dtype=logits.dtype)
        if bound_kcal_mol is None or direction is None:
            raise ValueError("bound_kcal_mol and direction are required when the hinge term is active")
        resolved_bound, resolved_direction = self.resolve_bounds(label_type_id, bound_kcal_mol, direction)
        predicted_ddg = expected_ddg_from_logits(logits, self.bin_centers)
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
        logits: torch.Tensor,
        label_type_id: torch.Tensor,
        target_bin: torch.Tensor,
        bound_kcal_mol: torch.Tensor | None = None,
        direction: torch.Tensor | None = None,
        iteration: LossIteration | None = None,
        target_ddg: torch.Tensor | None = None,
    ) -> DDGLossOutput:
        """
        logits: (B, num_bins) float, the bounded-ddG classification head's raw logits.
        label_type_id: (B,) long, values in {BOUNDED_ID, INEQ_ID, NB_ID}.
        target_bin: (B,) long, valid (and only used) where label_type_id == BOUNDED_ID.
        bound_kcal_mol, direction: (B,) float, required only when the hinge term is
            active; see `dl.losses.hinge_loss` and `resolve_bounds` for their contract.
        iteration: overrides `self.config.iteration` (mainly for tests).
        target_ddg: (B,) float, required only when `self.config.tail_reweight_enabled`
            (the classification head's own target is `target_bin`, not this) --
            see `dl.losses.tail_weighting.tail_weight`.
        """
        iteration = iteration or self.config.iteration

        bounded_mask = label_type_id == BOUNDED_ID
        hinge_mask = (label_type_id == INEQ_ID) | (label_type_id == NB_ID)
        n_bounded = count_label_type(label_type_id, BOUNDED_ID)
        n_ineq = count_label_type(label_type_id, INEQ_ID)
        n_nb = count_label_type(label_type_id, NB_ID)
        n_hinge = n_ineq + n_nb

        hinge_active = iteration == LossIteration.ITERATION_2_WITH_HINGE and n_hinge > 0

        bounded_loss = self.compute_bounded_term(logits, target_bin, bounded_mask, n_bounded, target_ddg=target_ddg)
        hinge_loss = (
            self.compute_hinge_term(logits, label_type_id, bound_kcal_mol, direction, hinge_mask, n_hinge)
            if hinge_active
            else torch.zeros((), device=logits.device, dtype=logits.dtype)
        )
        total_loss, bounded_weight, hinge_weight = self.combine_terms(
            bounded_loss, hinge_loss, n_bounded, n_hinge, hinge_active
        )

        return DDGLossOutput(
            total_loss=total_loss,
            bounded_loss=bounded_loss.detach(),
            hinge_loss=hinge_loss.detach(),
            bounded_weight=bounded_weight,
            hinge_weight=hinge_weight,
            n_bounded=n_bounded,
            n_ineq=n_ineq,
            n_nb=n_nb,
        )
