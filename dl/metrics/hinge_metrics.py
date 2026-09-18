"""Inequality/hinge head metrics (Implementation Spec sec. 5): correct-side
accuracy, tracked separately for "ineq" and "n.b." entries.

`torchmetrics.Metric` subclasses so a Lightning training loop can accumulate
them across an epoch's batches, one instance per (split, label type) pair.
Each instance's `update()` takes the *full* batch (all label types mixed)
plus resolves the ineq-missing-bound fallback / n.b. anchor internally, then
restricts itself to its own label type -- callers don't need to pre-filter.
"""

from __future__ import annotations

import torch
from torchmetrics import Metric

from dl.losses.config import nb_anchor_kcal_mol_from_noise_floor, outermost_bin_edge_kcal_mol
from dl.utils.bound_resolution import resolve_bounds
from dl.utils.label_codes import INEQ_ID, NB_ID


def is_correct_side(predicted_ddg: torch.Tensor, bound_kcal_mol: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    return direction * (predicted_ddg - bound_kcal_mol) >= 0


class HingeCorrectSideAccuracy(Metric):
    full_state_update = False

    def __init__(
        self,
        label_type_id: int,
        ineq_default_bound_kcal_mol: float = outermost_bin_edge_kcal_mol(),
        nb_anchor_kcal_mol: float = nb_anchor_kcal_mol_from_noise_floor(),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.label_type_id = label_type_id
        self.ineq_default_bound_kcal_mol = ineq_default_bound_kcal_mol
        self.nb_anchor_kcal_mol = nb_anchor_kcal_mol
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self,
        predicted_ddg: torch.Tensor,
        label_type_id: torch.Tensor,
        bound_kcal_mol: torch.Tensor,
        direction: torch.Tensor,
    ) -> None:
        resolved_bound, resolved_direction = resolve_bounds(
            label_type_id, bound_kcal_mol, direction, self.ineq_default_bound_kcal_mol, self.nb_anchor_kcal_mol
        )
        mask = label_type_id == self.label_type_id
        correct = is_correct_side(predicted_ddg, resolved_bound, resolved_direction)[mask]
        self.correct += correct.sum()
        self.total += correct.numel()

    def compute(self) -> torch.Tensor:
        if self.total == 0:
            return torch.tensor(float("nan"), device=self.total.device)
        return self.correct.float() / self.total


class IneqCorrectSideAccuracy(HingeCorrectSideAccuracy):
    def __init__(
        self,
        ineq_default_bound_kcal_mol: float = outermost_bin_edge_kcal_mol(),
        nb_anchor_kcal_mol: float = nb_anchor_kcal_mol_from_noise_floor(),
        **kwargs,
    ):
        super().__init__(INEQ_ID, ineq_default_bound_kcal_mol, nb_anchor_kcal_mol, **kwargs)


class NBCorrectSideAccuracy(HingeCorrectSideAccuracy):
    def __init__(
        self,
        ineq_default_bound_kcal_mol: float = outermost_bin_edge_kcal_mol(),
        nb_anchor_kcal_mol: float = nb_anchor_kcal_mol_from_noise_floor(),
        **kwargs,
    ):
        super().__init__(NB_ID, ineq_default_bound_kcal_mol, nb_anchor_kcal_mol, **kwargs)
