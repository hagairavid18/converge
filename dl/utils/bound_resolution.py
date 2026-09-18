"""Fill in the fallback ineq bound and the n.b. anchor/direction override,
shared by `dl.losses.ddg_loss` (training) and `dl.metrics.hinge_metrics`
(evaluation) so the two never compute a sample's effective bound differently.
"""

from __future__ import annotations

import torch

from dl.utils.label_codes import INEQ_ID, NB_ID


def resolve_ineq_bound(
    bound_kcal_mol: torch.Tensor,
    direction: torch.Tensor,
    ineq_mask: torch.Tensor,
    ineq_default_bound_kcal_mol: float,
) -> torch.Tensor:
    ineq_missing = ineq_mask & torch.isnan(bound_kcal_mol)
    default_bound = direction * ineq_default_bound_kcal_mol
    return torch.where(ineq_missing, default_bound, bound_kcal_mol)


def resolve_nb_bound_and_direction(
    bound_kcal_mol: torch.Tensor,
    direction: torch.Tensor,
    nb_mask: torch.Tensor,
    nb_anchor_kcal_mol: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    anchored_bound = torch.where(nb_mask, torch.full_like(bound_kcal_mol, nb_anchor_kcal_mol), bound_kcal_mol)
    forced_direction = torch.where(nb_mask, torch.ones_like(direction), direction)
    return anchored_bound, forced_direction


def resolve_bounds(
    label_type_id: torch.Tensor,
    bound_kcal_mol: torch.Tensor,
    direction: torch.Tensor,
    ineq_default_bound_kcal_mol: float,
    nb_anchor_kcal_mol: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    resolved_bound = bound_kcal_mol.clone().to(torch.float32)
    resolved_direction = direction.clone().to(torch.float32)
    resolved_bound = resolve_ineq_bound(
        resolved_bound, resolved_direction, label_type_id == INEQ_ID, ineq_default_bound_kcal_mol
    )
    resolved_bound, resolved_direction = resolve_nb_bound_and_direction(
        resolved_bound, resolved_direction, label_type_id == NB_ID, nb_anchor_kcal_mol
    )
    return resolved_bound, resolved_direction
