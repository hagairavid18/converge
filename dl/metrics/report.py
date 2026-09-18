"""Full per-split evaluation report (Implementation Spec sec. 5): bounded-
head accuracy/confusion, hinge-head correct-side accuracy, and the two
additional analyses. Every entry point here takes tensors for exactly one
split's evaluation set and returns one `MetricsReport` for it -- there is no
function anywhere in this module that accepts or produces a pooled-across-
splits result, since the spec requires splits never be pooled.

Runs the underlying `torchmetrics.Metric`s as one-shot: instantiate, a
single `update()` over the whole split, `compute()`. A Lightning training
loop instead keeps its own long-lived instances (one set per split) and
calls `update()` per step / `compute()` at epoch end -- see
`dl.metrics.classification_metrics` and `dl.metrics.hinge_metrics`.
"""

from __future__ import annotations

import torch
from pydantic import BaseModel, ConfigDict

from shared.constants import SplitName

from dl.metrics.analyses import alanine_scanning_vs_other, ddg_by_interface_region
from dl.metrics.classification_metrics import (
    BoundedBinAccuracy,
    BoundedBinConfusionMatrix,
    logits_to_pred_bin,
)
from dl.metrics.hinge_metrics import IneqCorrectSideAccuracy, NBCorrectSideAccuracy
from dl.utils.bin_utils import compute_bin_centers, expected_ddg_from_logits
from dl.utils.label_codes import BOUNDED_ID, INEQ_ID, NB_ID


class DDGEvalBatch(BaseModel):
    """One split's worth of model outputs + labels + metadata, as tensors.
    Never mix samples from two different `SplitName` values into one batch.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    logits: torch.Tensor
    label_type_id: torch.Tensor
    bin_target: torch.Tensor
    ddg_kcal_mol: torch.Tensor
    bound_kcal_mol: torch.Tensor
    direction: torch.Tensor
    interface_region_id: torch.Tensor
    is_alanine_scanning: torch.Tensor


class MetricsReport(BaseModel):
    split_name: SplitName
    n_bounded: int
    n_ineq: int
    n_nb: int
    bin_accuracy: float
    confusion_summary: dict
    hinge_accuracies: dict
    ddg_by_interface_region_predicted: dict
    ddg_by_interface_region_true: dict
    alanine_scanning_vs_other_predicted: dict
    alanine_scanning_vs_other_true: dict


def bounded_classification_report(
    logits: torch.Tensor, bin_target: torch.Tensor, bounded_mask: torch.Tensor, num_bins: int
) -> tuple[float, dict]:
    pred_bin = logits_to_pred_bin(logits[bounded_mask])
    true_bin = bin_target[bounded_mask]

    accuracy_metric = BoundedBinAccuracy().to(logits.device)
    accuracy_metric.update(pred_bin, true_bin)

    confusion_metric = BoundedBinConfusionMatrix(num_bins=num_bins).to(logits.device)
    confusion_metric.update(pred_bin, true_bin)

    return accuracy_metric.compute().item(), confusion_metric.compute()


def hinge_report(
    predicted_ddg: torch.Tensor,
    label_type_id: torch.Tensor,
    bound_kcal_mol: torch.Tensor,
    direction: torch.Tensor,
    ineq_default_bound_kcal_mol: float,
    nb_anchor_kcal_mol: float,
) -> dict:
    ineq_metric = IneqCorrectSideAccuracy(ineq_default_bound_kcal_mol, nb_anchor_kcal_mol).to(predicted_ddg.device)
    nb_metric = NBCorrectSideAccuracy(ineq_default_bound_kcal_mol, nb_anchor_kcal_mol).to(predicted_ddg.device)
    ineq_metric.update(predicted_ddg, label_type_id, bound_kcal_mol, direction)
    nb_metric.update(predicted_ddg, label_type_id, bound_kcal_mol, direction)
    return {
        "ineq_correct_side_accuracy": ineq_metric.compute().item(),
        "ineq_n": int((label_type_id == INEQ_ID).sum().item()),
        "nb_correct_side_accuracy": nb_metric.compute().item(),
        "nb_n": int((label_type_id == NB_ID).sum().item()),
    }


def compute_full_report(
    batch: DDGEvalBatch,
    split_name: SplitName,
    ineq_default_bound_kcal_mol: float,
    nb_anchor_kcal_mol: float,
    bin_edges_kcal_mol: list[float],
) -> MetricsReport:
    num_bins = len(bin_edges_kcal_mol) - 1
    bounded_mask = batch.label_type_id == BOUNDED_ID

    bin_acc, confusion_summary = bounded_classification_report(
        batch.logits, batch.bin_target, bounded_mask, num_bins
    )

    bin_centers = compute_bin_centers(bin_edges_kcal_mol)
    predicted_ddg = expected_ddg_from_logits(batch.logits, bin_centers)
    hinge_accuracies = hinge_report(
        predicted_ddg,
        batch.label_type_id,
        batch.bound_kcal_mol,
        batch.direction,
        ineq_default_bound_kcal_mol,
        nb_anchor_kcal_mol,
    )

    return MetricsReport(
        split_name=split_name,
        n_bounded=int(bounded_mask.sum().item()),
        n_ineq=int((batch.label_type_id == INEQ_ID).sum().item()),
        n_nb=int((batch.label_type_id == NB_ID).sum().item()),
        bin_accuracy=bin_acc,
        confusion_summary=confusion_summary,
        hinge_accuracies=hinge_accuracies,
        ddg_by_interface_region_predicted=ddg_by_interface_region(batch.interface_region_id, predicted_ddg),
        ddg_by_interface_region_true=ddg_by_interface_region(batch.interface_region_id, batch.ddg_kcal_mol),
        alanine_scanning_vs_other_predicted=alanine_scanning_vs_other(batch.is_alanine_scanning, predicted_ddg),
        alanine_scanning_vs_other_true=alanine_scanning_vs_other(batch.is_alanine_scanning, batch.ddg_kcal_mol),
    )


def compute_reports_for_all_splits(
    batches_by_split: dict[SplitName, DDGEvalBatch],
    ineq_default_bound_kcal_mol: float,
    nb_anchor_kcal_mol: float,
    bin_edges_kcal_mol: list[float],
) -> dict[SplitName, MetricsReport]:
    """Convenience wrapper that computes one independent `MetricsReport` per
    split. Each split's batch is processed in isolation -- nothing here ever
    concatenates tensors across splits.
    """
    return {
        split_name: compute_full_report(
            batch, split_name, ineq_default_bound_kcal_mol, nb_anchor_kcal_mol, bin_edges_kcal_mol
        )
        for split_name, batch in batches_by_split.items()
    }
