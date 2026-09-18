"""Batch-combination helpers shared by `dl.losses.ddg_loss.DDGLoss`
(classification bounded term) and `dl.losses.ddg_regression_loss.DDGRegressionLoss`
(regression bounded term) -- both dispatch the same way across label types
and combine the bounded/hinge terms identically.
"""

from __future__ import annotations

import torch


def count_label_type(label_type_id: torch.Tensor, label_id: int) -> int:
    return int((label_type_id == label_id).sum().item())


def masked_mean(per_sample: torch.Tensor, mask: torch.Tensor, count: int) -> torch.Tensor:
    return (per_sample * mask.to(per_sample.dtype)).sum() / count


def inverse_frequency_weights(n_a: int, n_b: int) -> tuple[float, float]:
    total = n_a + n_b
    return total / (2.0 * n_a), total / (2.0 * n_b)


def combine_bounded_and_hinge_terms(
    bounded_loss: torch.Tensor,
    hinge_loss: torch.Tensor,
    n_bounded: int,
    n_hinge: int,
    hinge_active: bool,
    apply_batch_imbalance_reweight: bool,
) -> tuple[torch.Tensor, float, float]:
    bounded_weight, hinge_weight = 1.0, 1.0
    if hinge_active and n_bounded > 0 and n_hinge > 0 and apply_batch_imbalance_reweight:
        bounded_weight, hinge_weight = inverse_frequency_weights(n_bounded, n_hinge)
    total_loss = bounded_weight * bounded_loss + hinge_weight * hinge_loss
    return total_loss, bounded_weight, hinge_weight
