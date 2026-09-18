"""Regression alternative to the bounded-label head metrics (Implementation
Spec sec. 5): MAE, RMSE, and Spearman correlation, reported alongside the
84%-within-1-kcal/mol noise-floor figure as a reference ceiling, same
pattern as `dl.metrics.classification_metrics`.

Thin `torchmetrics.Metric` subclasses wrapping the stock regression metrics
(no need to reimplement MAE/RMSE/Spearman), one instance per split. Callers
restrict `update()` to bounded entries only, same convention as
`BoundedBinAccuracy`/`BoundedBinConfusionMatrix`.
"""

from __future__ import annotations

import torch
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, SpearmanCorrCoef

from shared.constants import NOISE_FLOOR_AGREEMENT_FRACTION

from dl.utils.shape_utils import flatten_last_singleton_dim


class BoundedMAE(MeanAbsoluteError):
    """Mean absolute error for the bounded-ddG regression head."""

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        super().update(flatten_last_singleton_dim(preds), flatten_last_singleton_dim(target))


class BoundedRMSE(MeanSquaredError):
    """Root mean squared error for the bounded-ddG regression head
    (`squared=False` by default, unlike the stock `MeanSquaredError`).
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("squared", False)
        super().__init__(**kwargs)

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        super().update(flatten_last_singleton_dim(preds), flatten_last_singleton_dim(target))


class BoundedSpearmanCorrelation(SpearmanCorrCoef):
    """Spearman rank correlation for the bounded-ddG regression head."""

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        super().update(flatten_last_singleton_dim(preds), flatten_last_singleton_dim(target))


def compute_bounded_regression_report(
    predicted_ddg: torch.Tensor,
    target_ddg: torch.Tensor,
    bounded_mask: torch.Tensor,
    noise_floor_agreement_fraction: float = NOISE_FLOOR_AGREEMENT_FRACTION,
) -> dict:
    bounded_pred = predicted_ddg[bounded_mask]
    bounded_target = target_ddg[bounded_mask]

    mae_metric = BoundedMAE().to(predicted_ddg.device)
    rmse_metric = BoundedRMSE().to(predicted_ddg.device)
    spearman_metric = BoundedSpearmanCorrelation().to(predicted_ddg.device)

    n = int(bounded_pred.numel())
    if n == 0:
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "spearman_correlation": float("nan"),
            "noise_floor_agreement_fraction": noise_floor_agreement_fraction,
            "n": 0,
        }

    mae_metric.update(bounded_pred, bounded_target)
    rmse_metric.update(bounded_pred, bounded_target)
    spearman_metric.update(bounded_pred, bounded_target)

    spearman_value = spearman_metric.compute().item() if n >= 2 else float("nan")

    return {
        "mae": mae_metric.compute().item(),
        "rmse": rmse_metric.compute().item(),
        "spearman_correlation": spearman_value,
        "noise_floor_agreement_fraction": noise_floor_agreement_fraction,
        "n": n,
    }
