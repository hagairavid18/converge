"""Regression alternative to the bounded-label head metrics (Implementation
Spec sec. 5): MAE, RMSE, and Spearman correlation, reported alongside the
84%-within-1-kcal/mol noise-floor figure as a reference ceiling, same
pattern as `dl.metrics.classification_metrics`.

Thin `torchmetrics.Metric` subclasses wrapping the stock regression metrics
(no need to reimplement MAE/RMSE/Spearman), one instance per split. Callers
restrict `update()` to bounded entries only, same convention as
`BoundedBinAccuracy`/`BoundedBinConfusionMatrix`.

`BucketedMAE` is a dict-returning `Metric` in the same spirit as
`dl.metrics.classification_metrics.BoundedBinConfusionMatrix`: per-bucket
state accumulated via `index_add_`, `compute()` returning one MAE per
`dl.utils.constants.MAGNITUDE_BUCKET_NAMES` entry (regression-head-only
validation breakout, see `dl.training.lightning_module`).
"""

from __future__ import annotations

import torch
from torchmetrics import Metric
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, SpearmanCorrCoef

from shared.constants import NOISE_FLOOR_AGREEMENT_FRACTION

from dl.utils.constants import MAGNITUDE_BUCKET_BOUNDARIES_KCAL_MOL, MAGNITUDE_BUCKET_NAMES
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


def compute_magnitude_bucket_indices(target_ddg: torch.Tensor) -> torch.Tensor:
    lower, moderate_lower, moderate_upper, upper = MAGNITUDE_BUCKET_BOUNDARIES_KCAL_MOL
    indices = torch.full_like(target_ddg, fill_value=2, dtype=torch.long)
    indices[target_ddg >= moderate_upper] = 3
    indices[target_ddg >= upper] = 4
    indices[target_ddg <= moderate_lower] = 1
    indices[target_ddg <= lower] = 0
    return indices


class BucketedMAE(Metric):
    """MAE broken out by the 5 named magnitude buckets in
    `dl.utils.constants.MAGNITUDE_BUCKET_NAMES`, bucketed from the raw
    continuous `target` (not the 10 `DDG_NUM_BINS` classification bins), so
    this works identically under the regression head, where no classification
    bin is predicted at all.
    """

    full_state_update = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        num_buckets = len(MAGNITUDE_BUCKET_NAMES)
        self.add_state("abs_error_sum", default=torch.zeros(num_buckets), dist_reduce_fx="sum")
        self.add_state("count", default=torch.zeros(num_buckets), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        preds = flatten_last_singleton_dim(preds)
        target = flatten_last_singleton_dim(target)
        bucket_indices = compute_magnitude_bucket_indices(target)
        abs_error = (preds - target).abs()
        self.abs_error_sum.index_add_(0, bucket_indices, abs_error)
        self.count.index_add_(0, bucket_indices, torch.ones_like(abs_error))

    def compute(self) -> dict[str, float]:
        return {
            name: (self.abs_error_sum[i] / self.count[i]).item() if self.count[i] > 0 else float("nan")
            for i, name in enumerate(MAGNITUDE_BUCKET_NAMES)
        }


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
