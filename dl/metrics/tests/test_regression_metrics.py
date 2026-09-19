import math

import torch

from dl.metrics.regression_metrics import (
    BoundedMAE,
    BoundedRMSE,
    BoundedSpearmanCorrelation,
    BucketedMAE,
    compute_bounded_regression_report,
)
from dl.utils.constants import MAGNITUDE_BUCKET_NAMES


def test_bounded_mae_basic():
    metric = BoundedMAE()
    metric.update(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.5, 2.5, 2.0]))
    assert abs(metric.compute().item() - 0.6666667) < 1e-4


def test_bounded_rmse_is_root_mean_squared_error_not_mse():
    metric = BoundedRMSE()
    metric.update(torch.tensor([0.0, 0.0]), torch.tensor([3.0, 4.0]))
    assert abs(metric.compute().item() - 3.5355339) < 1e-4


def test_bounded_mae_accepts_bx1_shape():
    metric = BoundedMAE()
    metric.update(torch.tensor([[1.0], [2.0]]), torch.tensor([[1.0], [4.0]]))
    assert abs(metric.compute().item() - 1.0) < 1e-6


def test_bounded_spearman_perfect_rank_correlation():
    metric = BoundedSpearmanCorrelation()
    metric.update(torch.tensor([1.0, 2.0, 3.0, 4.0]), torch.tensor([10.0, 20.0, 30.0, 40.0]))
    assert abs(metric.compute().item() - 1.0) < 1e-4


def test_compute_bounded_regression_report_restricts_to_bounded_mask():
    predicted_ddg = torch.tensor([1.0, 2.0, 100.0, 3.0])
    target_ddg = torch.tensor([1.0, 2.0, float("nan"), 3.5])
    bounded_mask = torch.tensor([True, True, False, True])

    report = compute_bounded_regression_report(predicted_ddg, target_ddg, bounded_mask)
    assert report["n"] == 3
    assert report["mae"] < 1.0
    assert report["noise_floor_agreement_fraction"] == 0.84


def test_compute_bounded_regression_report_empty_mask_returns_nan():
    predicted_ddg = torch.tensor([1.0, 2.0])
    target_ddg = torch.tensor([1.0, 2.0])
    bounded_mask = torch.tensor([False, False])

    report = compute_bounded_regression_report(predicted_ddg, target_ddg, bounded_mask)
    assert report["n"] == 0
    assert math.isnan(report["mae"])
    assert math.isnan(report["spearman_correlation"])


def test_bucketed_mae_spans_all_five_buckets():
    metric = BucketedMAE()
    preds = torch.tensor([-2.0, -1.0, 0.5, 1.0, 4.0])
    target = torch.tensor([-3.0, -1.5, 0.0, 1.5, 3.0])

    metric.update(preds, target)
    result = metric.compute()

    assert set(result.keys()) == set(MAGNITUDE_BUCKET_NAMES)
    assert abs(result["large_destabilizing"] - 1.0) < 1e-6
    assert abs(result["moderate_destabilizing"] - 0.5) < 1e-6
    assert abs(result["near_zero"] - 0.5) < 1e-6
    assert abs(result["moderate_stabilizing"] - 0.5) < 1e-6
    assert abs(result["large_stabilizing"] - 1.0) < 1e-6


def test_bucketed_mae_empty_buckets_are_nan():
    metric = BucketedMAE()
    metric.update(torch.tensor([0.1, -0.2]), torch.tensor([0.2, -0.1]))
    result = metric.compute()

    assert not math.isnan(result["near_zero"])
    for bucket_name in ("large_destabilizing", "moderate_destabilizing", "moderate_stabilizing", "large_stabilizing"):
        assert math.isnan(result[bucket_name])


def test_bucketed_mae_boundary_values_assigned_to_correct_bucket():
    metric = BucketedMAE()
    boundary_targets = torch.tensor([-2.0, -1.0, 1.0, 2.0])
    metric.update(boundary_targets.clone(), boundary_targets)
    result = metric.compute()

    assert result["large_destabilizing"] == 0.0
    assert result["moderate_destabilizing"] == 0.0
    assert result["moderate_stabilizing"] == 0.0
    assert result["large_stabilizing"] == 0.0
    assert math.isnan(result["near_zero"])


def test_bucketed_mae_accumulates_across_updates():
    metric = BucketedMAE()
    metric.update(torch.tensor([0.0]), torch.tensor([0.0]))
    metric.update(torch.tensor([1.0]), torch.tensor([0.0]))
    result = metric.compute()
    assert abs(result["near_zero"] - 0.5) < 1e-6


def test_compute_bounded_regression_report_single_sample_spearman_is_nan():
    predicted_ddg = torch.tensor([1.0])
    target_ddg = torch.tensor([1.2])
    bounded_mask = torch.tensor([True])

    report = compute_bounded_regression_report(predicted_ddg, target_ddg, bounded_mask)
    assert report["n"] == 1
    assert math.isnan(report["spearman_correlation"])
