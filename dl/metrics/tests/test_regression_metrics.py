import math

import torch

from dl.metrics.regression_metrics import (
    BoundedMAE,
    BoundedRMSE,
    BoundedSpearmanCorrelation,
    compute_bounded_regression_report,
)


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


def test_compute_bounded_regression_report_single_sample_spearman_is_nan():
    predicted_ddg = torch.tensor([1.0])
    target_ddg = torch.tensor([1.2])
    bounded_mask = torch.tensor([True])

    report = compute_bounded_regression_report(predicted_ddg, target_ddg, bounded_mask)
    assert report["n"] == 1
    assert math.isnan(report["spearman_correlation"])
