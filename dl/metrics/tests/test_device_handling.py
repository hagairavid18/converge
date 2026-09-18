import torch

from dl.metrics.classification_metrics import BoundedBinAccuracy, BoundedBinConfusionMatrix
from dl.metrics.hinge_metrics import IneqCorrectSideAccuracy


def test_bounded_bin_accuracy_empty_compute_matches_state_device():
    metric = BoundedBinAccuracy().to("cpu")
    result = metric.compute()
    assert result.device.type == "cpu"


def test_hinge_metric_empty_compute_matches_state_device():
    metric = IneqCorrectSideAccuracy().to("cpu")
    result = metric.compute()
    assert result.device.type == "cpu"


def test_confusion_matrix_submodule_follows_to_device():
    metric = BoundedBinConfusionMatrix(num_bins=10).to("cpu")
    metric.update(torch.tensor([0, 1]), torch.tensor([0, 1]))
    summary = metric.compute()
    assert summary["n"] == 2
