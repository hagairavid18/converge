import torch

from dl.metrics.classification_metrics import (
    BoundedBinAccuracy,
    BoundedBinConfusionMatrix,
    logits_to_pred_bin,
)


def test_logits_to_pred_bin_argmax():
    logits = torch.tensor([[0.1, 0.9, 0.0], [5.0, 1.0, 1.0]])
    assert torch.equal(logits_to_pred_bin(logits), torch.tensor([1, 0]))


def test_bounded_bin_accuracy_accumulates_across_updates():
    metric = BoundedBinAccuracy()
    metric.update(torch.tensor([0, 1, 2]), torch.tensor([0, 1, 5]))
    metric.update(torch.tensor([3, 4]), torch.tensor([3, 3]))
    assert abs(metric.compute().item() - 3 / 5) < 1e-6


def test_bounded_bin_accuracy_empty_is_nan():
    metric = BoundedBinAccuracy()
    assert torch.isnan(metric.compute())


def test_bounded_bin_accuracy_reset():
    metric = BoundedBinAccuracy()
    metric.update(torch.tensor([0]), torch.tensor([1]))
    metric.reset()
    assert torch.isnan(metric.compute())


def test_confusion_matrix_exact_and_within_1_bin_fractions():
    metric = BoundedBinConfusionMatrix(num_bins=10)
    pred_bin = torch.tensor([5, 6, 9])
    true_bin = torch.tensor([5, 5, 0])
    metric.update(pred_bin, true_bin)
    summary = metric.compute()

    assert summary["n"] == 3
    assert summary["exact_fraction"] == 1 / 3
    assert summary["within_1_bin_fraction"] == 2 / 3
    assert summary["noise_floor_agreement_fraction"] == 0.84


def test_confusion_matrix_adjacent_vs_distant_histogram_ordering():
    metric = BoundedBinConfusionMatrix(num_bins=10)
    metric.update(torch.tensor([6, 9]), torch.tensor([5, 0]))
    summary = metric.compute()
    histogram = summary["distance_histogram"]
    assert histogram[1] > 0
    assert histogram[9] > 0
