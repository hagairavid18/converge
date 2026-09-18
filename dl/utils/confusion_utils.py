"""Pure tensor building blocks for turning a bin confusion matrix into the
adjacency-aware, noise-floor-referenced summary called for by the
Implementation Spec. Used inside `dl.metrics.classification_metrics` to layer
that summary on top of a `torchmetrics.classification.MulticlassConfusionMatrix`
result.
"""

from __future__ import annotations

import torch


def bin_distance_matrix(num_bins: int) -> torch.Tensor:
    bin_index = torch.arange(num_bins)
    return (bin_index.view(-1, 1) - bin_index.view(1, -1)).abs()


def confusion_mass_at_distance(confusion: torch.Tensor, distance: int) -> int:
    mask = bin_distance_matrix(confusion.shape[0]) == distance
    return int(confusion[mask].sum().item())


def confusion_distance_histogram(confusion: torch.Tensor, n: int) -> dict[int, float]:
    num_bins = confusion.shape[0]
    return {d: confusion_mass_at_distance(confusion, d) / n for d in range(num_bins)}


def empty_confusion_summary(noise_floor_agreement_fraction: float) -> dict:
    return {
        "exact_fraction": float("nan"),
        "within_1_bin_fraction": float("nan"),
        "noise_floor_agreement_fraction": noise_floor_agreement_fraction,
        "distance_histogram": {},
        "n": 0,
    }


def adjacency_aware_confusion_summary(confusion: torch.Tensor, noise_floor_agreement_fraction: float) -> dict:
    """`within_1_bin_fraction` is directly comparable to
    `noise_floor_agreement_fraction` (0.84): bin width == NOISE_FLOOR_KCAL_MOL,
    so "within 1 bin" here is exactly "within 1 kcal/mol" there.
    """
    n = int(confusion.sum().item())
    if n == 0:
        return empty_confusion_summary(noise_floor_agreement_fraction)

    distance_histogram = confusion_distance_histogram(confusion, n)
    exact_fraction = distance_histogram.get(0, 0.0)
    within_1_bin_fraction = exact_fraction + distance_histogram.get(1, 0.0)

    return {
        "exact_fraction": exact_fraction,
        "within_1_bin_fraction": within_1_bin_fraction,
        "noise_floor_agreement_fraction": noise_floor_agreement_fraction,
        "distance_histogram": distance_histogram,
        "n": n,
    }
