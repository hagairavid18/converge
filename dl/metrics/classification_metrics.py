"""Bounded-label head metrics (Implementation Spec sec. 5): accuracy and an
adjacency-aware confusion matrix over the DDG_NUM_BINS classification bins.

Implemented as `torchmetrics.Metric` subclasses (internal state, `update()`/
`compute()`) so a Lightning training loop can accumulate them across an
epoch's batches and read them back at epoch end, with one instance per split
(`SplitName.HELD_OUT_PDB` / `SAME_PDB_ALLOWED` never sharing state). Both
classes are importable as `dl.metrics.BoundedBinAccuracy` /
`dl.metrics.BoundedBinConfusionMatrix` for config-driven construction via
`dl.utils.factory.build_object`.
"""

from __future__ import annotations

import torch
from torchmetrics import Metric
from torchmetrics.classification import MulticlassConfusionMatrix

from shared.constants import DDG_NUM_BINS, NOISE_FLOOR_AGREEMENT_FRACTION

from dl.utils.confusion_utils import adjacency_aware_confusion_summary


def logits_to_pred_bin(logits: torch.Tensor) -> torch.Tensor:
    return logits.argmax(dim=-1)


class BoundedBinAccuracy(Metric):
    """Exact-bin accuracy for the bounded-ddG classification head."""

    full_state_update = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred_bin: torch.Tensor, true_bin: torch.Tensor) -> None:
        self.correct += (pred_bin == true_bin).sum()
        self.total += pred_bin.numel()

    def compute(self) -> torch.Tensor:
        if self.total == 0:
            return torch.tensor(float("nan"), device=self.total.device)
        return self.correct.float() / self.total


class BoundedBinConfusionMatrix(Metric):
    """Wraps `torchmetrics.classification.MulticlassConfusionMatrix` and
    layers the adjacency-aware, noise-floor-referenced summary
    (`dl.utils.confusion_utils.adjacency_aware_confusion_summary`) on top of
    its `compute()` output, so `compute()` here returns that summary dict
    directly rather than the raw matrix.
    """

    full_state_update = False

    def __init__(
        self,
        num_bins: int = DDG_NUM_BINS,
        noise_floor_agreement_fraction: float = NOISE_FLOOR_AGREEMENT_FRACTION,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.noise_floor_agreement_fraction = noise_floor_agreement_fraction
        self._confusion_matrix = MulticlassConfusionMatrix(num_classes=num_bins)

    def update(self, pred_bin: torch.Tensor, true_bin: torch.Tensor) -> None:
        self._confusion_matrix.update(pred_bin, true_bin)

    def compute(self) -> dict:
        confusion = self._confusion_matrix.compute().to(torch.long)
        return adjacency_aware_confusion_summary(confusion, self.noise_floor_agreement_fraction)

    def reset(self) -> None:
        super().reset()
        self._confusion_matrix.reset()
