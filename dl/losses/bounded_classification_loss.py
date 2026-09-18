"""Bin-distance-weighted classification loss for bounded-ddG entries.

Locked-in decision: the bounded-ddG head is a classifier over
`DDG_NUM_BINS` bins, not a regressor. Plain cross-entropy penalizes a 1-bin
miss as harshly (per unit of predicted probability mass) as a 9-bin miss,
which is wrong here: `NOISE_FLOOR_AGREEMENT_FRACTION` (0.84) says
independently-remeasured *same* mutations already land in adjacent bins
~16% of the time from measurement noise alone, since the bin width equals
`NOISE_FLOOR_KCAL_MOL`. This module instead uses an expected-
misclassification-cost loss: for predicted distribution p = softmax(logits)
and true bin t, `loss = sum_k p(k) * cost(t, k)`, where `cost` grows with
bin distance and is near-zero at distance 1. See
`build_bin_distance_cost_matrix` for the exact cost formula and rationale.
"""

from __future__ import annotations

import torch
from torch import nn

from dl.losses.config import DDGLossConfig


def adjacent_bin_cost(noise_floor_agreement_fraction: float) -> float:
    return 1.0 - noise_floor_agreement_fraction


def bin_distance_matrix(num_bins: int) -> torch.Tensor:
    bin_index = torch.arange(num_bins)
    return (bin_index.view(-1, 1) - bin_index.view(1, -1)).abs().to(torch.float32)


def build_bin_distance_cost_matrix(
    num_bins: int,
    noise_floor_agreement_fraction: float,
    distant_bin_cost_power: float = 2.0,
) -> torch.Tensor:
    """cost(t, k) = 0 at distance 0, `adjacent_bin_cost(...)` at distance 1,
    and `adjacent_bin_cost(...) + (distance - 1) ** distant_bin_cost_power`
    at distance >= 2. Shape (num_bins, num_bins).
    """
    distance = bin_distance_matrix(num_bins)
    adjacent_cost = adjacent_bin_cost(noise_floor_agreement_fraction)
    distant_cost = adjacent_cost + (distance - 1).clamp(min=0) ** distant_bin_cost_power
    return torch.where(distance >= 2, distant_cost, torch.where(distance == 1, adjacent_cost, 0.0))


class BinDistanceWeightedLoss(nn.Module):
    """Expected bin-distance-cost loss for the bounded-ddG classification
    head. Agnostic to label type -- callers mask out non-bounded rows
    (see `dl.losses.ddg_loss.DDGLoss`).
    """

    def __init__(self, config: DDGLossConfig | None = None):
        super().__init__()
        self.config = config or DDGLossConfig()
        cost_matrix = build_bin_distance_cost_matrix(
            num_bins=self.config.num_bins,
            noise_floor_agreement_fraction=self.config.noise_floor_agreement_fraction,
            distant_bin_cost_power=self.config.distant_bin_cost_power,
        )
        self.register_buffer("cost_matrix", cost_matrix, persistent=False)

    def _validate_logits_shape(self, logits: torch.Tensor) -> None:
        if logits.shape[-1] != self.config.num_bins:
            raise ValueError(
                f"expected logits with {self.config.num_bins} bins, got shape {tuple(logits.shape)}"
            )

    def forward(self, logits: torch.Tensor, target_bin: torch.Tensor) -> torch.Tensor:
        """logits: (B, num_bins) float. target_bin: (B,) long. Returns (B,) float."""
        self._validate_logits_shape(logits)
        probs = torch.softmax(logits, dim=-1)
        cost_matrix = self.cost_matrix.to(dtype=probs.dtype, device=probs.device)
        per_class_cost = cost_matrix[target_bin]
        return (probs * per_class_cost).sum(dim=-1)
