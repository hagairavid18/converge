"""Regression alternative to the bounded-ddG classification head
(Implementation Spec sec. 4): a dead-zone loss around the true ddG value.
Zero penalty within `regression_dead_zone_kcal_mol` of the true value
(sized to `NOISE_FLOOR_KCAL_MOL`, same as the classification head's
adjacent-bin near-zero cost); squared penalty beyond that (see
`DDGLossConfig.regression_penalty_growth` for the rationale).
"""

from __future__ import annotations

import torch
from torch import nn

from dl.losses.config import DDGLossConfig


def dead_zone_violation(predicted_ddg: torch.Tensor, target_ddg: torch.Tensor, dead_zone_kcal_mol: float) -> torch.Tensor:
    return torch.relu((predicted_ddg - target_ddg).abs() - dead_zone_kcal_mol)


def grow_penalty(violation: torch.Tensor, growth: str) -> torch.Tensor:
    return violation**2 if growth == "squared" else violation


class BoundedRegressionLoss(nn.Module):
    """Per-sample dead-zone regression loss for the bounded-ddG head."""

    def __init__(self, config: DDGLossConfig | None = None):
        super().__init__()
        self.config = config or DDGLossConfig()

    def forward(self, predicted_ddg: torch.Tensor, target_ddg: torch.Tensor) -> torch.Tensor:
        violation = dead_zone_violation(predicted_ddg, target_ddg, self.config.regression_dead_zone_kcal_mol)
        penalty = grow_penalty(violation, self.config.regression_penalty_growth)
        return self.config.regression_scale * penalty
