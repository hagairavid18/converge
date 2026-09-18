"""Hinge loss for "ineq" and "n.b." entries (Implementation Spec sec. 4,
iteration 2, gated behind `LossIteration.ITERATION_2_WITH_HINGE`).

Direction convention: `direction` is +1 for ">" (truth is at least this
destabilizing) and -1 for "<" (truth is at most this value); n.b. entries
are always +1 (binding abolished, far on the destabilizing side).
"""

from __future__ import annotations

import torch
from torch import nn

from dl.losses.config import DDGLossConfig


def signed_margin(predicted_ddg: torch.Tensor, bound_kcal_mol: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Positive when `predicted_ddg` is on the correct side of the bound."""
    return direction * (predicted_ddg - bound_kcal_mol)


class HingeLoss(nn.Module):
    """Per-sample one-sided hinge loss against a known ddG bound: zero when
    `predicted_ddg` is on the correct side, else growing linearly (rate
    `config.hinge_scale`) with the kcal/mol distance past the bound.
    """

    def __init__(self, config: DDGLossConfig | None = None):
        super().__init__()
        self.config = config or DDGLossConfig()

    def forward(
        self,
        predicted_ddg: torch.Tensor,
        bound_kcal_mol: torch.Tensor,
        direction: torch.Tensor,
    ) -> torch.Tensor:
        violation = torch.relu(-signed_margin(predicted_ddg, bound_kcal_mol, direction))
        return self.config.hinge_scale * violation
