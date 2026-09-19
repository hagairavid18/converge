"""Optional per-sample loss reweighting that up-weights large-magnitude
(tail) ddG targets relative to near-zero ones -- off by default, gated by
`DDGLossConfig.tail_reweight_enabled` (see `dl.losses.ddg_loss.DDGLoss` /
`dl.losses.ddg_regression_loss.DDGRegressionLoss`, which multiply their
bounded per-sample loss tensor by this before it reaches `masked_mean`).
"""

from __future__ import annotations

import torch


def tail_weight(target_ddg: torch.Tensor, alpha: float, reference_kcal_mol: float, cap: float) -> torch.Tensor:
    """`1 + alpha * min(|target_ddg| / reference_kcal_mol, cap)`: 1.0 (no-op)
    at `target_ddg == 0`, linear in `|target_ddg|`, saturating at
    `1 + alpha * cap` once `|target_ddg| >= reference_kcal_mol * cap`.
    """
    scaled_magnitude = target_ddg.abs() / reference_kcal_mol
    return 1.0 + alpha * torch.clamp(scaled_magnitude, max=cap)
