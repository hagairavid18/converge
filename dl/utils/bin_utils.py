"""Helpers built on the locked-in `DDG_BIN_EDGES_KCAL_MOL` scheme, shared by
`dl/losses` (hinge term) and `dl/metrics` (analyses that need a single
real-valued ddG estimate from the classification head).
"""

from __future__ import annotations

import torch


def compute_bin_centers(bin_edges_kcal_mol: list[float]) -> torch.Tensor:
    """Midpoint of each bin, shape (num_bins,)."""
    edges = torch.as_tensor(bin_edges_kcal_mol, dtype=torch.float32)
    return (edges[:-1] + edges[1:]) / 2.0


def expected_ddg_from_logits(logits: torch.Tensor, bin_centers: torch.Tensor) -> torch.Tensor:
    """Differentiable point estimate of ddG: softmax-weighted average of bin
    centers, given classification logits of shape (B, num_bins) and
    `bin_centers` of shape (num_bins,). Returns shape (B,).
    """
    probs = torch.softmax(logits, dim=-1)
    return probs @ bin_centers.to(dtype=probs.dtype, device=probs.device)
