"""Mutation-window pooling (Implementation Spec §3, weighting mechanism 2).

Reduces the per-residue difference representation to one fixed-size vector
using a smooth Gaussian falloff around the mutation site(s), normalized to
sum to 1 over non-padded residues. `sigma` is a fixed hyperparameter
(`ModelConfig.gaussian_sigma_angstrom`), not a learned parameter, per
CLAUDE.md's simplicity rule.

Distances are precomputed by pre-processing (`ModelInputs.mutation_distances`,
`[B, M, L]` -- Euclidean C-alpha distance from each of a sample's `M` point
mutations to every one of its `L` residues) rather than derived here from
raw coordinates. A sample with more than one mutated residue (`M > 1`) gets
one Gaussian per site, and a residue's window weight is the elementwise max
across its sites' Gaussians (`mutation_site_mask` marks which of the padded
`M` sites are real) -- a residue close to *any* mutated site is weighted
highly, rather than averaging sites.
"""

from __future__ import annotations

import torch
from torch import nn

from dl.models.config import ModelConfig
from dl.models.schemas import ModelInputs


def gaussian_falloff(distance: torch.Tensor, sigma: float) -> torch.Tensor:
    return torch.exp(-(distance**2) / (2 * sigma**2))


def combine_multi_site_weights(site_weights: torch.Tensor, mutation_site_mask: torch.Tensor) -> torch.Tensor:
    masked_weights = site_weights.masked_fill(~mutation_site_mask.unsqueeze(-1), 0.0)
    return masked_weights.max(dim=1).values


def normalize_pooling_weights(weights: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
    masked_weights = weights.masked_fill(~padding_mask, 0.0)
    return masked_weights / masked_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)


def pool_with_weights(residue_features: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (residue_features * weights.unsqueeze(-1)).sum(dim=1)


class MutationWindowPooling(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.sigma = config.gaussian_sigma_angstrom

    def forward(self, residue_features: torch.Tensor, inputs: ModelInputs) -> tuple[torch.Tensor, torch.Tensor]:
        site_weights = gaussian_falloff(inputs.mutation_distances, self.sigma)
        raw_weights = combine_multi_site_weights(site_weights, inputs.mutation_site_mask)
        pooling_weights = normalize_pooling_weights(raw_weights, inputs.padding_mask)
        pooled = pool_with_weights(residue_features, pooling_weights)
        return pooled, pooling_weights
