"""Mutation-window pooling (Implementation Spec §3, weighting mechanism 2).

Reduces the per-residue difference representation to one fixed-size vector.
Two strategies, selected via `ModelConfig.pooling_strategy` and built from
`POOLING_FACTORIES` -- a new strategy is a factory registered via
`register_pooling_factory`, not a change to `DDGPredictor`:

- `gaussian_distance` (the original mechanism): a fixed-sigma Gaussian
  falloff over pre-processing's precomputed per-mutation-site distances
  (`ModelInputs.mutation_distances`, real 3D C-alpha distance). A sample
  with multiple mutated residues gets one Gaussian per site, combined via
  elementwise max (`mutation_site_mask` marks which of the padded sites are
  real). `sigma` (`ModelConfig.gaussian_sigma_angstrom`) is a fixed
  hyperparameter, not learned.
- `norm_softmax`: a parameter-free softmax over each residue's
  difference-vector norm, divided by `ModelConfig.pooling_temperature`
  (fixed hyperparameter, not learned, default 1.0 = no softening). Confirmed
  empirically (see docs/future_work.md, the antigen ESM-2 and SaProt
  mutation-localization probes) that this norm spikes sharply at the mutated
  residue(s) and stays near zero elsewhere, so it needs no distance input
  and works identically under every `embedding_source_mode`, including
  `sequence_only`. Multi-point mutations are handled for free -- the softmax
  naturally puts mass on every residue with high difference norm, regardless
  of site count. The site-vs-elsewhere gap can be large enough (confirmed
  for the SaProt backend, see docs/future_work.md) that temperature 1.0
  collapses this into a near one-hot selection of a single residue rather
  than a real weighted window -- a temperature > 1 softens that back into an
  actual neighborhood average.
- `norm_softmax_by_chain_role`: the same `norm_softmax` mechanism, but run
  independently on the antigen residues (`ModelInputs.antigen_mask`) and the
  antibody (heavy+light) residues, each normalized to sum to 1 within its
  own side, then concatenated -- `POOLING_OUTPUT_DIM_MULTIPLIER` is 2 for
  this strategy, so `DDGPredictor` sizes the head's input accordingly. A
  single global softmax lets whichever side has the larger diff-norm
  suppress the other side to ~0 weight entirely (confirmed empirically, see
  docs/future_work.md); pooling each side separately guarantees both always
  contribute to the final representation, regardless of which side the
  mutation is actually on.

Both `norm_softmax` variants are normalized to sum to 1 over non-padded
residues (per side, for the split variant); `gaussian_distance` likewise.
"""

from __future__ import annotations

from typing import Callable, Dict

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


def norm_softmax_weights(residue_features: torch.Tensor, padding_mask: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    norms = residue_features.norm(dim=-1)
    masked_norms = norms.masked_fill(~padding_mask, float("-inf"))
    return torch.softmax(masked_norms / temperature, dim=1)


def pool_with_weights(residue_features: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (residue_features * weights.unsqueeze(-1)).sum(dim=1)


class GaussianDistancePooling(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.sigma = config.gaussian_sigma_angstrom

    def forward(self, residue_features: torch.Tensor, inputs: ModelInputs) -> tuple[torch.Tensor, torch.Tensor]:
        site_weights = gaussian_falloff(inputs.mutation_distances, self.sigma)
        raw_weights = combine_multi_site_weights(site_weights, inputs.mutation_site_mask)
        pooling_weights = normalize_pooling_weights(raw_weights, inputs.padding_mask)
        pooled = pool_with_weights(residue_features, pooling_weights)
        return pooled, pooling_weights


class NormSoftmaxPooling(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.temperature = config.pooling_temperature

    def forward(self, residue_features: torch.Tensor, inputs: ModelInputs) -> tuple[torch.Tensor, torch.Tensor]:
        pooling_weights = norm_softmax_weights(residue_features, inputs.padding_mask, self.temperature)
        pooled = pool_with_weights(residue_features, pooling_weights)
        return pooled, pooling_weights


class ChainRoleSplitPooling(nn.Module):
    OUTPUT_DIM_MULTIPLIER = 2

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.temperature = config.pooling_temperature

    def forward(self, residue_features: torch.Tensor, inputs: ModelInputs) -> tuple[torch.Tensor, torch.Tensor]:
        antigen_mask = inputs.antigen_mask & inputs.padding_mask
        antibody_mask = (~inputs.antigen_mask) & inputs.padding_mask

        antigen_weights = norm_softmax_weights(residue_features, antigen_mask, self.temperature)
        antibody_weights = norm_softmax_weights(residue_features, antibody_mask, self.temperature)

        antigen_pooled = pool_with_weights(residue_features, antigen_weights)
        antibody_pooled = pool_with_weights(residue_features, antibody_weights)
        pooled = torch.cat([antibody_pooled, antigen_pooled], dim=-1)

        # Diagnostic-only combined view: each residue's weight is its share
        # WITHIN its own side (each side sums to 1 independently), so this
        # sums to ~2 overall, not 1 -- not consumed by the loss/head, only
        # returned for inspection (see notebooks/pooling_*_probe.py).
        combined_weights = antigen_weights.masked_fill(~antigen_mask, 0.0) + antibody_weights.masked_fill(~antibody_mask, 0.0)
        return pooled, combined_weights


PoolingFactory = Callable[[ModelConfig], nn.Module]

POOLING_FACTORIES: Dict[str, PoolingFactory] = {
    "gaussian_distance": GaussianDistancePooling,
    "norm_softmax": NormSoftmaxPooling,
    "norm_softmax_by_chain_role": ChainRoleSplitPooling,
}

POOLING_OUTPUT_DIM_MULTIPLIER: Dict[str, int] = {
    "gaussian_distance": 1,
    "norm_softmax": 1,
    "norm_softmax_by_chain_role": ChainRoleSplitPooling.OUTPUT_DIM_MULTIPLIER,
}


def register_pooling_factory(name: str, factory: PoolingFactory, output_dim_multiplier: int = 1) -> None:
    POOLING_FACTORIES[name] = factory
    POOLING_OUTPUT_DIM_MULTIPLIER[name] = output_dim_multiplier


def build_pooling(name: str, config: ModelConfig) -> nn.Module:
    if name not in POOLING_FACTORIES:
        raise ValueError(f"Unknown pooling strategy '{name}' - register it via register_pooling_factory() first")
    return POOLING_FACTORIES[name](config)


def pooling_output_dim_multiplier(name: str) -> int:
    return POOLING_OUTPUT_DIM_MULTIPLIER.get(name, 1)
