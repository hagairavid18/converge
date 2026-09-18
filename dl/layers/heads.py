"""Prediction heads attached to the pooled representation (Implementation Spec §3).

Two head types share the same pooled vector, selected via
`ModelConfig.active_heads` (`["classification"]`, `["regression"]`, or
both) and built from `HEAD_FACTORIES` -- a new head type is a factory
registered via `register_head_factory`, not a change to `DDGPredictor`.

- `classification`: logits over `DDG_NUM_BINS` bins (iteration 1's
  bounded-ddG head).
- `regression`: `head_outputs["regression"]` is the predicted ddG in
  kcal/mol, shape `[B]` (one scalar per sample, squeezed from the
  underlying `Linear(input_dim, 1)`) -- for the losses/metrics workstream's
  margin/dead-zone regression loss to consume directly.
"""

from __future__ import annotations

from typing import Callable, Dict

import torch
from torch import nn

from dl.models.config import ModelConfig

HeadFactory = Callable[[ModelConfig, int], nn.Module]


def build_classification_head(config: ModelConfig, input_dim: int) -> nn.Module:
    return nn.Linear(input_dim, config.num_ddg_bins)


class RegressionHead(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, pooled_representation: torch.Tensor) -> torch.Tensor:
        return self.linear(pooled_representation).squeeze(-1)


def build_regression_head(config: ModelConfig, input_dim: int) -> nn.Module:
    return RegressionHead(input_dim)


HEAD_FACTORIES: Dict[str, HeadFactory] = {
    "classification": build_classification_head,
    "regression": build_regression_head,
}


def register_head_factory(name: str, factory: HeadFactory) -> None:
    HEAD_FACTORIES[name] = factory


def build_head(name: str, config: ModelConfig, input_dim: int) -> nn.Module:
    if name not in HEAD_FACTORIES:
        raise ValueError(f"Unknown prediction head '{name}' - register it via register_head_factory() first")
    return HEAD_FACTORIES[name](config, input_dim)


class PredictionHeads(nn.Module):
    def __init__(self, config: ModelConfig, input_dim: int):
        super().__init__()
        self.heads = nn.ModuleDict({name: build_head(name, config, input_dim) for name in config.active_heads})

    def forward(self, pooled_representation):
        return {name: head(pooled_representation) for name, head in self.heads.items()}
