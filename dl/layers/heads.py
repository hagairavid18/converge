"""Prediction heads attached to the pooled representation (Implementation Spec §3).

Iteration 1 (`shared.constants.ACTIVE_LOSS_ITERATION`) uses only the
classification head over `DDG_NUM_BINS`. The set of active heads is
config-driven (`ModelConfig.active_heads`) and built from `HEAD_FACTORIES`,
so a later hinge/inequality head (iteration 2, owned by the losses/metrics
workstream) can be registered via `register_head_factory` and attached to
the same pooled vector without restructuring the backbone.
"""

from __future__ import annotations

from typing import Callable, Dict

from torch import nn

from dl.models.config import ModelConfig

HeadFactory = Callable[[ModelConfig, int], nn.Module]


def build_classification_head(config: ModelConfig, input_dim: int) -> nn.Module:
    return nn.Linear(input_dim, config.num_ddg_bins)


HEAD_FACTORIES: Dict[str, HeadFactory] = {
    "classification": build_classification_head,
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
