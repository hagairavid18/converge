"""Resolves the `model`/`loss`/`metrics` YAML config sections into
constructed objects via `dl.utils.factory.build_object`.

`DDGPredictor` and `DDGLoss` each take a single `config` object rather than
flat kwargs, so `build_configured_object` wraps the section's flat `params`
dict into the right Pydantic config type before calling `build_object`.
Metric classes (`dl.metrics`) take flat kwargs directly and need no such
wrapping.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any, Optional, Type

import dl.losses
import dl.models

from dl.losses.config import DDGLossConfig
from dl.models.config import ModelConfig
from dl.utils.factory import build_object


def build_configured_object(module: ModuleType, section: dict, config_cls: Optional[Type] = None) -> Any:
    class_name = section["class_name"]
    params = dict(section.get("params", {}))
    if config_cls is not None:
        params = {"config": config_cls(**params)}
    return build_object(module, class_name, **params)


def build_model_from_config(section: dict) -> Any:
    return build_configured_object(dl.models, section, config_cls=ModelConfig)


def build_loss_from_config(section: dict) -> Any:
    return build_configured_object(dl.losses, section, config_cls=DDGLossConfig)


def build_metric_from_config(metrics_module: ModuleType, section: dict) -> Any:
    return build_configured_object(metrics_module, section, config_cls=None)
