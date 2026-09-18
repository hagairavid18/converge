"""Generic config-driven object construction, used by the Lightning module to
build its loss and metrics from the config sections `train.py` passes through.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

from pydantic import BaseModel, Field


def build_object(module: ModuleType, class_name: str, **params: Any) -> Any:
    return getattr(module, class_name)(**params)


class ClassNameParams(BaseModel):
    class_name: str
    params: dict = Field(default_factory=dict)
