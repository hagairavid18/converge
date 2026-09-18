"""A full run configuration (Implementation Spec §4b): one YAML file per run,
loaded into this `RunConfig`, with a clearly separated section per submodule
(`dataset`, `model`, `loss`, `metrics`, `trainer`). See `dl/configs/*.yaml`
for examples.

`model`, `loss`, and `metrics` each carry a `class_name` + `params` shape
(`dl.utils.factory.ClassNameParams`) resolved via `dl.training.object_building`,
so swapping architectures/losses/metrics is a config change, not a code
change. `dataset` and `trainer` are plain typed sections since there is only
one dataset pipeline and one trainer shape to configure.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel

from dl.datasets.config import DatasetConfig
from dl.utils.factory import ClassNameParams


class TrainerConfig(BaseModel):
    max_epochs: int = 50
    learning_rate: float = 1e-3
    accelerator: str = "auto"
    devices: int = 1
    comet_project_name: str = "skempi-ddg"
    experiment_name: str = "run"


class RunConfig(BaseModel):
    dataset: DatasetConfig = DatasetConfig()
    model: ClassNameParams = ClassNameParams(class_name="DDGPredictor")
    loss: ClassNameParams = ClassNameParams(class_name="DDGLoss")
    metrics: ClassNameParams = ClassNameParams(class_name="BoundedBinAccuracy")
    trainer: TrainerConfig = TrainerConfig()


def load_run_config(path: str | Path) -> RunConfig:
    with open(path) as config_file:
        raw_config = yaml.safe_load(config_file)
    return RunConfig(**raw_config)
