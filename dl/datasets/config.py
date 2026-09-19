"""Dataset/dataloader configuration section of a run config (see
`dl.training.run_config.RunConfig`).
"""

from __future__ import annotations

from pydantic import BaseModel

from shared.constants import SplitName


class DatasetConfig(BaseModel):
    split: SplitName = SplitName.HELD_OUT_PDB
    batch_size: int = 16
    num_workers: int = 0
