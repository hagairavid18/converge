"""Dataset/dataloader configuration section of a run config (see
`dl.training.run_config.RunConfig`).
"""

from __future__ import annotations

from pydantic import BaseModel

from shared.constants import RANDOM_SEED, SplitName


class DatasetConfig(BaseModel):
    split: SplitName = SplitName.HELD_OUT_PDB
    batch_size: int = 16
    num_workers: int = 0
    force_synthetic: bool = False
    synthetic_sequence_length: int = 24
    synthetic_num_train_samples: int = 64
    synthetic_num_val_samples: int = 16
    synthetic_seed: int = RANDOM_SEED
