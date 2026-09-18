"""Builds the train/val `Dataset` pair for a run (Implementation Spec §4b).

`DataLoader` construction (batch size, shuffle, num_workers, `collate_fn`)
is deliberately not done here -- that happens in `train.py` at the repo
root, driven by the run config's `dataset` section, so `dl/datasets/` stays
limited to `Dataset` classes and their construction.

Uses `dl.datasets.mutation_csv_dataset.MutationCsvDataset` once real
processed data is present under `shared.constants.PROCESSED_DATA_DIR` /
`EMBEDDING_CACHE_DIR` for the configured split, falling back to
`dl.datasets.synthetic_dataset.SyntheticDDGDataset` otherwise -- consistent
with how the rest of this workstream unblocks itself ahead of the
pre-processing pipeline. `dataset.force_synthetic: true` in a run config
keeps training on synthetic data even once real data appears.
"""

from __future__ import annotations

from torch.utils.data import Dataset

from dl.datasets.config import DatasetConfig
from dl.datasets.mutation_csv_dataset import MutationCsvDataset
from dl.datasets.synthetic_dataset import SyntheticDDGDataset
from dl.models.config import ModelConfig
from shared.constants import EMBEDDING_CACHE_DIR, SPLIT_FILES


def is_real_embedding_file(path) -> bool:
    return path.is_file() and path.suffix == ".pt"


def real_split_files_present(dataset_config: DatasetConfig) -> bool:
    split_files = SPLIT_FILES[dataset_config.split]
    return split_files["train"].exists() and split_files["val"].exists()


def real_embeddings_present() -> bool:
    return EMBEDDING_CACHE_DIR.exists() and any(is_real_embedding_file(path) for path in EMBEDDING_CACHE_DIR.rglob("*"))


def real_data_present(dataset_config: DatasetConfig) -> bool:
    return real_split_files_present(dataset_config) and real_embeddings_present()


def should_use_synthetic_data(dataset_config: DatasetConfig) -> bool:
    return dataset_config.force_synthetic or not real_data_present(dataset_config)


def build_synthetic_dataset(dataset_config: DatasetConfig, model_config: ModelConfig, num_bins: int, num_samples: int, seed: int) -> SyntheticDDGDataset:
    return SyntheticDDGDataset(model_config, num_bins, num_samples, dataset_config.synthetic_sequence_length, seed)


def build_real_datasets(dataset_config: DatasetConfig) -> tuple[Dataset, Dataset]:
    split_files = SPLIT_FILES[dataset_config.split]
    return MutationCsvDataset(split_files["train"]), MutationCsvDataset(split_files["val"])


def build_train_and_val_datasets(dataset_config: DatasetConfig, model_config: ModelConfig, num_bins: int) -> tuple[Dataset, Dataset]:
    if not should_use_synthetic_data(dataset_config):
        return build_real_datasets(dataset_config)
    train_dataset = build_synthetic_dataset(
        dataset_config, model_config, num_bins, dataset_config.synthetic_num_train_samples, dataset_config.synthetic_seed
    )
    val_dataset = build_synthetic_dataset(
        dataset_config, model_config, num_bins, dataset_config.synthetic_num_val_samples, dataset_config.synthetic_seed + 1
    )
    return train_dataset, val_dataset
