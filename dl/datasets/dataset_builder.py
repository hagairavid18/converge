"""Builds the train/val `Dataset` pair for a run (Implementation Spec §4b).

`DataLoader` construction (batch size, shuffle, num_workers, `collate_fn`)
is deliberately not done here -- that happens in `train.py` at the repo
root, driven by the run config's `dataset` section, so `dl/datasets/` stays
limited to `Dataset` classes and their construction.

Builds `dl.datasets.mutation_csv_dataset.MutationCsvDataset` from the real
processed data under `shared.constants.PROCESSED_DATA_DIR` /
`EMBEDDING_CACHE_DIR` for the configured split, raising `FileNotFoundError`
if either isn't present yet.

`build_real_datasets` resolves the run's active `LossIteration` (via
`resolve_loss_iteration`, from `loss.params.iteration` in the run config,
falling back to `shared.constants.ACTIVE_LOSS_ITERATION`) and applies it only
to the train dataset's record filtering -- the val dataset is always
constructed with `LossIteration.ITERATION_1_BOUNDED_ONLY` regardless, so
ineq/n.b. rows never leak into validation even when a config opts into
`ITERATION_2_WITH_HINGE` for training.
"""

from __future__ import annotations

from torch.utils.data import Dataset

from data.homology_dedup import normalize_protein_name
from dl.datasets.config import DatasetConfig
from dl.datasets.mutation_csv_dataset import MutationCsvDataset
from dl.models.config import ModelConfig
from shared.constants import ACTIVE_LOSS_ITERATION, EMBEDDING_CACHE_DIR, SPLIT_FILES, LossIteration


def is_real_embedding_file(path) -> bool:
    return path.is_file() and path.suffix == ".pt"


def real_split_files_present(dataset_config: DatasetConfig) -> bool:
    split_files = SPLIT_FILES[dataset_config.split]
    return split_files["train"].exists() and split_files["val"].exists()


def real_embeddings_present() -> bool:
    return EMBEDDING_CACHE_DIR.exists() and any(is_real_embedding_file(path) for path in EMBEDDING_CACHE_DIR.rglob("*"))


def real_data_present(dataset_config: DatasetConfig) -> bool:
    return real_split_files_present(dataset_config) and real_embeddings_present()


def require_real_data_present(dataset_config: DatasetConfig) -> None:
    if real_data_present(dataset_config):
        return
    raise FileNotFoundError(
        f"No processed data found for split '{dataset_config.split}'. Expected split CSVs under "
        f"{SPLIT_FILES[dataset_config.split]['train'].parent} and cached embeddings under {EMBEDDING_CACHE_DIR}. "
        "Run the data/ pre-processing pipeline before training."
    )


def resolve_loss_iteration(loss_params: dict) -> LossIteration:
    return LossIteration(loss_params.get("iteration", ACTIVE_LOSS_ITERATION))


def build_real_datasets(dataset_config: DatasetConfig, iteration: LossIteration) -> tuple[Dataset, Dataset]:
    split_files = SPLIT_FILES[dataset_config.split]
    train_dataset = MutationCsvDataset(split_files["train"], iteration=iteration)
    train_complex_names = frozenset(normalize_protein_name(record.complex_name or "") for record in train_dataset.records)
    val_dataset = MutationCsvDataset(
        split_files["val"], iteration=LossIteration.ITERATION_1_BOUNDED_ONLY, train_complex_names=train_complex_names
    )
    return train_dataset, val_dataset


def build_train_and_val_datasets(
    dataset_config: DatasetConfig,
    model_config: ModelConfig,
    num_bins: int,
    iteration: LossIteration = ACTIVE_LOSS_ITERATION,
) -> tuple[Dataset, Dataset]:
    require_real_data_present(dataset_config)
    return build_real_datasets(dataset_config, iteration)
