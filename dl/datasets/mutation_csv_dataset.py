"""Real SKEMPI ΔΔG dataset (Implementation Spec §4b).

Reads per-sample metadata from a `shared.constants.SPLIT_FILES` CSV -- one
row per `shared.constants.MutationRecord` (one row per sample_id; a
multi-point mutant's substituted residues live in its `mutations` column,
JSON-encoded as a list of `shared.constants.PointMutation`) -- and joins in
the corresponding per-residue tensors cached under
`shared.constants.EMBEDDING_CACHE_DIR`.

Cache file contract, confirmed with the pre-processing workstream: one file
per sample, `EMBEDDING_CACHE_DIR / f"{sample_id}.pt"`, loadable with
`torch.load` into a dict already shaped for `dl.models.schemas.ModelInputs`
(`mutation_distances` included, precomputed by pre-processing from every
point mutation's `flat_residue_index` -- this dataset never needs to touch
that field itself). The cache also carries `wt_structure_plddt_diagnostic`
(deferred validation analysis, see `docs/future_work.md`) and
`wt_real_structure_features` (unused fallback for an alternative WT-encoding
design that wasn't chosen -- the wild type is confirmed to also run through
ESMFold, same as the mutant); `select_model_fields` drops both, along with
any other extra cache keys, before constructing `ModelInputs`.

`shared.constants.ACTIVE_LOSS_ITERATION == ITERATION_1_BOUNDED_ONLY` means
non-bounded samples are dropped entirely at construction time, per that
constant's own comment ("ineq/n.b. masked out of loss and batches").
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from dl.models.schemas import PER_RESIDUE_FIELDS
from dl.utils.label_codes import LABEL_TYPE_TO_ID
from shared.constants import ACTIVE_LOSS_ITERATION, EMBEDDING_CACHE_DIR, LabelType, LossIteration, MutationRecord


def embedding_cache_path(sample_id: str) -> Path:
    return EMBEDDING_CACHE_DIR / f"{sample_id}.pt"


def load_cached_residue_tensors(sample_id: str) -> dict:
    return torch.load(embedding_cache_path(sample_id))


def select_model_fields(cached: dict) -> dict:
    """Cache keys for an inactive `EmbeddingSourceMode` may be entirely
    absent (e.g. pre-processing skips ESMFold altogether under
    `SEQUENCE_ONLY`), not merely `None` -- `.get(...)` keeps every
    per-residue field present (as `None` when absent) so collation can
    always look it up, matching `dl.models.synthetic`'s always-present-keys
    shape. `mutation_distances` is mode-independent and always required, so
    it is indexed directly rather than defaulted.
    """
    selected = {key: cached.get(key) for key in PER_RESIDUE_FIELDS}
    selected["mutation_distances"] = cached["mutation_distances"]
    return selected


def coerce_field_value(key: str, value):
    if pd.isna(value):
        return None
    if key == "source_publication":
        return str(value)
    if key == "mutations":
        return json.loads(value)
    return value


def clean_row_for_record(row: dict) -> dict:
    return {key: coerce_field_value(key, value) for key, value in row.items() if key in MutationRecord.model_fields}


def load_records(csv_path: str | Path) -> list[MutationRecord]:
    frame = pd.read_csv(csv_path)
    return [MutationRecord(**clean_row_for_record(row)) for row in frame.to_dict(orient="records")]


def keep_bounded_only(records: list[MutationRecord]) -> list[MutationRecord]:
    return [record for record in records if record.label_type == LabelType.BOUNDED]


def filter_records_for_active_iteration(records: list[MutationRecord]) -> list[MutationRecord]:
    if ACTIVE_LOSS_ITERATION == LossIteration.ITERATION_1_BOUNDED_ONLY:
        return keep_bounded_only(records)
    return records


def label_fields_for_record(record: MutationRecord) -> dict:
    return {
        "label_type_id": torch.tensor(LABEL_TYPE_TO_ID[record.label_type], dtype=torch.long),
        "target_bin": torch.tensor(int(record.ddg_bin) if record.ddg_bin is not None else 0, dtype=torch.long),
    }


class MutationCsvDataset(Dataset):
    def __init__(self, csv_path: str | Path):
        self.records = filter_records_for_active_iteration(load_records(csv_path))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[dict, dict]:
        record = self.records[index]
        input_fields = select_model_fields(load_cached_residue_tensors(record.sample_id))
        return input_fields, label_fields_for_record(record)
