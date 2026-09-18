"""Real SKEMPI ΔΔG dataset (Implementation Spec §4b).

Reads per-sample metadata from a `shared.constants.SPLIT_FILES` CSV -- one
row per `shared.constants.MutationRecord` (one row per sample_id; a
multi-point mutant's substituted residues live in its `mutations` column,
JSON-encoded as a list of `shared.constants.PointMutation`) -- and joins in
the corresponding per-residue tensors cached under
`shared.constants.EMBEDDING_CACHE_DIR`.

Cache file contract, confirmed with the pre-processing workstream: one file
per sample (`data.embedding_pipeline.cache.sample_embedding_cache_path`),
loadable into a dict already shaped for `dl.models.schemas.ModelInputs`
(`mutation_distances` included, precomputed by pre-processing from every
point mutation's `flat_residue_index` -- this dataset never needs to touch
that field itself). The cache also carries `wt_structure_plddt_diagnostic`
(deferred validation analysis, see `docs/future_work.md`) and
`wt_real_structure_features` (unused fallback for an alternative WT-encoding
design that wasn't chosen -- the wild type is confirmed to also run through
ESMFold, same as the mutant); `select_model_fields` drops both, along with
any other extra cache keys, before constructing `ModelInputs`.

On-the-fly extraction: a deliberate deviation from the Implementation
Spec's "precomputed offline before training" (logged in
`docs/future_work.md`) -- since this machine doesn't train at scale, there
is no need to require the full cache to be pre-populated.
`load_or_compute_cached_tensors` calls the pre-processing workstream's own
per-entry entry point
(`data.embedding_pipeline.entry_embeddings.compute_and_cache_entry_embeddings`)
to compute and write a sample's cache file (this can be slow -- e.g.
ESMFold on CPU -- under `EmbeddingSourceMode`s that need it) whenever it
doesn't exist yet, or exists but was computed under a more restrictive
`ACTIVE_EMBEDDING_SOURCE_MODE` that didn't need fields the current one
does (`cache_satisfies_active_mode`) -- e.g. a cache written under
`SEQUENCE_ONLY` has no `wt_structure_embedding` at all, so if the mode is
later widened to `STRUCTURE_AND_SEQUENCE`, that stale cache is
recomputed rather than silently reused. Otherwise the existing file is
loaded as-is.

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

from data.embedding_pipeline.cache import load_sample_embedding_bundle, sample_embedding_bundle_exists
from data.embedding_pipeline.entry_embeddings import compute_and_cache_entry_embeddings
from data.processed_io import load_sample_chain_maps
from dl.models.schemas import PER_RESIDUE_FIELDS
from dl.utils.embedding_mode import uses_sequence_embedding, uses_structure_embedding
from dl.utils.label_codes import LABEL_TYPE_TO_ID
from shared.constants import ACTIVE_EMBEDDING_SOURCE_MODE, ACTIVE_LOSS_ITERATION, LabelType, LossIteration, MutationRecord


def cache_satisfies_active_mode(cached: dict) -> bool:
    """A cache file computed under a more restrictive `EmbeddingSourceMode`
    (e.g. `SEQUENCE_ONLY`) exists but is missing the fields a since-widened
    active mode (e.g. `STRUCTURE_AND_SEQUENCE`) now needs -- existence alone
    isn't enough to trust a cache file as current.
    """
    mode = ACTIVE_EMBEDDING_SOURCE_MODE
    if uses_structure_embedding(mode) and "wt_structure_embedding" not in cached:
        return False
    if uses_sequence_embedding(mode) and "wt_sequence_embedding" not in cached:
        return False
    return True


def load_or_compute_cached_tensors(record: MutationRecord, sample_chain_maps: dict[str, dict[str, str]]) -> dict:
    if sample_embedding_bundle_exists(record.sample_id):
        cached = load_sample_embedding_bundle(record.sample_id)
        if cache_satisfies_active_mode(cached):
            return cached
    compute_and_cache_entry_embeddings(record, sample_chain_maps[record.sample_id])
    return load_sample_embedding_bundle(record.sample_id)


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
        self.sample_chain_maps = load_sample_chain_maps()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[dict, dict]:
        record = self.records[index]
        cached = load_or_compute_cached_tensors(record, self.sample_chain_maps)
        return select_model_fields(cached), label_fields_for_record(record)
