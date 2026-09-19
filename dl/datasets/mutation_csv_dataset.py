"""Real SKEMPI ΔΔG dataset (Implementation Spec §4b).

Reads per-sample metadata from a `shared.constants.SPLIT_FILES` CSV -- one
row per `shared.constants.MutationRecord` (one row per sample_id; a
multi-point mutant's substituted residues live in its `mutations` column,
JSON-encoded as a list of `shared.constants.PointMutation`) -- and joins in
the corresponding per-residue tensors cached under
`shared.constants.EMBEDDING_CACHE_DIR`.

Cache file contract, confirmed with the pre-processing workstream: per
`data.embedding_pipeline.cache`, wild-type tensors are cached once per
`pdb_id` and mutant tensors once per `sample_id`, split further into
`structure/` and `sequence/` (so one backbone can be regenerated
independently of the other).
`data.embedding_pipeline.entry_embeddings.compute_and_cache_entry_embeddings`
loads/computes whichever of those four pieces this sample needs and merges
them into one dict already shaped for `dl.models.schemas.ModelInputs`
(`mutation_distances` included, precomputed by pre-processing from every
point mutation's `flat_residue_index` -- this dataset never needs to touch
that field itself). The merged dict also carries `wt_structure_plddt_diagnostic`
(deferred validation analysis, see `docs/future_work.md`) and
`wt_real_structure_features` (unused fallback for an alternative WT-encoding
design that wasn't chosen -- the wild type is confirmed to also run through
ESMFold, same as the mutant) and `residue_coordinates` (pipeline-internal,
used only to derive `mutation_distances`); `select_model_fields` drops all
three, along with any other extra cache keys, before constructing
`ModelInputs`.

On-the-fly extraction: a deliberate deviation from the Implementation
Spec's "precomputed offline before training" (logged in
`docs/future_work.md`) -- since this machine doesn't train at scale, there
is no need to require the full cache to be pre-populated.
`load_or_compute_cached_tensors` calls
`compute_and_cache_entry_embeddings`, which computes and writes whichever
cache pieces don't exist yet, or exist but were computed under a more
restrictive `ACTIVE_EMBEDDING_SOURCE_MODE` that didn't need fields the
current one does -- e.g. a cache written under `SEQUENCE_ONLY` has no
`wt_structure_embedding` at all, so if the mode is later widened to
`STRUCTURE_AND_SEQUENCE`, that stale piece is recomputed rather than
silently reused. Already-current pieces are loaded as-is, not recomputed.

`MutationCsvDataset`'s `iteration` (defaulting to
`shared.constants.ACTIVE_LOSS_ITERATION`) selects this at construction time:
`ITERATION_1_BOUNDED_ONLY` drops non-bounded samples entirely, per that
constant's own comment ("ineq/n.b. masked out of loss and batches"); see
`dl.datasets.dataset_builder.build_real_datasets` for how the train/val split
resolves and enforces this per dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from data.embedding_pipeline.entry_embeddings import compute_and_cache_entry_embeddings
from data.homology_dedup import normalize_protein_name
from data.processed_io import load_sample_chain_maps
from dl.models.schemas import PER_RESIDUE_FIELDS
from dl.utils.label_codes import LABEL_TYPE_TO_ID, MUTATION_SIDE_TO_ID
from shared.constants import ACTIVE_LOSS_ITERATION, ChainRole, LabelType, LossIteration, MutationRecord


def load_or_compute_cached_tensors(record: MutationRecord, sample_chain_maps: dict[str, dict[str, str]]) -> dict:
    return compute_and_cache_entry_embeddings(record, sample_chain_maps[record.sample_id])


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


def filter_records_for_active_iteration(
    records: list[MutationRecord], iteration: LossIteration = ACTIVE_LOSS_ITERATION
) -> list[MutationRecord]:
    if iteration == LossIteration.ITERATION_1_BOUNDED_ONLY:
        return keep_bounded_only(records)
    return records


def metadata_fields_for_record(record: MutationRecord) -> dict:
    return {"temperature_kelvin": torch.tensor(record.temperature_kelvin, dtype=torch.float32)}


def ineq_direction_to_signed(ineq_direction: str | None) -> float:
    if ineq_direction == ">":
        return 1.0
    if ineq_direction == "<":
        return -1.0
    return 0.0


def hinge_bound_kcal_mol_for_record(record: MutationRecord) -> float:
    if record.label_type == LabelType.INEQ and record.ddg_kcal_mol is not None:
        return record.ddg_kcal_mol
    return float("nan")


def hinge_direction_for_record(record: MutationRecord) -> float:
    if record.label_type == LabelType.INEQ:
        return ineq_direction_to_signed(record.ineq_direction)
    return 0.0


def classify_mutation_side(record: MutationRecord) -> str:
    roles = {mutation.chain_role for mutation in record.mutations}
    if roles == {ChainRole.ANTIGEN}:
        return "antigen_only"
    if roles <= {ChainRole.HEAVY, ChainRole.LIGHT}:
        return "antibody_only"
    return "both"


def mutation_side_id_for_record(record: MutationRecord) -> int:
    return MUTATION_SIDE_TO_ID[classify_mutation_side(record)]


def label_fields_for_record(record: MutationRecord, train_complex_names: frozenset[str]) -> dict:
    return {
        "label_type_id": torch.tensor(LABEL_TYPE_TO_ID[record.label_type], dtype=torch.long),
        "target_bin": torch.tensor(int(record.ddg_bin) if record.ddg_bin is not None else 0, dtype=torch.long),
        "target_ddg": torch.tensor(record.ddg_kcal_mol if record.ddg_kcal_mol is not None else 0.0, dtype=torch.float32),
        "bound_kcal_mol": torch.tensor(hinge_bound_kcal_mol_for_record(record), dtype=torch.float32),
        "direction": torch.tensor(hinge_direction_for_record(record), dtype=torch.float32),
        "complex_seen_in_train": torch.tensor(
            normalize_protein_name(record.complex_name or "") in train_complex_names, dtype=torch.bool
        ),
        "mutation_side_id": torch.tensor(mutation_side_id_for_record(record), dtype=torch.long),
    }


class MutationCsvDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        iteration: LossIteration = ACTIVE_LOSS_ITERATION,
        train_complex_names: frozenset[str] = frozenset(),
    ):
        self.records = filter_records_for_active_iteration(load_records(csv_path), iteration)
        self.sample_chain_maps = load_sample_chain_maps()
        self.train_complex_names = train_complex_names

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[dict, dict]:
        record = self.records[index]
        cached = load_or_compute_cached_tensors(record, self.sample_chain_maps)
        input_fields = {**select_model_fields(cached), **metadata_fields_for_record(record)}
        return input_fields, label_fields_for_record(record, self.train_complex_names)
