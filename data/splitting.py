"""Held-out-complex and same-complex-allowed train/val split assignment.

Both splits group on `MutationRecord.complex_name` (normalized via
`data.homology_dedup.normalize_protein_name`) -- the actual identity of "the
same antibody-antigen pair," since SKEMPI frequently re-crystallizes the same
complex under a different `pdb_id` with a different mutation (verified: 18%
of `held_out_pdb`'s old `pdb_id`-grouped val rows shared a `complex_name` with
a train row under a different PDB code). `pdb_id` is never used as a grouping
key here any more.

`held_out_pdb` groups every record of a complex onto the same side (no
complex appears on both sides at all). `same_pdb_allowed` is a two-tier
assignment: a subset of complexes is accumulated (by sample count, in stable
hash order) into an entirely held-out bucket until it reaches
`SAME_PDB_ALLOWED_NEW_COMPLEX_FRACTION` of the total sample count -- so val
always contains some genuinely new complexes -- then every other ("shared")
complex's samples are individually split at
`SAME_PDB_ALLOWED_SHARED_SAMPLE_FRACTION`, so the same complex (even the same
PDB) can appear on both sides with a different mutation, as before.

Homology-group siblings (see `data.homology_dedup`) are discarded from the
record set entirely by `data.pipeline` before either split function ever
runs, so a given `(complex_name, mutation set)` never appears twice here.
"""

from __future__ import annotations

import csv
import hashlib
import json

from data.homology_dedup import normalize_protein_name
from shared.constants import (
    MutationRecord,
    RANDOM_SEED,
    SPLIT_FILES,
    SplitName,
)
from data.utils.constants import (
    DEFAULT_VAL_FRACTION,
    SAME_PDB_ALLOWED_NEW_COMPLEX_FRACTION,
    SAME_PDB_ALLOWED_SHARED_SAMPLE_FRACTION,
)


def _stable_unit_interval_hash(key: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(1 << 64)


def _assign_subset(key: str, seed: int, val_fraction: float) -> str:
    return "val" if _stable_unit_interval_hash(key, seed) < val_fraction else "train"


def _record_complex_key(record: MutationRecord) -> str:
    return normalize_protein_name(record.complex_name or "")


def assign_held_out_pdb_split(
    records: list[MutationRecord], val_fraction: float = DEFAULT_VAL_FRACTION, seed: int = RANDOM_SEED
) -> dict[str, str]:
    complex_keys = sorted({_record_complex_key(record) for record in records})
    complex_to_subset = {complex_key: _assign_subset(complex_key, seed, val_fraction) for complex_key in complex_keys}
    return {record.sample_id: complex_to_subset[_record_complex_key(record)] for record in records}


def _held_out_complexes(records: list[MutationRecord], new_complex_fraction: float, seed: int) -> set[str]:
    complex_sample_counts: dict[str, int] = {}
    for record in records:
        complex_key = _record_complex_key(record)
        complex_sample_counts[complex_key] = complex_sample_counts.get(complex_key, 0) + 1

    ordered_complex_keys = sorted(complex_sample_counts, key=lambda complex_key: _stable_unit_interval_hash(complex_key, seed))
    target_sample_count = new_complex_fraction * len(records)

    held_out: set[str] = set()
    accumulated_sample_count = 0
    for complex_key in ordered_complex_keys:
        if accumulated_sample_count >= target_sample_count:
            break
        held_out.add(complex_key)
        accumulated_sample_count += complex_sample_counts[complex_key]
    return held_out


def assign_same_pdb_allowed_split(
    records: list[MutationRecord],
    new_complex_fraction: float = SAME_PDB_ALLOWED_NEW_COMPLEX_FRACTION,
    shared_sample_fraction: float = SAME_PDB_ALLOWED_SHARED_SAMPLE_FRACTION,
    seed: int = RANDOM_SEED,
) -> dict[str, str]:
    held_out_complexes = _held_out_complexes(records, new_complex_fraction, seed)
    assignment: dict[str, str] = {}
    for record in records:
        if _record_complex_key(record) in held_out_complexes:
            assignment[record.sample_id] = "val"
        else:
            assignment[record.sample_id] = _assign_subset(record.sample_id, seed, shared_sample_fraction)
    return assignment


def apply_split_memberships(records: list[MutationRecord]) -> None:
    held_out_pdb_assignment = assign_held_out_pdb_split(records)
    same_pdb_allowed_assignment = assign_same_pdb_allowed_split(records)
    for record in records:
        record.split_membership[SplitName.HELD_OUT_PDB.value] = held_out_pdb_assignment[record.sample_id]
        record.split_membership[SplitName.SAME_PDB_ALLOWED.value] = same_pdb_allowed_assignment[record.sample_id]


def mutation_record_csv_fieldnames() -> list[str]:
    return [name for name in MutationRecord.model_fields if name != "split_membership"]


def mutation_record_to_csv_row(record: MutationRecord) -> dict:
    """Flattens a `MutationRecord` to one CSV row. `mutations` (a list of
    `PointMutation`, always >= 1) has no natural single-column
    representation, so it is JSON-encoded into one string column -- the
    simplest option that survives a CSV round-trip without ambiguity;
    consumers should `json.loads` that column back into a list of dicts
    shaped like `PointMutation`.
    """
    dumped = record.model_dump(mode="json")
    dumped.pop("split_membership", None)
    dumped["mutations"] = json.dumps(dumped["mutations"])
    return dumped


def records_for_split_subset(records: list[MutationRecord], split: SplitName, subset: str) -> list[MutationRecord]:
    return [record for record in records if record.split_membership.get(split.value) == subset]


def write_split_files(records: list[MutationRecord]) -> None:
    fieldnames = mutation_record_csv_fieldnames()
    for split in SplitName:
        for subset in ("train", "val"):
            subset_records = records_for_split_subset(records, split, subset)
            destination = SPLIT_FILES[split][subset]
            destination.parent.mkdir(parents=True, exist_ok=True)
            with open(destination, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for record in sorted(subset_records, key=lambda r: r.sample_id):
                    writer.writerow(mutation_record_to_csv_row(record))
