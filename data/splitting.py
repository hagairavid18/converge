"""Held-out-PDB and same-PDB-allowed train/val split assignment.

Both splits partition on `MutationRecord.sample_id` (one measurement -- one
ddG value, one or more `PointMutation`s -- see `data.record_builder`), so a
whole entry always lands on one side of a split. They differ in the grouping
key used before the train/val partition: held-out-PDB groups by `pdb_id`
first (so no PDB appears on both sides at all); same-PDB-allowed partitions
`sample_id`s directly (so the same PDB can appear on both sides, but a given
mutation entry never does).

Homology-group siblings (see `data.homology_dedup`) are then forced into
validation on both splits, never training, as a deliberate consistency
check: one representative per group is left to the normal hash-based
assignment above, and every other member of the group is overridden into
val -- at the single-sample level on *both* splits, including held-out-PDB.
This means held-out-PDB's "no PDB in both" invariant is deliberately broken
for exactly these sibling records: their purpose (comparing predictions on
nominally-the-same mutation deposited under a different PDB) isn't the
generalization-to-an-unseen-PDB purpose that invariant exists for, so
forcing their whole PDB to val (an earlier approach) was solving the wrong
problem -- it cascaded badly whenever a sibling happened to share a PDB with
many unrelated ("hub") mutations. Every other (non-sibling) sample from that
same PDB is unaffected and stays wherever the ordinary per-PDB hash puts it,
train included.
"""

from __future__ import annotations

import csv
import hashlib
import json

from shared.constants import (
    MutationRecord,
    RANDOM_SEED,
    SPLIT_FILES,
    SplitName,
)
from data.homology_dedup import homology_sibling_sample_ids
from data.utils.constants import DEFAULT_VAL_FRACTION


def _stable_unit_interval_hash(key: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(1 << 64)


def _assign_subset(key: str, seed: int, val_fraction: float) -> str:
    return "val" if _stable_unit_interval_hash(key, seed) < val_fraction else "train"


def assign_held_out_pdb_split(
    records: list[MutationRecord], val_fraction: float = DEFAULT_VAL_FRACTION, seed: int = RANDOM_SEED
) -> dict[str, str]:
    pdb_ids = sorted({record.pdb_id for record in records})
    pdb_to_subset = {pdb_id: _assign_subset(pdb_id, seed, val_fraction) for pdb_id in pdb_ids}
    assignment = {record.sample_id: pdb_to_subset[record.pdb_id] for record in records}

    for sample_id in homology_sibling_sample_ids(records):
        assignment[sample_id] = "val"

    return assignment


def assign_same_pdb_allowed_split(
    records: list[MutationRecord], val_fraction: float = DEFAULT_VAL_FRACTION, seed: int = RANDOM_SEED
) -> dict[str, str]:
    sample_ids = sorted({record.sample_id for record in records})
    assignment = {sample_id: _assign_subset(sample_id, seed, val_fraction) for sample_id in sample_ids}

    for sample_id in homology_sibling_sample_ids(records):
        assignment[sample_id] = "val"

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
