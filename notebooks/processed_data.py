"""Loader for real processed MutationRecord data.

Real records live as four per-split CSVs named in `shared.constants.SPLIT_FILES`: two
independent splitting schemes (`held_out_pdb`, `same_pdb_allowed`), each divided into
train/val. The same underlying row is assigned independently under both schemes (e.g.
train under `held_out_pdb`, val under `same_pdb_allowed`), so it is written out verbatim
into two of the four files -- naively concatenating all four would double-count every
record for population-level dataset exploration. This module concatenates, tags each row
with which split/subset it was read from, deduplicates full rows (see `deduplicate_rows`
for why `sample_id` alone is not a safe dedup key), then explodes each sample's
JSON-encoded `mutations` column into one row per point mutation -- a SKEMPI multi-mutant
is one CSV row but several distinct mutated residues, each with its own
`interface_region`/`is_alanine_scanning`/residue identity, which is the granularity the
EDA notebooks in this directory analyze.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shared.constants import SPLIT_FILES, SPLIT_SUBSETS, SplitName  # noqa: E402

SPLIT_SCHEME_COLUMN = "split_scheme"
SPLIT_SUBSET_COLUMN = "split_subset"
SAMPLE_LEVEL_COLUMNS = [
    "sample_id",
    "pdb_id",
    "complex_name",
    "label_type",
    "ddg_kcal_mol",
    "ddg_bin",
    "ineq_direction",
    "temperature_kelvin",
]


def iter_split_file_locations() -> list[tuple[SplitName, str, Path]]:
    return [
        (split, subset, SPLIT_FILES[split][subset])
        for split in SplitName
        for subset in SPLIT_SUBSETS
    ]


def read_split_csv(split: SplitName, subset: str, path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame[SPLIT_SCHEME_COLUMN] = split.value
    frame[SPLIT_SUBSET_COLUMN] = subset
    return frame


def read_existing_split_frames() -> list[pd.DataFrame]:
    return [
        read_split_csv(split, subset, path)
        for split, subset, path in iter_split_file_locations()
        if path.exists()
    ]


def raise_missing_split_files_error() -> None:
    expected_paths = [str(path) for _, _, path in iter_split_file_locations()]
    raise FileNotFoundError(
        f"None of the expected processed split files exist yet: {expected_paths}. "
        "Run the pre-processing pipeline first (see README.md)."
    )


def deduplicate_rows(data: pd.DataFrame) -> pd.DataFrame:
    """Collapse rows that are identical across every `MutationRecord` field.

    A given sample is assigned independently under both splitting schemes (e.g. train
    under `held_out_pdb`, val under `same_pdb_allowed`), so its row is written out
    verbatim into two of the four source files. Matching on `sample_id` alone is safe
    here since it uniquely identifies a sample-level row (unlike the exploded
    per-mutation rows produced later by `explode_mutations`).
    """
    return data.drop_duplicates(subset=["sample_id"], keep="first").reset_index(drop=True)


def explode_mutations(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for record in data.to_dict("records"):
        for mutation in json.loads(record["mutations"]):
            rows.append({**{col: record[col] for col in SAMPLE_LEVEL_COLUMNS}, **mutation})
    return pd.DataFrame(rows)


def load_processed_dataframe() -> pd.DataFrame:
    frames = read_existing_split_frames()
    if not frames:
        raise_missing_split_files_error()
    combined = deduplicate_rows(pd.concat(frames, ignore_index=True))
    return explode_mutations(combined)
