"""Reading and writing the processed `MutationRecord` dataset."""

from __future__ import annotations

import json
from pathlib import Path

from shared.constants import PROCESSED_DATA_DIR, MutationRecord
from data.utils.io import read_jsonl, write_jsonl

PROCESSED_RECORDS_PATH = PROCESSED_DATA_DIR / "mutation_records.jsonl"
SAMPLE_CHAIN_MAP_PATH = PROCESSED_DATA_DIR / "sample_chain_maps.json"


def save_mutation_records(records: list[MutationRecord], destination: Path = PROCESSED_RECORDS_PATH) -> None:
    write_jsonl(records, destination, lambda record: record.model_dump_json())


def load_mutation_records(source: Path = PROCESSED_RECORDS_PATH) -> list[MutationRecord]:
    return read_jsonl(source, MutationRecord.model_validate_json)


def save_sample_chain_maps(
    sample_chain_maps: dict[str, dict[str, str]], destination: Path = SAMPLE_CHAIN_MAP_PATH
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8") as f:
        json.dump(sample_chain_maps, f, indent=2, sort_keys=True)


def load_sample_chain_maps(source: Path = SAMPLE_CHAIN_MAP_PATH) -> dict[str, dict[str, str]]:
    with open(source, encoding="utf-8") as f:
        return json.load(f)
