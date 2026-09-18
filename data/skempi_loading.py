"""Loading the raw SKEMPI CSV and filtering to the antibody-antigen subset."""

from __future__ import annotations

import csv
from pathlib import Path

from shared.constants import (
    ANTIBODY_ANTIGEN_HALLMARK_COLUMN,
    ANTIBODY_ANTIGEN_HALLMARK_VALUE,
    SKEMPI_RAW_CSV,
)
from data.utils.constants import HOLD_OUT_TYPE_SEPARATOR, SKEMPI_CSV_DELIMITER


def load_skempi_rows(csv_path: Path = SKEMPI_RAW_CSV) -> list[dict]:
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=SKEMPI_CSV_DELIMITER)
        return list(reader)


def row_matches_antibody_antigen_hallmark(row: dict) -> bool:
    hallmark_tokens = row.get(ANTIBODY_ANTIGEN_HALLMARK_COLUMN, "").split(HOLD_OUT_TYPE_SEPARATOR)
    return ANTIBODY_ANTIGEN_HALLMARK_VALUE in hallmark_tokens


def filter_antibody_antigen_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row_matches_antibody_antigen_hallmark(row)]


def load_antibody_antigen_rows(csv_path: Path = SKEMPI_RAW_CSV) -> list[dict]:
    return filter_antibody_antigen_rows(load_skempi_rows(csv_path))
