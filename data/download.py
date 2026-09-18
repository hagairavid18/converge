"""Download the raw SKEMPI 2.0 mutation/affinity table (the CSV, from
SKEMPI's own site) and the PDB structures it references (from RCSB, see
`data.structures` for why RCSB rather than SKEMPI's bundled structures.tgz).
"""

from __future__ import annotations

from pathlib import Path

from shared.constants import SKEMPI_DOWNLOAD_URL, SKEMPI_RAW_CSV
from data.structures import skempi_structure_path
from data.utils.constants import RCSB_PDB_DOWNLOAD_URL_TEMPLATE
from data.utils.io import download_file


def download_skempi_csv(overwrite: bool = False) -> Path:
    return download_file(SKEMPI_DOWNLOAD_URL, SKEMPI_RAW_CSV, overwrite=overwrite)


def download_structure(pdb_id: str, overwrite: bool = False) -> Path:
    url = RCSB_PDB_DOWNLOAD_URL_TEMPLATE.format(pdb_id=pdb_id.upper())
    return download_file(url, skempi_structure_path(pdb_id), overwrite=overwrite)


def ensure_structures_for_pdb_ids(pdb_ids: list[str], overwrite: bool = False) -> list[str]:
    failed_pdb_ids = []
    for pdb_id in sorted(set(pdb_ids)):
        try:
            download_structure(pdb_id, overwrite=overwrite)
        except Exception:
            failed_pdb_ids.append(pdb_id)
    return failed_pdb_ids
