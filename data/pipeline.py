"""End-to-end orchestration: download -> load/filter -> build records ->
flat-index assignment -> discard homology siblings -> splits -> write
processed dataset.
"""

from __future__ import annotations

from pydantic import BaseModel

from data.download import download_skempi_csv, ensure_structures_for_pdb_ids
from data.flat_indexing import assign_flat_residue_indices
from data.homology_dedup import discard_homology_siblings
from data.processed_io import save_mutation_records, save_sample_chain_maps
from data.record_builder import build_all_mutation_records
from data.skempi_loading import load_antibody_antigen_rows
from data.splitting import apply_split_memberships, write_split_files
from data.utils.constants import COLUMN_PDB
from shared.constants import MutationRecord


class PipelineReport(BaseModel):
    num_antibody_antigen_rows: int
    num_samples: int
    num_homology_sibling_samples_discarded: int
    num_skipped_mutation_tokens: int
    num_row_errors: int
    num_structure_download_failures: int
    skipped_mutation_tokens: list[tuple[str, str]]
    row_errors: list[str]


def run_preprocessing_pipeline(download: bool = True) -> tuple[list[MutationRecord], PipelineReport]:
    if download:
        download_skempi_csv()

    rows = load_antibody_antigen_rows()

    failed_structure_downloads = []
    if download:
        pdb_ids = [row[COLUMN_PDB].split("_")[0] for row in rows]
        failed_structure_downloads = ensure_structures_for_pdb_ids(pdb_ids)

    build_result = build_all_mutation_records(rows)
    assign_flat_residue_indices(build_result.records, build_result.sample_chain_maps)
    records = discard_homology_siblings(build_result.records)
    num_homology_sibling_samples_discarded = len(build_result.records) - len(records)
    apply_split_memberships(records)

    save_mutation_records(records)
    write_split_files(records)
    save_sample_chain_maps(build_result.sample_chain_maps)

    report = PipelineReport(
        num_antibody_antigen_rows=len(rows),
        num_samples=len(records),
        num_homology_sibling_samples_discarded=num_homology_sibling_samples_discarded,
        num_skipped_mutation_tokens=len(build_result.skipped_mutation_tokens),
        num_row_errors=len(build_result.row_errors),
        num_structure_download_failures=len(failed_structure_downloads),
        skipped_mutation_tokens=build_result.skipped_mutation_tokens,
        row_errors=build_result.row_errors,
    )
    return records, report
