"""Entry point for the (heavy, ML-backed) embedding-computation stage,
separate from `data.pipeline` (metadata/splits only, no ML) so a CPU-only
dev run can exercise the fast stage fully and the slow stage on a small
subset -- see the pipeline report for what actually ran end-to-end locally.
"""

from __future__ import annotations

from data.embedding_pipeline.entry_embeddings import compute_and_cache_entry_embeddings
from data.processed_io import load_mutation_records, load_sample_chain_maps
from shared.constants import MutationRecord


def run_embedding_pipeline(
    records: list[MutationRecord] | None = None,
    sample_chain_maps: dict[str, dict[str, str]] | None = None,
    sample_id_limit: int | None = None,
) -> list[str]:
    records = records if records is not None else load_mutation_records()
    sample_chain_maps = sample_chain_maps if sample_chain_maps is not None else load_sample_chain_maps()

    records_to_process = records[:sample_id_limit] if sample_id_limit is not None else records

    processed_sample_ids = []
    for record in records_to_process:
        compute_and_cache_entry_embeddings(record, sample_chain_maps[record.sample_id])
        processed_sample_ids.append(record.sample_id)
    return processed_sample_ids
