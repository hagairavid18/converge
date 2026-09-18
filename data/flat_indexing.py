"""Assigns each `PointMutation.flat_residue_index`: the 0-based index of
that mutated residue within its sample's concatenated (heavy+light+antigen)
per-residue cache tensor -- see `dl.datasets.mutation_csv_dataset`, which
indexes `residue_coordinates` directly by this value to locate a Gaussian
pooling center. Needs only real structure (no ML backends), so it runs in
the fast metadata pass, before embedding computation.
"""

from __future__ import annotations

from data.chain_roles import ordered_chain_ids_for_sample
from data.structures import chain_residue_position_index, get_chain, load_structure
from shared.constants import MutationRecord


def compute_chain_offsets(structure, chain_ids: list[str]) -> dict[str, int]:
    offsets = {}
    cumulative = 0
    for chain_id in chain_ids:
        offsets[chain_id] = cumulative
        cumulative += len(chain_residue_position_index(get_chain(structure, chain_id)))
    return offsets


def assign_flat_residue_indices(
    records: list[MutationRecord], sample_chain_maps: dict[str, dict[str, str]]
) -> None:
    structures_by_pdb_id = {}
    offsets_by_sample_id: dict[str, dict[str, int]] = {}
    position_index_by_sample_chain: dict[tuple[str, str], dict] = {}

    for record in records:
        chain_map = sample_chain_maps[record.sample_id]
        if record.sample_id not in offsets_by_sample_id:
            structure = structures_by_pdb_id.setdefault(record.pdb_id, load_structure(record.pdb_id))
            ordered_chain_ids = ordered_chain_ids_for_sample(chain_map)
            offsets_by_sample_id[record.sample_id] = compute_chain_offsets(structure, ordered_chain_ids)

        structure = structures_by_pdb_id[record.pdb_id]
        for mutation in record.mutations:
            cache_key = (record.sample_id, mutation.chain_id)
            if cache_key not in position_index_by_sample_chain:
                position_index_by_sample_chain[cache_key] = chain_residue_position_index(
                    get_chain(structure, mutation.chain_id)
                )

            insertion_code = (mutation.insertion_code or "").upper()
            local_index = position_index_by_sample_chain[cache_key][(mutation.residue_position, insertion_code)]
            chain_offset = offsets_by_sample_id[record.sample_id][mutation.chain_id]
            mutation.flat_residue_index = chain_offset + local_index
