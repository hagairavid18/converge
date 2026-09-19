"""Per-entry SaProt embedding computation and caching, into its own
`structure_saprot/` cache namespace (`data.embedding_pipeline.cache`).
`entry_embeddings.apply_saprot_structure_override` calls
`compute_wt_saprot_bundle`/`compute_mut_saprot_bundle` (via this same
`load_or_compute_bundle` caching) as an opt-in swap for the production
IgFold/zero-placeholder structure embedding, gated on
`shared.constants.USE_SAPROT_STRUCTURE`. See `docs/future_work.md` for the
validated timing.

WT bundle: per `pdb_id`, every chain's SaProt embedding with its real 3Di
token throughout. Mutant bundle: per `sample_id`, each mutated chain has its
mutated residue's 3Di token replaced with SaProt's own "structure unknown"
token (`saprot_backend.SAPROT_STRUCTURE_UNKNOWN_TOKEN`) -- SKEMPI provides no
separately-solved mutant structure, the same WT-geometry-only assumption
`entry_embeddings.py` already documents, made explicit at the token level
here. Foldseek's 3Di output is cached in-process per `pdb_id` (`lru_cache`)
since both bundles need it and it's shared across every sample of that PDB.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch

from data.chain_roles import group_mutations_by_chain_id, ordered_chain_ids_for_sample
from data.embedding_pipeline.cache import load_or_compute_bundle, saprot_mut_cache_path, saprot_wt_cache_path
from data.embedding_pipeline.foldseek_util import compute_3di_sequences_by_chain
from data.embedding_pipeline.saprot_backend import compute_saprot_embeddings
from data.structures import chain_residue_position_index, chain_sequence, get_chain, load_structure, skempi_structure_path
from shared.constants import ACTIVE_SAPROT_HIDDEN_DIM, MutationRecord, PointMutation


def _align_to_real_sequence(real_sequence: str, aa_sequence: str, di_sequence: str) -> tuple[str, str]:
    """Foldseek parses the PDB file directly and can include a residue our
    own `data.structures.chain_sequence` excludes as non-standard/hetero --
    observed for an N-terminal PCA (pyroglutamate) residue on 2 of 55 PDBs
    in this dataset (3L5X chain H, 4GXU chains M/N), which Foldseek reads as
    a plain glutamate and prepends. Since every downstream consumer
    (`PointMutation.flat_residue_index`, `mutation_distances`, the model's
    other embedding branches) indexes residues via `chain_sequence`'s
    numbering, a silent length mismatch here would misalign the SaProt
    embedding against every other per-residue field for that chain -- so
    this trims Foldseek's sequence down to `chain_sequence`'s indexing
    (only ever a small, exact prefix trim; anything else raises rather than
    guessing) instead of assuming they already agree.
    """
    if aa_sequence == real_sequence:
        return aa_sequence, di_sequence
    offset = len(aa_sequence) - len(real_sequence)
    if offset > 0 and aa_sequence[offset:] == real_sequence:
        return aa_sequence[offset:], di_sequence[offset:]
    raise ValueError(f"Foldseek sequence cannot be aligned to the real chain sequence: real={real_sequence!r} foldseek={aa_sequence!r}")


@lru_cache(maxsize=64)
def _di_sequences_for_pdb(pdb_id: str) -> dict[str, tuple[str, str]]:
    raw = compute_3di_sequences_by_chain(skempi_structure_path(pdb_id))
    structure = load_structure(pdb_id)
    return {
        chain_id: _align_to_real_sequence(chain_sequence(get_chain(structure, chain_id)), aa_sequence, di_sequence)
        for chain_id, (aa_sequence, di_sequence) in raw.items()
    }


def _as_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(array)).float()


def _mutated_local_positions(chain, mutations: list[PointMutation]) -> frozenset[int]:
    position_index = chain_residue_position_index(chain)
    return frozenset(position_index[(mutation.residue_position, (mutation.insertion_code or "").upper())] for mutation in mutations)


def compute_wt_saprot_bundle(pdb_id: str, ordered_chain_ids: list[str]) -> dict[str, torch.Tensor]:
    di_sequences_by_chain = _di_sequences_for_pdb(pdb_id)
    embeddings = [compute_saprot_embeddings(*di_sequences_by_chain[chain_id]) for chain_id in ordered_chain_ids]
    return {"wt_saprot_embedding": _as_tensor(np.concatenate(embeddings, axis=0))}


def compute_mut_saprot_bundle(
    pdb_id: str, ordered_chain_ids: list[str], mutations_by_chain_id: dict[str, list[PointMutation]]
) -> dict[str, torch.Tensor]:
    di_sequences_by_chain = _di_sequences_for_pdb(pdb_id)
    structure = load_structure(pdb_id)
    embeddings = []
    for chain_id in ordered_chain_ids:
        aa_sequence, di_sequence = di_sequences_by_chain[chain_id]
        mutations = mutations_by_chain_id.get(chain_id, [])
        masked_positions = _mutated_local_positions(get_chain(structure, chain_id), mutations) if mutations else frozenset()
        embeddings.append(compute_saprot_embeddings(aa_sequence, di_sequence, masked_positions))
    return {"mut_saprot_embedding": _as_tensor(np.concatenate(embeddings, axis=0))}


def compute_and_cache_saprot_entry_embeddings(record: MutationRecord, chain_map: dict[str, str]) -> dict[str, torch.Tensor]:
    ordered_chain_ids = ordered_chain_ids_for_sample(chain_map)
    mutations_by_chain_id = group_mutations_by_chain_id(record.mutations)

    wt_bundle = load_or_compute_bundle(
        saprot_wt_cache_path(record.pdb_id),
        ("wt_saprot_embedding",),
        lambda: compute_wt_saprot_bundle(record.pdb_id, ordered_chain_ids),
        {"wt_saprot_embedding": ACTIVE_SAPROT_HIDDEN_DIM},
    )
    mut_bundle = load_or_compute_bundle(
        saprot_mut_cache_path(record.sample_id),
        ("mut_saprot_embedding",),
        lambda: compute_mut_saprot_bundle(record.pdb_id, ordered_chain_ids, mutations_by_chain_id),
        {"mut_saprot_embedding": ACTIVE_SAPROT_HIDDEN_DIM},
    )
    return {**wt_bundle, **mut_bundle}


def compute_and_cache_all_saprot_embeddings(records: list[MutationRecord], sample_chain_maps: dict[str, dict[str, str]]) -> None:
    for record in records:
        compute_and_cache_saprot_entry_embeddings(record, sample_chain_maps[record.sample_id])
