"""Per-entry (sample_id) embedding computation: for every chain (heavy,
light, antigen) of the complex, compute wild-type-branch and mutant-branch
embeddings/confidences, concatenate them into one whole-complex per-residue
tensor set (see `data.chain_roles.ordered_chain_ids_for_sample` for the
fixed concatenation order), and cache one file per sample -- matching the
`dl.datasets.mutation_csv_dataset` cache contract (see
`data.embedding_pipeline.cache`).

Each chain is still run through the structure/sequence models separately,
never concatenated before inference, per the Implementation Spec's "as
separate chains, not concatenated" -- only the resulting per-residue
*outputs* are concatenated afterwards, for storage/batching convenience.

Per-residue interface-region weighting was dropped from this cache: the
model's pooling is pure distance-based Gaussian falloff (no consumer of a
whole-chain interface classification remains -- see
`data.mutation_tokens.interface_region_from_code` for the only place that
classification is still used, directly from SKEMPI's own column, for
`PointMutation.interface_region`). In its place, `mutation_distances` caches
the raw (not sigma-weighted) per-residue Euclidean distance from each point
mutation's wild-type CA coordinate to every residue in the complex, so the
Gaussian falloff can be applied live at train time with a configurable sigma
without recomputing distances from scratch.

`shared.constants.ACTIVE_EMBEDDING_SOURCE_MODE` gates which backend(s)
actually run, not just which fields the model reads: under `SEQUENCE_ONLY`,
ESMFold is never invoked at all (it is the ~100x-larger, much slower model),
so `wt`/`mut_structure_*` keys are simply absent from a sample's cache file
rather than computed and discarded -- `dl.models.schemas.ModelInputs`
already treats those fields as `Optional`/`None` for exactly this case.
Structure embeddings for samples processed under `SEQUENCE_ONLY` do not
exist until deliberately backfilled later (e.g. on a GPU); this is an
accepted, deliberate gap, not an oversight. Symmetrically, `STRUCTURE_ONLY`
skips ESM-2. `real_structure_features` and `residue_coordinates` are cheap,
non-ML, real-geometry outputs and are always computed regardless of mode.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from data.chain_roles import ordered_chain_ids_for_sample
from data.embedding_pipeline.cache import save_sample_embedding_bundle
from data.embedding_pipeline.mutant_sequence import mutant_sequence_for_chain
from data.embedding_pipeline.real_structure_features import compute_real_structure_features
from data.embedding_pipeline.sequence_backend import (
    compute_sequence_embeddings,
    compute_sequence_pseudo_log_likelihood,
)
from data.embedding_pipeline.structure_backend import (
    compute_mutant_structural_embedding,
    compute_wild_type_structural_embedding,
)
from data.structures import chain_ca_coordinates, chain_sequence, get_chain, load_structure
from shared.constants import ACTIVE_EMBEDDING_SOURCE_MODE, EmbeddingSourceMode, MutationRecord, PointMutation


def should_compute_sequence_embeddings() -> bool:
    return ACTIVE_EMBEDDING_SOURCE_MODE != EmbeddingSourceMode.STRUCTURE_ONLY


def should_compute_structure_embeddings() -> bool:
    return ACTIVE_EMBEDDING_SOURCE_MODE != EmbeddingSourceMode.SEQUENCE_ONLY


def group_mutations_by_chain_id(mutations: list[PointMutation]) -> dict[str, list[PointMutation]]:
    grouped = defaultdict(list)
    for mutation in mutations:
        grouped[mutation.chain_id].append(mutation)
    return grouped


def compute_wild_type_chain_bundle(structure, chain_id: str) -> dict:
    chain = get_chain(structure, chain_id)
    sequence = chain_sequence(chain)
    bundle = {
        "real_structure_features": compute_real_structure_features(structure, chain_id),
        "residue_coordinates": chain_ca_coordinates(chain),
    }

    if should_compute_sequence_embeddings():
        pseudo_log_likelihood = compute_sequence_pseudo_log_likelihood(sequence)
        bundle["sequence_embedding"] = compute_sequence_embeddings(sequence)
        bundle["sequence_confidence"] = np.exp(pseudo_log_likelihood)
        bundle["sequence_pseudo_log_likelihood"] = pseudo_log_likelihood

    if should_compute_structure_embeddings():
        structural_embedding, structural_plddt_diagnostic = compute_wild_type_structural_embedding(sequence)
        bundle["structure_embedding"] = structural_embedding
        bundle["structure_plddt_diagnostic"] = structural_plddt_diagnostic

    return bundle


def compute_mutant_chain_bundle(structure, chain_id: str, mutations: list[PointMutation]) -> dict:
    chain = get_chain(structure, chain_id)
    mutant_sequence = mutant_sequence_for_chain(chain, mutations)
    bundle = {}

    if should_compute_sequence_embeddings():
        pseudo_log_likelihood = compute_sequence_pseudo_log_likelihood(mutant_sequence)
        bundle["sequence_embedding"] = compute_sequence_embeddings(mutant_sequence)
        bundle["sequence_confidence"] = np.exp(pseudo_log_likelihood)
        bundle["sequence_pseudo_log_likelihood"] = pseudo_log_likelihood

    if should_compute_structure_embeddings():
        structural_embedding, structural_confidence = compute_mutant_structural_embedding(mutant_sequence)
        bundle["structure_embedding"] = structural_embedding
        bundle["structure_confidence"] = structural_confidence

    return bundle


def _concatenate_field(chain_bundles: list[dict], field: str) -> np.ndarray:
    return np.concatenate([bundle[field] for bundle in chain_bundles], axis=0)


def _as_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(array))


def compute_mutation_distances(residue_coordinates: np.ndarray, mutations: list[PointMutation]) -> np.ndarray:
    """Raw per-residue Euclidean distance (Angstroms) from each mutation's
    wild-type CA coordinate to every residue in the sample's concatenated
    complex. Shape `[num_mutations, L]`, ordered the same as `mutations`.
    """
    mutation_site_coordinates = residue_coordinates[[m.flat_residue_index for m in mutations]]
    return np.linalg.norm(residue_coordinates[None, :, :] - mutation_site_coordinates[:, None, :], axis=-1)


def build_sample_tensor_bundle(
    wt_chain_bundles: list[dict], mut_chain_bundles: list[dict], mutations: list[PointMutation]
) -> dict[str, torch.Tensor]:
    residue_coordinates = _concatenate_field(wt_chain_bundles, "residue_coordinates")
    tensor_bundle = {
        "residue_coordinates": _as_tensor(residue_coordinates).float(),
        "mutation_distances": _as_tensor(compute_mutation_distances(residue_coordinates, mutations)).float(),
        "wt_real_structure_features": _as_tensor(
            _concatenate_field(wt_chain_bundles, "real_structure_features")
        ).float(),
    }

    if should_compute_sequence_embeddings():
        tensor_bundle["wt_sequence_embedding"] = _as_tensor(
            _concatenate_field(wt_chain_bundles, "sequence_embedding")
        ).float()
        tensor_bundle["wt_sequence_confidence"] = _as_tensor(
            _concatenate_field(wt_chain_bundles, "sequence_confidence")
        ).float()
        tensor_bundle["wt_sequence_pseudo_log_likelihood"] = _as_tensor(
            _concatenate_field(wt_chain_bundles, "sequence_pseudo_log_likelihood")
        ).float()
        tensor_bundle["mut_sequence_embedding"] = _as_tensor(
            _concatenate_field(mut_chain_bundles, "sequence_embedding")
        ).float()
        tensor_bundle["mut_sequence_confidence"] = _as_tensor(
            _concatenate_field(mut_chain_bundles, "sequence_confidence")
        ).float()
        tensor_bundle["mut_sequence_pseudo_log_likelihood"] = _as_tensor(
            _concatenate_field(mut_chain_bundles, "sequence_pseudo_log_likelihood")
        ).float()

    if should_compute_structure_embeddings():
        tensor_bundle["wt_structure_embedding"] = _as_tensor(
            _concatenate_field(wt_chain_bundles, "structure_embedding")
        ).float()
        tensor_bundle["wt_structure_plddt_diagnostic"] = _as_tensor(
            _concatenate_field(wt_chain_bundles, "structure_plddt_diagnostic")
        ).float()
        tensor_bundle["mut_structure_embedding"] = _as_tensor(
            _concatenate_field(mut_chain_bundles, "structure_embedding")
        ).float()
        tensor_bundle["mut_structure_confidence"] = _as_tensor(
            _concatenate_field(mut_chain_bundles, "structure_confidence")
        ).float()

    return tensor_bundle


def compute_and_cache_entry_embeddings(record: MutationRecord, chain_map: dict[str, str]) -> None:
    structure = load_structure(record.pdb_id)
    mutations_by_chain_id = group_mutations_by_chain_id(record.mutations)

    ordered_chain_ids = ordered_chain_ids_for_sample(chain_map)
    wt_chain_bundles = [compute_wild_type_chain_bundle(structure, chain_id) for chain_id in ordered_chain_ids]
    mut_chain_bundles = [
        compute_mutant_chain_bundle(structure, chain_id, mutations_by_chain_id.get(chain_id, []))
        for chain_id in ordered_chain_ids
    ]

    tensor_bundle = build_sample_tensor_bundle(wt_chain_bundles, mut_chain_bundles, record.mutations)
    save_sample_embedding_bundle(record.sample_id, tensor_bundle)


def compute_and_cache_all_embeddings(
    records: list[MutationRecord], sample_chain_maps: dict[str, dict[str, str]]
) -> None:
    for record in records:
        compute_and_cache_entry_embeddings(record, sample_chain_maps[record.sample_id])
