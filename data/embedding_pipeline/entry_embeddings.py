"""Per-entry (sample_id) embedding computation: for every chain (heavy,
light, antigen) of the complex, compute wild-type-branch and mutant-branch
embeddings/confidences, concatenate them into one whole-complex per-residue
tensor set (see `data.chain_roles.ordered_chain_ids_for_sample` for the
fixed concatenation order), and cache one file per sample -- matching the
`dl.datasets.mutation_csv_dataset` cache contract (see
`data.embedding_pipeline.cache`).

Structure embeddings use a mixed backbone by `ChainRole` (see
`data.embedding_pipeline.structure_backend`): IgFold for heavy/light chains
(called once for both together, since IgFold models VH-VL pairing jointly),
ESMFold for the antigen chain. Sequence embeddings (ESM-2) are unchanged and
still run per chain independently, uniformly across all three roles. Either
way, chains are never concatenated before inference -- only the resulting
per-residue *outputs* are concatenated afterwards, for storage/batching
convenience.

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
neither IgFold nor ESMFold is invoked at all, so `wt`/`mut_structure_*` keys
are simply absent from a sample's cache file rather than computed and
discarded -- `dl.models.schemas.ModelInputs` already treats those fields as
`Optional`/`None` for exactly this case. Structure embeddings for samples
processed under `SEQUENCE_ONLY` do not exist until deliberately backfilled
later; this is an accepted, deliberate gap, not an oversight. Symmetrically,
`STRUCTURE_ONLY` skips ESM-2. `real_structure_features` and
`residue_coordinates` are cheap, non-ML, real-geometry outputs and are
always computed regardless of mode.
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
    compute_mutant_antibody_structural_embeddings,
    compute_mutant_structural_embedding,
    compute_wild_type_antibody_structural_embeddings,
    compute_wild_type_structural_embedding,
)
from data.structures import chain_ca_coordinates, chain_sequence, get_chain, load_structure
from shared.constants import (
    ACTIVE_EMBEDDING_SOURCE_MODE,
    ChainRole,
    EmbeddingSourceMode,
    MutationRecord,
    PointMutation,
)

ANTIBODY_CHAIN_ROLES = (ChainRole.HEAVY, ChainRole.LIGHT)


def should_compute_sequence_embeddings() -> bool:
    return ACTIVE_EMBEDDING_SOURCE_MODE != EmbeddingSourceMode.STRUCTURE_ONLY


def should_compute_structure_embeddings() -> bool:
    return ACTIVE_EMBEDDING_SOURCE_MODE != EmbeddingSourceMode.SEQUENCE_ONLY


def group_mutations_by_chain_id(mutations: list[PointMutation]) -> dict[str, list[PointMutation]]:
    grouped = defaultdict(list)
    for mutation in mutations:
        grouped[mutation.chain_id].append(mutation)
    return grouped


def invert_chain_map(chain_map: dict[str, str]) -> dict[str, ChainRole]:
    return {chain_id: ChainRole(role_value) for role_value, chain_ids in chain_map.items() for chain_id in chain_ids}


def split_chain_ids_by_backbone(
    ordered_chain_ids: list[str], chain_role_by_id: dict[str, ChainRole]
) -> tuple[list[str], list[str]]:
    antibody_chain_ids = [c for c in ordered_chain_ids if chain_role_by_id[c] in ANTIBODY_CHAIN_ROLES]
    antigen_chain_ids = [c for c in ordered_chain_ids if chain_role_by_id[c] == ChainRole.ANTIGEN]
    return antibody_chain_ids, antigen_chain_ids


def compute_sequence_features(sequence: str) -> dict:
    pseudo_log_likelihood = compute_sequence_pseudo_log_likelihood(sequence)
    return {
        "sequence_embedding": compute_sequence_embeddings(sequence),
        "sequence_confidence": np.exp(pseudo_log_likelihood),
        "sequence_pseudo_log_likelihood": pseudo_log_likelihood,
    }


def compute_wild_type_structure_features(structure, antibody_chain_ids: list[str], antigen_chain_ids: list[str]) -> dict[str, dict]:
    features: dict[str, dict] = {}
    if not should_compute_structure_embeddings():
        return features

    if antibody_chain_ids:
        chain_sequences = {c: chain_sequence(get_chain(structure, c)) for c in antibody_chain_ids}
        for chain_id, (embedding, plddt_diagnostic) in compute_wild_type_antibody_structural_embeddings(
            chain_sequences
        ).items():
            features[chain_id] = {"structure_embedding": embedding, "structure_plddt_diagnostic": plddt_diagnostic}

    for chain_id in antigen_chain_ids:
        sequence = chain_sequence(get_chain(structure, chain_id))
        embedding, plddt_diagnostic = compute_wild_type_structural_embedding(sequence)
        features[chain_id] = {"structure_embedding": embedding, "structure_plddt_diagnostic": plddt_diagnostic}

    return features


def compute_mutant_structure_features(
    structure,
    antibody_chain_ids: list[str],
    antigen_chain_ids: list[str],
    mutations_by_chain_id: dict[str, list[PointMutation]],
) -> dict[str, dict]:
    features: dict[str, dict] = {}
    if not should_compute_structure_embeddings():
        return features

    if antibody_chain_ids:
        chain_sequences = {
            c: mutant_sequence_for_chain(get_chain(structure, c), mutations_by_chain_id.get(c, []))
            for c in antibody_chain_ids
        }
        for chain_id, (embedding, confidence) in compute_mutant_antibody_structural_embeddings(
            chain_sequences
        ).items():
            features[chain_id] = {"structure_embedding": embedding, "structure_confidence": confidence}

    for chain_id in antigen_chain_ids:
        mutant_sequence = mutant_sequence_for_chain(get_chain(structure, chain_id), mutations_by_chain_id.get(chain_id, []))
        embedding, confidence = compute_mutant_structural_embedding(mutant_sequence)
        features[chain_id] = {"structure_embedding": embedding, "structure_confidence": confidence}

    return features


def compute_wild_type_chain_bundle(structure, chain_id: str, structure_features_by_chain: dict[str, dict]) -> dict:
    chain = get_chain(structure, chain_id)
    sequence = chain_sequence(chain)
    bundle = {
        "real_structure_features": compute_real_structure_features(structure, chain_id),
        "residue_coordinates": chain_ca_coordinates(chain),
    }

    if should_compute_sequence_embeddings():
        bundle.update(compute_sequence_features(sequence))

    if chain_id in structure_features_by_chain:
        bundle.update(structure_features_by_chain[chain_id])

    return bundle


def compute_mutant_chain_bundle(
    structure, chain_id: str, mutations: list[PointMutation], structure_features_by_chain: dict[str, dict]
) -> dict:
    chain = get_chain(structure, chain_id)
    mutant_sequence = mutant_sequence_for_chain(chain, mutations)
    bundle = {}

    if should_compute_sequence_embeddings():
        bundle.update(compute_sequence_features(mutant_sequence))

    if chain_id in structure_features_by_chain:
        bundle.update(structure_features_by_chain[chain_id])

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
    chain_role_by_id = invert_chain_map(chain_map)
    antibody_chain_ids, antigen_chain_ids = split_chain_ids_by_backbone(ordered_chain_ids, chain_role_by_id)

    wt_structure_features = compute_wild_type_structure_features(structure, antibody_chain_ids, antigen_chain_ids)
    mut_structure_features = compute_mutant_structure_features(
        structure, antibody_chain_ids, antigen_chain_ids, mutations_by_chain_id
    )

    wt_chain_bundles = [
        compute_wild_type_chain_bundle(structure, chain_id, wt_structure_features) for chain_id in ordered_chain_ids
    ]
    mut_chain_bundles = [
        compute_mutant_chain_bundle(structure, chain_id, mutations_by_chain_id.get(chain_id, []), mut_structure_features)
        for chain_id in ordered_chain_ids
    ]

    tensor_bundle = build_sample_tensor_bundle(wt_chain_bundles, mut_chain_bundles, record.mutations)
    save_sample_embedding_bundle(record.sample_id, tensor_bundle)


def compute_and_cache_all_embeddings(
    records: list[MutationRecord], sample_chain_maps: dict[str, dict[str, str]]
) -> None:
    for record in records:
        compute_and_cache_entry_embeddings(record, sample_chain_maps[record.sample_id])
