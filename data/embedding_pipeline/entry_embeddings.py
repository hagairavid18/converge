"""Per-entry embedding computation: for every chain (heavy, light, antigen)
of a complex, compute wild-type-branch and mutant-branch embeddings/
confidences, concatenate them into one whole-complex per-residue tensor set
(see `data.chain_roles.ordered_chain_ids_for_sample` for the fixed
concatenation order), and cache them via `data.embedding_pipeline.cache`.

Structure embeddings: IgFold for antibody chains (heavy/light, one call for
both together); the antigen chain gets a zero-placeholder instead of a real
structure-model call (see `data.embedding_pipeline.structure_backend` for
why -- ESMFold was tried there and dropped for memory/complexity reasons).
Sequence embeddings (ESM-2) run per chain independently, uniformly across
all three roles. Chains are never concatenated before inference -- only the
resulting per-residue *outputs* are concatenated afterwards, for
storage/batching convenience.

`shared.constants.USE_SAPROT_STRUCTURE` (env var `SKEMPI_USE_SAPROT_STRUCTURE`)
swaps that IgFold/zero-placeholder structure embedding for SaProt's instead,
uniformly across all three chain roles including the antigen (see
`apply_saprot_structure_override` and `data.embedding_pipeline.
saprot_entry_embeddings`) -- the one real structural signal option for the
antigen chain, since IgFold can't handle non-antibody sequences and ESMFold
OOM'd at this scale (see `docs/future_work.md`). Real-geometry fields
(`residue_coordinates`, `wt_real_structure_features`, `mutation_distances`)
are always computed the normal way regardless of this flag -- it only
changes which model produced `wt`/`mut_structure_embedding`.

The wild-type branch is identical for every sample that shares the same
`pdb_id` (it doesn't depend on which mutation that particular sample
represents), so it is computed and cached once per `pdb_id`
(`data.embedding_pipeline.cache.wt_structure_cache_path` /
`wt_sequence_cache_path`) and reused across every sample of that complex,
rather than recomputed per sample. The mutant branch is sample-specific and
is cached per `sample_id`. Structure and sequence outputs are cached
separately (`structure/` vs `sequence/`), so one backbone can be
regenerated independently of the other.

Per-residue interface-region weighting was dropped from this cache: the
model's pooling is pure distance-based Gaussian falloff (no consumer of a
whole-chain interface classification remains -- see
`data.mutation_tokens.interface_region_from_code` for the only place that
classification is still used, directly from SKEMPI's own column, for
`PointMutation.interface_region`). In its place, `mutation_distances` caches
the raw (not sigma-weighted) per-residue Euclidean distance from each point
mutation's wild-type CA coordinate to every residue in the complex, so the
Gaussian falloff can be applied live at train time with a configurable sigma
without recomputing distances from scratch. It lives in the mutant-branch
structure bundle (it is sample-specific, since which residue is mutated
varies per sample) but is always computed regardless of
`ACTIVE_EMBEDDING_SOURCE_MODE` (it's cheap, geometry-only, not a model
output).

`shared.constants.ACTIVE_EMBEDDING_SOURCE_MODE` gates which backend(s)
actually run, not just which fields the model reads: under `SEQUENCE_ONLY`,
neither IgFold nor the zero-placeholder path runs, so `wt`/`mut_structure_embedding`
keys are simply absent from the cache rather than computed and discarded --
`wt/mut_sequence_*` bundles are skipped outright under `STRUCTURE_ONLY`,
likewise. `dl.models.schemas.ModelInputs` already treats absent fields as
`Optional`/`None` for exactly this case. `real_structure_features` and
`residue_coordinates` are cheap, non-ML, real-geometry outputs and are
always computed regardless of mode, in the wild-type structure bundle.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from data.chain_roles import group_mutations_by_chain_id, ordered_chain_ids_for_sample
from data.embedding_pipeline.cache import (
    bundle_is_current,
    load_or_compute_bundle,
    mut_sequence_cache_path,
    mut_structure_cache_path,
    saprot_mut_cache_path,
    saprot_wt_cache_path,
    wt_sequence_cache_path,
    wt_structure_cache_path,
)
from data.embedding_pipeline.mutant_sequence import mutant_sequence_for_chain
from data.embedding_pipeline.real_structure_features import compute_real_structure_features
from data.embedding_pipeline.saprot_entry_embeddings import compute_mut_saprot_bundle, compute_wt_saprot_bundle
from data.embedding_pipeline.sequence_backend import (
    compute_sequence_embeddings,
    compute_sequence_pseudo_log_likelihood,
)
from data.embedding_pipeline.structure_backend import (
    compute_mutant_antibody_structural_embeddings,
    compute_wild_type_antibody_structural_embeddings,
    compute_zero_structure_placeholder,
)
from data.structures import chain_ca_coordinates, chain_sequence, get_chain, load_structure
from shared.constants import (
    ACTIVE_EMBEDDING_SOURCE_MODE,
    ACTIVE_ESM2_HIDDEN_DIM,
    ACTIVE_SAPROT_HIDDEN_DIM,
    USE_SAPROT_STRUCTURE,
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


def invert_chain_map(chain_map: dict[str, str]) -> dict[str, ChainRole]:
    return {chain_id: ChainRole(role_value) for role_value, chain_ids in chain_map.items() for chain_id in chain_ids}


def compute_sequence_features(sequence: str) -> dict:
    pseudo_log_likelihood = compute_sequence_pseudo_log_likelihood(sequence)
    return {
        "sequence_embedding": compute_sequence_embeddings(sequence),
        "sequence_confidence": np.exp(pseudo_log_likelihood),
        "sequence_pseudo_log_likelihood": pseudo_log_likelihood,
    }


def _concatenate_field(chain_bundles: list[dict], field: str) -> np.ndarray:
    return np.concatenate([bundle[field] for bundle in chain_bundles], axis=0)


def _as_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(array)).float()


def compute_mutation_distances(residue_coordinates: np.ndarray, mutations: list[PointMutation]) -> np.ndarray:
    """Raw per-residue Euclidean distance (Angstroms) from each mutation's
    wild-type CA coordinate to every residue in the sample's concatenated
    complex. Shape `[num_mutations, L]`, ordered the same as `mutations`.
    """
    mutation_site_coordinates = residue_coordinates[[m.flat_residue_index for m in mutations]]
    return np.linalg.norm(residue_coordinates[None, :, :] - mutation_site_coordinates[:, None, :], axis=-1)


def _structure_embeddings_by_chain(
    ordered_chain_ids: list[str],
    antibody_sequences_by_chain_id: dict[str, str],
    fallback_sequence_for_chain: Callable[[str], str],
    antibody_embed_fn: Callable[[dict[str, str]], dict[str, tuple[np.ndarray, np.ndarray]]],
) -> tuple[np.ndarray, np.ndarray]:
    """Shared routing for both branches: IgFold (one call) for antibody
    chains, a zero placeholder per chain for everything else.
    """
    embedding_by_chain: dict[str, np.ndarray] = {}
    confidence_by_chain: dict[str, np.ndarray] = {}
    if antibody_sequences_by_chain_id:
        for chain_id, (embedding, confidence) in antibody_embed_fn(antibody_sequences_by_chain_id).items():
            embedding_by_chain[chain_id] = embedding
            confidence_by_chain[chain_id] = confidence

    for chain_id in ordered_chain_ids:
        if chain_id not in embedding_by_chain:
            embedding_by_chain[chain_id], confidence_by_chain[chain_id] = compute_zero_structure_placeholder(
                len(fallback_sequence_for_chain(chain_id))
            )

    return (
        np.concatenate([embedding_by_chain[cid] for cid in ordered_chain_ids], axis=0),
        np.concatenate([confidence_by_chain[cid] for cid in ordered_chain_ids], axis=0),
    )


def compute_antigen_mask(
    coordinates_by_chain: dict[str, np.ndarray], ordered_chain_ids: list[str], chain_role_by_id: dict[str, ChainRole]
) -> np.ndarray:
    """Derives each chain's residue count from the same `chain_ca_coordinates`
    output `residue_coordinates` is built from (rather than recomputing it
    via `chain_sequence` separately), so `antigen_mask` is guaranteed to
    line up 1:1 with `residue_coordinates` by construction -- the same
    length-mismatch risk `data.embedding_pipeline.saprot_entry_embeddings.
    _align_to_real_sequence` had to fix for Foldseek is trivially avoided
    here by not introducing a second, independent length source at all.
    """
    return np.concatenate(
        [
            np.full(len(coordinates_by_chain[cid]), chain_role_by_id[cid] == ChainRole.ANTIGEN, dtype=bool)
            for cid in ordered_chain_ids
        ],
        axis=0,
    )


def compute_wt_structure_bundle(
    structure, ordered_chain_ids: list[str], chain_role_by_id: dict[str, ChainRole]
) -> dict[str, torch.Tensor]:
    """Per-PDB wild-type bundle: real-geometry fields always, structure
    embedding/confidence only under a structure-using `EmbeddingSourceMode`.
    """
    chains = {cid: get_chain(structure, cid) for cid in ordered_chain_ids}
    coordinates_by_chain = {cid: chain_ca_coordinates(chains[cid]) for cid in ordered_chain_ids}
    bundle = {
        "residue_coordinates": _as_tensor(np.concatenate([coordinates_by_chain[cid] for cid in ordered_chain_ids], axis=0)),
        "wt_real_structure_features": _as_tensor(
            np.concatenate([compute_real_structure_features(structure, cid) for cid in ordered_chain_ids], axis=0)
        ),
        "antigen_mask": torch.from_numpy(compute_antigen_mask(coordinates_by_chain, ordered_chain_ids, chain_role_by_id)),
    }
    if not should_compute_structure_embeddings():
        return bundle

    antibody_ids = [cid for cid in ordered_chain_ids if chain_role_by_id[cid] in ANTIBODY_CHAIN_ROLES]
    antibody_sequences = {cid: chain_sequence(chains[cid]) for cid in antibody_ids}
    embeddings, plddts = _structure_embeddings_by_chain(
        ordered_chain_ids,
        antibody_sequences,
        lambda cid: chain_sequence(chains[cid]),
        compute_wild_type_antibody_structural_embeddings,
    )
    bundle["wt_structure_embedding"] = _as_tensor(embeddings)
    bundle["wt_structure_plddt_diagnostic"] = _as_tensor(plddts)
    return bundle


def compute_mut_structure_bundle(
    structure,
    ordered_chain_ids: list[str],
    chain_role_by_id: dict[str, ChainRole],
    mutations_by_chain_id: dict[str, list[PointMutation]],
    mutations: list[PointMutation],
    residue_coordinates: np.ndarray,
) -> dict[str, torch.Tensor]:
    """Per-sample mutant bundle: `mutation_distances` always (cheap,
    geometry-only), structure embedding/confidence only under a
    structure-using `EmbeddingSourceMode`.
    """
    bundle = {"mutation_distances": _as_tensor(compute_mutation_distances(residue_coordinates, mutations))}
    if not should_compute_structure_embeddings():
        return bundle

    def mutant_sequence_for(cid: str) -> str:
        return mutant_sequence_for_chain(get_chain(structure, cid), mutations_by_chain_id.get(cid, []))

    antibody_ids = [cid for cid in ordered_chain_ids if chain_role_by_id[cid] in ANTIBODY_CHAIN_ROLES]
    antibody_sequences = {cid: mutant_sequence_for(cid) for cid in antibody_ids}
    embeddings, confidences = _structure_embeddings_by_chain(
        ordered_chain_ids,
        antibody_sequences,
        mutant_sequence_for,
        compute_mutant_antibody_structural_embeddings,
    )
    bundle["mut_structure_embedding"] = _as_tensor(embeddings)
    bundle["mut_structure_confidence"] = _as_tensor(confidences)
    return bundle


def compute_wt_sequence_bundle(structure, ordered_chain_ids: list[str]) -> dict[str, torch.Tensor]:
    """Per-PDB wild-type ESM-2 bundle."""
    chain_features = [compute_sequence_features(chain_sequence(get_chain(structure, cid))) for cid in ordered_chain_ids]
    return {
        "wt_sequence_embedding": _as_tensor(_concatenate_field(chain_features, "sequence_embedding")),
        "wt_sequence_confidence": _as_tensor(_concatenate_field(chain_features, "sequence_confidence")),
        "wt_sequence_pseudo_log_likelihood": _as_tensor(_concatenate_field(chain_features, "sequence_pseudo_log_likelihood")),
    }


def compute_mut_sequence_bundle(
    structure, ordered_chain_ids: list[str], mutations_by_chain_id: dict[str, list[PointMutation]]
) -> dict[str, torch.Tensor]:
    """Per-sample mutant ESM-2 bundle."""
    chain_features = [
        compute_sequence_features(mutant_sequence_for_chain(get_chain(structure, cid), mutations_by_chain_id.get(cid, [])))
        for cid in ordered_chain_ids
    ]
    return {
        "mut_sequence_embedding": _as_tensor(_concatenate_field(chain_features, "sequence_embedding")),
        "mut_sequence_confidence": _as_tensor(_concatenate_field(chain_features, "sequence_confidence")),
        "mut_sequence_pseudo_log_likelihood": _as_tensor(_concatenate_field(chain_features, "sequence_pseudo_log_likelihood")),
    }


def _lazy_structure_loader(pdb_id: str) -> Callable[[], object]:
    cell: dict = {}

    def get_structure():
        if "structure" not in cell:
            cell["structure"] = load_structure(pdb_id)
        return cell["structure"]

    return get_structure


def _per_residue_standardize(embedding: torch.Tensor) -> torch.Tensor:
    """Per-residue-vector z-score standardization (mean 0, std 1 across the
    feature dim), parameter-free (`elementwise_affine=False` -- no new
    learned weights, per CLAUDE.md's simplicity rule). Applied identically
    and independently to the wt and mut tensors, so an unchanged residue
    (near-identical raw wt/mut vectors) still standardizes to near-identical
    vectors -- the mutation-localization property of the difference is
    preserved, only each vector's own scale/offset changes.
    """
    return torch.nn.functional.layer_norm(embedding, embedding.shape[-1:])


def apply_saprot_structure_override(
    merged: dict[str, torch.Tensor], record: MutationRecord, ordered_chain_ids: list[str], mutations_by_chain_id: dict[str, list[PointMutation]]
) -> None:
    """Replaces the IgFold/zero-placeholder `wt_structure_embedding` /
    `mut_structure_embedding` (already computed and cached under
    `structure/`, unconditionally, above) with the SaProt-backed ones from
    the separate `structure_saprot/` cache namespace, when
    `shared.constants.USE_SAPROT_STRUCTURE` is set. Keeping the two cache
    namespaces separate (rather than writing SaProt output into
    `structure/`) avoids a dimension-mismatch staleness trap if a later run
    switches the backend back. Also drops the now-stale IgFold-derived
    confidence/diagnostic keys -- there is no SaProt-derived confidence
    signal yet, so the SaProt-backed branch is left unweighted (see
    `dl.layers.confidence_weighting`), same as any other `None` confidence.

    The raw cached SaProt tensors are standardized (`_per_residue_standardize`)
    on the way out, not before caching -- so the cache stays the raw model
    output and the transform is free to change without invalidating it.
    """
    wt_saprot = load_or_compute_bundle(
        saprot_wt_cache_path(record.pdb_id),
        ("wt_saprot_embedding",),
        lambda: compute_wt_saprot_bundle(record.pdb_id, ordered_chain_ids),
        {"wt_saprot_embedding": ACTIVE_SAPROT_HIDDEN_DIM},
    )
    mut_saprot = load_or_compute_bundle(
        saprot_mut_cache_path(record.sample_id),
        ("mut_saprot_embedding",),
        lambda: compute_mut_saprot_bundle(record.pdb_id, ordered_chain_ids, mutations_by_chain_id),
        {"mut_saprot_embedding": ACTIVE_SAPROT_HIDDEN_DIM},
    )
    merged["wt_structure_embedding"] = _per_residue_standardize(wt_saprot["wt_saprot_embedding"])
    merged["mut_structure_embedding"] = _per_residue_standardize(mut_saprot["mut_saprot_embedding"])
    merged.pop("wt_structure_plddt_diagnostic", None)
    merged.pop("mut_structure_confidence", None)


def compute_and_cache_entry_embeddings(record: MutationRecord, chain_map: dict[str, str]) -> dict[str, torch.Tensor]:
    """Ensures every cache piece this sample needs (wild-type structure/
    sequence, keyed by `pdb_id`; mutant structure/sequence, keyed by
    `sample_id`) exists and is current, computing+caching only whichever
    are missing or stale, and returns them merged into one dict shaped for
    `dl.datasets.mutation_csv_dataset.select_model_fields`. The PDB
    structure file is only loaded from disk if at least one piece actually
    needs (re)computing.
    """
    ordered_chain_ids = ordered_chain_ids_for_sample(chain_map)
    chain_role_by_id = invert_chain_map(chain_map)
    mutations_by_chain_id = group_mutations_by_chain_id(record.mutations)
    get_structure = _lazy_structure_loader(record.pdb_id)

    structure_required = ("wt_structure_embedding",) if should_compute_structure_embeddings() else ()
    wt_structure = load_or_compute_bundle(
        wt_structure_cache_path(record.pdb_id),
        structure_required,
        lambda: compute_wt_structure_bundle(get_structure(), ordered_chain_ids, chain_role_by_id),
    )
    mut_structure = load_or_compute_bundle(
        mut_structure_cache_path(record.sample_id),
        ("mut_structure_embedding",) if should_compute_structure_embeddings() else (),
        lambda: compute_mut_structure_bundle(
            get_structure(),
            ordered_chain_ids,
            chain_role_by_id,
            mutations_by_chain_id,
            record.mutations,
            wt_structure["residue_coordinates"].numpy(),
        ),
    )
    merged = {**wt_structure, **mut_structure}
    if should_compute_structure_embeddings() and USE_SAPROT_STRUCTURE:
        apply_saprot_structure_override(merged, record, ordered_chain_ids, mutations_by_chain_id)

    if should_compute_sequence_embeddings():
        wt_sequence = load_or_compute_bundle(
            wt_sequence_cache_path(record.pdb_id),
            ("wt_sequence_embedding",),
            lambda: compute_wt_sequence_bundle(get_structure(), ordered_chain_ids),
            {"wt_sequence_embedding": ACTIVE_ESM2_HIDDEN_DIM},
        )
        mut_sequence = load_or_compute_bundle(
            mut_sequence_cache_path(record.sample_id),
            ("mut_sequence_embedding",),
            lambda: compute_mut_sequence_bundle(get_structure(), ordered_chain_ids, mutations_by_chain_id),
            {"mut_sequence_embedding": ACTIVE_ESM2_HIDDEN_DIM},
        )
        merged.update(wt_sequence)
        merged.update(mut_sequence)

    return merged


def entry_embeddings_are_cached(record: MutationRecord, chain_map: dict[str, str]) -> bool:
    """Read-only check mirroring `compute_and_cache_entry_embeddings`'s
    cache-hit condition, without computing or loading a PDB structure --
    for callers (e.g. `notebooks/pooling_analysis_lib.py`) that want to
    prioritize already-cached samples over triggering new model calls.
    """
    structure_required = ("wt_structure_embedding",) if should_compute_structure_embeddings() else ()
    if not bundle_is_current(wt_structure_cache_path(record.pdb_id), structure_required):
        return False
    if not bundle_is_current(
        mut_structure_cache_path(record.sample_id), ("mut_structure_embedding",) if should_compute_structure_embeddings() else ()
    ):
        return False
    if should_compute_structure_embeddings() and USE_SAPROT_STRUCTURE:
        if not bundle_is_current(
            saprot_wt_cache_path(record.pdb_id), ("wt_saprot_embedding",), {"wt_saprot_embedding": ACTIVE_SAPROT_HIDDEN_DIM}
        ):
            return False
        if not bundle_is_current(
            saprot_mut_cache_path(record.sample_id), ("mut_saprot_embedding",), {"mut_saprot_embedding": ACTIVE_SAPROT_HIDDEN_DIM}
        ):
            return False
    if not should_compute_sequence_embeddings():
        return True
    if not bundle_is_current(
        wt_sequence_cache_path(record.pdb_id), ("wt_sequence_embedding",), {"wt_sequence_embedding": ACTIVE_ESM2_HIDDEN_DIM}
    ):
        return False
    return bundle_is_current(
        mut_sequence_cache_path(record.sample_id),
        ("mut_sequence_embedding",),
        {"mut_sequence_embedding": ACTIVE_ESM2_HIDDEN_DIM},
    )


def compute_and_cache_all_embeddings(
    records: list[MutationRecord], sample_chain_maps: dict[str, dict[str, str]]
) -> None:
    for record in records:
        compute_and_cache_entry_embeddings(record, sample_chain_maps[record.sample_id])
