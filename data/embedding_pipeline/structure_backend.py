"""Structure-model embeddings and per-residue structural confidence.

IgFold only, for antibody chains (`ChainRole.HEAVY`/`LIGHT` -- purpose-built
for antibody variable-domain prediction from sequence alone, called once
for both together since IgFold models VH-VL pairing jointly). IgFold cannot
process arbitrary (non-antibody) sequences, so the antigen chain gets no
structure model call at all: `compute_zero_structure_placeholder` returns an
all-zero embedding/confidence of the same shape instead. ESMFold was tried
for the antigen chain and dropped -- its O(L^2) triangular attention OOM'd
folding one real (long) chain even on a 32GB V100 -- so this is a deliberate
simplification, not an oversight; see `docs/future_work.md`.

Both the wild-type and mutant branches run IgFold for antibody chains: the
wild-type branch for a dimensionally-compatible embedding but reports no
confidence (per the spec, only sequence confidence feeds the wild-type
branch's confidence weighting); the mutant branch reports a bounded,
higher-is-better pseudo-confidence derived from IgFold's prmsd via
`exp(-prmsd)`.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
from igfold import IgFoldRunner

from data.embedding_pipeline.device import resolve_device

STRUCTURE_EMBEDDING_COMMON_DIM = 1024


def pad_structure_embedding_to_common_dim(embedding: np.ndarray) -> np.ndarray:
    missing = STRUCTURE_EMBEDDING_COMMON_DIM - embedding.shape[-1]
    if missing < 0:
        raise ValueError(
            f"structure embedding dim {embedding.shape[-1]} exceeds "
            f"STRUCTURE_EMBEDDING_COMMON_DIM={STRUCTURE_EMBEDDING_COMMON_DIM}"
        )
    if missing == 0:
        return embedding
    return np.pad(embedding, ((0, 0), (0, missing)))


def compute_zero_structure_placeholder(sequence_length: int) -> tuple[np.ndarray, np.ndarray]:
    """Stand-in for a chain IgFold cannot fold (the antigen): an all-zero
    embedding/confidence of the same shape a real structure-model call would
    produce, so it concatenates cleanly with the antibody chains' real
    IgFold output. A deliberate, documented placeholder (see module
    docstring), not a computed value -- revisit if a structure signal for
    the antigen chain turns out to matter.
    """
    return (
        np.zeros((sequence_length, STRUCTURE_EMBEDDING_COMMON_DIM), dtype=np.float32),
        np.zeros(sequence_length, dtype=np.float32),
    )


@lru_cache(maxsize=1)
def _load_igfold_runner() -> IgFoldRunner:
    return IgFoldRunner(device=resolve_device())


def _run_igfold(chain_sequences: dict[str, str]):
    runner = _load_igfold_runner()
    with torch.no_grad():
        return runner.embed(sequences=chain_sequences)


def _split_by_chain_lengths(per_residue_array: np.ndarray, chain_sequences: dict[str, str]) -> dict[str, np.ndarray]:
    split: dict[str, np.ndarray] = {}
    offset = 0
    for chain_id, sequence in chain_sequences.items():
        length = len(sequence)
        split[chain_id] = per_residue_array[offset : offset + length]
        offset += length
    return split


def compute_mutant_antibody_structural_embeddings(
    chain_sequences: dict[str, str],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Antibody-branch (heavy/light) mutant structure embeddings via IgFold,
    one call for all chains passed together. Returns, per chain id,
    (embedding[L, 1024] zero-padded from IgFold's native 64-dim,
    confidence[L] in (0, 1] via `exp(-prmsd)`).
    """
    output = _run_igfold(chain_sequences)
    embedding_by_chain = _split_by_chain_lengths(output.structure_embs[0].cpu().numpy(), chain_sequences)
    prmsd_by_chain = _split_by_chain_lengths(output.prmsd[0].mean(axis=-1).cpu().numpy(), chain_sequences)
    return {
        chain_id: (pad_structure_embedding_to_common_dim(embedding_by_chain[chain_id]), np.exp(-prmsd_by_chain[chain_id]))
        for chain_id in chain_sequences
    }


def compute_wild_type_antibody_structural_embeddings(
    chain_sequences: dict[str, str],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Antibody-branch (heavy/light) wild-type structure embeddings via
    IgFold, dimensionally compatible with the mutant branch's output.
    Returns, per chain id, (embedding[L, 1024], prmsd_diagnostic[L]) --
    captured for a deferred validation check (see `docs/future_work.md`),
    never fed to the model as a confidence input.
    """
    output = _run_igfold(chain_sequences)
    embedding_by_chain = _split_by_chain_lengths(output.structure_embs[0].cpu().numpy(), chain_sequences)
    prmsd_by_chain = _split_by_chain_lengths(output.prmsd[0].mean(axis=-1).cpu().numpy(), chain_sequences)
    return {
        chain_id: (pad_structure_embedding_to_common_dim(embedding_by_chain[chain_id]), np.exp(-prmsd_by_chain[chain_id]))
        for chain_id in chain_sequences
    }
