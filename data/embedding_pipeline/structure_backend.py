"""Structure-model embeddings and per-residue structural confidence.

Mixed backbone, by `shared.constants.ChainRole` (a deliberate, confirmed
reintroduction of mixed-model complexity -- see `docs/future_work.md`):
IgFold for antibody chains (`HEAVY`/`LIGHT` -- purpose-built for antibody
variable-domain prediction from sequence alone, much lighter/faster than
ESMFold) and ESMFold for the `ANTIGEN` chain (IgFold does not handle
arbitrary protein sequences). Which function a caller uses is how that
routing happens -- `data.embedding_pipeline.entry_embeddings` picks IgFold's
antibody functions for heavy/light chains and ESMFold's for the antigen.

Both backbones follow the same two-branch treatment: the wild-type branch
runs the model too (for a dimensionally-compatible embedding) but reports no
confidence; the mutant branch reports confidence (ESMFold: pLDDT; IgFold:
prmsd, converted to a bounded, higher-is-better pseudo-confidence via
`exp(-prmsd)` -- flagged in the pipeline report as a design choice, since
prmsd is a predicted RMSD in Angstroms, lower-is-better and unbounded, not
directly comparable to pLDDT's [0, 1] higher-is-better scale).

IgFold's per-residue embedding dimension (64) is much smaller than ESMFold's
(1024), and both feed the same per-sample concatenated tensor, so IgFold's
output is zero-padded up to `STRUCTURE_EMBEDDING_COMMON_DIM` before it
leaves this module -- the simplest resolution, per the pipeline report.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
from igfold import IgFoldRunner
from transformers import AutoTokenizer, EsmForProteinFolding

from data.embedding_pipeline.device import resolve_device
from shared.constants import ESMFOLD_CHECKPOINT

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


@lru_cache(maxsize=1)
def _load_esmfold_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(ESMFOLD_CHECKPOINT)
    model = EsmForProteinFolding.from_pretrained(ESMFOLD_CHECKPOINT, low_cpu_mem_usage=True)
    model.to(resolve_device())
    model.eval()
    return tokenizer, model


def _run_esmfold(sequence: str):
    tokenizer, model = _load_esmfold_model_and_tokenizer()
    device = resolve_device()
    input_ids = tokenizer([sequence], return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    with torch.no_grad():
        return model(input_ids)


def compute_mutant_structural_embedding(sequence: str) -> tuple[np.ndarray, np.ndarray]:
    """Antigen-branch mutant structure embedding via ESMFold. Returns
    (per_residue_embedding[L, 1024], per_residue_plddt[L], in [0, 1])."""
    output = _run_esmfold(sequence)
    per_residue_embedding = output.s_s[0].cpu().numpy()
    per_residue_plddt = output.plddt[0].mean(axis=-1).cpu().numpy()
    return per_residue_embedding, per_residue_plddt


def compute_wild_type_structural_embedding(sequence: str) -> tuple[np.ndarray, np.ndarray]:
    """Antigen-branch wild-type structure embedding via ESMFold. Returns
    (per_residue_embedding[L, 1024], per_residue_plddt_diagnostic[L]).

    Run on the real (not mutated) sequence so it is dimensionally compatible
    with `compute_mutant_structural_embedding`'s output for the model's
    branch-subtraction step -- a deliberate reconciliation with
    `dl.models.schemas.ModelInputs` (which expects `wt_structure_embedding`
    in the same `structure_embed_dim` as `mut_structure_embedding`), flagged
    in the pipeline report as a place where two readings of the spec's
    "wild-type structure: use the real PDB structure directly" wording could
    diverge -- see `data.embedding_pipeline.real_structure_features` for the
    alternative (real-coordinate-derived, not model-predicted) features,
    cached alongside this under a separate key rather than discarded.

    The pLDDT is ESMFold's internal self-assessed confidence in this same
    (otherwise unused) wild-type run. Per the spec, the wild-type branch
    gets no structural confidence in the model's confidence-weighting (that
    still only uses sequence confidence for this branch) -- this value is
    captured purely as a diagnostic for a deferred validation check, see
    `docs/future_work.md`, and must not be wired into the model as a
    confidence input.
    """
    output = _run_esmfold(sequence)
    per_residue_embedding = output.s_s[0].cpu().numpy()
    per_residue_plddt = output.plddt[0].mean(axis=-1).cpu().numpy()
    return per_residue_embedding, per_residue_plddt


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
    one call for all chains passed together (IgFold models heavy/light
    pairing jointly). Returns, per chain id, (embedding[L, 1024] zero-padded
    from IgFold's native 64-dim, confidence[L] in (0, 1] via `exp(-prmsd)`).
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
    IgFold, dimensionally compatible with the mutant branch's output (see
    `compute_wild_type_structural_embedding`'s docstring for why the
    wild-type branch still runs the model). Returns, per chain id,
    (embedding[L, 1024], prmsd_diagnostic[L]) -- the diagnostic mirrors
    `compute_wild_type_structural_embedding`'s `plddt_diagnostic`: captured
    for the same deferred validation check, never fed to the model as a
    confidence input.
    """
    output = _run_igfold(chain_sequences)
    embedding_by_chain = _split_by_chain_lengths(output.structure_embs[0].cpu().numpy(), chain_sequences)
    prmsd_by_chain = _split_by_chain_lengths(output.prmsd[0].mean(axis=-1).cpu().numpy(), chain_sequences)
    return {
        chain_id: (pad_structure_embedding_to_common_dim(embedding_by_chain[chain_id]), np.exp(-prmsd_by_chain[chain_id]))
        for chain_id in chain_sequences
    }
