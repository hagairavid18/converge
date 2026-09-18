"""Structure-model (ESMFold) mutant-branch embeddings and per-residue
structural confidence (pLDDT), branching on
`shared.constants.ACTIVE_STRUCTURE_BACKEND` so a later swap to IgFold
requires no pipeline rework -- only a new branch here.

Only the mutant branch runs a structure prediction (see
`data.embedding_pipeline.real_structure_features` for the wild-type branch, which
uses the real PDB structure directly per the Implementation Spec).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
from transformers import AutoTokenizer, EsmForProteinFolding

from data.embedding_pipeline.device import resolve_device
from shared.constants import ACTIVE_STRUCTURE_BACKEND, ESMFOLD_CHECKPOINT, StructureBackend


@lru_cache(maxsize=1)
def _load_esmfold_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(ESMFOLD_CHECKPOINT)
    model = EsmForProteinFolding.from_pretrained(ESMFOLD_CHECKPOINT, low_cpu_mem_usage=True)
    model.to(resolve_device())
    model.eval()
    return tokenizer, model


def _require_esmfold_backend() -> None:
    if ACTIVE_STRUCTURE_BACKEND != StructureBackend.ESMFOLD:
        raise NotImplementedError(f"Structure backend {ACTIVE_STRUCTURE_BACKEND} is not implemented yet")


def _run_esmfold(sequence: str):
    tokenizer, model = _load_esmfold_model_and_tokenizer()
    device = resolve_device()
    input_ids = tokenizer([sequence], return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    with torch.no_grad():
        return model(input_ids)


def compute_mutant_structural_embedding(sequence: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns (per_residue_embedding[L, D], per_residue_plddt[L], in [0, 1])."""
    _require_esmfold_backend()
    output = _run_esmfold(sequence)
    per_residue_embedding = output.s_s[0].cpu().numpy()
    per_residue_plddt = output.plddt[0].mean(axis=-1).cpu().numpy()
    return per_residue_embedding, per_residue_plddt


def compute_wild_type_structural_embedding(sequence: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns (per_residue_embedding[L, D], per_residue_plddt_diagnostic[L]).

    The embedding is run on the real (not mutated) sequence so it is
    dimensionally compatible with `compute_mutant_structural_embedding`'s
    output for the model's branch-subtraction step -- a deliberate
    reconciliation with `dl.models.schemas.ModelInputs` (which expects
    `wt_structure_embedding` in the same `structure_embed_dim` as
    `mut_structure_embedding`), flagged in the pipeline report as a place
    where two readings of the spec's "wild-type structure: use the real PDB
    structure directly" wording could diverge -- see
    `data.embedding_pipeline.real_structure_features` for the alternative
    (real-coordinate-derived, not model-predicted) features, cached
    alongside this under a separate key rather than discarded.

    The pLDDT is ESMFold's internal self-assessed confidence in this same
    (otherwise unused) wild-type run. Per the spec, the wild-type branch
    gets no structural confidence in the model's confidence-weighting (that
    still only uses sequence confidence for this branch) -- this value is
    captured purely as a diagnostic for a deferred validation check (does
    ESMFold's own confidence on the *true* sequence correlate with anything
    useful), see `docs/future_work.md`, and must not be wired into the model
    as a confidence input.
    """
    _require_esmfold_backend()
    output = _run_esmfold(sequence)
    per_residue_embedding = output.s_s[0].cpu().numpy()
    per_residue_plddt = output.plddt[0].mean(axis=-1).cpu().numpy()
    return per_residue_embedding, per_residue_plddt
