"""Read/write of the precomputed residue-tensor cache under
`EMBEDDING_CACHE_DIR`, split two ways:

- by model (`structure/` vs `sequence/`, plus `structure_saprot/` for the
  SaProt structure-embedding backend, see `data.embedding_pipeline.
  saprot_entry_embeddings` and `shared.constants.USE_SAPROT_STRUCTURE`), so
  one backbone can be swapped and regenerated (e.g. `rm -rf
  data/embeddings/structure`) without touching the others' cache.
- by wild-type vs. mutant (`wt/` vs `mut/`). A sample's wild-type complex
  is shared with every other sample measuring a different mutation on the
  *same* PDB (`shared.constants.MutationRecord.pdb_id`), so wild-type
  bundles are keyed and deduplicated by `pdb_id` instead of being
  recomputed and stored once per sample -- `data.processed_io` shows
  1211 samples span only 55 unique PDBs, so this avoids up to ~1156
  redundant wild-type structure/sequence model calls (and the matching
  redundant storage) across the full dataset. Mutant bundles are still
  keyed by `sample_id`, since the mutant sequence is sample-specific.

Every bundle is one `torch.save`-able dict, written via a same-directory
temp file + atomic `os.replace` so concurrent cache-warming jobs computing
the same key can never leave a torn/partially-written file for a reader to
load, and so the "last writer wins" on a race rather than corrupting the
file.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import torch

from shared.constants import EMBEDDING_CACHE_DIR


def _safe_id(raw_id: str) -> str:
    return raw_id.replace("/", "_")


def wt_structure_cache_path(pdb_id: str) -> Path:
    return EMBEDDING_CACHE_DIR / "structure" / "wt" / f"{_safe_id(pdb_id)}.pt"


def mut_structure_cache_path(sample_id: str) -> Path:
    return EMBEDDING_CACHE_DIR / "structure" / "mut" / f"{_safe_id(sample_id)}.pt"


def wt_sequence_cache_path(pdb_id: str) -> Path:
    return EMBEDDING_CACHE_DIR / "sequence" / "wt" / f"{_safe_id(pdb_id)}.pt"


def mut_sequence_cache_path(sample_id: str) -> Path:
    return EMBEDDING_CACHE_DIR / "sequence" / "mut" / f"{_safe_id(sample_id)}.pt"


def saprot_wt_cache_path(pdb_id: str) -> Path:
    return EMBEDDING_CACHE_DIR / "structure_saprot" / "wt" / f"{_safe_id(pdb_id)}.pt"


def saprot_mut_cache_path(sample_id: str) -> Path:
    return EMBEDDING_CACHE_DIR / "structure_saprot" / "mut" / f"{_safe_id(sample_id)}.pt"


def save_bundle(path: Path, tensors: dict[str, torch.Tensor]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    torch.save(tensors, tmp_path)
    os.replace(tmp_path, path)
    return path


def load_bundle(path: Path) -> dict[str, torch.Tensor]:
    return torch.load(path)


def bundle_exists(path: Path) -> bool:
    return path.exists()


def bundle_is_current(
    path: Path, required_keys: tuple[str, ...], expected_last_dim_by_key: dict[str, int] | None = None
) -> bool:
    """A bundle computed under a more restrictive `EmbeddingSourceMode`
    (e.g. `SEQUENCE_ONLY`) may exist but be missing fields a since-widened
    mode now needs -- existence alone isn't enough to trust it as current.

    `expected_last_dim_by_key` catches a second, subtler staleness case:
    a bundle computed under a *different* active checkpoint (e.g. the ESM-2
    small-vs-full checkpoint switched by `SKEMPI_USE_SMALL_CHECKPOINTS`) has
    every required key present but at the wrong embedding dimension -- this
    silently produced a real dimension-mismatch crash once two processes
    (one with the small checkpoint active, one with the full checkpoint)
    wrote wild-type and mutant bundles for the same sample under different
    settings, so key presence alone is not sufficient here either.
    """
    if not bundle_exists(path):
        return False
    cached = load_bundle(path)
    if not all(key in cached for key in required_keys):
        return False
    for key, expected_dim in (expected_last_dim_by_key or {}).items():
        if key in cached and cached[key].shape[-1] != expected_dim:
            return False
    return True


def load_or_compute_bundle(
    path: Path, required_keys: tuple[str, ...], compute_fn, expected_last_dim_by_key: dict[str, int] | None = None
) -> dict[str, torch.Tensor]:
    if bundle_is_current(path, required_keys, expected_last_dim_by_key):
        return load_bundle(path)
    tensors = compute_fn()
    save_bundle(path, tensors)
    return tensors
