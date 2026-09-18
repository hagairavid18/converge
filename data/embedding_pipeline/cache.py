"""Read/write of the precomputed per-sample residue-tensor cache in
`EMBEDDING_CACHE_DIR`, one `torch.save`-able dict per `sample_id`, matching
`dl.datasets.mutation_csv_dataset.embedding_cache_path` /
`load_cached_residue_tensors`.
"""

from __future__ import annotations

from pathlib import Path

import torch

from shared.constants import EMBEDDING_CACHE_DIR


def sample_embedding_cache_path(sample_id: str) -> Path:
    safe_sample_id = sample_id.replace("/", "_")
    return EMBEDDING_CACHE_DIR / f"{safe_sample_id}.pt"


def save_sample_embedding_bundle(sample_id: str, tensors: dict[str, torch.Tensor]) -> Path:
    destination = sample_embedding_cache_path(sample_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensors, destination)
    return destination


def load_sample_embedding_bundle(sample_id: str) -> dict[str, torch.Tensor]:
    return torch.load(sample_embedding_cache_path(sample_id))


def sample_embedding_bundle_exists(sample_id: str) -> bool:
    return sample_embedding_cache_path(sample_id).exists()
