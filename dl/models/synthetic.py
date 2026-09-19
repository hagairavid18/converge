"""Synthetic `ModelInputs` generator, matching the shapes/dtypes documented
in `dl.models.schemas`. Used to validate the architecture (overfitting
sanity check, `dl.training.sanity_check`) before real pre-processing output
exists, and by `dl.datasets.dataset_builder` as a fallback data source.

Generates a random number (1..`max_mutation_sites`) of mutation sites per
sample, with a distance profile that is zero at each site's own residue and
grows with residue-index distance from it -- not real 3D geometry, but
enough structure to exercise the multi-site max-pooling in
`dl.layers.pooling` meaningfully (padded via
`mutation_distances`/`mutation_site_mask`).
"""

from __future__ import annotations

import torch

from dl.models.config import ModelConfig
from dl.models.schemas import ModelInputs
from dl.utils.constants import METADATA_FIELD_NORMALIZATION_STATS
from dl.utils.embedding_mode import uses_sequence_embedding, uses_structure_embedding
from shared.constants import RANDOM_SEED


def make_synthetic_inputs(
    config: ModelConfig,
    batch_size: int,
    sequence_length: int,
    seed: int = RANDOM_SEED,
    max_mutation_sites: int = 3,
) -> ModelInputs:
    generator = torch.Generator().manual_seed(seed)
    mode = config.embedding_source_mode

    structure_kwargs = (
        _synthetic_structure_embeddings(config, batch_size, sequence_length, generator)
        if uses_structure_embedding(mode)
        else {}
    )
    sequence_kwargs = (
        _synthetic_sequence_embeddings(config, batch_size, sequence_length, generator)
        if uses_sequence_embedding(mode)
        else {}
    )

    mutation_distances, mutation_site_mask = _synthetic_mutation_distances(
        batch_size, sequence_length, max_mutation_sites, generator
    )
    metadata_kwargs = (
        _synthetic_metadata_fields(config, batch_size, generator) if config.extra_metadata_fields else {}
    )

    return ModelInputs(
        padding_mask=torch.ones(batch_size, sequence_length, dtype=torch.bool),
        mutation_distances=mutation_distances,
        mutation_site_mask=mutation_site_mask,
        antigen_mask=_synthetic_antigen_mask(batch_size, sequence_length),
        **structure_kwargs,
        **sequence_kwargs,
        **metadata_kwargs,
    )


def _synthetic_antigen_mask(batch_size: int, sequence_length: int) -> torch.Tensor:
    """Last third of the (arbitrary, synthetic) sequence is "antigen", the
    rest "antibody" -- realism doesn't matter here, only that both sides are
    non-empty so `dl.layers.pooling.ChainRoleSplitPooling` has something
    real to pool on each side.
    """
    antigen_start = max(1, (2 * sequence_length) // 3)
    mask = torch.zeros(sequence_length, dtype=torch.bool)
    mask[antigen_start:] = True
    return mask.unsqueeze(0).expand(batch_size, -1)


def _synthetic_mutation_distances(
    batch_size: int, sequence_length: int, max_mutation_sites: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    site_counts = torch.randint(1, max_mutation_sites + 1, (batch_size,), generator=generator)
    distances = torch.zeros(batch_size, max_mutation_sites, sequence_length)
    mask = torch.zeros(batch_size, max_mutation_sites, dtype=torch.bool)
    residue_positions = torch.arange(sequence_length, dtype=torch.float32)
    for sample_index in range(batch_size):
        count = int(site_counts[sample_index])
        centers = torch.randperm(sequence_length, generator=generator)[:count]
        for site_index, center in enumerate(centers):
            distances[sample_index, site_index] = (residue_positions - center).abs()
        mask[sample_index, :count] = True
    return distances, mask


def _synthetic_structure_embeddings(config: ModelConfig, batch_size: int, sequence_length: int, generator: torch.Generator) -> dict:
    return {
        "wt_structure_embedding": torch.randn(batch_size, sequence_length, config.structure_embed_dim, generator=generator),
        "mut_structure_embedding": torch.randn(batch_size, sequence_length, config.structure_embed_dim, generator=generator),
        "mut_structure_confidence": torch.rand(batch_size, sequence_length, generator=generator),
    }


def _synthetic_metadata_fields(config: ModelConfig, batch_size: int, generator: torch.Generator) -> dict:
    fields = {}
    for field in config.extra_metadata_fields:
        mean, std = METADATA_FIELD_NORMALIZATION_STATS[field]
        fields[field] = torch.normal(mean, std, size=(batch_size,), generator=generator)
    return fields


def _synthetic_sequence_embeddings(config: ModelConfig, batch_size: int, sequence_length: int, generator: torch.Generator) -> dict:
    return {
        "wt_sequence_embedding": torch.randn(batch_size, sequence_length, config.sequence_embed_dim, generator=generator),
        "mut_sequence_embedding": torch.randn(batch_size, sequence_length, config.sequence_embed_dim, generator=generator),
        "wt_sequence_confidence": torch.rand(batch_size, sequence_length, generator=generator),
        "mut_sequence_confidence": torch.rand(batch_size, sequence_length, generator=generator),
    }
