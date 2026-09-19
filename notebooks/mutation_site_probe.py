"""Mutation-site localization probe (validation experiment, not production code).

Question: does the per-residue embedding-difference signal (mutant minus
wild-type, under whichever `shared.constants.EmbeddingSourceMode` is
currently active) actually localize around the true mutated residue, or is
it noisy/uninformative for that purpose?

Trains a minimal linear probe -- a single `Linear(embedding_dim, 1)` scorer
shared across every residue position, applied to that residue's difference
vector, softmax over positions within a sample, cross-entropy against the
true mutated position -- deliberately as simple as possible: the question
is whether the *signal* is there at all, not whether a bigger model can dig
it out. Compared against two baselines that need no training: guessing a
random position, and picking the residue with the largest raw difference
norm.

A plain manual training loop, not PyTorch Lightning or the `class_name`/
`params` config integration `train.py` uses -- this is a one-off validation
script (see CLAUDE.md's `notebooks/` convention), not part of the training
pipeline, and per-sample sequences have a different length `L`, which would
need padding infrastructure Lightning-style batching doesn't buy anything
for here.

Restricted to single-point-mutation samples (unambiguous target position).
Reuses whatever real cached samples already exist under
`EMBEDDING_CACHE_DIR`, computing more on the fly via
`dl.datasets.mutation_csv_dataset.load_or_compute_cached_tensors` (the same
on-demand path the real training dataset uses, including its check that a
cache computed under a since-widened `ACTIVE_EMBEDDING_SOURCE_MODE` gets
recomputed rather than reused stale) up to `TARGET_NUM_SAMPLES`, deduping by
`sample_id` across the four split files (the same underlying sample can
appear in more than one split file under SKEMPI's two splitting schemes).

Run: uv run python notebooks/mutation_site_probe.py
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn.functional as F
from torch import nn

from data.embedding_pipeline.entry_embeddings import entry_embeddings_are_cached  # noqa: E402
from data.processed_io import load_sample_chain_maps  # noqa: E402
from dl.datasets.mutation_csv_dataset import (  # noqa: E402
    filter_records_for_active_iteration,
    load_or_compute_cached_tensors,
    load_records,
)
from dl.utils.embedding_mode import uses_sequence_embedding, uses_structure_embedding  # noqa: E402
from shared.constants import ACTIVE_EMBEDDING_SOURCE_MODE, RANDOM_SEED, SPLIT_FILES, SplitName  # noqa: E402

TARGET_NUM_SAMPLES = 32
TRAIN_FRACTION = 0.8
NUM_EPOCHS = 300
LEARNING_RATE = 1e-2


@dataclass
class ProbeSample:
    sample_id: str
    difference: torch.Tensor
    true_position: int


def single_mutation_records(csv_path: Path) -> list:
    records = filter_records_for_active_iteration(load_records(csv_path))
    return [record for record in records if len(record.mutations) == 1]


def collect_unique_single_mutation_records() -> list:
    seen_sample_ids: set[str] = set()
    unique_records = []
    for split in SplitName:
        for subset in ("train", "val"):
            for record in single_mutation_records(SPLIT_FILES[split][subset]):
                if record.sample_id not in seen_sample_ids:
                    seen_sample_ids.add(record.sample_id)
                    unique_records.append(record)
    return unique_records


def prioritize_already_cached(records: list, sample_chain_maps: dict) -> list:
    """Already-cached-and-current samples first, so the probe mostly reuses
    fast disk loads and only falls back to slow on-the-fly computation
    (e.g. ESMFold) for however many more are needed to reach
    `TARGET_NUM_SAMPLES`.
    """
    return sorted(records, key=lambda record: not entry_embeddings_are_cached(record, sample_chain_maps[record.sample_id]))


def branch_embedding(cached: dict, prefix: str) -> torch.Tensor:
    mode = ACTIVE_EMBEDDING_SOURCE_MODE
    parts = []
    if uses_structure_embedding(mode):
        parts.append(cached[f"{prefix}_structure_embedding"])
    if uses_sequence_embedding(mode):
        parts.append(cached[f"{prefix}_sequence_embedding"])
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)


def build_probe_sample(record, sample_chain_maps: dict) -> ProbeSample:
    cached = load_or_compute_cached_tensors(record, sample_chain_maps)
    difference = branch_embedding(cached, "mut") - branch_embedding(cached, "wt")
    return ProbeSample(sample_id=record.sample_id, difference=difference, true_position=record.mutations[0].flat_residue_index)


def collect_probe_samples(target_num_samples: int) -> list[ProbeSample]:
    sample_chain_maps = load_sample_chain_maps()
    records = prioritize_already_cached(collect_unique_single_mutation_records(), sample_chain_maps)[:target_num_samples]
    samples = []
    for index, record in enumerate(records):
        samples.append(build_probe_sample(record, sample_chain_maps))
        print(f"  [{index + 1}/{len(records)}] cached {record.sample_id}")
    return samples


def train_val_split(samples: list[ProbeSample], train_fraction: float, seed: int) -> tuple[list[ProbeSample], list[ProbeSample]]:
    shuffled = samples.copy()
    random.Random(seed).shuffle(shuffled)
    split_index = max(1, int(len(shuffled) * train_fraction))
    return shuffled[:split_index], shuffled[split_index:]


class LinearMutationSiteProbe(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.scorer = nn.Linear(embedding_dim, 1)

    def forward(self, difference: torch.Tensor) -> torch.Tensor:
        return self.scorer(difference).squeeze(-1)


def train_probe(probe: LinearMutationSiteProbe, train_samples: list[ProbeSample]) -> None:
    optimizer = torch.optim.Adam(probe.parameters(), lr=LEARNING_RATE)
    for _ in range(NUM_EPOCHS):
        optimizer.zero_grad()
        losses = [
            F.cross_entropy(probe(sample.difference).unsqueeze(0), torch.tensor([sample.true_position]))
            for sample in train_samples
        ]
        loss = torch.stack(losses).mean()
        loss.backward()
        optimizer.step()


def norm_baseline_prediction(sample: ProbeSample) -> int:
    return int(sample.difference.norm(dim=-1).argmax())


def random_baseline_prediction(sample: ProbeSample, rng: random.Random) -> int:
    return rng.randrange(sample.difference.shape[0])


def probe_prediction(probe: LinearMutationSiteProbe, sample: ProbeSample) -> int:
    with torch.no_grad():
        return int(probe(sample.difference).argmax())


@dataclass
class LocalizationReport:
    name: str
    accuracy: float
    mean_absolute_error_residues: float

    def __str__(self) -> str:
        return f"{self.name}: accuracy={self.accuracy:.3f}, mean_|error|={self.mean_absolute_error_residues:.1f} residues"


def summarize(name: str, samples: list[ProbeSample], predictions: list[int]) -> LocalizationReport:
    errors = [abs(prediction - sample.true_position) for prediction, sample in zip(predictions, samples)]
    accuracy = sum(error == 0 for error in errors) / len(samples)
    mean_error = sum(errors) / len(samples)
    return LocalizationReport(name, accuracy, mean_error)


def evaluate(probe: LinearMutationSiteProbe, samples: list[ProbeSample], rng: random.Random) -> list[LocalizationReport]:
    return [
        summarize("trained linear probe", samples, [probe_prediction(probe, sample) for sample in samples]),
        summarize("norm-of-difference baseline (no training)", samples, [norm_baseline_prediction(sample) for sample in samples]),
        summarize("random baseline", samples, [random_baseline_prediction(sample, rng) for sample in samples]),
    ]


def main() -> None:
    print(f"Active embedding source mode: {ACTIVE_EMBEDDING_SOURCE_MODE.value}")
    print(f"Collecting up to {TARGET_NUM_SAMPLES} single-point-mutation samples (computing on the fly if not cached)...")
    samples = collect_probe_samples(TARGET_NUM_SAMPLES)
    print(f"Collected {len(samples)} samples.")

    train_samples, val_samples = train_val_split(samples, TRAIN_FRACTION, RANDOM_SEED)
    print(f"Train: {len(train_samples)}, val: {len(val_samples)}")

    embedding_dim = train_samples[0].difference.shape[-1]
    probe = LinearMutationSiteProbe(embedding_dim)
    train_probe(probe, train_samples)

    rng = random.Random(RANDOM_SEED)
    print("\n-- Train set --")
    for report in evaluate(probe, train_samples, rng):
        print(report)
    print("\n-- Held-out val set --")
    for report in evaluate(probe, val_samples, rng):
        print(report)


if __name__ == "__main__":
    main()
