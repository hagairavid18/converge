"""SaProt mutation-site localization probe (validation experiment, not
production code).

Same question as `mutation_site_probe.py`'s baseline check (does the raw
mutant-minus-wild-type difference-vector norm spike at the true mutated
residue, with no training at all), but restricted to the SaProt-backed
`wt_structure_embedding`/`mut_structure_embedding` specifically (not
concatenated with sequence), to check the same localization property holds
for `shared.constants.USE_SAPROT_STRUCTURE` before trusting it in a real
training run.

Run: SKEMPI_USE_SAPROT_STRUCTURE=1 SKEMPI_USE_SMALL_CHECKPOINTS=0 uv run python notebooks/saprot_diff_localization_probe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch

from data.embedding_pipeline.entry_embeddings import entry_embeddings_are_cached  # noqa: E402
from data.processed_io import load_sample_chain_maps  # noqa: E402
from dl.datasets.mutation_csv_dataset import (  # noqa: E402
    filter_records_for_active_iteration,
    load_or_compute_cached_tensors,
    load_records,
)
from shared.constants import RANDOM_SEED, SPLIT_FILES, SplitName, USE_SAPROT_STRUCTURE  # noqa: E402

TARGET_NUM_SAMPLES = 100
TOP_K = 10


def single_mutation_records(csv_path) -> list:
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
    return sorted(records, key=lambda record: not entry_embeddings_are_cached(record, sample_chain_maps[record.sample_id]))


def main() -> None:
    print("USE_SAPROT_STRUCTURE:", USE_SAPROT_STRUCTURE)
    if not USE_SAPROT_STRUCTURE:
        raise RuntimeError("Run with SKEMPI_USE_SAPROT_STRUCTURE=1 -- otherwise this just re-tests IgFold/zero-placeholder.")

    sample_chain_maps = load_sample_chain_maps()
    records = prioritize_already_cached(collect_unique_single_mutation_records(), sample_chain_maps)[:TARGET_NUM_SAMPLES]

    site_norms, elsewhere_norms = [], []
    argmax_hits = 0
    top_k_distances, random_baseline_distances = [], []
    rng = np.random.default_rng(RANDOM_SEED)

    for index, record in enumerate(records):
        cached = load_or_compute_cached_tensors(record, sample_chain_maps)
        difference = cached["mut_structure_embedding"] - cached["wt_structure_embedding"]
        diff_norm = difference.norm(dim=-1)
        true_position = record.mutations[0].flat_residue_index

        site_norms.append(diff_norm[true_position].item())
        elsewhere_mask = torch.ones_like(diff_norm, dtype=torch.bool)
        elsewhere_mask[true_position] = False
        elsewhere_norms.append(diff_norm[elsewhere_mask].mean().item())

        argmax_hits += int(diff_norm.argmax().item() == true_position)

        distances = cached["mutation_distances"][0]
        top_k_positions = diff_norm.topk(min(TOP_K, diff_norm.numel())).indices
        top_k_distances.append(distances[top_k_positions].mean().item())
        random_positions = torch.from_numpy(rng.choice(diff_norm.numel(), size=min(TOP_K, diff_norm.numel()), replace=False))
        random_baseline_distances.append(distances[random_positions].mean().item())

        print(f"  [{index + 1}/{len(records)}] {record.sample_id}: site_norm={diff_norm[true_position].item():.3f}")

    n = len(records)
    print(f"\nn_samples={n}")
    print(f"mean diff-norm AT mutation site: {np.mean(site_norms):.3f}")
    print(f"mean diff-norm elsewhere:        {np.mean(elsewhere_norms):.3f}")
    print(f"argmax-of-diff-norm localization accuracy: {argmax_hits}/{n} ({100 * argmax_hits / n:.1f}%)")
    print(f"top-{TOP_K} diff-norm positions, mean real 3D distance to true site: {np.mean(top_k_distances):.2f} A")
    print(f"random baseline, mean real 3D distance to true site:                {np.mean(random_baseline_distances):.2f} A")


if __name__ == "__main__":
    main()
