"""Pooling-weight concentration probe (validation experiment, not production
code).

The mutation-localization probes (`mutation_site_probe.py`,
`saprot_diff_localization_probe.py`) check whether the *raw difference norm*
peaks at the true mutated residue. This checks the actual thing that
matters for the model: the real `dl.layers.pooling.norm_softmax_weights`
output, computed the same way `DDGPredictor.forward` does (through
confidence weighting and the fused embedding, not the raw per-branch
difference alone), for both structure backends -- does the softmax actually
concentrate most of its mass near the mutation, or is it spread thin across
many residues despite the correct residue having the single largest score?

Run:
  uv run python notebooks/pooling_weight_concentration_probe.py             # IgFold/zero backend
  SKEMPI_USE_SAPROT_STRUCTURE=1 uv run python notebooks/pooling_weight_concentration_probe.py   # SaProt backend
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
from dl.datasets.collation import collate_ddg_batch  # noqa: E402
from dl.datasets.mutation_csv_dataset import (  # noqa: E402
    filter_records_for_active_iteration,
    label_fields_for_record,
    load_or_compute_cached_tensors,
    load_records,
    select_model_fields,
)
from dl.models.config import ModelConfig  # noqa: E402
from dl.models.ddg_model import DDGPredictor  # noqa: E402
from shared.constants import SPLIT_FILES, SplitName, USE_SAPROT_STRUCTURE  # noqa: E402

TARGET_NUM_SAMPLES = 30
TOP_K = 5


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


def participation_ratio(weights: torch.Tensor) -> float:
    """Effective number of residues the softmax is actually spreading mass
    over -- `1 / sum(w^2)`, minimized (=1) when all mass sits on one residue,
    maximized (=L) when uniform over all L residues.
    """
    return (1.0 / (weights**2).sum()).item()


def main() -> None:
    label = "SaProt" if USE_SAPROT_STRUCTURE else "IgFold/zero-placeholder"
    print(f"backend: {label} (USE_SAPROT_STRUCTURE={USE_SAPROT_STRUCTURE})")

    sample_chain_maps = load_sample_chain_maps()
    records = prioritize_already_cached(collect_unique_single_mutation_records(), sample_chain_maps)[:TARGET_NUM_SAMPLES]

    config = ModelConfig(embedding_source_mode="structure_only", active_heads=(), pooling_strategy="norm_softmax")
    model = DDGPredictor(config)
    model.eval()

    weight_at_site, max_weight, top_k_mass, eff_n = [], [], [], []
    with torch.no_grad():
        for index, record in enumerate(records):
            cached = load_or_compute_cached_tensors(record, sample_chain_maps)
            batch = [(select_model_fields(cached), label_fields_for_record(record, frozenset()))]
            inputs, _ = collate_ddg_batch(batch)
            out = model(inputs)
            weights = out["pooling_weights"][0]
            true_position = record.mutations[0].flat_residue_index

            weight_at_site.append(weights[true_position].item())
            max_weight.append(weights.max().item())
            top_k_mass.append(weights.topk(min(TOP_K, weights.numel())).values.sum().item())
            eff_n.append(participation_ratio(weights))
            print(
                f"  [{index + 1}/{len(records)}] {record.sample_id}: "
                f"L={weights.numel()} w_at_site={weights[true_position].item():.4f} "
                f"w_max={weights.max().item():.4f} argmax_is_site={weights.argmax().item() == true_position} "
                f"eff_n={participation_ratio(weights):.1f}"
            )

    n = len(records)
    print(f"\nn_samples={n}, backend={label}")
    print(f"mean weight AT true mutation site: {np.mean(weight_at_site):.4f}")
    print(f"mean max weight (any residue):     {np.mean(max_weight):.4f}")
    print(f"mean top-{TOP_K} cumulative weight:  {np.mean(top_k_mass):.4f}")
    print(f"mean effective participation (1/sum(w^2)), lower=sharper: {np.mean(eff_n):.1f}")
    print(f"argmax-of-weight == true site: {sum(1 for i in range(n) if max_weight[i] == weight_at_site[i])}/{n}")


if __name__ == "__main__":
    main()
