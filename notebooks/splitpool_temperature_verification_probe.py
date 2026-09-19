"""Re-verification of `norm_softmax_by_chain_role` pooling at temperature=5.0
(validation experiment, not production code) -- the confirmed current
architecture (`structure_only`, SaProt, `pooling_strategy:
norm_softmax_by_chain_role`, `pooling_temperature: 5.0`).

Unlike flat `norm_softmax`, `ChainRoleSplitPooling` runs one softmax over the
antigen residues and a separate one over the antibody residues (see
`dl.layers.pooling.ChainRoleSplitPooling`), so both sides always contribute
by construction -- the "does the other chain get real weight" question that
motivated the original temperature sweep (`pooling_temperature_probe.py`) no
longer applies. What can still go wrong at a given temperature is sharpness:
each side's own softmax could still collapse to one-hot (temperature too low)
or wash out to near-uniform (temperature too high). This checks, per side,
independently: the participation ratio (`1/sum(w^2)` restricted to that
side's residues, which sum to 1 on their own) and whether the true mutation
site still gets the top weight within its own side.

Run:
  SKEMPI_USE_SAPROT_STRUCTURE=1 uv run python notebooks/splitpool_temperature_verification_probe.py
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
from data.homology_dedup import normalize_protein_name  # noqa: E402
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
from shared.constants import ChainRole, SPLIT_FILES, SplitName, USE_SAPROT_STRUCTURE  # noqa: E402

TARGET_NUM_SAMPLES = 30
TEMPERATURES = [1.0, 5.0, 10.0]
_NO_TRAIN_COMPLEXES: frozenset[str] = frozenset()


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
    total = weights.sum()
    if total.item() == 0.0:
        return float("nan")
    normalized = weights / total
    return (1.0 / (normalized**2).sum()).item()


def main() -> None:
    if not USE_SAPROT_STRUCTURE:
        print("WARNING: SKEMPI_USE_SAPROT_STRUCTURE=1 not set -- this will probe the IgFold/zero backend, not SaProt.")

    sample_chain_maps = load_sample_chain_maps()
    records = prioritize_already_cached(collect_unique_single_mutation_records(), sample_chain_maps)[:TARGET_NUM_SAMPLES]

    for temperature in TEMPERATURES:
        config = ModelConfig(
            embedding_source_mode="structure_only",
            active_heads=(),
            pooling_strategy="norm_softmax_by_chain_role",
            pooling_temperature=temperature,
        )
        model = DDGPredictor(config)
        model.eval()

        antigen_participation, antibody_participation = [], []
        antigen_argmax_hits, antigen_n = 0, 0
        antibody_argmax_hits, antibody_n = 0, 0

        with torch.no_grad():
            for record in records:
                cached = load_or_compute_cached_tensors(record, sample_chain_maps)
                batch = [(select_model_fields(cached), label_fields_for_record(record, _NO_TRAIN_COMPLEXES))]
                inputs, _ = collate_ddg_batch(batch)
                out = model(inputs)
                weights = out["pooling_weights"][0]
                antigen_mask = inputs.antigen_mask[0]
                true_position = record.mutations[0].flat_residue_index
                mutation_on_antigen = record.mutations[0].chain_role == ChainRole.ANTIGEN

                antigen_weights = weights[antigen_mask]
                antibody_weights = weights[~antigen_mask]
                antigen_participation.append(participation_ratio(antigen_weights))
                antibody_participation.append(participation_ratio(antibody_weights))

                if mutation_on_antigen:
                    antigen_n += 1
                    local_true_position = int(antigen_mask[:true_position].sum().item())
                    antigen_argmax_hits += int(antigen_weights.argmax().item() == local_true_position)
                else:
                    antibody_n += 1
                    local_true_position = int((~antigen_mask)[:true_position].sum().item())
                    antibody_argmax_hits += int(antibody_weights.argmax().item() == local_true_position)

        print(f"\n--- temperature={temperature} ---")
        print(
            f"  antigen side:  mean participation={np.mean(antigen_participation):.2f}"
            f"  argmax-hits (mutation on antigen, n={antigen_n})={antigen_argmax_hits}/{antigen_n}"
        )
        print(
            f"  antibody side: mean participation={np.mean(antibody_participation):.2f}"
            f"  argmax-hits (mutation on antibody, n={antibody_n})={antibody_argmax_hits}/{antibody_n}"
        )


if __name__ == "__main__":
    main()
