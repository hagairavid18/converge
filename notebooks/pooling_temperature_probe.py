"""Pooling-temperature offline probe (validation experiment, not production
code).

Extends `pooling_weight_concentration_probe.py`'s finding (norm_softmax
collapses to a near one-hot selection at temperature 1.0) by sweeping
`ModelConfig.pooling_temperature` and checking, for each value: how sharp
the resulting weights are (participation ratio), whether the true mutation
site still gets the most weight, and -- the specific check requested --
whether the *other* chain role (the one the mutation did NOT land on) still
gets a meaningful share of the pooling weight, or whether it's still
effectively zeroed out regardless of temperature.

Run:
  uv run python notebooks/pooling_temperature_probe.py                                   # IgFold/zero backend
  SKEMPI_USE_SAPROT_STRUCTURE=1 uv run python notebooks/pooling_temperature_probe.py      # SaProt backend
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch

from data.chain_roles import ordered_chain_ids_for_sample
from data.embedding_pipeline.entry_embeddings import entry_embeddings_are_cached  # noqa: E402
from data.processed_io import load_sample_chain_maps  # noqa: E402
from data.structures import chain_sequence, get_chain, load_structure  # noqa: E402
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
TEMPERATURES = [1.0, 3.0, 5.0, 10.0, 20.0]


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


def antigen_mask_for_sample(chain_map: dict, pdb_id: str) -> torch.Tensor:
    ordered_chain_ids = ordered_chain_ids_for_sample(chain_map)
    antigen_chain_ids = set(chain_map.get(ChainRole.ANTIGEN.value, ""))
    structure = load_structure(pdb_id)
    is_antigen = []
    for chain_id in ordered_chain_ids:
        chain_length = len(chain_sequence(get_chain(structure, chain_id)))
        is_antigen.extend([chain_id in antigen_chain_ids] * chain_length)
    return torch.tensor(is_antigen, dtype=torch.bool)


def participation_ratio(weights: torch.Tensor) -> float:
    return (1.0 / (weights**2).sum()).item()


def main() -> None:
    label = "SaProt" if USE_SAPROT_STRUCTURE else "IgFold/zero-placeholder"
    print(f"backend: {label} (USE_SAPROT_STRUCTURE={USE_SAPROT_STRUCTURE})")

    sample_chain_maps = load_sample_chain_maps()
    records = prioritize_already_cached(collect_unique_single_mutation_records(), sample_chain_maps)[:TARGET_NUM_SAMPLES]

    for temperature in TEMPERATURES:
        config = ModelConfig(
            embedding_source_mode="structure_only", active_heads=(), pooling_strategy="norm_softmax", pooling_temperature=temperature
        )
        model = DDGPredictor(config)
        model.eval()

        eff_n, argmax_hits, antigen_weight_when_mutation_is_antibody, antibody_weight_when_mutation_is_antigen = [], 0, [], []
        with torch.no_grad():
            for record in records:
                cached = load_or_compute_cached_tensors(record, sample_chain_maps)
                batch = [(select_model_fields(cached), label_fields_for_record(record, frozenset()))]
                inputs, _ = collate_ddg_batch(batch)
                out = model(inputs)
                weights = out["pooling_weights"][0]
                true_position = record.mutations[0].flat_residue_index

                mutation_chain_role = record.mutations[0].chain_role
                is_antigen = antigen_mask_for_sample(sample_chain_maps[record.sample_id], record.pdb_id)
                antigen_weight = weights[is_antigen].sum().item()
                antibody_weight = weights[~is_antigen].sum().item()

                eff_n.append(participation_ratio(weights))
                argmax_hits += int(weights.argmax().item() == true_position)
                if mutation_chain_role == ChainRole.ANTIGEN:
                    antibody_weight_when_mutation_is_antigen.append(antibody_weight)
                else:
                    antigen_weight_when_mutation_is_antibody.append(antigen_weight)

        n = len(records)
        print(f"\n--- temperature={temperature} ---")
        print(f"  mean effective participation (1/sum(w^2)): {np.mean(eff_n):.2f}")
        print(f"  argmax-of-weight == true site: {argmax_hits}/{n}")
        if antigen_weight_when_mutation_is_antibody:
            print(
                f"  mutation on antibody (n={len(antigen_weight_when_mutation_is_antibody)}): "
                f"mean weight mass still on ANTIGEN chain = {np.mean(antigen_weight_when_mutation_is_antibody):.4f}"
            )
        if antibody_weight_when_mutation_is_antigen:
            print(
                f"  mutation on antigen (n={len(antibody_weight_when_mutation_is_antigen)}): "
                f"mean weight mass still on ANTIBODY chain(s) = {np.mean(antibody_weight_when_mutation_is_antigen):.4f}"
            )


if __name__ == "__main__":
    main()
