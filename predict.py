"""Config + checkpoint driven ΔΔG inference script.

    uv run python predict.py --config dl/configs/<name>.yaml --ckpt path/to.ckpt --input data/splits/same_pdb_allowed_val.csv

`--input` is a CSV shaped like `data/splits/*.csv` (a `MutationRecord` per
row, `sample_id`s already known to the data pipeline). Reuses `train.py`'s
data/model-building blocks, so no scoring logic is duplicated here.

Beyond the predicted ddG, reports per row whether that row's `complex_name`
was seen during the checkpoint's training split -- the seen-complex/
new-mutation vs. unseen-complex distinction tracked in `docs/future_work.md`.
"""

from __future__ import annotations

import argparse
import sys

import torch

from data.homology_dedup import normalize_protein_name
from dl.datasets.collation import collate_ddg_batch
from dl.datasets.mutation_csv_dataset import MutationCsvDataset, load_records
from dl.losses.config import DDGLossConfig
from dl.training.lightning_module import resolve_active_head_name
from dl.training.object_building import build_model_from_config
from dl.training.run_config import RunConfig, load_run_config
from dl.utils.bin_utils import compute_bin_centers, expected_ddg_from_logits
from dl.utils.checkpoint_io import load_model_weights, require_existing_file
from dl.utils.device import get_default_device
from dl.utils.shape_utils import flatten_last_singleton_dim
from shared.constants import SPLIT_FILES, LossIteration, MutationRecord, SplitName


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict ddG for mutation records with a trained checkpoint.")
    parser.add_argument("--config", required=True, help="Run config YAML matching the checkpoint's architecture (a dl/configs/*.yaml file).")
    parser.add_argument("--ckpt", required=True, help="Path to a PyTorch Lightning checkpoint (.ckpt) produced by train.py.")
    parser.add_argument("--input", required=True, help="CSV of one or more rows matching the MutationRecord schema (same shape as data/splits/*.csv).")
    return parser.parse_args()


def load_prediction_dataset(input_path) -> MutationCsvDataset:
    return MutationCsvDataset(input_path, iteration=LossIteration.ITERATION_2_WITH_HINGE)


def build_predictor(run_config: RunConfig, ckpt_path, device: torch.device) -> torch.nn.Module:
    model = build_model_from_config(run_config.model.model_dump())
    load_model_weights(model, ckpt_path, device)
    model.to(device)
    model.eval()
    return model


def predicted_ddg_from_head_output(head_output: torch.Tensor, active_head: str, run_config: RunConfig) -> torch.Tensor:
    if active_head == "regression":
        return flatten_last_singleton_dim(head_output)
    loss_config = DDGLossConfig(**run_config.loss.params)
    bin_centers = compute_bin_centers(loss_config.bin_edges_kcal_mol)
    return expected_ddg_from_logits(head_output, bin_centers)


def train_complex_names_for_split(split: SplitName) -> frozenset[str]:
    train_records = load_records(SPLIT_FILES[split]["train"])
    return frozenset(normalize_protein_name(record.complex_name or "") for record in train_records)


def print_prediction(record: MutationRecord, predicted_ddg: float, train_complex_names: frozenset[str]) -> None:
    print(f"sample_id={record.sample_id} complex='{record.complex_name}': predicted ddG = {predicted_ddg:.3f} kcal/mol")
    normalized_complex_name = normalize_protein_name(record.complex_name or "")
    if normalized_complex_name in train_complex_names:
        print(f"  complex '{record.complex_name}' was SEEN during training for this checkpoint (seen-complex/new-mutation regime)")
    else:
        print(f"  complex '{record.complex_name}' was NOT seen during training for this checkpoint (unseen-complex regime)")


def main() -> None:
    args = parse_args()
    config_path = require_existing_file(args.config, "Config file")
    ckpt_path = require_existing_file(args.ckpt, "Checkpoint file")
    input_path = require_existing_file(args.input, "Input CSV")

    run_config = load_run_config(config_path)
    device = get_default_device()

    dataset = load_prediction_dataset(input_path)
    if len(dataset) == 0:
        sys.exit(f"No rows found in {input_path}")

    model = build_predictor(run_config, ckpt_path, device)
    active_head = resolve_active_head_name(run_config.model.model_dump())

    inputs, _ = collate_ddg_batch([dataset[index] for index in range(len(dataset))])
    inputs = inputs.to(device)

    with torch.no_grad():
        head_output = model(inputs)["head_outputs"][active_head]
        predicted_ddg = predicted_ddg_from_head_output(head_output, active_head, run_config)

    train_complex_names = train_complex_names_for_split(run_config.dataset.split)
    for record, prediction in zip(dataset.records, predicted_ddg.tolist()):
        print_prediction(record, prediction, train_complex_names)


if __name__ == "__main__":
    main()
