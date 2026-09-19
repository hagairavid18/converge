"""Config + checkpoint driven ΔΔG inference script (plan section 5).

    uv run python predict.py --config dl/configs/<name>.yaml --ckpt path/to.ckpt --input data/splits/same_pdb_allowed_val.csv

Reuses the same building blocks `train.py` and `dl.training.lightning_module`
already use for data loading, embedding-cache reuse, and model construction --
no scoring logic is duplicated here: `dl.datasets.mutation_csv_dataset.
MutationCsvDataset` (which itself calls `data.embedding_pipeline.
entry_embeddings.compute_and_cache_entry_embeddings`, filling in only
whichever embedding-cache pieces are missing and reusing everything already
cached), `dl.datasets.collation.collate_ddg_batch`, and
`dl.training.object_building.build_model_from_config`.

`--input` must be a CSV with the same columns as a `shared.constants.
MutationRecord`-per-row split file (see `data/splits/*.csv`). Every row must
already be a known `sample_id` with a chain map registered by the data
pipeline (`data.processed_io.load_sample_chain_maps`) -- inference here means
scoring an existing SKEMPI mutation record against a trained checkpoint, not
an arbitrary novel sequence with no precomputed structure/chain metadata.

Beyond the predicted ddG, this also reports -- per input row, for the
specific checkpoint pointed at -- whether the row's `complex_name` was seen
during that checkpoint's training split (`dataset.split` in `--config`,
looked up via `shared.constants.SPLIT_FILES`). This is the seen-complex/
new-mutation vs. unseen-complex evaluation-regime distinction tracked
throughout `docs/future_work.md` and `dl.training.lightning_module`'s val
breakouts -- the script never silently assumes one regime, it always states
which applies for the checkpoint the user pointed it at.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from data.homology_dedup import normalize_protein_name
from dl.datasets.collation import collate_ddg_batch
from dl.datasets.mutation_csv_dataset import MutationCsvDataset, load_records
from dl.losses.config import DDGLossConfig
from dl.training.lightning_module import resolve_active_head_name
from dl.training.object_building import build_model_from_config
from dl.training.run_config import RunConfig, load_run_config
from dl.utils.bin_utils import compute_bin_centers, expected_ddg_from_logits
from dl.utils.shape_utils import flatten_last_singleton_dim
from shared.constants import SPLIT_FILES, LossIteration, MutationRecord, SplitName


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict ddG for mutation records with a trained checkpoint.")
    parser.add_argument("--config", required=True, help="Run config YAML matching the checkpoint's architecture (a dl/configs/*.yaml file).")
    parser.add_argument("--ckpt", required=True, help="Path to a PyTorch Lightning checkpoint (.ckpt) produced by train.py.")
    parser.add_argument("--input", required=True, help="CSV of one or more rows matching the MutationRecord schema (same shape as data/splits/*.csv).")
    return parser.parse_args()


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def require_existing_file(path_str: str, description: str) -> Path:
    path = Path(path_str)
    if not path.is_file():
        sys.exit(f"{description} not found: {path}")
    return path


def load_prediction_dataset(input_path: Path) -> MutationCsvDataset:
    return MutationCsvDataset(input_path, iteration=LossIteration.ITERATION_2_WITH_HINGE)


def strip_model_submodule_prefix(state_dict: dict) -> dict:
    """`DDGLightningModule` stores the `DDGPredictor` as `self.model`, so its
    checkpointed `state_dict` keys are prefixed `model.` -- this rebuilds a
    plain `DDGPredictor` state dict from that, ignoring any other logged
    submodule (losses/metrics carry no learnable state of their own).
    """
    prefix = "model."
    return {key[len(prefix):]: value for key, value in state_dict.items() if key.startswith(prefix)}


def load_model_weights(model: torch.nn.Module, ckpt_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(strip_model_submodule_prefix(checkpoint["state_dict"]))


def build_predictor(run_config: RunConfig, ckpt_path: Path, device: torch.device) -> torch.nn.Module:
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
    device = resolve_device()

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
