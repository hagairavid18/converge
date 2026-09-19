"""Final round wrap-up: current-best-model bucketed MAE vs. a naive
per-complex-mean baseline (validation experiment, not production code).

"Current best model" ("SOTA") = `dl/configs/
structure_saprot_only_linear_splitpool_temp5_same_pdb_allowed_20ep.yaml`
(SaProt structure-only embeddings, linear regression head,
`norm_softmax_by_chain_role` ("splitpool") pooling at temperature 5.0,
`same_pdb_allowed` split, 20 epochs). Two independent training runs of this
exact config exist as checkpoints (see `mutation_side_val_reliability_probe.py`
for how they were identified from each `.ckpt`'s embedded
`hyper_parameters`, since no config metadata is stored next to checkpoints):
`4411e9ff0f8f467cb17c5ab60e51a926/checkpoints/epoch=19-step=940.ckpt` (val
MAE 0.843, the better of the two) and
`6a29445812a440f2b2d70d64cb289ae0/checkpoints/epoch=19-step=940.ckpt` (val
MAE 1.113). This probe uses the better one as SOTA.

Both checkpoints predate the `BucketedMAE` validation metric
(`dl.metrics.regression_metrics.BucketedMAE`, wired into
`dl.training.lightning_module.DDGLightningModule` as
`val_metric_mae_by_bucket/{bucket_name}/{split}`), so getting the per-bucket
breakdown requires re-running validation for this checkpoint under the
*current* code -- the checkpoint weights are unaffected by intervening code
changes, only which metrics get computed/logged at eval time changes. This
reuses the exact same reconstruction and evaluation machinery `train.py`
uses for `--mode eval` (`DDGLightningModule.load_from_checkpoint` off the
matching run config, `pl.Trainer.validate(...)`) rather than a hand-rolled
forward loop, so the result is exactly what an `eval` run of this config
against this checkpoint would log.

Goal 1 (SOTA): run that validation pass and read off the 5
`val_metric_mae_by_bucket/{bucket_name}/same_pdb_allowed` values plus the
pooled `val_metric/same_pdb_allowed`, as a sanity check that the right
checkpoint/data loaded (expect overall MAE close to the previously-recorded
0.843).

Goal 2 (naive baseline): for every bounded `same_pdb_allowed_val.csv` row,
the "prediction" is the mean bounded `ddg_kcal_mol` of that row's complex's
*training* rows (`same_pdb_allowed_train.csv`, grouped by
`data.homology_dedup.normalize_protein_name(complex_name)`), falling back to
the global mean bounded train ddG for a complex with zero train presence at
all (the `same_pdb_allowed` split's held-out "new complex" bucket -- see
`data/splitting.py`). Each val row is bucketed by its own actual
`target_ddg` using the same boundary logic as `BucketedMAE`
(`dl.metrics.regression_metrics.compute_magnitude_bucket_indices` /
`dl.utils.constants.MAGNITUDE_BUCKET_BOUNDARIES_KCAL_MOL`), so the two
reports are directly comparable bucket-for-bucket.

Sample counts per bucket are computed once from the val dataset's own
bounded targets (`MutationCsvDataset` for `same_pdb_allowed_val` is already
bounded-only, per `LossIteration.ITERATION_1_BOUNDED_ONLY`) and shared
between the SOTA and baseline reports, since both are evaluated over
exactly the same val samples.

Requires the exact env vars the checkpoint was trained under -- inferred
in `mutation_side_val_reliability_probe.py` from the regression head's
input width baked into the checkpoint (2560 = 2x SaProt's *full* 650M
hidden dim, 1280, not the 480-dim CPU-dev checkpoint) -- set at the top of
this file (before any project import resolves `shared.constants`) so this
runs standalone as:

    uv run python notebooks/final_sota_vs_baseline_histogram_probe.py

Output: prints both reports and writes
`notebooks/final_sota_vs_baseline_histogram_results.json`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ["SKEMPI_USE_SAPROT_STRUCTURE"] = "1"
os.environ["SKEMPI_USE_SMALL_CHECKPOINTS"] = "0"

import json  # noqa: E402

import pytorch_lightning as pl  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from data.homology_dedup import normalize_protein_name  # noqa: E402
from dl.datasets.collation import collate_ddg_batch  # noqa: E402
from dl.datasets.dataset_builder import build_real_datasets, resolve_loss_iteration  # noqa: E402
from dl.datasets.mutation_csv_dataset import keep_bounded_only, load_records  # noqa: E402
from dl.metrics.regression_metrics import compute_magnitude_bucket_indices  # noqa: E402
from dl.training.lightning_module import DDGLightningModule  # noqa: E402
from dl.training.run_config import RunConfig, load_run_config  # noqa: E402
from dl.utils.constants import MAGNITUDE_BUCKET_NAMES  # noqa: E402
from shared.constants import MutationRecord, SPLIT_FILES, SplitName  # noqa: E402

CONFIG_PATH = _REPO_ROOT / "dl/configs/structure_saprot_only_linear_splitpool_temp5_same_pdb_allowed_20ep.yaml"
SOTA_CHECKPOINT_RELATIVE = "4411e9ff0f8f467cb17c5ab60e51a926/checkpoints/epoch=19-step=940.ckpt"
SOTA_CHECKPOINT_PATH = _REPO_ROOT / ".cometml-runs/skempi-ddg" / SOTA_CHECKPOINT_RELATIVE
PRIOR_SOTA_VAL_MAE = 0.843
SANITY_TOLERANCE = 0.05

RESULTS_PATH = Path(__file__).resolve().parent / "final_sota_vs_baseline_histogram_results.json"

BATCH_SIZE = 16


def check_env() -> None:
    if os.environ.get("SKEMPI_USE_SAPROT_STRUCTURE") != "1" or os.environ.get("SKEMPI_USE_SMALL_CHECKPOINTS") != "0":
        sys.exit("This checkpoint requires SKEMPI_USE_SAPROT_STRUCTURE=1 and SKEMPI_USE_SMALL_CHECKPOINTS=0.")


def load_checkpoint_hyperparameters(ckpt_path: Path) -> dict:
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return checkpoint["hyper_parameters"]


def confirm_checkpoint_matches_config(hyperparameters: dict, run_config: RunConfig) -> None:
    actual_model_params = hyperparameters["model_config"]["params"]
    mismatches = {
        key: (actual_model_params.get(key), expected)
        for key, expected in run_config.model.params.items()
        if actual_model_params.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Checkpoint hyperparameters do not match {CONFIG_PATH.name}: {mismatches}")
    if hyperparameters["split_name"] != run_config.dataset.split.value:
        raise ValueError(
            f"Checkpoint split_name {hyperparameters['split_name']!r} != {run_config.dataset.split.value!r}"
        )


def build_lightning_module_from_checkpoint(run_config: RunConfig, ckpt_path: Path) -> DDGLightningModule:
    hyperparameters = load_checkpoint_hyperparameters(ckpt_path)
    confirm_checkpoint_matches_config(hyperparameters, run_config)
    kwargs = dict(
        model_config=run_config.model.model_dump(),
        loss_config=run_config.loss.model_dump(),
        metrics_config=run_config.metrics.model_dump(),
        learning_rate=run_config.trainer.learning_rate,
        weight_decay=run_config.trainer.weight_decay,
        split_name=run_config.dataset.split,
    )
    module = DDGLightningModule.load_from_checkpoint(str(ckpt_path), map_location="cpu", **kwargs)
    module.eval()
    return module


def build_val_loader(run_config: RunConfig) -> DataLoader:
    iteration = resolve_loss_iteration(run_config.loss.params)
    _, val_dataset = build_real_datasets(run_config.dataset, iteration)
    return DataLoader(
        val_dataset,
        batch_size=run_config.dataset.batch_size,
        shuffle=False,
        num_workers=run_config.dataset.num_workers,
        collate_fn=collate_ddg_batch,
    )


def run_sota_validation(run_config: RunConfig, ckpt_path: Path) -> tuple[dict, DataLoader]:
    module = build_lightning_module_from_checkpoint(run_config, ckpt_path)
    val_loader = build_val_loader(run_config)
    trainer = pl.Trainer(accelerator="cpu", devices=1, logger=False, enable_progress_bar=False, enable_checkpointing=False)
    results = trainer.validate(module, dataloaders=val_loader)
    return results[0], val_loader


def extract_sota_metrics(logged: dict, split_name: str) -> tuple[float, dict[str, float]]:
    overall_mae = float(logged[f"val_metric/{split_name}"])
    bucket_mae = {
        bucket_name: float(logged[f"val_metric_mae_by_bucket/{bucket_name}/{split_name}"])
        for bucket_name in MAGNITUDE_BUCKET_NAMES
        if f"val_metric_mae_by_bucket/{bucket_name}/{split_name}" in logged
    }
    return overall_mae, bucket_mae


def bounded_records(csv_path: Path) -> list[MutationRecord]:
    return keep_bounded_only(load_records(csv_path))


def complex_mean_ddg(records: list[MutationRecord]) -> dict[str, float]:
    totals: dict[str, list[float]] = {}
    for record in records:
        complex_key = normalize_protein_name(record.complex_name or "")
        totals.setdefault(complex_key, []).append(record.ddg_kcal_mol)
    return {complex_key: sum(values) / len(values) for complex_key, values in totals.items()}


def global_mean_ddg(records: list[MutationRecord]) -> float:
    values = [record.ddg_kcal_mol for record in records]
    return sum(values) / len(values)


def baseline_prediction_for_record(
    record: MutationRecord, complex_means: dict[str, float], global_mean: float
) -> float:
    complex_key = normalize_protein_name(record.complex_name or "")
    return complex_means.get(complex_key, global_mean)


def bucket_indices_for_records(records: list[MutationRecord]) -> torch.Tensor:
    targets = torch.tensor([record.ddg_kcal_mol for record in records], dtype=torch.float32)
    return compute_magnitude_bucket_indices(targets)


def compute_bucket_counts(records: list[MutationRecord]) -> dict[str, int]:
    bucket_indices = bucket_indices_for_records(records)
    counts = {name: 0 for name in MAGNITUDE_BUCKET_NAMES}
    for index in bucket_indices.tolist():
        counts[MAGNITUDE_BUCKET_NAMES[index]] += 1
    return counts


def compute_baseline_report(train_records: list[MutationRecord], val_records: list[MutationRecord]) -> tuple[float, dict[str, float]]:
    complex_means = complex_mean_ddg(train_records)
    global_mean = global_mean_ddg(train_records)

    bucket_indices = bucket_indices_for_records(val_records)
    abs_error_sum = {name: 0.0 for name in MAGNITUDE_BUCKET_NAMES}
    bucket_counts = {name: 0 for name in MAGNITUDE_BUCKET_NAMES}
    total_abs_error = 0.0

    for record, bucket_index in zip(val_records, bucket_indices.tolist()):
        prediction = baseline_prediction_for_record(record, complex_means, global_mean)
        abs_error = abs(prediction - record.ddg_kcal_mol)
        bucket_name = MAGNITUDE_BUCKET_NAMES[bucket_index]
        abs_error_sum[bucket_name] += abs_error
        bucket_counts[bucket_name] += 1
        total_abs_error += abs_error

    bucket_mae = {
        name: (abs_error_sum[name] / bucket_counts[name] if bucket_counts[name] > 0 else None)
        for name in MAGNITUDE_BUCKET_NAMES
    }
    overall_mae = total_abs_error / len(val_records)
    return overall_mae, bucket_mae


def none_if_nan(value: float) -> float | None:
    return None if value != value else value


def run_sanity_checks(sota_overall_mae: float, sota_bucket_n: dict[str, int], total_bounded_val: int) -> None:
    if abs(sota_overall_mae - PRIOR_SOTA_VAL_MAE) > SANITY_TOLERANCE:
        print(
            f"WARNING: sota_overall_mae={sota_overall_mae:.4f} differs from the previously-recorded "
            f"{PRIOR_SOTA_VAL_MAE} by more than {SANITY_TOLERANCE} -- check checkpoint/data loading."
        )
    else:
        print(f"Sanity check passed: sota_overall_mae={sota_overall_mae:.4f} ~= prior {PRIOR_SOTA_VAL_MAE}")
    bucket_sum = sum(sota_bucket_n.values())
    if bucket_sum != total_bounded_val:
        print(f"WARNING: bucket counts sum to {bucket_sum}, expected {total_bounded_val}")
    else:
        print(f"Sanity check passed: bucket counts sum to {bucket_sum} (== total bounded val rows)")


def print_report(title: str, overall_mae: float, bucket_mae: dict, bucket_n: dict[str, int]) -> None:
    print(f"\n=== {title} ===")
    for name in MAGNITUDE_BUCKET_NAMES:
        mae_value = bucket_mae[name]
        mae_str = f"{mae_value:.4f}" if mae_value is not None else "n/a"
        print(f"  {name:22s} n={bucket_n[name]:3d}  MAE={mae_str}")
    print(f"  {'overall':22s} n={sum(bucket_n.values()):3d}  MAE={overall_mae:.4f}")


def main() -> None:
    check_env()

    run_config = load_run_config(CONFIG_PATH)
    logged, val_loader = run_sota_validation(run_config, SOTA_CHECKPOINT_PATH)
    sota_overall_mae, sota_bucket_mae = extract_sota_metrics(logged, run_config.dataset.split.value)

    val_records: list[MutationRecord] = val_loader.dataset.records
    sota_bucket_n = compute_bucket_counts(val_records)

    train_records = bounded_records(SPLIT_FILES[SplitName.SAME_PDB_ALLOWED]["train"])
    baseline_overall_mae, baseline_bucket_mae = compute_baseline_report(train_records, val_records)
    baseline_bucket_n = compute_bucket_counts(val_records)

    run_sanity_checks(sota_overall_mae, sota_bucket_n, len(val_records))
    print_report("SOTA (4411e9ff..., structure_saprot_only_linear_splitpool_temp5_same_pdb_allowed_20ep)", sota_overall_mae, sota_bucket_mae, sota_bucket_n)
    print_report("Naive per-complex-mean baseline", baseline_overall_mae, baseline_bucket_mae, baseline_bucket_n)

    results = {
        "sota_checkpoint": SOTA_CHECKPOINT_RELATIVE,
        "sota_overall_mae": sota_overall_mae,
        "sota_bucket_mae": {name: none_if_nan(sota_bucket_mae.get(name, float("nan"))) for name in MAGNITUDE_BUCKET_NAMES},
        "sota_bucket_n": sota_bucket_n,
        "baseline_overall_mae": baseline_overall_mae,
        "baseline_bucket_mae": baseline_bucket_mae,
        "baseline_bucket_n": baseline_bucket_n,
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
