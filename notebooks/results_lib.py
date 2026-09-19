"""Shared computation library for `notebooks/04_results.ipynb`.

Consolidates `classical_regressors_probe.py` (classical-regressor baselines on
the model's frozen pooled representation) and
`final_sota_vs_baseline_histogram_probe.py` (bucketed MAE vs. a naive
per-complex-mean baseline) into one module.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from torch.utils.data import DataLoader

from data.homology_dedup import normalize_protein_name
from dl.datasets.collation import collate_ddg_batch
from dl.datasets.config import DatasetConfig
from dl.datasets.dataset_builder import build_real_datasets, build_train_and_val_datasets, resolve_loss_iteration
from dl.datasets.mutation_csv_dataset import keep_bounded_only, load_records
from dl.metrics.regression_metrics import compute_magnitude_bucket_indices
from dl.models.config import ModelConfig
from dl.models.ddg_model import DDGPredictor
from dl.training.lightning_module import DDGLightningModule
from dl.training.run_config import RunConfig, load_run_config
from dl.utils.constants import MAGNITUDE_BUCKET_NAMES
from dl.utils.label_codes import BOUNDED_ID
from shared.constants import DDG_NUM_BINS, MutationRecord, SPLIT_FILES, SplitName

DEFAULT_RESULTS_JSON_PATH = Path(__file__).resolve().parent / "final_sota_vs_baseline_histogram_results.json"

TRAIN_CHAIN_ROLE_MIX = {"antigen_only": 0.343, "antibody_only": 0.599, "both": 0.058}
VAL_CHAIN_ROLE_MIX = {"antigen_only": 0.267, "antibody_only": 0.653, "both": 0.079}

# ---------------------------------------------------------------------------
# Classical-regressor baseline on the frozen pooled representation
# ---------------------------------------------------------------------------

CLASSICAL_REGRESSORS = {
    "dummy_mean": DummyRegressor(strategy="mean"),
    "linear": LinearRegression(),
    "ridge_alpha10": Ridge(alpha=10.0),
    "knn_k10": KNeighborsRegressor(n_neighbors=10),
    "random_forest": RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42),
    "hist_gradient_boosting": HistGradientBoostingRegressor(random_state=42),
    "svr_rbf": SVR(kernel="rbf"),
}


def build_frozen_pooled_model_config(embedding_source_mode: str = "structure_only") -> ModelConfig:
    return ModelConfig(
        embedding_source_mode=embedding_source_mode,
        active_heads=(),
        pooling_strategy="norm_softmax_by_chain_role",
        pooling_temperature=5.0,
    )


def extract_pooled_features(model: DDGPredictor, loader: DataLoader) -> dict[str, np.ndarray]:
    pooled, target_ddg, label_type_id, complex_seen_in_train = [], [], [], []
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            pooled.append(outputs["pooled_representation"].numpy())
            target_ddg.append(labels.target_ddg.numpy())
            label_type_id.append(labels.label_type_id.numpy())
            complex_seen_in_train.append(labels.complex_seen_in_train.numpy())
    return {
        "pooled": np.concatenate(pooled),
        "target_ddg": np.concatenate(target_ddg),
        "label_type_id": np.concatenate(label_type_id),
        "complex_seen_in_train": np.concatenate(complex_seen_in_train),
    }


def bounded_only(fields: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    mask = fields["label_type_id"] == BOUNDED_ID
    return {key: value[mask] for key, value in fields.items()}


def build_pooled_feature_splits(
    model: DDGPredictor, model_config: ModelConfig, dataset_config: DatasetConfig
) -> tuple[dict, dict, dict]:
    train_ds, val_ds = build_train_and_val_datasets(dataset_config, model_config, DDG_NUM_BINS)
    train_loader = DataLoader(train_ds, batch_size=dataset_config.batch_size, shuffle=False, collate_fn=collate_ddg_batch)
    val_loader = DataLoader(val_ds, batch_size=dataset_config.batch_size, shuffle=False, collate_fn=collate_ddg_batch)

    train_fields = bounded_only(extract_pooled_features(model, train_loader))
    val_fields = bounded_only(extract_pooled_features(model, val_loader))
    val_new_complex = {key: value[~val_fields["complex_seen_in_train"]] for key, value in val_fields.items()}
    val_seen_complex = {key: value[val_fields["complex_seen_in_train"]] for key, value in val_fields.items()}
    return train_fields, val_new_complex, val_seen_complex


def evaluate_regressor(
    name: str,
    regressor,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_new_complex_x: np.ndarray,
    val_new_complex_y: np.ndarray,
    val_seen_complex_x: np.ndarray,
    val_seen_complex_y: np.ndarray,
) -> dict:
    regressor.fit(train_x, train_y)
    return {
        "regressor": name,
        "train_mae": mean_absolute_error(train_y, regressor.predict(train_x)),
        "val_new_complex_mae": mean_absolute_error(val_new_complex_y, regressor.predict(val_new_complex_x)),
        "val_complex_seen_in_train_mae": mean_absolute_error(val_seen_complex_y, regressor.predict(val_seen_complex_x)),
    }


def run_classical_regressor_comparison(
    embedding_source_mode: str = "structure_only", split: SplitName = SplitName.SAME_PDB_ALLOWED, batch_size: int = 32
) -> pd.DataFrame:
    model_config = build_frozen_pooled_model_config(embedding_source_mode)
    model = DDGPredictor(model_config)
    model.eval()
    dataset_config = DatasetConfig(split=split, batch_size=batch_size)
    train_fields, val_new_complex, val_seen_complex = build_pooled_feature_splits(model, model_config, dataset_config)

    scaler = StandardScaler().fit(train_fields["pooled"])
    train_x = scaler.transform(train_fields["pooled"])
    val_new_complex_x = scaler.transform(val_new_complex["pooled"])
    val_seen_complex_x = scaler.transform(val_seen_complex["pooled"])

    rows = [
        evaluate_regressor(
            name,
            regressor,
            train_x,
            train_fields["target_ddg"],
            val_new_complex_x,
            val_new_complex["target_ddg"],
            val_seen_complex_x,
            val_seen_complex["target_ddg"],
        )
        for name, regressor in CLASSICAL_REGRESSORS.items()
    ]
    return pd.DataFrame(rows).set_index("regressor")


# ---------------------------------------------------------------------------
# Bucketed MAE vs. a naive per-complex-mean baseline
# ---------------------------------------------------------------------------


def load_bucketed_results(path: Path = DEFAULT_RESULTS_JSON_PATH) -> dict:
    return json.loads(path.read_text())


def bucketed_results_to_dataframe(results: dict) -> pd.DataFrame:
    rows = [
        {
            "bucket": bucket,
            "n": results["sota_bucket_n"][bucket],
            "model_mae": results["sota_bucket_mae"][bucket],
            "baseline_mae": results["baseline_bucket_mae"][bucket],
        }
        for bucket in MAGNITUDE_BUCKET_NAMES
    ]
    rows.append(
        {
            "bucket": "overall",
            "n": sum(results["sota_bucket_n"].values()),
            "model_mae": results["sota_overall_mae"],
            "baseline_mae": results["baseline_overall_mae"],
        }
    )
    return pd.DataFrame(rows).set_index("bucket")


def plot_bucketed_mae_bar_chart(bucket_df: pd.DataFrame) -> plt.Figure:
    model_color, baseline_color = "#2a78d6", "#898781"
    x = np.arange(len(bucket_df))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, bucket_df["model_mae"], width, label="current-best model", color=model_color)
    ax.bar(x + width / 2, bucket_df["baseline_mae"], width, label="naive per-complex-mean baseline", color=baseline_color)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{bucket}\n(n={n})" for bucket, n in zip(bucket_df.index, bucket_df["n"])], rotation=0)
    ax.set_ylabel("MAE (kcal/mol)")
    ax.set_title("Bucketed MAE: model vs. naive baseline (same_pdb_allowed val)")
    ax.legend()
    fig.tight_layout()
    return fig


def check_env_matches_checkpoint() -> bool:
    import os

    return os.environ.get("SKEMPI_USE_SAPROT_STRUCTURE") == "1" and os.environ.get("SKEMPI_USE_SMALL_CHECKPOINTS") == "0"


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
        raise ValueError(f"Checkpoint hyperparameters do not match {run_config}: {mismatches}")
    if hyperparameters["split_name"] != run_config.dataset.split.value:
        raise ValueError(f"Checkpoint split_name {hyperparameters['split_name']!r} != {run_config.dataset.split.value!r}")


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
        totals.setdefault(normalize_protein_name(record.complex_name or ""), []).append(record.ddg_kcal_mol)
    return {complex_key: sum(values) / len(values) for complex_key, values in totals.items()}


def global_mean_ddg(records: list[MutationRecord]) -> float:
    return sum(record.ddg_kcal_mol for record in records) / len(records)


def baseline_prediction_for_record(record: MutationRecord, complex_means: dict[str, float], global_mean: float) -> float:
    return complex_means.get(normalize_protein_name(record.complex_name or ""), global_mean)


def bucket_indices_for_records(records: list[MutationRecord]) -> torch.Tensor:
    targets = torch.tensor([record.ddg_kcal_mol for record in records], dtype=torch.float32)
    return compute_magnitude_bucket_indices(targets)


def compute_bucket_counts(records: list[MutationRecord]) -> dict[str, int]:
    counts = {name: 0 for name in MAGNITUDE_BUCKET_NAMES}
    for index in bucket_indices_for_records(records).tolist():
        counts[MAGNITUDE_BUCKET_NAMES[index]] += 1
    return counts


def compute_baseline_report(train_records: list[MutationRecord], val_records: list[MutationRecord]) -> tuple[float, dict]:
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
        name: (abs_error_sum[name] / bucket_counts[name] if bucket_counts[name] > 0 else None) for name in MAGNITUDE_BUCKET_NAMES
    }
    return total_abs_error / len(val_records), bucket_mae


def recompute_bucketed_results(config_path: Path, checkpoint_path: Path) -> dict:
    run_config = load_run_config(config_path)
    logged, val_loader = run_sota_validation(run_config, checkpoint_path)
    sota_overall_mae, sota_bucket_mae = extract_sota_metrics(logged, run_config.dataset.split.value)

    val_records: list[MutationRecord] = val_loader.dataset.records
    sota_bucket_n = compute_bucket_counts(val_records)

    train_records = bounded_records(SPLIT_FILES[SplitName.SAME_PDB_ALLOWED]["train"])
    baseline_overall_mae, baseline_bucket_mae = compute_baseline_report(train_records, val_records)

    return {
        "sota_checkpoint": str(checkpoint_path),
        "sota_overall_mae": sota_overall_mae,
        "sota_bucket_mae": sota_bucket_mae,
        "sota_bucket_n": sota_bucket_n,
        "baseline_overall_mae": baseline_overall_mae,
        "baseline_bucket_mae": baseline_bucket_mae,
        "baseline_bucket_n": sota_bucket_n,
    }


# ---------------------------------------------------------------------------
# Train/val chain-role composition caveat (mutation_side_val_reliability_probe.py)
# ---------------------------------------------------------------------------


def chain_role_mix_comparison_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "train (n=855)": TRAIN_CHAIN_ROLE_MIX,
            "val (n=202)": VAL_CHAIN_ROLE_MIX,
        }
    ).round(3)
