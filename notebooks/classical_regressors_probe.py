"""One-off check: is a learned head (linear/MLP) even necessary, or does a
classical (non-neural) regressor on the same pooled feature do as well?

Everything upstream of the prediction head -- embedding fusion, confidence
weighting, subtraction, and pooling -- is parameter-free, so
`DDGPredictor(config)` with `active_heads=()` gives a fixed, un-trained
`pooled_representation` per sample. This script extracts that vector for
every train/val row once, then fits several scikit-learn regressors directly
on it, using the same new-complex / complex-seen-in-train breakdown as the
real training loop (`dl.training.lightning_module`) so results are directly
comparable to the Comet MAE numbers from the neural heads. Defaults match the
confirmed current architecture (`structure_only`, `norm_softmax_by_chain_role`,
temperature=5.0, `same_pdb_allowed` split); override via CLI args (see
`main`'s `sys.argv` handling) to probe other configurations.

Run:
  SKEMPI_USE_SAPROT_STRUCTURE=1 uv run python notebooks/classical_regressors_probe.py
"""

from __future__ import annotations

import sys

import numpy as np
import torch
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from torch.utils.data import DataLoader

from dl.datasets.collation import collate_ddg_batch
from dl.datasets.config import DatasetConfig
from dl.datasets.dataset_builder import build_train_and_val_datasets
from dl.models.config import ModelConfig
from dl.models.ddg_model import DDGPredictor
from dl.utils.label_codes import BOUNDED_ID
from shared.constants import DDG_NUM_BINS, SplitName

REGRESSORS = {
    "dummy_mean": DummyRegressor(strategy="mean"),
    "linear": LinearRegression(),
    "ridge_alpha10": Ridge(alpha=10.0),
    "knn_k10": KNeighborsRegressor(n_neighbors=10),
    "random_forest": RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42),
    "hist_gradient_boosting": HistGradientBoostingRegressor(random_state=42),
    "svr_rbf": SVR(kernel="rbf"),
}


def extract_pooled_features(model: DDGPredictor, loader: DataLoader) -> dict[str, np.ndarray]:
    pooled, target_ddg, label_type_id, complex_seen_in_train = [], [], [], []
    model.eval()
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


def evaluate_regressor(name, regressor, train_x, train_y, val_new_complex_x, val_new_complex_y, val_seen_complex_x, val_seen_complex_y) -> None:
    regressor.fit(train_x, train_y)
    train_mae = mean_absolute_error(train_y, regressor.predict(train_x))
    new_complex_mae = mean_absolute_error(val_new_complex_y, regressor.predict(val_new_complex_x))
    seen_complex_mae = mean_absolute_error(val_seen_complex_y, regressor.predict(val_seen_complex_x))
    print(f"{name:24s} train={train_mae:.3f}  val_new_complex={new_complex_mae:.3f}  val_complex_seen_in_train={seen_complex_mae:.3f}")


def main() -> None:
    embedding_source_mode = sys.argv[1] if len(sys.argv) > 1 else "structure_only"
    split = SplitName(sys.argv[2]) if len(sys.argv) > 2 else SplitName.SAME_PDB_ALLOWED
    print(f"embedding_source_mode={embedding_source_mode}  split={split.value}")
    model_config = ModelConfig(
        embedding_source_mode=embedding_source_mode,
        active_heads=(),
        pooling_strategy="norm_softmax_by_chain_role",
        pooling_temperature=5.0,
    )
    dataset_config = DatasetConfig(split=split, batch_size=32)
    model = DDGPredictor(model_config)

    train_ds, val_ds = build_train_and_val_datasets(dataset_config, model_config, DDG_NUM_BINS)
    train_loader = DataLoader(train_ds, batch_size=dataset_config.batch_size, shuffle=False, collate_fn=collate_ddg_batch)
    val_loader = DataLoader(val_ds, batch_size=dataset_config.batch_size, shuffle=False, collate_fn=collate_ddg_batch)

    train_fields = bounded_only(extract_pooled_features(model, train_loader))
    val_fields = bounded_only(extract_pooled_features(model, val_loader))
    val_new_complex = {key: value[~val_fields["complex_seen_in_train"]] for key, value in val_fields.items()}
    val_seen_complex = {key: value[val_fields["complex_seen_in_train"]] for key, value in val_fields.items()}

    print(
        f"n_train={len(train_fields['target_ddg'])}  "
        f"n_val_new_complex={len(val_new_complex['target_ddg'])}  "
        f"n_val_complex_seen_in_train={len(val_seen_complex['target_ddg'])}"
    )

    scaler = StandardScaler().fit(train_fields["pooled"])
    train_x = scaler.transform(train_fields["pooled"])
    val_new_complex_x = scaler.transform(val_new_complex["pooled"])
    val_seen_complex_x = scaler.transform(val_seen_complex["pooled"])

    for name, regressor in REGRESSORS.items():
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


if __name__ == "__main__":
    main()
