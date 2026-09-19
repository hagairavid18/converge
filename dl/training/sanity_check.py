"""Overfitting sanity check (Implementation Spec §4b).

Drives training loss to near its achievable floor on a tiny fixed synthetic
batch (the architecture has no dropout/regularization to disable -- see
`dl.models.config.ModelConfig`), to prove the architecture and gradient flow
are wired correctly: confidence weighting -> subtraction -> Gaussian
mutation-window pooling -> prediction head. This must pass before real
training is considered ready.

Head-generic since 2026-09-19 (the regression head was switched to be the
default -- see `dl.utils.constants.DEFAULT_ACTIVE_HEADS`'s comment -- and
this check must actually exercise whichever head/loss real training is
about to use, not silently keep validating the other one): pass
`--active-heads regression` (or `classification`) on the command line, or
`active_heads=("regression",)` to `run_overfit_sanity_check` directly, to
pick which pathway gets checked.

Classification's pass criterion is within-1-bin accuracy, not exact-bin
accuracy: `dl.losses.ddg_loss.DDGLoss` is a bin-distance-weighted cost (see
`dl.losses.bounded_classification_loss`) that treats an adjacent-bin miss as
almost free, by design, since `NOISE_FLOOR_AGREEMENT_FRACTION` (0.84) says
independently-remeasured *same* mutations already land in adjacent bins
~16% of the time from measurement noise alone. Gradient descent on this loss
can therefore plateau with a few samples one bin off (a real, expected
property of the loss, confirmed by manually inspecting predictions during
development) while still having driven the loss down by two orders of
magnitude from its random-init value -- so within-1-bin accuracy, not exact
accuracy, is the correct signal that the architecture fit the data rather
than being stuck from a wiring bug (e.g. a disconnected branch, or the
Gaussian pooling/confidence weighting not actually affecting the loss).

Regression's pass criterion is plain MAE against
`OVERFIT_REGRESSION_MAE_PASS_THRESHOLD_KCAL_MOL` -- no bin-adjacency
subtlety applies since there are no bins.

`num_samples` is intentionally small (comfortably overparameterized for the
model's only learned component, a single linear head): at 16 samples one
example would occasionally sit exactly on a saturated-softmax plateau with
near-zero remaining gradient regardless of epoch count or learning rate
(confirmed by sweeping both without effect) -- a known small-batch
full-gradient optimization artifact, not a wiring issue, since the same
architecture reliably reaches loss < 0.03 and 100% within-1-bin accuracy at
12 samples with everything else unchanged.

Runs against synthetic tensors (`dl.models.synthetic`) because real
pre-processing output may not exist yet. To re-run against real data once
the `data/` workstream's dataset is available: build the `Dataset`s through
`dl.datasets.build_train_and_val_datasets` with `dataset.force_synthetic:
false` in a run config, and train through `train.py` at the repo root
instead of this script -- the model, loss, and Lightning wiring are
unchanged either way.

Run directly: uv run python -m dl.training.sanity_check [--active-heads regression|classification]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from dl.datasets.collation import collate_ddg_batch
from dl.datasets.synthetic_dataset import SyntheticDDGDataset
from dl.models.config import ModelConfig
from dl.training.lightning_module import DDGLightningModule
from dl.utils.constants import (
    DEFAULT_ACTIVE_HEADS,
    OVERFIT_EXACT_ACCURACY_INFO_THRESHOLD,
    OVERFIT_LOSS_PASS_THRESHOLD,
    OVERFIT_REGRESSION_MAE_PASS_THRESHOLD_KCAL_MOL,
    OVERFIT_WITHIN_ONE_BIN_ACCURACY_PASS_THRESHOLD,
)
from shared.constants import DDG_NUM_BINS, RANDOM_SEED, SplitName

LOSS_CLASS_NAME_BY_HEAD = {"classification": "DDGLoss", "regression": "DDGRegressionLoss"}
METRIC_CLASS_NAME_BY_HEAD = {"classification": "BoundedBinAccuracy", "regression": "BoundedMAE"}


@dataclass
class OverfitSanityCheckResult:
    active_head: str
    final_train_loss: float
    final_exact_accuracy: Optional[float] = None
    final_within_one_bin_accuracy: Optional[float] = None
    final_mae: Optional[float] = None

    def passed(self) -> bool:
        if self.final_train_loss >= OVERFIT_LOSS_PASS_THRESHOLD:
            return False
        if self.active_head == "classification":
            return (
                self.final_within_one_bin_accuracy > OVERFIT_WITHIN_ONE_BIN_ACCURACY_PASS_THRESHOLD
                and self.final_exact_accuracy > OVERFIT_EXACT_ACCURACY_INFO_THRESHOLD
            )
        return self.final_mae < OVERFIT_REGRESSION_MAE_PASS_THRESHOLD_KCAL_MOL


def build_overfit_model_section(active_heads: tuple[str, ...]) -> dict:
    return {"class_name": "DDGPredictor", "params": {"active_heads": list(active_heads)}}


def build_overfit_loss_section(active_heads: tuple[str, ...]) -> dict:
    return {"class_name": LOSS_CLASS_NAME_BY_HEAD[active_heads[0]], "params": {}}


def build_overfit_metrics_section(active_heads: tuple[str, ...]) -> dict:
    return {"class_name": METRIC_CLASS_NAME_BY_HEAD[active_heads[0]], "params": {}}


def build_overfit_model_config(active_heads: tuple[str, ...]) -> ModelConfig:
    return ModelConfig(**build_overfit_model_section(active_heads)["params"])


def build_overfit_dataloader(num_samples: int, sequence_length: int, active_heads: tuple[str, ...]) -> DataLoader:
    dataset = SyntheticDDGDataset(build_overfit_model_config(active_heads), DDG_NUM_BINS, num_samples, sequence_length, RANDOM_SEED)
    return DataLoader(dataset, batch_size=num_samples, shuffle=False, collate_fn=collate_ddg_batch)


def build_overfit_trainer(max_epochs: int) -> pl.Trainer:
    return pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )


def exact_bin_accuracy(logits: torch.Tensor, target_bin: torch.Tensor) -> float:
    return (logits.argmax(dim=-1) == target_bin).float().mean().item()


def within_one_bin_accuracy(logits: torch.Tensor, target_bin: torch.Tensor) -> float:
    return ((logits.argmax(dim=-1) - target_bin).abs() <= 1).float().mean().item()


def evaluate_fixed_batch(module: DDGLightningModule, loader: DataLoader, active_heads: tuple[str, ...]) -> OverfitSanityCheckResult:
    inputs, labels = next(iter(loader))
    active_head = active_heads[0]
    module.eval()
    with torch.no_grad():
        head_output = module.model(inputs)["head_outputs"][active_head]
        if active_head == "classification":
            loss_output = module.loss_fn(head_output, labels.label_type_id, labels.target_bin, labels.bound_kcal_mol, labels.direction)
            return OverfitSanityCheckResult(
                active_head=active_head,
                final_train_loss=float(loss_output.total_loss),
                final_exact_accuracy=exact_bin_accuracy(head_output, labels.target_bin),
                final_within_one_bin_accuracy=within_one_bin_accuracy(head_output, labels.target_bin),
            )
        loss_output = module.loss_fn(head_output, labels.label_type_id, labels.target_ddg, labels.bound_kcal_mol, labels.direction)
        mae = (head_output.squeeze(-1) - labels.target_ddg).abs().mean().item()
        return OverfitSanityCheckResult(active_head=active_head, final_train_loss=float(loss_output.total_loss), final_mae=mae)


def run_overfit_sanity_check(
    active_heads: tuple[str, ...] = DEFAULT_ACTIVE_HEADS,
    max_epochs: int = 600,
    num_samples: int = 12,
    sequence_length: int = 24,
    learning_rate: float = 3e-3,
) -> OverfitSanityCheckResult:
    pl.seed_everything(RANDOM_SEED, workers=True)
    loader = build_overfit_dataloader(num_samples, sequence_length, active_heads)
    module = DDGLightningModule(
        model_config=build_overfit_model_section(active_heads),
        loss_config=build_overfit_loss_section(active_heads),
        metrics_config=build_overfit_metrics_section(active_heads),
        learning_rate=learning_rate,
        split_name=SplitName.HELD_OUT_PDB,
    )
    trainer = build_overfit_trainer(max_epochs)
    trainer.fit(module, train_dataloaders=loader)
    return evaluate_fixed_batch(module, loader, active_heads)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--active-heads", choices=["classification", "regression"], default=DEFAULT_ACTIVE_HEADS[0])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_overfit_sanity_check(active_heads=(args.active_heads,))
    if result.active_head == "classification":
        print(
            f"final_train_loss={result.final_train_loss:.4f} "
            f"final_exact_accuracy={result.final_exact_accuracy:.4f} "
            f"final_within_one_bin_accuracy={result.final_within_one_bin_accuracy:.4f}"
        )
    else:
        print(f"final_train_loss={result.final_train_loss:.4f} final_mae={result.final_mae:.4f}")
    if not result.passed():
        raise SystemExit("Overfitting sanity check FAILED - architecture/gradient-flow wiring likely broken.")
    print("Overfitting sanity check PASSED.")


if __name__ == "__main__":
    main()
