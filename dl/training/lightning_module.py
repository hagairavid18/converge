"""PyTorch Lightning training loop for `DDGPredictor` (Implementation Spec §3
+ §4b; CLAUDE.md: PyTorch Lightning, not a raw training loop).

`model`, `loss`, and `metrics` are each resolved from a `class_name` +
`params` config section via `dl.training.object_building`, so swapping any
of the three is a config change (see `dl/configs/*.yaml`), not a code
change.

Bounded-vs-hinge dispatch, the bin-distance-weighted classification cost,
and the hinge term all live in `dl.losses.ddg_loss.DDGLoss` -- this module
only wires its inputs/outputs to the model and the logger.

Per `shared/constants.py` ("report all downstream metrics separately per
split, never pooled"), a run trains and validates against exactly one
`shared.constants.SplitName` at a time (`dataset.split` in the run config).
This module keeps distinct train/val metric *instances* (never sharing
accumulated state) and labels every logged value with that split's name, so
results from `held_out_pdb` and `same_pdb_allowed` runs are never displayed
or averaged together even if compared side by side later. (If separate
*simultaneous* per-split validation dataloaders within one run turn out to
be what's wanted instead, that would need Lightning's multi-dataloader
validation support layered on top of this -- flagging as an open question.)

`dl.metrics` does not export its `torchmetrics.Metric` classes yet (no
`dl/metrics/__init__.py`), so metric construction falls back to a plain
`torchmetrics.Accuracy` with a warning until it does; nothing else here
needs to change once it's ready.
"""

from __future__ import annotations

import warnings
from typing import Optional

import pytorch_lightning as pl
import torch
import torchmetrics

import dl.metrics
from dl.datasets.schemas import DDGLabels
from dl.models.schemas import ModelInputs
from dl.training.object_building import build_loss_from_config, build_metric_from_config, build_model_from_config
from dl.utils.label_codes import BOUNDED_ID
from shared.constants import SplitName


def build_fallback_accuracy(num_classes: int) -> torchmetrics.Metric:
    return torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)


def build_bounded_metric(metrics_section: dict, num_classes: int) -> torchmetrics.Metric:
    try:
        return build_metric_from_config(dl.metrics, metrics_section)
    except (AttributeError, ImportError) as error:
        warnings.warn(
            f"dl.metrics.{metrics_section.get('class_name')} not available yet ({error}); "
            "falling back to torchmetrics.Accuracy"
        )
        return build_fallback_accuracy(num_classes)


def select_bounded_rows(logits: torch.Tensor, labels: DDGLabels) -> tuple[torch.Tensor, torch.Tensor]:
    bounded_mask = labels.label_type_id == BOUNDED_ID
    predicted_bin = logits.argmax(dim=-1)
    return predicted_bin[bounded_mask], labels.target_bin[bounded_mask]


class DDGLightningModule(pl.LightningModule):
    def __init__(
        self,
        model_config: dict,
        loss_config: dict,
        metrics_config: dict,
        learning_rate: float = 1e-3,
        split_name: SplitName = SplitName.HELD_OUT_PDB,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.split_name = split_name
        self.model = build_model_from_config(model_config)
        self.loss_fn = build_loss_from_config(loss_config)
        num_bins = self.model.config.num_ddg_bins
        self.train_metric = build_bounded_metric(metrics_config, num_bins)
        self.val_metric = build_bounded_metric(metrics_config, num_bins)
        self.save_hyperparameters(
            {
                "model_config": model_config,
                "loss_config": loss_config,
                "metrics_config": metrics_config,
                "learning_rate": learning_rate,
                "split_name": split_name.value,
            }
        )

    def forward(self, inputs: ModelInputs) -> dict:
        return self.model(inputs)

    def training_step(self, batch: tuple[ModelInputs, DDGLabels], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, self.train_metric, "train")

    def validation_step(self, batch: tuple[ModelInputs, DDGLabels], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, self.val_metric, "val")

    def _shared_step(self, batch: tuple[ModelInputs, DDGLabels], metric: torchmetrics.Metric, stage: str) -> torch.Tensor:
        inputs, labels = batch
        logits = self.model(inputs)["head_outputs"]["classification"]
        loss_output = self.loss_fn(logits, labels.label_type_id, labels.target_bin, labels.bound_kcal_mol, labels.direction)
        self._update_metric(metric, logits, labels)
        self.log(f"{stage}_loss/{self.split_name.value}", loss_output.total_loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss_output.total_loss

    def _update_metric(self, metric: torchmetrics.Metric, logits: torch.Tensor, labels: DDGLabels) -> None:
        predicted_bin, target_bin = select_bounded_rows(logits, labels)
        if predicted_bin.numel() > 0:
            metric.update(predicted_bin, target_bin)

    def on_train_epoch_end(self) -> None:
        self._log_and_reset_metric(self.train_metric, "train")

    def on_validation_epoch_end(self) -> None:
        self._log_and_reset_metric(self.val_metric, "val")

    def _log_and_reset_metric(self, metric: torchmetrics.Metric, stage: str) -> None:
        self.log(f"{stage}_acc/{self.split_name.value}", metric.compute(), prog_bar=True)
        metric.reset()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def transfer_batch_to_device(
        self, batch: tuple[ModelInputs, DDGLabels], device: torch.device, dataloader_idx: int
    ) -> tuple[ModelInputs, DDGLabels]:
        inputs, labels = batch
        return inputs.to(device), labels.to(device)
