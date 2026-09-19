"""PyTorch Lightning training loop for `DDGPredictor` (Implementation Spec §3
+ §4b; CLAUDE.md: PyTorch Lightning, not a raw training loop).

`model`, `loss`, and `metrics` are each resolved from a `class_name` +
`params` config section via `dl.training.object_building`, so swapping any
of the three is a config change (see `dl/configs/*.yaml`), not a code
change. Which head this module trains against (`"classification"` or
`"regression"`) is likewise read from `model_config["params"]["active_heads"]`
(falling back to `dl.utils.constants.DEFAULT_ACTIVE_HEADS`, currently
regression -- see that constant's comment): exactly one head must be active,
since a bounded-ddG entry has exactly one target/loss/metric shape per run,
not two trained simultaneously.

Bounded-vs-hinge dispatch, the bin-distance-weighted classification cost
(or the dead-zone regression cost), and the hinge term all live in
`dl.losses.ddg_loss.DDGLoss` / `dl.losses.ddg_regression_loss.
DDGRegressionLoss` -- this module only wires its inputs/outputs to the
model and the logger.

Per `shared/constants.py` ("report all downstream metrics separately per
split, never pooled"), a run trains and validates against exactly one
`shared.constants.SplitName` at a time (`dataset.split` in the run config).
This module keeps distinct metric *instances* per split and per breakout
(never sharing accumulated state) and labels every logged value with that
split's name, so results from `held_out_pdb` and `same_pdb_allowed` runs are
never displayed or averaged together even if compared side by side later.

Beyond the pooled `{stage}_metric/{split}` value, validation also reports two
further breakouts, using the same configured metric class (so switching MAE
for RMSE, say, changes all of them consistently):
- `val_metric_high_abs_ddg/{split}` (regression only, since it needs the raw
  signed target `DDGLabels.target_ddg`): restricted to
  `|target_ddg| >= dl.utils.constants.HIGH_ABS_DDG_THRESHOLD_KCAL_MOL`, to
  watch large-effect predictions separately from the (much more numerous)
  near-zero-effect ones.
- `val_metric_new_complex/{split}` / `val_metric_complex_seen_in_train/{split}`
  (head-agnostic, driven by `DDGLabels.complex_seen_in_train`): splits val
  rows by whether their `complex_name` (the actual antibody-antigen pair
  identity -- not literal `pdb_id`, since the same complex can be
  re-crystallized under a different PDB code) also appears in the training
  split. On `held_out_pdb` this is now vacuous in the other direction
  (`complex_seen_in_train` is `False` for every val row by construction, so
  only `new_complex` ever logs); on `same_pdb_allowed` both buckets log real,
  comparably-sized rows -- `data.splitting.assign_same_pdb_allowed_split`'s
  two-tier design guarantees some complexes are held out entirely (feeding
  `new_complex`) alongside the usual same-complex-different-mutation mixing
  (feeding `complex_seen_in_train`) -- the comparison the user actually cares
  about.

- `val_metric_antigen_only/{split}` / `val_metric_antibody_only/{split}`
  (head-agnostic, driven by `DDGLabels.mutation_side_id`): splits val rows by
  whether every mutation in the record sits on the antigen chain, or every
  mutation sits on an antibody chain (heavy or light) -- per
  `dl.datasets.mutation_csv_dataset.classify_mutation_side`. Motivated by an
  earlier investigation (`notebooks/mutation_side_val_reliability_probe.py`)
  finding that the current best model's MAE on antigen-side mutations is
  meaningfully worse than on antibody-side mutations (roughly 1.2x-1.5x
  across two checkpoints) -- a gap invisible in the pooled val metric alone.
  The rare mixed-chain "both" group (~6-8% of records) is not given its own
  metric, since antigen-only vs. antibody-only are the two buckets the
  investigation's finding is actually about.

- `val_metric_mae_by_bucket/{bucket_name}/{split}` (regression only, same
  reason as `high_abs_ddg` above -- it needs the raw signed target
  `DDGLabels.target_ddg`): a finer breakdown of the same bounded-entry MAE
  into 5 named magnitude buckets (`dl.utils.constants.MAGNITUDE_BUCKET_NAMES`,
  `dl.metrics.BucketedMAE`) rather than the pooled figure alone, for
  visibility into tail (large-effect) vs. center (near-zero) performance
  specifically -- a model could look fine on the pooled MAE while doing much
  worse on the (much rarer) large-magnitude mutations, which this breakout
  would expose. Unlike the other breakouts here, `BucketedMAE.compute()`
  returns a dict (one MAE per bucket) rather than a scalar, so
  `_log_and_reset_dict_metric` logs one value per non-empty bucket instead of
  a single pooled value.

(Homology-group duplicates -- the same mutation re-deposited under a
different PDB -- are discarded from the dataset entirely by
`data.homology_dedup.discard_homology_siblings`, not routed to val, so there
is no separate "genuine held-out vs. sibling" breakout here anymore.)

Any breakout with zero qualifying rows in a given epoch (e.g. no
high-|ddG| sample landed in a small val set) is simply not logged that
epoch, rather than calling a torchmetrics `compute()` with no accumulated
state -- `_seen` tracks this per log key independently of whatever
`torchmetrics.Metric` subclass is configured.
"""

from __future__ import annotations

import math
import warnings

import pytorch_lightning as pl
import torch
import torchmetrics

import dl.metrics
from dl.datasets.schemas import DDGLabels
from dl.models.schemas import ModelInputs
from dl.training.object_building import build_loss_from_config, build_metric_from_config, build_model_from_config
from dl.utils.constants import DEFAULT_ACTIVE_HEADS, HIGH_ABS_DDG_THRESHOLD_KCAL_MOL
from dl.utils.label_codes import BOUNDED_ID, MUTATION_SIDE_ANTIBODY_ONLY, MUTATION_SIDE_ANTIGEN_ONLY
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


def resolve_active_head_name(model_config: dict) -> str:
    active_heads = tuple(model_config.get("params", {}).get("active_heads", DEFAULT_ACTIVE_HEADS))
    if len(active_heads) != 1:
        raise ValueError(
            f"DDGLightningModule trains against exactly one active head, got {active_heads} "
            "-- set model.params.active_heads to a single-element list in the run config."
        )
    return active_heads[0]


def select_bounded_rows_classification(
    logits: torch.Tensor, labels: DDGLabels
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bounded_mask = labels.label_type_id == BOUNDED_ID
    predicted_bin = logits.argmax(dim=-1)
    return predicted_bin[bounded_mask], labels.target_bin[bounded_mask], bounded_mask


def select_bounded_rows_regression(
    predicted_ddg: torch.Tensor, labels: DDGLabels
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bounded_mask = labels.label_type_id == BOUNDED_ID
    return predicted_ddg[bounded_mask], labels.target_ddg[bounded_mask], bounded_mask


def restrict_to_high_abs_effect(
    predicted: torch.Tensor, target: torch.Tensor, threshold: float = HIGH_ABS_DDG_THRESHOLD_KCAL_MOL
) -> tuple[torch.Tensor, torch.Tensor]:
    high_mask = target.abs() >= threshold
    return predicted[high_mask], target[high_mask]


def restrict_by_complex_seen_in_train(
    predicted: torch.Tensor, target: torch.Tensor, complex_seen_in_train: torch.Tensor, seen: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = complex_seen_in_train if seen else ~complex_seen_in_train
    return predicted[mask], target[mask]


def restrict_by_mutation_side(
    predicted: torch.Tensor, target: torch.Tensor, mutation_side_id: torch.Tensor, side_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = mutation_side_id == side_id
    return predicted[mask], target[mask]


class DDGLightningModule(pl.LightningModule):
    def __init__(
        self,
        model_config: dict,
        loss_config: dict,
        metrics_config: dict,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        split_name: SplitName = SplitName.HELD_OUT_PDB,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.split_name = split_name
        self.active_head = resolve_active_head_name(model_config)
        self.model = build_model_from_config(model_config)
        self.loss_fn = build_loss_from_config(loss_config)
        num_bins = self.model.config.num_ddg_bins

        self.train_metric = build_bounded_metric(metrics_config, num_bins)
        self.val_metric = build_bounded_metric(metrics_config, num_bins)
        self.train_large_effect_metric = (
            build_bounded_metric(metrics_config, num_bins) if self.active_head == "regression" else None
        )
        self.val_large_effect_metric = (
            build_bounded_metric(metrics_config, num_bins) if self.active_head == "regression" else None
        )
        self.val_new_complex_metric = build_bounded_metric(metrics_config, num_bins)
        self.val_complex_seen_metric = build_bounded_metric(metrics_config, num_bins)
        self.val_antigen_only_metric = build_bounded_metric(metrics_config, num_bins)
        self.val_antibody_only_metric = build_bounded_metric(metrics_config, num_bins)
        self.val_bucketed_mae_metric = dl.metrics.BucketedMAE() if self.active_head == "regression" else None
        self._seen: dict[str, bool] = {}

        self.save_hyperparameters(
            {
                "model_config": model_config,
                "loss_config": loss_config,
                "metrics_config": metrics_config,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "split_name": split_name.value,
            }
        )

    def forward(self, inputs: ModelInputs) -> dict:
        return self.model(inputs)

    def _compute_step_core(
        self, batch: tuple[ModelInputs, DDGLabels]
    ) -> tuple[object, torch.Tensor, torch.Tensor, torch.Tensor, DDGLabels]:
        inputs, labels = batch
        head_output = self.model(inputs)["head_outputs"][self.active_head]
        if self.active_head == "classification":
            loss_output = self.loss_fn(
                head_output,
                labels.label_type_id,
                labels.target_bin,
                labels.bound_kcal_mol,
                labels.direction,
                target_ddg=labels.target_ddg,
            )
            predicted, target, bounded_mask = select_bounded_rows_classification(head_output, labels)
        else:
            loss_output = self.loss_fn(head_output, labels.label_type_id, labels.target_ddg, labels.bound_kcal_mol, labels.direction)
            predicted, target, bounded_mask = select_bounded_rows_regression(head_output, labels)
        return loss_output, predicted, target, bounded_mask, labels

    def _update_metric(self, log_key: str, metric: torchmetrics.Metric, predicted: torch.Tensor, target: torch.Tensor) -> None:
        if predicted.numel() == 0:
            return
        metric.update(predicted, target)
        self._seen[log_key] = True

    def _log_and_reset_metric(self, log_key: str, metric: torchmetrics.Metric, prog_bar: bool = False) -> None:
        if self._seen.get(log_key):
            self.log(log_key, metric.compute(), prog_bar=prog_bar)
        metric.reset()
        self._seen[log_key] = False

    def _log_and_reset_dict_metric(self, seen_key: str, log_name_prefix: str, metric: torchmetrics.Metric) -> None:
        if self._seen.get(seen_key):
            for bucket_name, value in metric.compute().items():
                if not math.isnan(value):
                    self.log(f"{log_name_prefix}/{bucket_name}/{self.split_name.value}", value)
        metric.reset()
        self._seen[seen_key] = False

    def training_step(self, batch: tuple[ModelInputs, DDGLabels], batch_idx: int) -> torch.Tensor:
        loss_output, predicted, target, _, _ = self._compute_step_core(batch)
        self._update_metric(f"train_metric/{self.split_name.value}", self.train_metric, predicted, target)
        if self.train_large_effect_metric is not None:
            high_predicted, high_target = restrict_to_high_abs_effect(predicted, target)
            self._update_metric(
                f"train_metric_high_abs_ddg/{self.split_name.value}", self.train_large_effect_metric, high_predicted, high_target
            )
        self.log(f"train_loss/{self.split_name.value}", loss_output.total_loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss_output.total_loss

    def validation_step(self, batch: tuple[ModelInputs, DDGLabels], batch_idx: int) -> torch.Tensor:
        loss_output, predicted, target, bounded_mask, labels = self._compute_step_core(batch)
        self._update_metric(f"val_metric/{self.split_name.value}", self.val_metric, predicted, target)
        if self.val_large_effect_metric is not None:
            high_predicted, high_target = restrict_to_high_abs_effect(predicted, target)
            self._update_metric(
                f"val_metric_high_abs_ddg/{self.split_name.value}", self.val_large_effect_metric, high_predicted, high_target
            )
        complex_seen_in_train = labels.complex_seen_in_train[bounded_mask]
        new_predicted, new_target = restrict_by_complex_seen_in_train(predicted, target, complex_seen_in_train, seen=False)
        self._update_metric(f"val_metric_new_complex/{self.split_name.value}", self.val_new_complex_metric, new_predicted, new_target)
        seen_predicted, seen_target = restrict_by_complex_seen_in_train(predicted, target, complex_seen_in_train, seen=True)
        self._update_metric(
            f"val_metric_complex_seen_in_train/{self.split_name.value}", self.val_complex_seen_metric, seen_predicted, seen_target
        )
        mutation_side_id = labels.mutation_side_id[bounded_mask]
        antigen_predicted, antigen_target = restrict_by_mutation_side(
            predicted, target, mutation_side_id, MUTATION_SIDE_ANTIGEN_ONLY
        )
        self._update_metric(
            f"val_metric_antigen_only/{self.split_name.value}", self.val_antigen_only_metric, antigen_predicted, antigen_target
        )
        antibody_predicted, antibody_target = restrict_by_mutation_side(
            predicted, target, mutation_side_id, MUTATION_SIDE_ANTIBODY_ONLY
        )
        self._update_metric(
            f"val_metric_antibody_only/{self.split_name.value}", self.val_antibody_only_metric, antibody_predicted, antibody_target
        )
        if self.val_bucketed_mae_metric is not None:
            self._update_metric("val_metric_mae_by_bucket", self.val_bucketed_mae_metric, predicted, target)
        self.log(f"val_loss/{self.split_name.value}", loss_output.total_loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss_output.total_loss

    def on_train_epoch_end(self) -> None:
        self._log_and_reset_metric(f"train_metric/{self.split_name.value}", self.train_metric, prog_bar=True)
        if self.train_large_effect_metric is not None:
            self._log_and_reset_metric(f"train_metric_high_abs_ddg/{self.split_name.value}", self.train_large_effect_metric)

    def on_validation_epoch_end(self) -> None:
        self._log_and_reset_metric(f"val_metric/{self.split_name.value}", self.val_metric, prog_bar=True)
        if self.val_large_effect_metric is not None:
            self._log_and_reset_metric(f"val_metric_high_abs_ddg/{self.split_name.value}", self.val_large_effect_metric)
        self._log_and_reset_metric(f"val_metric_new_complex/{self.split_name.value}", self.val_new_complex_metric)
        self._log_and_reset_metric(f"val_metric_complex_seen_in_train/{self.split_name.value}", self.val_complex_seen_metric)
        self._log_and_reset_metric(f"val_metric_antigen_only/{self.split_name.value}", self.val_antigen_only_metric)
        self._log_and_reset_metric(f"val_metric_antibody_only/{self.split_name.value}", self.val_antibody_only_metric)
        if self.val_bucketed_mae_metric is not None:
            self._log_and_reset_dict_metric(
                "val_metric_mae_by_bucket", "val_metric_mae_by_bucket", self.val_bucketed_mae_metric
            )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def transfer_batch_to_device(
        self, batch: tuple[ModelInputs, DDGLabels], device: torch.device, dataloader_idx: int
    ) -> tuple[ModelInputs, DDGLabels]:
        inputs, labels = batch
        return inputs.to(device), labels.to(device)
