"""Config-driven training/evaluation entry point (Implementation Spec §4b).

    uv run python train.py --config dl/configs/structure_only.yaml
    uv run python train.py --config dl/configs/structure_only.yaml --mode eval --ckpt path/to.ckpt

Reads a YAML run config (`dl.training.run_config.RunConfig`) and builds the
`Dataset`s (`dl.datasets`), `DataLoader`s, `DDGLightningModule` (which itself
resolves the `model`/`loss`/`metrics` sections into objects), and `Trainer`
from it, then either fits or validates depending on `--mode` (inferred as
"eval" when `--ckpt` is passed and `--mode` is omitted).

Real processed data is read through `dl.datasets.build_train_and_val_datasets`,
which falls back to synthetic data when nothing is present yet under
`shared.constants.PROCESSED_DATA_DIR`/`EMBEDDING_CACHE_DIR` for the
configured split -- see `dl/datasets/dataset_builder.py` for exactly what a
real `Dataset` needs to produce once the `data/` workstream is ready.
"""

from __future__ import annotations

import argparse
from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from dl.datasets.collation import collate_ddg_batch
from dl.datasets.dataset_builder import build_train_and_val_datasets, resolve_loss_iteration
from dl.models.config import ModelConfig
from dl.training.lightning_module import DDGLightningModule
from dl.training.run_config import RunConfig, load_run_config
from dl.utils.comet_logging import build_comet_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["train", "eval"], default=None)
    parser.add_argument("--ckpt", default=None)
    return parser.parse_args()


def resolve_mode(args: argparse.Namespace) -> str:
    if args.mode is not None:
        return args.mode
    return "eval" if args.ckpt else "train"


def build_dataloaders(run_config: RunConfig) -> tuple[DataLoader, DataLoader]:
    model_config = ModelConfig(**run_config.model.params)
    num_bins = model_config.num_ddg_bins
    iteration = resolve_loss_iteration(run_config.loss.params)
    train_dataset, val_dataset = build_train_and_val_datasets(run_config.dataset, model_config, num_bins, iteration)
    train_loader = DataLoader(
        train_dataset,
        batch_size=run_config.dataset.batch_size,
        shuffle=True,
        num_workers=run_config.dataset.num_workers,
        collate_fn=collate_ddg_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=run_config.dataset.batch_size,
        shuffle=False,
        num_workers=run_config.dataset.num_workers,
        collate_fn=collate_ddg_batch,
    )
    return train_loader, val_loader


def build_lightning_module(run_config: RunConfig, ckpt_path: Optional[str]) -> DDGLightningModule:
    kwargs = dict(
        model_config=run_config.model.model_dump(),
        loss_config=run_config.loss.model_dump(),
        metrics_config=run_config.metrics.model_dump(),
        learning_rate=run_config.trainer.learning_rate,
        weight_decay=run_config.trainer.weight_decay,
        split_name=run_config.dataset.split,
    )
    if ckpt_path:
        return DDGLightningModule.load_from_checkpoint(ckpt_path, **kwargs)
    return DDGLightningModule(**kwargs)


def build_trainer(run_config: RunConfig) -> pl.Trainer:
    logger = build_comet_logger(run_config.trainer.experiment_name, run_config.trainer.comet_project_name)
    return pl.Trainer(
        accelerator=run_config.trainer.accelerator,
        devices=run_config.trainer.devices,
        max_epochs=run_config.trainer.max_epochs,
        logger=logger or False,
    )


def run_training(module: DDGLightningModule, trainer: pl.Trainer, train_loader: DataLoader, val_loader: DataLoader) -> None:
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)


def run_evaluation(module: DDGLightningModule, trainer: pl.Trainer, val_loader: DataLoader) -> None:
    trainer.validate(module, dataloaders=val_loader)


def main() -> None:
    args = parse_args()
    run_config = load_run_config(args.config)
    train_loader, val_loader = build_dataloaders(run_config)
    module = build_lightning_module(run_config, args.ckpt)
    trainer = build_trainer(run_config)
    if resolve_mode(args) == "eval":
        run_evaluation(module, trainer, val_loader)
    else:
        run_training(module, trainer, train_loader, val_loader)


if __name__ == "__main__":
    main()
