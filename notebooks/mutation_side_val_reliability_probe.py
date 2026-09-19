"""Mutation-side validation-reliability probe (validation experiment, not
production code).

Background: `data/splitting.py` assigns train/val splits purely by protein-
complex identity (`complex_name`), with no stratification by which chain a
mutation sits on (antibody heavy/light vs. antigen -- each
`shared.constants.PointMutation` carries a `chain_role`). Classifying every
`same_pdb_allowed_{train,val}.csv` row as `antigen_only` (all mutations
`chain_role == "antigen"`), `antibody_only` (all `"heavy"`/`"light"`), or
`both` shows a real skew: train (n=855) is 34.3%/59.9%/5.8%
antigen_only/antibody_only/both, val (n=202) is 26.7%/65.3%/7.9% -- a
7.6-point antigen_only gap.

Question this probe answers: does that skew meaningfully bias val MAE as an
estimate of full-distribution performance? If the current best model's
per-side MAE is similar across groups, the skew barely matters -- val's
aggregate MAE would look about the same under either mix. If per-side MAE
differs a lot, val (under-representing antigen_only relative to train) is
a biased/less-reliable estimate, so a back-of-envelope reweight of val's
aggregate MAE by train's mix (using the measured per-group MAEs as weights)
would move noticeably from val's actual aggregate MAE.

"Current best model" = `dl/configs/
structure_saprot_only_linear_splitpool_temp5_same_pdb_allowed_20ep.yaml`
(SaProt structure-only embeddings, linear regression head,
`norm_softmax_by_chain_role` ("splitpool") pooling at temperature 5.0,
`same_pdb_allowed` split, 20 epochs).

Checkpoint identification: no config metadata is stored next to checkpoints
under `.cometml-runs/skempi-ddg/*/checkpoints/*.ckpt`, so the matching
checkpoint is found by inspecting each `.ckpt`'s embedded
`hyper_parameters` (from `DDGLightningModule.save_hyperparameters`, see
`dl/training/lightning_module.py`) for `model_config.params` matching this
config's `embedding_source_mode: structure_only`, `pooling_strategy:
norm_softmax_by_chain_role`, `pooling_temperature: 5.0`,
`regression_hidden_dims: []`, `active_heads: ["regression"]`,
`split_name: same_pdb_allowed`, at the highest epoch (target: 20 epochs).

Two checkpoints match ALL of the above identically:
`4411e9ff0f8f467cb17c5ab60e51a926/checkpoints/epoch=19-step=940.ckpt` and
`6a29445812a440f2b2d70d64cb289ae0/checkpoints/epoch=19-step=940.ckpt`.
Their weights differ (confirmed two distinct completed training runs of the
same config, not a duplicated file) and nothing else on disk breaks the
tie. The corresponding config file's own mtime (13:00:44) sits only ~10
minutes before the first checkpoint's mtime (13:10:30) -- consistent with
"write config, immediately launch its run" -- while the second checkpoint
finishes 32 minutes later still (13:42:23), suggesting a second, later rerun
of the identical config; this is circumstantial, not proof. Rather than
guess which one is "the" model, this probe evaluates BOTH and reports them
side by side -- if they tell the same qualitative story, the ambiguity does
not matter for the decision this probe informs.

Requires the exact env vars the checkpoint was trained under -- inferred
from the regression head's input width baked into the checkpoint, 2560 = 2
(the `norm_softmax_by_chain_role` output-dim multiplier) x 1280 (SaProt's
*full* 650M-checkpoint hidden dim, not the 480-dim CPU-dev checkpoint):

    SKEMPI_USE_SAPROT_STRUCTURE=1 SKEMPI_USE_SMALL_CHECKPOINTS=0 \\
        uv run python notebooks/mutation_side_val_reliability_probe.py

A handful of val samples (pdb_id 1CZ8) have only a stale 480-dim SaProt
cache on disk under `data/embeddings/structure_saprot/`; these are
recomputed on the fly at the correct 1280-dim width by
`data.embedding_pipeline.cache.bundle_is_current`'s own dimension check --
the same on-the-fly-recompute behavior `dl.datasets.mutation_csv_dataset`
always relies on, not special-cased here.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from data.homology_dedup import normalize_protein_name  # noqa: E402
from dl.datasets.collation import collate_ddg_batch  # noqa: E402
from dl.datasets.mutation_csv_dataset import MutationCsvDataset  # noqa: E402
from dl.models.config import ModelConfig  # noqa: E402
from dl.models.ddg_model import DDGPredictor  # noqa: E402
from dl.utils.label_codes import BOUNDED_ID  # noqa: E402
from shared.constants import ChainRole, MutationRecord, SPLIT_FILES, SplitName  # noqa: E402

CANDIDATE_CHECKPOINTS = {
    "4411e9ff...(finished 13:10:30, ~10min after config mtime)": Path(
        ".cometml-runs/skempi-ddg/4411e9ff0f8f467cb17c5ab60e51a926/checkpoints/epoch=19-step=940.ckpt"
    ),
    "6a294458...(finished 13:42:23, likely a later rerun)": Path(
        ".cometml-runs/skempi-ddg/6a29445812a440f2b2d70d64cb289ae0/checkpoints/epoch=19-step=940.ckpt"
    ),
}

EXPECTED_MODEL_PARAMS = {
    "embedding_source_mode": "structure_only",
    "pooling_strategy": "norm_softmax_by_chain_role",
    "pooling_temperature": 5.0,
    "regression_hidden_dims": [],
    "active_heads": ["regression"],
}
EXPECTED_SPLIT_NAME = "same_pdb_allowed"

TRAIN_MIX_FRACTIONS = {"antigen_only": 0.343, "antibody_only": 0.599, "both": 0.058}
BATCH_SIZE = 16


def check_saprot_env() -> None:
    if os.environ.get("SKEMPI_USE_SAPROT_STRUCTURE") != "1":
        sys.exit("Set SKEMPI_USE_SAPROT_STRUCTURE=1 -- this checkpoint was trained on SaProt structure embeddings.")
    if os.environ.get("SKEMPI_USE_SMALL_CHECKPOINTS") != "0":
        sys.exit(
            "Set SKEMPI_USE_SMALL_CHECKPOINTS=0 -- this checkpoint's head width (2560) matches SaProt's "
            "full 650M-checkpoint hidden dim (1280), not the 480-dim CPU-dev checkpoint."
        )


def classify_chain_group(record: MutationRecord) -> str:
    roles = {mutation.chain_role for mutation in record.mutations}
    if roles == {ChainRole.ANTIGEN}:
        return "antigen_only"
    if roles <= {ChainRole.HEAVY, ChainRole.LIGHT}:
        return "antibody_only"
    return "both"


def report_group_composition(name: str, records: list[MutationRecord]) -> None:
    counts: dict[str, int] = {"antigen_only": 0, "antibody_only": 0, "both": 0}
    for record in records:
        counts[classify_chain_group(record)] += 1
    total = len(records)
    print(f"{name} (n={total}): " + ", ".join(f"{group}={count} ({100 * count / total:.1f}%)" for group, count in counts.items()))


def load_hyperparameters(ckpt_path: Path) -> dict:
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return checkpoint["hyper_parameters"]


def confirm_hyperparameters_match(hyperparameters: dict) -> None:
    model_params = hyperparameters["model_config"]["params"]
    mismatches = {
        key: (model_params.get(key), expected)
        for key, expected in EXPECTED_MODEL_PARAMS.items()
        if model_params.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Checkpoint hyperparameters do not match the target config: {mismatches}")
    if hyperparameters["split_name"] != EXPECTED_SPLIT_NAME:
        raise ValueError(f"Checkpoint split_name {hyperparameters['split_name']!r} != {EXPECTED_SPLIT_NAME!r}")


def strip_model_prefix(state_dict: dict) -> dict:
    return {key[len("model.") :]: value for key, value in state_dict.items() if key.startswith("model.")}


def build_model_from_checkpoint(ckpt_path: Path) -> DDGPredictor:
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hyperparameters = checkpoint["hyper_parameters"]
    confirm_hyperparameters_match(hyperparameters)
    config = ModelConfig(**hyperparameters["model_config"]["params"])
    model = DDGPredictor(config)
    model.load_state_dict(strip_model_prefix(checkpoint["state_dict"]), strict=True)
    model.eval()
    return model


def build_val_dataset_with_train_complex_names() -> tuple[MutationCsvDataset, MutationCsvDataset]:
    split_files = SPLIT_FILES[SplitName.SAME_PDB_ALLOWED]
    train_dataset = MutationCsvDataset(split_files["train"])
    train_complex_names = frozenset(normalize_protein_name(record.complex_name or "") for record in train_dataset.records)
    val_dataset = MutationCsvDataset(split_files["val"], train_complex_names=train_complex_names)
    return train_dataset, val_dataset


@dataclass
class GroupErrors:
    errors: dict[str, list[float]] = field(default_factory=lambda: {"antigen_only": [], "antibody_only": [], "both": []})

    def add(self, group: str, abs_error: float) -> None:
        self.errors[group].append(abs_error)

    def mean(self, group: str) -> float:
        values = self.errors[group]
        return sum(values) / len(values) if values else float("nan")

    def n(self, group: str) -> int:
        return len(self.errors[group])

    def overall_mean(self) -> float:
        all_values = [value for values in self.errors.values() for value in values]
        return sum(all_values) / len(all_values)

    def overall_n(self) -> int:
        return sum(self.n(group) for group in self.errors)


@torch.no_grad()
def run_inference(model: DDGPredictor, val_dataset: MutationCsvDataset) -> GroupErrors:
    loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_ddg_batch)
    group_errors = GroupErrors()
    record_offset = 0
    for inputs, labels in loader:
        batch_records = val_dataset.records[record_offset : record_offset + labels.label_type_id.shape[0]]
        record_offset += labels.label_type_id.shape[0]

        predicted_ddg = model(inputs)["head_outputs"]["regression"]
        bounded_mask = labels.label_type_id == BOUNDED_ID
        abs_error = (predicted_ddg - labels.target_ddg).abs()

        for index, record in enumerate(batch_records):
            if not bool(bounded_mask[index]):
                continue
            group_errors.add(classify_chain_group(record), float(abs_error[index]))
    return group_errors


def reweighted_hypothetical_mae(group_errors: GroupErrors, mix_fractions: dict[str, float]) -> float:
    return sum(mix_fractions[group] * group_errors.mean(group) for group in mix_fractions)


def print_mae_report(label: str, group_errors: GroupErrors) -> None:
    print(f"\n=== {label} ===")
    for group in ("antigen_only", "antibody_only", "both"):
        print(f"  {group:14s} n={group_errors.n(group):3d}  MAE={group_errors.mean(group):.3f}")
    actual_val_mae = group_errors.overall_mean()
    hypothetical_val_mae = reweighted_hypothetical_mae(group_errors, TRAIN_MIX_FRACTIONS)
    print(f"  {'overall':14s} n={group_errors.overall_n():3d}  MAE={actual_val_mae:.3f}")
    print(f"  actual val MAE (val's own {26.7}/{65.3}/{7.9}% mix):        {actual_val_mae:.4f}")
    print(f"  hypothetical val MAE (train's {34.3}/{59.9}/{5.8}% mix):    {hypothetical_val_mae:.4f}")
    print(f"  delta (hypothetical - actual):                            {hypothetical_val_mae - actual_val_mae:+.4f}")
    group_means = {group: group_errors.mean(group) for group in ("antigen_only", "antibody_only", "both")}
    finite_means = [v for v in group_means.values() if v == v]
    if len(finite_means) >= 2:
        print(f"  group MAE range: min={min(finite_means):.3f} max={max(finite_means):.3f} ratio={max(finite_means) / min(finite_means):.2f}x")


def main() -> None:
    check_saprot_env()

    train_dataset, val_dataset = build_val_dataset_with_train_complex_names()
    print("Chain-role composition (bounded-only rows actually used for MAE, may differ slightly from the")
    print("all-label-type composition quoted in this script's docstring, which is over every row):")
    report_group_composition("train (bounded-only)", train_dataset.records)
    report_group_composition("val (bounded-only)", val_dataset.records)

    for label, relative_path in CANDIDATE_CHECKPOINTS.items():
        ckpt_path = _REPO_ROOT / relative_path
        model = build_model_from_checkpoint(ckpt_path)
        group_errors = run_inference(model, val_dataset)
        print_mae_report(label, group_errors)


if __name__ == "__main__":
    main()
