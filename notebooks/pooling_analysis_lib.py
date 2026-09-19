"""Shared computation library for `notebooks/03_model.ipynb`.

Consolidates the reusable logic previously duplicated across six deleted one-off
validation scripts: `mutation_site_probe.py`, `saprot_diff_localization_probe.py`,
`pooling_weight_concentration_probe.py`, `pooling_temperature_probe.py`,
`splitpool_temperature_verification_probe.py`, and `visualize_pooling_norms_3d.py`.
Everything here is upstream of the model's one learned component (the regression
head), so a freshly-initialized `DDGPredictor` -- no trained checkpoint needed --
reproduces exactly the pooling weights/diff-norms a trained model would use.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import py3Dmol
import torch
import torch.nn.functional as F
from torch import nn

from data.chain_roles import ordered_chain_ids_for_sample
from data.embedding_pipeline.entry_embeddings import entry_embeddings_are_cached
from data.structures import chain_residue_sequence, get_chain, load_structure, skempi_structure_path
from dl.datasets.collation import collate_ddg_batch
from dl.datasets.mutation_csv_dataset import (
    filter_records_for_active_iteration,
    label_fields_for_record,
    load_or_compute_cached_tensors,
    load_records,
    select_model_fields,
)
from dl.models.config import ModelConfig
from dl.models.ddg_model import DDGPredictor
from dl.utils.embedding_mode import uses_sequence_embedding, uses_structure_embedding
from shared.constants import ACTIVE_EMBEDDING_SOURCE_MODE, ChainRole, MutationRecord, SPLIT_FILES, SplitName

_NO_TRAIN_COMPLEXES: frozenset[str] = frozenset()

# ---------------------------------------------------------------------------
# Record selection & caching
# ---------------------------------------------------------------------------


def single_mutation_records(csv_path: Path) -> list[MutationRecord]:
    records = filter_records_for_active_iteration(load_records(csv_path))
    return [record for record in records if len(record.mutations) == 1]


def collect_unique_single_mutation_records() -> list[MutationRecord]:
    seen_sample_ids: set[str] = set()
    unique_records = []
    for split in SplitName:
        for subset in ("train", "val"):
            for record in single_mutation_records(SPLIT_FILES[split][subset]):
                if record.sample_id not in seen_sample_ids:
                    seen_sample_ids.add(record.sample_id)
                    unique_records.append(record)
    return unique_records


def prioritize_already_cached(records: list[MutationRecord], sample_chain_maps: dict) -> list[MutationRecord]:
    """Already-cached-and-current samples first, so probes mostly reuse fast
    disk loads and only fall back to slow on-the-fly computation for however
    many more are needed.
    """
    return sorted(records, key=lambda record: not entry_embeddings_are_cached(record, sample_chain_maps[record.sample_id]))


def split_records_by_mutation_side(records: list[MutationRecord]) -> tuple[list[MutationRecord], list[MutationRecord]]:
    antibody_records = [r for r in records if r.mutations[0].chain_role != ChainRole.ANTIGEN]
    antigen_records = [r for r in records if r.mutations[0].chain_role == ChainRole.ANTIGEN]
    return antibody_records, antigen_records


def has_local_pdb_and_cached_embeddings(record: MutationRecord, sample_chain_maps: dict) -> bool:
    return skempi_structure_path(record.pdb_id).exists() and entry_embeddings_are_cached(
        record, sample_chain_maps[record.sample_id]
    )


def pick_examples_with_local_structure(sample_chain_maps: dict, count: int) -> list[MutationRecord]:
    candidates = collect_unique_single_mutation_records()
    picked, seen_pdb_ids = [], set()
    for record in candidates:
        if record.pdb_id in seen_pdb_ids:
            continue
        if has_local_pdb_and_cached_embeddings(record, sample_chain_maps):
            picked.append(record)
            seen_pdb_ids.add(record.pdb_id)
        if len(picked) == count:
            break
    return picked


# ---------------------------------------------------------------------------
# Raw embedding-difference-norm localization (no training)
# ---------------------------------------------------------------------------


def saprot_diff_norm(cached: dict) -> torch.Tensor:
    return (cached["mut_structure_embedding"] - cached["wt_structure_embedding"]).norm(dim=-1)


@dataclass
class LocalizationSummary:
    label: str
    n: int
    accuracy: float
    mean_site_norm: float
    mean_elsewhere_norm: float
    mean_topk_distance_angstrom: float
    mean_random_distance_angstrom: float


def summarize_localization(
    records: list[MutationRecord], sample_chain_maps: dict, label: str, top_k: int = 10, seed: int = 0
) -> LocalizationSummary:
    rng = np.random.default_rng(seed)
    site_norms, elsewhere_norms, topk_distances, random_distances, hits = [], [], [], [], 0
    for record in records:
        cached = load_or_compute_cached_tensors(record, sample_chain_maps)
        diff_norm = saprot_diff_norm(cached)
        true_position = record.mutations[0].flat_residue_index

        site_norms.append(diff_norm[true_position].item())
        elsewhere_mask = torch.ones_like(diff_norm, dtype=torch.bool)
        elsewhere_mask[true_position] = False
        elsewhere_norms.append(diff_norm[elsewhere_mask].mean().item())
        hits += int(diff_norm.argmax().item() == true_position)

        distances = cached["mutation_distances"][0]
        topk_positions = diff_norm.topk(min(top_k, diff_norm.numel())).indices
        topk_distances.append(distances[topk_positions].mean().item())
        random_positions = torch.from_numpy(rng.choice(diff_norm.numel(), size=min(top_k, diff_norm.numel()), replace=False))
        random_distances.append(distances[random_positions].mean().item())

    n = len(records)
    return LocalizationSummary(
        label=label,
        n=n,
        accuracy=hits / n,
        mean_site_norm=float(np.mean(site_norms)),
        mean_elsewhere_norm=float(np.mean(elsewhere_norms)),
        mean_topk_distance_angstrom=float(np.mean(topk_distances)),
        mean_random_distance_angstrom=float(np.mean(random_distances)),
    )


def localization_report(records_by_label: dict[str, list[MutationRecord]], sample_chain_maps: dict) -> pd.DataFrame:
    results = [summarize_localization(records, sample_chain_maps, label) for label, records in records_by_label.items()]
    return pd.DataFrame([vars(result) for result in results]).set_index("label")


# ---------------------------------------------------------------------------
# Trained linear probe vs. training-free baselines
# ---------------------------------------------------------------------------


@dataclass
class ProbeSample:
    sample_id: str
    difference: torch.Tensor
    true_position: int


def branch_embedding(cached: dict, prefix: str) -> torch.Tensor:
    mode = ACTIVE_EMBEDDING_SOURCE_MODE
    parts = []
    if uses_structure_embedding(mode):
        parts.append(cached[f"{prefix}_structure_embedding"])
    if uses_sequence_embedding(mode):
        parts.append(cached[f"{prefix}_sequence_embedding"])
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)


def build_probe_sample(record: MutationRecord, sample_chain_maps: dict) -> ProbeSample:
    cached = load_or_compute_cached_tensors(record, sample_chain_maps)
    difference = branch_embedding(cached, "mut") - branch_embedding(cached, "wt")
    return ProbeSample(record.sample_id, difference, record.mutations[0].flat_residue_index)


def train_val_probe_split(
    samples: list[ProbeSample], train_fraction: float, seed: int
) -> tuple[list[ProbeSample], list[ProbeSample]]:
    shuffled = samples.copy()
    random.Random(seed).shuffle(shuffled)
    split_index = max(1, int(len(shuffled) * train_fraction))
    return shuffled[:split_index], shuffled[split_index:]


class LinearMutationSiteProbe(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.scorer = nn.Linear(embedding_dim, 1)

    def forward(self, difference: torch.Tensor) -> torch.Tensor:
        return self.scorer(difference).squeeze(-1)


def train_probe(probe: LinearMutationSiteProbe, train_samples: list[ProbeSample], num_epochs: int, learning_rate: float) -> None:
    optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)
    for _ in range(num_epochs):
        optimizer.zero_grad()
        losses = [
            F.cross_entropy(probe(sample.difference).unsqueeze(0), torch.tensor([sample.true_position]))
            for sample in train_samples
        ]
        torch.stack(losses).mean().backward()
        optimizer.step()


def norm_baseline_prediction(sample: ProbeSample) -> int:
    return int(sample.difference.norm(dim=-1).argmax())


def random_baseline_prediction(sample: ProbeSample, rng: random.Random) -> int:
    return rng.randrange(sample.difference.shape[0])


def probe_prediction(probe: LinearMutationSiteProbe, sample: ProbeSample) -> int:
    with torch.no_grad():
        return int(probe(sample.difference).argmax())


def summarize_predictions(name: str, samples: list[ProbeSample], predictions: list[int]) -> dict:
    errors = [abs(prediction - sample.true_position) for prediction, sample in zip(predictions, samples)]
    return {
        "method": name,
        "n": len(samples),
        "accuracy": sum(error == 0 for error in errors) / len(samples),
        "mean_abs_error_residues": sum(errors) / len(samples),
    }


def evaluate_linear_probe_localization(
    records: list[MutationRecord],
    sample_chain_maps: dict,
    target_num_samples: int = 32,
    train_fraction: float = 0.8,
    seed: int = 0,
    num_epochs: int = 300,
    learning_rate: float = 1e-2,
) -> pd.DataFrame:
    prioritized = prioritize_already_cached(records, sample_chain_maps)[:target_num_samples]
    samples = [build_probe_sample(record, sample_chain_maps) for record in prioritized]
    train_samples, val_samples = train_val_probe_split(samples, train_fraction, seed)

    probe = LinearMutationSiteProbe(train_samples[0].difference.shape[-1])
    train_probe(probe, train_samples, num_epochs, learning_rate)

    rng = random.Random(seed)
    rows = []
    for split_name, split_samples in [("train", train_samples), ("held-out val", val_samples)]:
        methods = [
            ("trained linear probe", [probe_prediction(probe, sample) for sample in split_samples]),
            ("norm-of-difference baseline (untrained)", [norm_baseline_prediction(sample) for sample in split_samples]),
            ("random baseline", [random_baseline_prediction(sample, rng) for sample in split_samples]),
        ]
        for method_name, predictions in methods:
            rows.append({"split": split_name, **summarize_predictions(method_name, split_samples, predictions)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pooling mechanism: participation ratio & concentration
# ---------------------------------------------------------------------------


def build_ddg_model(pooling_strategy: str, pooling_temperature: float) -> DDGPredictor:
    config = ModelConfig(
        embedding_source_mode="structure_only",
        active_heads=(),
        pooling_strategy=pooling_strategy,
        pooling_temperature=pooling_temperature,
    )
    model = DDGPredictor(config)
    model.eval()
    return model


def compute_pooling_output(model: DDGPredictor, record: MutationRecord, sample_chain_maps: dict):
    cached = load_or_compute_cached_tensors(record, sample_chain_maps)
    batch = [(select_model_fields(cached), label_fields_for_record(record, _NO_TRAIN_COMPLEXES))]
    inputs, _ = collate_ddg_batch(batch)
    with torch.no_grad():
        out = model(inputs)
    return out["pooling_weights"][0], inputs


def participation_ratio(weights: torch.Tensor) -> float:
    """Effective number of residues the softmax spreads mass over --
    `1 / sum(w^2)` on weights renormalized to sum to 1, minimized (=1) when
    all mass sits on one residue, maximized (=L) when uniform over L residues.
    """
    total = weights.sum()
    if total.item() == 0.0:
        return float("nan")
    normalized = weights / total
    return (1.0 / (normalized**2).sum()).item()


def side_mask_for_mutation(inputs, mutation_on_antigen: bool) -> torch.Tensor:
    antigen_mask = inputs.antigen_mask[0]
    return antigen_mask if mutation_on_antigen else ~antigen_mask


def local_true_position(side_mask: torch.Tensor, true_position: int) -> int:
    return int(side_mask[:true_position].sum().item())


def splitpool_localization_row(model: DDGPredictor, record: MutationRecord, sample_chain_maps: dict) -> dict:
    weights, inputs = compute_pooling_output(model, record, sample_chain_maps)
    mutation_on_antigen = record.mutations[0].chain_role == ChainRole.ANTIGEN
    side_mask = side_mask_for_mutation(inputs, mutation_on_antigen)
    side_weights = weights[side_mask]
    true_position = record.mutations[0].flat_residue_index
    return dict(
        sample_id=record.sample_id,
        side="antigen" if mutation_on_antigen else "antibody",
        side_length=int(side_mask.sum().item()),
        participation_ratio=participation_ratio(side_weights),
        argmax_is_true_site=bool(side_weights.argmax().item() == local_true_position(side_mask, true_position)),
    )


def run_splitpool_localization(records: list[MutationRecord], sample_chain_maps: dict, temperature: float) -> pd.DataFrame:
    model = build_ddg_model("norm_softmax_by_chain_role", temperature)
    return pd.DataFrame([splitpool_localization_row(model, record, sample_chain_maps) for record in records])


def summarize_splitpool_localization(splitpool_df: pd.DataFrame) -> pd.DataFrame:
    return splitpool_df.groupby("side").agg(
        n=("sample_id", "count"),
        mean_side_length=("side_length", "mean"),
        mean_participation_ratio=("participation_ratio", "mean"),
        argmax_hit_rate=("argmax_is_true_site", "mean"),
    )


def participation_ratio_report(records: list[MutationRecord], sample_chain_maps: dict, temperature: float) -> pd.DataFrame:
    return summarize_splitpool_localization(run_splitpool_localization(records, sample_chain_maps, temperature))


# ---------------------------------------------------------------------------
# Temperature sweep
# ---------------------------------------------------------------------------


def flat_localization_row(model: DDGPredictor, record: MutationRecord, sample_chain_maps: dict) -> dict:
    weights, inputs = compute_pooling_output(model, record, sample_chain_maps)
    antigen_mask = inputs.antigen_mask[0]
    mutation_on_antigen = record.mutations[0].chain_role == ChainRole.ANTIGEN
    other_side_mask = ~antigen_mask if mutation_on_antigen else antigen_mask
    true_position = record.mutations[0].flat_residue_index
    return dict(
        participation_ratio=participation_ratio(weights),
        argmax_is_true_site=bool(weights.argmax().item() == true_position),
        other_chain_weight_mass=weights[other_side_mask].sum().item(),
    )


def run_flat_softmax_sweep(records: list[MutationRecord], sample_chain_maps: dict, temperatures: list[float]) -> pd.DataFrame:
    rows = []
    for temperature in temperatures:
        model = build_ddg_model("norm_softmax", temperature)
        for record in records:
            rows.append({"temperature": temperature, **flat_localization_row(model, record, sample_chain_maps)})
    return pd.DataFrame(rows)


def summarize_flat_softmax_sweep(flat_sweep_df: pd.DataFrame) -> pd.DataFrame:
    return flat_sweep_df.groupby("temperature").agg(
        mean_participation_ratio=("participation_ratio", "mean"),
        argmax_hit_rate=("argmax_is_true_site", "mean"),
        mean_other_chain_weight_mass=("other_chain_weight_mass", "mean"),
    )


def run_splitpool_sweep(records: list[MutationRecord], sample_chain_maps: dict, temperatures: list[float]) -> pd.DataFrame:
    frames = []
    for temperature in temperatures:
        frame = run_splitpool_localization(records, sample_chain_maps, temperature)
        frame["temperature"] = temperature
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def summarize_splitpool_sweep(splitpool_sweep_df: pd.DataFrame) -> pd.DataFrame:
    return splitpool_sweep_df.groupby(["temperature", "side"]).agg(
        mean_participation_ratio=("participation_ratio", "mean"), argmax_hit_rate=("argmax_is_true_site", "mean")
    )


def plot_flat_softmax_temperature_tradeoff(flat_sweep_summary: pd.DataFrame, chosen_temperature: float) -> plt.Figure:
    antibody_color, antigen_color, chosen_color = "#2a78d6", "#eb6834", "#3f9142"
    fig, (ax_participation, ax_leakage) = plt.subplots(1, 2, figsize=(11, 4))

    ax_participation.plot(flat_sweep_summary.index, flat_sweep_summary["mean_participation_ratio"], marker="o", color=antibody_color)
    ax_participation.axvline(chosen_temperature, color=chosen_color, linestyle="--", label=f"chosen T={chosen_temperature}")
    ax_participation.set_yscale("log")
    ax_participation.set_xlabel("pooling_temperature")
    ax_participation.set_ylabel("mean participation ratio (log scale)")
    ax_participation.set_title("Sharpness: one-hot (low T) → washed out (high T)")
    ax_participation.legend()

    ax_leakage.plot(flat_sweep_summary.index, flat_sweep_summary["mean_other_chain_weight_mass"], marker="o", color=antigen_color)
    ax_leakage.axvline(chosen_temperature, color=chosen_color, linestyle="--", label=f"chosen T={chosen_temperature}")
    ax_leakage.set_xlabel("pooling_temperature")
    ax_leakage.set_ylabel("mean weight mass on the NON-mutated chain")
    ax_leakage.set_title("A single global softmax can zero out the other chain")
    ax_leakage.legend()

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3D structure visualization
# ---------------------------------------------------------------------------


def flat_residue_list(record: MutationRecord, chain_map: dict) -> list[tuple[str, int, str]]:
    structure = load_structure(record.pdb_id)
    flat = []
    for chain_id in ordered_chain_ids_for_sample(chain_map):
        chain = get_chain(structure, chain_id)
        for raw_position, insertion_code, _one_letter, _residue in chain_residue_sequence(chain):
            flat.append((chain_id, raw_position, insertion_code))
    return flat


def moving_average_1d(values: torch.Tensor, kernel_size: int) -> torch.Tensor:
    kernel_size = min(kernel_size, values.numel())
    if kernel_size <= 1:
        return values
    kernel = torch.ones(kernel_size) / kernel_size
    pad_left = kernel_size // 2
    pad_right = kernel_size - 1 - pad_left
    padded = F.pad(values.view(1, 1, -1), (pad_left, pad_right), mode="replicate")
    return F.conv1d(padded, kernel.view(1, 1, -1)).view(-1)


def smooth_weights_by_chain(weights: torch.Tensor, chain_ids: list[str], kernel_size: int = 5) -> torch.Tensor:
    """Visualization-only smoothing: a light moving average over each chain's
    sequence-adjacent residues, applied purely to the COLOR signal so the 3D
    gradient reads continuously along the backbone instead of one sharp spike.
    This is not a claim about the actual (unsmoothed) pooling weights the
    model uses elsewhere -- see `participation_ratio_report` for those.
    """
    smoothed_chunks, start = [], 0
    for _chain_id, group in itertools.groupby(chain_ids):
        length = sum(1 for _ in group)
        smoothed_chunks.append(moving_average_1d(weights[start : start + length], kernel_size))
        start += length
    return torch.cat(smoothed_chunks)


def weight_to_hex(weight: float, vmin: float, vmax: float) -> str:
    normalized = 0.0 if vmax == vmin else (weight - vmin) / (vmax - vmin)
    return mcolors.to_hex(matplotlib.colormaps["viridis"](normalized))


def render_sample_view(
    record: MutationRecord, sample_chain_maps: dict, weights: torch.Tensor, smoothing_kernel_size: int = 5
) -> tuple[py3Dmol.view, float, float]:
    chain_map = sample_chain_maps[record.sample_id]
    flat_residues = flat_residue_list(record, chain_map)
    assert len(flat_residues) == weights.numel(), "flat residue list must align 1:1 with pooling weights"

    chain_ids = [chain_id for chain_id, _, _ in flat_residues]
    color_weights = smooth_weights_by_chain(weights, chain_ids, smoothing_kernel_size)
    vmin, vmax = color_weights.min().item(), color_weights.max().item()
    pdb_text = skempi_structure_path(record.pdb_id).read_text()

    view = py3Dmol.view(width=900, height=650)
    view.addModel(pdb_text, "pdb")
    view.setStyle({}, {"cartoon": {"color": "white"}})
    for (chain_id, raw_position, insertion_code), weight in zip(flat_residues, color_weights.tolist()):
        selector = {"chain": chain_id, "resi": raw_position}
        if insertion_code:
            selector["icode"] = insertion_code
        view.addStyle(selector, {"cartoon": {"color": weight_to_hex(weight, vmin, vmax)}})

    for mutation in record.mutations:
        selector = {"chain": mutation.chain_id, "resi": mutation.residue_position}
        if mutation.insertion_code:
            selector["icode"] = mutation.insertion_code
        view.addStyle(selector, {"stick": {"color": "magenta", "radius": 0.35}})
    view.zoomTo()
    return view, vmin, vmax


def render_and_save_sample(
    record: MutationRecord, sample_chain_maps: dict, weights: torch.Tensor, output_dir: Path, smoothing_kernel_size: int = 5
) -> Path:
    view, _vmin, _vmax = render_sample_view(record, sample_chain_maps, weights, smoothing_kernel_size)
    out_path = output_dir / f"pooling_norms_3d_{record.sample_id}.html"
    view.write_html(str(out_path))
    return out_path


def pooling_weight_colorbar_figure(vmin: float, vmax: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 0.7))
    normalize = mcolors.Normalize(vmin=vmin, vmax=vmax)
    mappable = matplotlib.cm.ScalarMappable(norm=normalize, cmap="viridis")
    fig.colorbar(mappable, cax=ax, orientation="horizontal", label="pooling weight (low → high)")
    fig.tight_layout()
    return fig
