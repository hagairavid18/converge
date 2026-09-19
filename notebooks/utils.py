"""Shared helpers for the notebooks in this directory.

Three sections: (1) loading real processed MutationRecord data for
`01_data_exploration.ipynb`, (2) mutation-localization / pooling-mechanism
analysis for `03_model.ipynb`, (3) classical-regressor baselines and bucketed
MAE reporting for `04_results.ipynb`. Consolidated from what used to be
`processed_data.py`, `pooling_analysis_lib.py`, and `results_lib.py`.
"""

from __future__ import annotations

import itertools
import json
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import py3Dmol
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from torch import nn
from torch.utils.data import DataLoader

from data.chain_roles import ordered_chain_ids_for_sample
from data.embedding_pipeline.entry_embeddings import entry_embeddings_are_cached
from data.homology_dedup import normalize_protein_name
from data.structures import chain_residue_sequence, get_chain, load_structure, skempi_structure_path
from dl.datasets.collation import collate_ddg_batch
from dl.datasets.config import DatasetConfig
from dl.datasets.dataset_builder import build_real_datasets, build_train_and_val_datasets, resolve_loss_iteration
from dl.datasets.mutation_csv_dataset import (
    filter_records_for_active_iteration,
    keep_bounded_only,
    label_fields_for_record,
    load_or_compute_cached_tensors,
    load_records,
    select_model_fields,
)
from dl.metrics.regression_metrics import compute_magnitude_bucket_indices
from dl.models.config import ModelConfig
from dl.models.ddg_model import DDGPredictor
from dl.training.lightning_module import DDGLightningModule
from dl.training.run_config import RunConfig, load_run_config
from dl.utils.constants import MAGNITUDE_BUCKET_NAMES
from dl.utils.embedding_mode import uses_sequence_embedding, uses_structure_embedding
from dl.utils.label_codes import BOUNDED_ID
from shared.constants import (
    ACTIVE_EMBEDDING_SOURCE_MODE,
    DDG_NUM_BINS,
    ChainRole,
    MutationRecord,
    SPLIT_FILES,
    SPLIT_SUBSETS,
    SplitName,
)

# ===========================================================================
# 1. Real processed-data loading (01_data_exploration.ipynb)
# ===========================================================================

SPLIT_SCHEME_COLUMN = "split_scheme"
SPLIT_SUBSET_COLUMN = "split_subset"
SAMPLE_LEVEL_COLUMNS = [
    "sample_id",
    "pdb_id",
    "complex_name",
    "label_type",
    "ddg_kcal_mol",
    "ddg_bin",
    "ineq_direction",
    "temperature_kelvin",
]


def iter_split_file_locations() -> list[tuple[SplitName, str, Path]]:
    return [(split, subset, SPLIT_FILES[split][subset]) for split in SplitName for subset in SPLIT_SUBSETS]


def read_split_csv(split: SplitName, subset: str, path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame[SPLIT_SCHEME_COLUMN] = split.value
    frame[SPLIT_SUBSET_COLUMN] = subset
    return frame


def read_existing_split_frames() -> list[pd.DataFrame]:
    return [read_split_csv(split, subset, path) for split, subset, path in iter_split_file_locations() if path.exists()]


def raise_missing_split_files_error() -> None:
    expected_paths = [str(path) for _, _, path in iter_split_file_locations()]
    raise FileNotFoundError(
        f"None of the expected processed split files exist yet: {expected_paths}. "
        "Run the pre-processing pipeline first (see README.md)."
    )


def deduplicate_rows(data: pd.DataFrame) -> pd.DataFrame:
    """A sample is assigned independently under both splitting schemes, so its row is
    written verbatim into two of the four source files; `sample_id` alone is a safe
    dedup key here since it uniquely identifies a sample-level row.
    """
    return data.drop_duplicates(subset=["sample_id"], keep="first").reset_index(drop=True)


def explode_mutations(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for record in data.to_dict("records"):
        for mutation in json.loads(record["mutations"]):
            rows.append({**{col: record[col] for col in SAMPLE_LEVEL_COLUMNS}, **mutation})
    return pd.DataFrame(rows)


def load_processed_dataframe() -> pd.DataFrame:
    frames = read_existing_split_frames()
    if not frames:
        raise_missing_split_files_error()
    combined = deduplicate_rows(pd.concat(frames, ignore_index=True))
    return explode_mutations(combined)


# ===========================================================================
# 2. Mutation-localization & pooling-mechanism analysis (03_model.ipynb)
# ===========================================================================

_NO_TRAIN_COMPLEXES: frozenset[str] = frozenset()


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


def train_val_probe_split(samples: list[ProbeSample], train_fraction: float, seed: int) -> tuple[list[ProbeSample], list[ProbeSample]]:
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


GREY_COLOR = "#b0b0b0"
MUTATED_SIDE_COLORMAP = "YlOrRd"
SIDE_COLORMAPS = {"antibody": "Blues", "antigen": "YlOrRd"}
COLOR_GAMMA = 0.45


def weight_to_hex(weight: float, vmin: float, vmax: float, colormap: str = MUTATED_SIDE_COLORMAP, gamma: float = COLOR_GAMMA) -> str:
    """`gamma < 1` compresses the top of the range and expands the bottom/middle,
    so residues with meaningful-but-not-maximal weight are still visibly distinct
    from the near-zero floor -- a plain linear scale on a sharply peaked softmax
    makes everything but the single top residue collapse to nearly the same shade.
    """
    normalized = 0.0 if vmax == vmin else (weight - vmin) / (vmax - vmin)
    return mcolors.to_hex(matplotlib.colormaps[colormap](normalized**gamma))


def chain_side_for_residues(flat_residues: list[tuple[str, int, str]], chain_map: dict) -> list[str]:
    antigen_chain_ids = set(chain_map.get(ChainRole.ANTIGEN.value, ""))
    return ["antigen" if chain_id in antigen_chain_ids else "antibody" for chain_id, _, _ in flat_residues]


def mutated_side_for_record(record: MutationRecord) -> str:
    return "antigen" if record.mutations[0].chain_role == ChainRole.ANTIGEN else "antibody"


def per_side_color_ranges(color_weights: torch.Tensor, residue_sides: list[str]) -> dict[str, tuple[float, float]]:
    """Each side is normalized against its OWN min/max, independently of the
    other -- `ChainRoleSplitPooling` runs a separate softmax per side, so the
    two sides' raw weight magnitudes are not comparable on a shared scale.
    Under `structure_only` embeddings the non-mutated side's diff-norm is
    exactly zero by construction (each chain is encoded independently, so a
    mutation on one chain cannot change the other's embedding) -- its own
    min/max will both be ~0, so it renders as one near-uniform shade of its
    side's colormap rather than a fabricated gradient. That flatness is the
    real finding, not a display bug.
    """
    sides = torch.tensor([side == "antigen" for side in residue_sides])
    return {
        side: (color_weights[mask].min().item(), color_weights[mask].max().item())
        for side, mask in (("antibody", ~sides), ("antigen", sides))
        if mask.any()
    }


def residue_colors_for_record(
    record: MutationRecord, sample_chain_maps: dict, weights: torch.Tensor, smoothing_kernel_size: int = 7
) -> tuple[list[tuple[str, int, str]], list[str], dict[str, tuple[float, float]], str]:
    chain_map = sample_chain_maps[record.sample_id]
    flat_residues = flat_residue_list(record, chain_map)
    assert len(flat_residues) == weights.numel(), "flat residue list must align 1:1 with pooling weights"

    chain_ids = [chain_id for chain_id, _, _ in flat_residues]
    color_weights = smooth_weights_by_chain(weights, chain_ids, smoothing_kernel_size)
    residue_sides = chain_side_for_residues(flat_residues, chain_map)
    mutated_side = mutated_side_for_record(record)
    color_ranges = per_side_color_ranges(color_weights, residue_sides)

    colors = [
        weight_to_hex(weight, *color_ranges[side], colormap=SIDE_COLORMAPS[side])
        for weight, side in zip(color_weights.tolist(), residue_sides)
    ]
    return flat_residues, colors, color_ranges, mutated_side


def mutated_residue_ca_coordinate(record: MutationRecord) -> tuple[float, float, float]:
    mutation = record.mutations[0]
    structure = load_structure(record.pdb_id)
    chain = get_chain(structure, mutation.chain_id)
    insertion_code = (mutation.insertion_code or "").upper()
    for raw_position, code, _one_letter, residue in chain_residue_sequence(chain):
        if raw_position == mutation.residue_position and code.upper() == insertion_code:
            coord = residue["CA"].get_coord()
            return float(coord[0]), float(coord[1]), float(coord[2])
    raise ValueError(f"Mutated residue not found in structure for {record.sample_id}")


def apply_sample_style(
    view: py3Dmol.view, record: MutationRecord, sample_chain_maps: dict, weights: torch.Tensor, viewer=None, smoothing_kernel_size: int = 7
) -> dict[str, tuple[float, float]]:
    flat_residues, colors, color_ranges, _mutated_side = residue_colors_for_record(record, sample_chain_maps, weights, smoothing_kernel_size)
    kwargs = {} if viewer is None else {"viewer": viewer}

    pdb_text = skempi_structure_path(record.pdb_id).read_text()
    view.addModel(pdb_text, "pdb", **kwargs)
    view.setStyle({}, {"cartoon": {"color": GREY_COLOR}}, **kwargs)
    for (chain_id, raw_position, insertion_code), color in zip(flat_residues, colors):
        selector = {"chain": chain_id, "resi": raw_position}
        if insertion_code:
            selector["icode"] = insertion_code
        view.addStyle(selector, {"cartoon": {"color": color}}, **kwargs)

    for mutation in record.mutations:
        selector = {"chain": mutation.chain_id, "resi": mutation.residue_position}
        if mutation.insertion_code:
            selector["icode"] = mutation.insertion_code
        view.addStyle(selector, {"stick": {"color": "magenta", "radius": 0.35}}, **kwargs)
    view.zoomTo(**({"viewer": viewer} if viewer is not None else {}))
    return color_ranges


def render_sample_view(
    record: MutationRecord, sample_chain_maps: dict, weights: torch.Tensor, smoothing_kernel_size: int = 7
) -> tuple[py3Dmol.view, dict[str, tuple[float, float]]]:
    view = py3Dmol.view(width=900, height=650)
    color_ranges = apply_sample_style(view, record, sample_chain_maps, weights, smoothing_kernel_size=smoothing_kernel_size)
    return view, color_ranges


def render_and_save_sample(
    record: MutationRecord, sample_chain_maps: dict, weights: torch.Tensor, output_dir: Path, smoothing_kernel_size: int = 7
) -> Path:
    view, _color_ranges = render_sample_view(record, sample_chain_maps, weights, smoothing_kernel_size)
    out_path = output_dir / f"pooling_norms_3d_{record.sample_id}.html"
    view.write_html(str(out_path))
    return out_path


def render_gallery_grid(
    examples: list[tuple[MutationRecord, torch.Tensor]],
    sample_chain_maps: dict,
    output_path: Path,
    cols: int = 3,
    smoothing_kernel_size: int = 7,
) -> Path:
    """One combined HTML page with all examples side by side (a py3Dmol
    viewergrid), each cell labeled with its sample id and interface region and
    using the same grey-antibody / colored-mutated-side convention as
    `render_sample_view`.
    """
    rows = (len(examples) + cols - 1) // cols
    view = py3Dmol.view(width=380 * cols, height=380 * rows, viewergrid=(rows, cols), linked=False)
    for index, (record, weights) in enumerate(examples):
        cell = (index // cols, index % cols)
        apply_sample_style(view, record, sample_chain_maps, weights, viewer=cell, smoothing_kernel_size=smoothing_kernel_size)
        interface_region = record.mutations[0].interface_region
        region_label = interface_region.value if interface_region is not None else "unknown"
        label_text = f"{record.sample_id}\n{mutated_side_for_record(record)} side, {region_label}"
        x, y, z = mutated_residue_ca_coordinate(record)
        view.addLabel(
            label_text,
            {
                "position": {"x": x, "y": y, "z": z},
                "backgroundColor": "black",
                "backgroundOpacity": 0.7,
                "fontColor": "white",
                "fontSize": 12,
                "inFront": True,
            },
            viewer=cell,
        )
    view.write_html(str(output_path))
    return output_path


def pooling_weight_colorbar_figure(color_ranges: dict[str, tuple[float, float]], mutated_side: str | None = None) -> plt.Figure:
    fig, axes = plt.subplots(len(color_ranges), 1, figsize=(5, 0.9 * len(color_ranges)))
    axes = [axes] if len(color_ranges) == 1 else list(axes)
    for ax, (side, (vmin, vmax)) in zip(axes, color_ranges.items()):
        normalize = mcolors.Normalize(vmin=vmin, vmax=vmax)
        mappable = matplotlib.cm.ScalarMappable(norm=normalize, cmap=SIDE_COLORMAPS[side])
        tag = " (mutated)" if side == mutated_side else " (not mutated -- expected ~flat)"
        fig.colorbar(mappable, cax=ax, orientation="horizontal", label=f"{side}{tag} pooling weight (low → high)")
    fig.tight_layout()
    return fig


# ===========================================================================
# 3. Classical-regressor baselines & bucketed MAE (04_results.ipynb)
# ===========================================================================

DEFAULT_RESULTS_JSON_PATH = Path(__file__).resolve().parent / "final_sota_vs_baseline_histogram_results.json"

TRAIN_CHAIN_ROLE_MIX = {"antigen_only": 0.343, "antibody_only": 0.599, "both": 0.058}
VAL_CHAIN_ROLE_MIX = {"antigen_only": 0.267, "antibody_only": 0.653, "both": 0.079}

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


def build_pooled_feature_splits(model: DDGPredictor, model_config: ModelConfig, dataset_config: DatasetConfig) -> tuple[dict, dict, dict]:
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
    embedding_source_mode: str = "structure_only",
    split: SplitName = SplitName.SAME_PDB_ALLOWED,
    batch_size: int = 32,
    regressor_names: tuple[str, ...] | None = None,
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

    names = regressor_names or tuple(CLASSICAL_REGRESSORS)
    rows = [
        evaluate_regressor(
            name,
            CLASSICAL_REGRESSORS[name],
            train_x,
            train_fields["target_ddg"],
            val_new_complex_x,
            val_new_complex["target_ddg"],
            val_seen_complex_x,
            val_seen_complex["target_ddg"],
        )
        for name in names
    ]
    return pd.DataFrame(rows).set_index("regressor")


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


def chain_role_mix_comparison_table() -> pd.DataFrame:
    return pd.DataFrame({"train (n=855)": TRAIN_CHAIN_ROLE_MIX, "val (n=202)": VAL_CHAIN_ROLE_MIX}).round(3)
