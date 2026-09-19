"""Renders the real per-residue pooling weights (`dl.layers.pooling`) for one
mutation sample directly on its WT 3D structure, as an interactive HTML file.

This is the structural counterpart to `03_mutation_localization_rationale.ipynb`:
that notebook shows numerically that the mut-wt embedding-difference norm (and
the softmax pooling weight built from it) concentrates at the true mutation
site; this script lets you see it on the actual PDB. Residues are colored by
their pooling weight (a `viridis` ramp: dark/blue = low weight, yellow = high
weight); the true mutated residue is additionally drawn as magenta sticks.

Run (requires the SaProt embedding cache, already populated for this repo's
1211 samples with the full-size checkpoint):
  SKEMPI_USE_SAPROT_STRUCTURE=1 SKEMPI_USE_SMALL_CHECKPOINTS=0 \
    uv run python notebooks/visualize_pooling_norms_3d.py [sample_id ...]

With no arguments, picks a few already-cached single-point-mutation samples
automatically and writes one HTML file per sample.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
import matplotlib.colors as mcolors
import py3Dmol
import torch

from data.chain_roles import ordered_chain_ids_for_sample  # noqa: E402
from data.embedding_pipeline.entry_embeddings import entry_embeddings_are_cached  # noqa: E402
from data.processed_io import load_sample_chain_maps  # noqa: E402
from data.structures import chain_residue_sequence, get_chain, load_structure, skempi_structure_path  # noqa: E402
from dl.datasets.collation import collate_ddg_batch  # noqa: E402
from dl.datasets.mutation_csv_dataset import (  # noqa: E402
    filter_records_for_active_iteration,
    label_fields_for_record,
    load_or_compute_cached_tensors,
    load_records,
    select_model_fields,
)
from dl.models.config import ModelConfig  # noqa: E402
from dl.models.ddg_model import DDGPredictor  # noqa: E402
from shared.constants import SPLIT_FILES, SplitName  # noqa: E402

NUM_EXAMPLES = 2
OUTPUT_DIR = _REPO_ROOT / "notebooks"


def single_mutation_records(csv_path) -> list:
    records = filter_records_for_active_iteration(load_records(csv_path))
    return [record for record in records if len(record.mutations) == 1]


def collect_unique_single_mutation_records() -> list:
    seen_sample_ids: set[str] = set()
    unique_records = []
    for split in SplitName:
        for subset in ("train", "val"):
            for record in single_mutation_records(SPLIT_FILES[split][subset]):
                if record.sample_id not in seen_sample_ids:
                    seen_sample_ids.add(record.sample_id)
                    unique_records.append(record)
    return unique_records


def has_local_pdb_and_cached_embeddings(record, sample_chain_maps: dict) -> bool:
    return skempi_structure_path(record.pdb_id).exists() and entry_embeddings_are_cached(
        record, sample_chain_maps[record.sample_id]
    )


def pick_examples(sample_chain_maps: dict, count: int) -> list:
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


def compute_pooling_weights(record, sample_chain_maps: dict) -> torch.Tensor:
    config = ModelConfig(
        embedding_source_mode="structure_only",
        active_heads=(),
        pooling_strategy="norm_softmax_by_chain_role",
        pooling_temperature=5.0,
    )
    model = DDGPredictor(config)
    model.eval()
    cached = load_or_compute_cached_tensors(record, sample_chain_maps)
    batch = [(select_model_fields(cached), label_fields_for_record(record, frozenset()))]
    inputs, _ = collate_ddg_batch(batch)
    with torch.no_grad():
        out = model(inputs)
    return out["pooling_weights"][0]


def flat_residue_list(record, chain_map: dict) -> list[tuple[str, int, str]]:
    structure = load_structure(record.pdb_id)
    flat = []
    for chain_id in ordered_chain_ids_for_sample(chain_map):
        chain = get_chain(structure, chain_id)
        for raw_position, insertion_code, _one_letter, _residue in chain_residue_sequence(chain):
            flat.append((chain_id, raw_position, insertion_code))
    return flat


def weight_to_hex(weight: float, vmin: float, vmax: float) -> str:
    normalized = 0.0 if vmax == vmin else (weight - vmin) / (vmax - vmin)
    return mcolors.to_hex(matplotlib.colormaps["viridis"](normalized))


def render_sample(record, sample_chain_maps: dict) -> Path:
    chain_map = sample_chain_maps[record.sample_id]
    weights = compute_pooling_weights(record, sample_chain_maps)
    flat_residues = flat_residue_list(record, chain_map)
    assert len(flat_residues) == weights.numel(), "flat residue list must align 1:1 with pooling weights"

    vmin, vmax = weights.min().item(), weights.max().item()
    pdb_text = skempi_structure_path(record.pdb_id).read_text()

    view = py3Dmol.view(width=900, height=650)
    view.addModel(pdb_text, "pdb")
    view.setStyle({}, {"cartoon": {"color": "white"}})
    for (chain_id, raw_position, insertion_code), weight in zip(flat_residues, weights.tolist()):
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

    mutation_label = ",".join(f"{m.chain_id}{m.residue_position}" for m in record.mutations)
    out_path = OUTPUT_DIR / f"pooling_norms_3d_{record.sample_id}.html"
    view.write_html(str(out_path))
    print(f"{record.sample_id} ({record.pdb_id}, mutation {mutation_label}): "
          f"L={weights.numel()} w_at_site={weights[record.mutations[0].flat_residue_index]:.4f} "
          f"w_max={weights.max().item():.4f} -> {out_path}")
    return out_path


def main() -> None:
    sample_chain_maps = load_sample_chain_maps()
    sample_ids = sys.argv[1:]
    records_by_id = {r.sample_id: r for r in collect_unique_single_mutation_records()}

    if sample_ids:
        records = [records_by_id[sample_id] for sample_id in sample_ids]
    else:
        records = pick_examples(sample_chain_maps, NUM_EXAMPLES)

    if not records:
        raise SystemExit("No single-point-mutation sample found with both a local PDB file and cached embeddings.")

    for record in records:
        render_sample(record, sample_chain_maps)


if __name__ == "__main__":
    main()
