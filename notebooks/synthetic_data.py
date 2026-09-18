"""Synthetic MutationRecord generator used by the EDA notebooks in this directory.

The pre-processing workstream (see `data/` and `shared.constants.PROCESSED_DATA_DIR`)
runs independently and in parallel, and may not have produced real processed SKEMPI
records yet. This module fabricates a small, schema-valid dataset so the notebooks here
are runnable end-to-end right now. Every numeric assumption below (region means,
alanine-scanning shift, label-type mix) is an illustrative guess, not a measured
quantity, and carries zero weight once real data is available.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shared.constants import (  # noqa: E402
    ChainRole,
    InterfaceRegion,
    LabelType,
    MutationRecord,
    NOISE_FLOOR_KCAL_MOL,
    ddg_to_bin,
)

AMINO_ACIDS = np.array(list("ACDEFGHIKLMNPQRSTVWY"))

REGION_MEAN_DDG_KCAL_MOL: dict[str, float] = {
    InterfaceRegion.CORE.value: 1.3,
    InterfaceRegion.SUPPORT.value: 0.9,
    InterfaceRegion.RIM.value: 0.4,
    InterfaceRegion.SURFACE.value: 0.05,
    InterfaceRegion.INTERIOR.value: 0.3,
}

REGION_WEIGHTS: dict[str, float] = {
    InterfaceRegion.CORE.value: 0.22,
    InterfaceRegion.SUPPORT.value: 0.18,
    InterfaceRegion.RIM.value: 0.25,
    InterfaceRegion.SURFACE.value: 0.28,
    InterfaceRegion.INTERIOR.value: 0.07,
}

LABEL_TYPE_WEIGHTS: dict[str, float] = {
    LabelType.BOUNDED.value: 0.70,
    LabelType.INEQ.value: 0.22,
    LabelType.NB.value: 0.08,
}

ALANINE_SCANNING_DDG_SHIFT_KCAL_MOL = 0.6
ALANINE_SCANNING_FRACTION = 0.35
INEQ_NUMERIC_BOUND_REPORT_RATE = 0.5
DDG_NOISE_SCALE_KCAL_MOL = NOISE_FLOOR_KCAL_MOL * 0.9

CHAIN_ID_BY_ROLE = {
    ChainRole.HEAVY.value: "H",
    ChainRole.LIGHT.value: "L",
    ChainRole.ANTIGEN.value: "A",
}


def normalized_probabilities(weights: dict[str, float]) -> np.ndarray:
    values = np.array(list(weights.values()), dtype=float)
    return values / values.sum()


def sample_weighted_categories(weights: dict[str, float], size: int, rng: np.random.Generator) -> np.ndarray:
    categories = np.array(list(weights.keys()))
    return rng.choice(categories, size=size, p=normalized_probabilities(weights))


def sample_regions(n: int, rng: np.random.Generator) -> np.ndarray:
    return sample_weighted_categories(REGION_WEIGHTS, n, rng)


def sample_label_types(n: int, rng: np.random.Generator) -> np.ndarray:
    return sample_weighted_categories(LABEL_TYPE_WEIGHTS, n, rng)


def sample_alanine_scanning_flags(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.random(n) < ALANINE_SCANNING_FRACTION


def sample_chain_roles(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(np.array([role.value for role in ChainRole]), size=n)


def region_baseline_mean_ddg(regions: np.ndarray) -> np.ndarray:
    return pd.Series(regions).map(REGION_MEAN_DDG_KCAL_MOL).to_numpy()


def apply_alanine_scanning_shift(mean_ddg: np.ndarray, is_alanine_scanning: np.ndarray) -> np.ndarray:
    return mean_ddg + is_alanine_scanning * ALANINE_SCANNING_DDG_SHIFT_KCAL_MOL


def sample_raw_ddg(mean_ddg: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(loc=mean_ddg, scale=DDG_NOISE_SCALE_KCAL_MOL, size=mean_ddg.shape[0])


def nonzero_sign(values: np.ndarray) -> np.ndarray:
    signs = np.sign(values)
    return np.where(signs == 0, 1.0, signs)


def push_ineq_values_into_tail(raw_ddg: np.ndarray, label_types: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    is_ineq = label_types == LabelType.INEQ.value
    tail_shift = rng.uniform(0.5, 2.0, size=raw_ddg.shape[0]) * nonzero_sign(raw_ddg)
    return np.where(is_ineq, raw_ddg + tail_shift, raw_ddg)


def resample_alanine_wt_conflicts(wt_residues: np.ndarray, is_alanine_scanning: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    non_alanine_residues = AMINO_ACIDS[AMINO_ACIDS != "A"]
    conflicts = is_alanine_scanning & (wt_residues == "A")
    wt_residues = wt_residues.copy()
    wt_residues[conflicts] = rng.choice(non_alanine_residues, size=conflicts.sum())
    return wt_residues


def sample_wt_residues(n: int, is_alanine_scanning: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    wt_residues = rng.choice(AMINO_ACIDS, size=n)
    return resample_alanine_wt_conflicts(wt_residues, is_alanine_scanning, rng)


def sample_mutant_residues(wt_residues: np.ndarray, is_alanine_scanning: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    wt_indices = np.searchsorted(AMINO_ACIDS, wt_residues)
    offsets = rng.integers(1, AMINO_ACIDS.shape[0], size=wt_residues.shape[0])
    other_residues = AMINO_ACIDS[(wt_indices + offsets) % AMINO_ACIDS.shape[0]]
    return np.where(is_alanine_scanning, "A", other_residues)


def resolve_bounded_ddg(raw_ddg: np.ndarray, label_types: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    is_bounded = label_types == LabelType.BOUNDED.value
    ddg_values = np.where(is_bounded, np.round(raw_ddg, 3), np.nan)
    ddg_bins = np.full(raw_ddg.shape[0], np.nan)
    ddg_bins[is_bounded] = np.vectorize(ddg_to_bin)(ddg_values[is_bounded])
    return ddg_values, ddg_bins


def resolve_ineq_direction(raw_ddg: np.ndarray, label_types: np.ndarray) -> np.ndarray:
    is_ineq = label_types == LabelType.INEQ.value
    direction = np.where(raw_ddg >= 0, ">", "<")
    return np.where(is_ineq, direction, None)


def apply_ineq_best_effort_bound(ddg_values: np.ndarray, raw_ddg: np.ndarray, label_types: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    is_ineq = label_types == LabelType.INEQ.value
    reports_bound = is_ineq & (rng.random(label_types.shape[0]) < INEQ_NUMERIC_BOUND_REPORT_RATE)
    return np.where(reports_bound, np.round(raw_ddg, 3), ddg_values)


def resolve_ddg_fields(raw_ddg: np.ndarray, label_types: np.ndarray, rng: np.random.Generator) -> dict[str, np.ndarray]:
    ddg_values, ddg_bins = resolve_bounded_ddg(raw_ddg, label_types)
    ddg_values = apply_ineq_best_effort_bound(ddg_values, raw_ddg, label_types, rng)
    ineq_directions = resolve_ineq_direction(raw_ddg, label_types)
    return {"ddg_kcal_mol": ddg_values, "ddg_bin": ddg_bins, "ineq_direction": ineq_directions}


def sample_pdb_ids(pool_size: int, n: int, rng: np.random.Generator) -> np.ndarray:
    pool = np.array([f"SYN{i:04d}" for i in range(pool_size)])
    return rng.choice(pool, size=n)


def as_optional_float(value: float) -> float | None:
    return None if pd.isna(value) else float(value)


def as_optional_int(value: float) -> int | None:
    return None if pd.isna(value) else int(value)


def as_optional_str(value: object) -> str | None:
    return None if value is None else str(value)


def build_mutation_record(index: int, fields: dict[str, np.ndarray]) -> MutationRecord:
    chain_role = fields["chain_role"][index]
    return MutationRecord(
        sample_id=f"SYN{index:05d}",
        pdb_id=fields["pdb_id"][index],
        complex_name=None,
        chain_id=CHAIN_ID_BY_ROLE[chain_role],
        chain_role=chain_role,
        wt_residue=fields["wt_residue"][index],
        mutant_residue=fields["mutant_residue"][index],
        residue_position=int(fields["residue_position"][index]),
        aligned_interface_position=int(fields["aligned_interface_position"][index]),
        label_type=fields["label_type"][index],
        ddg_kcal_mol=as_optional_float(fields["ddg_kcal_mol"][index]),
        ddg_bin=as_optional_int(fields["ddg_bin"][index]),
        ineq_direction=as_optional_str(fields["ineq_direction"][index]),
        interface_region=fields["interface_region"][index],
        is_alanine_scanning=bool(fields["is_alanine_scanning"][index]),
        split_membership={},
        source_publication="synthetic",
        notes="synthetic record for pre-training EDA -- not real SKEMPI data",
    )


def generate_synthetic_fields(n: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    regions = sample_regions(n, rng)
    label_types = sample_label_types(n, rng)
    is_alanine_scanning = sample_alanine_scanning_flags(n, rng)

    mean_ddg = apply_alanine_scanning_shift(region_baseline_mean_ddg(regions), is_alanine_scanning)
    raw_ddg = push_ineq_values_into_tail(sample_raw_ddg(mean_ddg, rng), label_types, rng)
    ddg_fields = resolve_ddg_fields(raw_ddg, label_types, rng)

    wt_residues = sample_wt_residues(n, is_alanine_scanning, rng)
    mutant_residues = sample_mutant_residues(wt_residues, is_alanine_scanning, rng)

    fields = {
        "interface_region": regions,
        "label_type": label_types,
        "is_alanine_scanning": is_alanine_scanning,
        "wt_residue": wt_residues,
        "mutant_residue": mutant_residues,
        "chain_role": sample_chain_roles(n, rng),
        "pdb_id": sample_pdb_ids(60, n, rng),
        "residue_position": rng.integers(1, 450, size=n),
        "aligned_interface_position": rng.integers(1, 450, size=n),
    }
    fields.update(ddg_fields)
    return fields


def generate_synthetic_records(n: int = 1000, seed: int = 42) -> list[MutationRecord]:
    """Generate `n` synthetic MutationRecord instances, validated by construction."""
    fields = generate_synthetic_fields(n, seed)
    return [build_mutation_record(i, fields) for i in range(n)]


def generate_synthetic_dataframe(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    records = generate_synthetic_records(n=n, seed=seed)
    return pd.DataFrame([record.model_dump(mode="json") for record in records])
