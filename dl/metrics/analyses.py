"""Post-training evaluation analyses (Implementation Spec sec. 5, "two
additional analyses"): mutation location vs. ddG, and alanine-scanning vs.
other single-point substitutions. These operate on model predictions/labels
(not raw pre-processing data) and are reusable functions, not a notebook --
the pre-training raw-EDA versions of the same two analyses live in the
data-exploration workstream's notebooks.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy import stats

from dl.utils.region_codes import ID_TO_REGION, UNKNOWN_REGION_ID


def to_finite_numpy(values: torch.Tensor) -> np.ndarray:
    """`ddg_kcal_mol` may be NaN for entries without a well-defined real
    value (e.g. "ineq"/"n.b." ground truth); those are dropped here rather
    than silently poisoning a mean/std/test with NaN.
    """
    array = values.detach().cpu().numpy()
    return array[np.isfinite(array)]


def describe_group(values: torch.Tensor) -> dict:
    array = to_finite_numpy(values)
    if array.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "median": float("nan")}
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
    }


def region_name(region_id: int) -> str:
    return ID_TO_REGION[region_id].value if region_id != UNKNOWN_REGION_ID else "unknown"


def ddg_by_interface_region(
    interface_region_id: torch.Tensor, ddg_kcal_mol: torch.Tensor
) -> dict[str, dict]:
    """ddG distribution grouped by `InterfaceRegion` (support/core/rim/
    surface/interior). Rows with `UNKNOWN_REGION_ID` (region unavailable)
    are reported under a separate "unknown" key rather than silently dropped.
    """
    return {
        region_name(region_id): describe_group(ddg_kcal_mol[interface_region_id == region_id])
        for region_id in torch.unique(interface_region_id).tolist()
    }


def mannwhitney_comparison(sample_a: np.ndarray, sample_b: np.ndarray) -> dict:
    if sample_a.size == 0 or sample_b.size == 0:
        return {"u_statistic": float("nan"), "p_value": float("nan")}
    test_result = stats.mannwhitneyu(sample_a, sample_b, alternative="two-sided")
    return {"u_statistic": float(test_result.statistic), "p_value": float(test_result.pvalue)}


def alanine_scanning_vs_other(
    is_alanine_scanning: torch.Tensor, ddg_kcal_mol: torch.Tensor
) -> dict:
    """ddG distribution comparison: alanine-scanning (X->A) mutations vs.
    other single-point substitutions, with a Mann-Whitney U test (does not
    assume normality) for whether the two distributions differ.
    """
    alanine_mask = is_alanine_scanning.to(torch.bool)
    return {
        "alanine_scanning": describe_group(ddg_kcal_mol[alanine_mask]),
        "other_substitutions": describe_group(ddg_kcal_mol[~alanine_mask]),
        "mannwhitney": mannwhitney_comparison(
            to_finite_numpy(ddg_kcal_mol[alanine_mask]), to_finite_numpy(ddg_kcal_mol[~alanine_mask])
        ),
    }
