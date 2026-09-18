import torch

from dl.metrics.report import DDGEvalBatch, compute_full_report, compute_reports_for_all_splits
from dl.utils.label_codes import BOUNDED_ID, INEQ_ID, NB_ID
from dl.utils.region_codes import REGION_TO_ID, UNKNOWN_REGION_ID
from shared.constants import DDG_BIN_EDGES_KCAL_MOL, DDG_NUM_BINS, InterfaceRegion, SplitName

BIN_EDGES = list(DDG_BIN_EDGES_KCAL_MOL)
INEQ_DEFAULT_BOUND = BIN_EDGES[-1]
NB_ANCHOR = BIN_EDGES[-1] + 1.0


def make_perfect_batch() -> DDGEvalBatch:
    num_samples = 5
    target_bin = torch.tensor([0, 2, 4, 6, 9])
    logits = torch.full((num_samples, DDG_NUM_BINS), -20.0)
    logits[torch.arange(num_samples), target_bin] = 20.0
    return DDGEvalBatch(
        logits=logits,
        label_type_id=torch.tensor([BOUNDED_ID] * num_samples),
        bin_target=target_bin,
        ddg_kcal_mol=torch.tensor([-4.5, -2.5, -0.5, 1.5, 4.5]),
        bound_kcal_mol=torch.full((num_samples,), float("nan")),
        direction=torch.zeros(num_samples),
        interface_region_id=torch.tensor(
            [
                REGION_TO_ID[InterfaceRegion.CORE],
                REGION_TO_ID[InterfaceRegion.CORE],
                REGION_TO_ID[InterfaceRegion.SURFACE],
                UNKNOWN_REGION_ID,
                REGION_TO_ID[InterfaceRegion.RIM],
            ]
        ),
        is_alanine_scanning=torch.tensor([True, False, True, False, False]),
    )


def make_poor_batch() -> DDGEvalBatch:
    num_samples = 5
    target_bin = torch.tensor([0, 0, 0, 0, 0])
    logits = torch.full((num_samples, DDG_NUM_BINS), -20.0)
    logits[torch.arange(num_samples), 9] = 20.0
    return DDGEvalBatch(
        logits=logits,
        label_type_id=torch.tensor([BOUNDED_ID, BOUNDED_ID, INEQ_ID, NB_ID, BOUNDED_ID]),
        bin_target=target_bin,
        ddg_kcal_mol=torch.tensor([-4.5, -4.5, float("nan"), float("nan"), -4.5]),
        bound_kcal_mol=torch.tensor([float("nan"), float("nan"), 1.0, float("nan"), float("nan")]),
        direction=torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0]),
        interface_region_id=torch.tensor([UNKNOWN_REGION_ID] * num_samples),
        is_alanine_scanning=torch.tensor([False] * num_samples),
    )


def test_compute_full_report_perfect_batch_has_full_accuracy():
    report = compute_full_report(make_perfect_batch(), SplitName.HELD_OUT_PDB, INEQ_DEFAULT_BOUND, NB_ANCHOR, BIN_EDGES)
    assert report.bin_accuracy == 1.0
    assert report.confusion_summary["exact_fraction"] == 1.0
    assert report.n_bounded == 5


def test_compute_full_report_poor_batch_has_low_accuracy():
    report = compute_full_report(make_poor_batch(), SplitName.SAME_PDB_ALLOWED, INEQ_DEFAULT_BOUND, NB_ANCHOR, BIN_EDGES)
    assert report.bin_accuracy == 0.0
    assert report.n_ineq == 1
    assert report.n_nb == 1


def test_compute_full_report_without_regression_pred_has_no_regression_summary():
    report = compute_full_report(make_perfect_batch(), SplitName.HELD_OUT_PDB, INEQ_DEFAULT_BOUND, NB_ANCHOR, BIN_EDGES)
    assert report.regression_summary is None


def test_compute_full_report_with_regression_pred_adds_regression_summary():
    batch = make_perfect_batch()
    batch.regression_pred = batch.ddg_kcal_mol.clone()
    report = compute_full_report(batch, SplitName.HELD_OUT_PDB, INEQ_DEFAULT_BOUND, NB_ANCHOR, BIN_EDGES)
    assert report.regression_summary is not None
    assert report.regression_summary["mae"] == 0.0
    assert report.regression_summary["n"] == 5


def test_reports_for_all_splits_are_independent_and_never_pooled():
    reports = compute_reports_for_all_splits(
        {
            SplitName.HELD_OUT_PDB: make_perfect_batch(),
            SplitName.SAME_PDB_ALLOWED: make_poor_batch(),
        },
        INEQ_DEFAULT_BOUND,
        NB_ANCHOR,
        BIN_EDGES,
    )
    assert reports[SplitName.HELD_OUT_PDB].bin_accuracy == 1.0
    assert reports[SplitName.SAME_PDB_ALLOWED].bin_accuracy == 0.0
    assert reports[SplitName.HELD_OUT_PDB].split_name == SplitName.HELD_OUT_PDB
    assert reports[SplitName.SAME_PDB_ALLOWED].split_name == SplitName.SAME_PDB_ALLOWED
