import torch

from dl.metrics.hinge_metrics import IneqCorrectSideAccuracy, NBCorrectSideAccuracy, is_correct_side
from dl.utils.label_codes import BOUNDED_ID, INEQ_ID, NB_ID


def test_is_correct_side_basic():
    predicted = torch.tensor([3.0, 1.0])
    bound = torch.tensor([2.0, 2.0])
    direction = torch.tensor([1.0, 1.0])
    assert torch.equal(is_correct_side(predicted, bound, direction), torch.tensor([True, False]))


def test_ineq_metric_only_scores_ineq_entries():
    metric = IneqCorrectSideAccuracy()
    predicted_ddg = torch.tensor([3.0, 3.0, -10.0])
    label_type_id = torch.tensor([INEQ_ID, BOUNDED_ID, NB_ID])
    bound_kcal_mol = torch.tensor([2.0, float("nan"), float("nan")])
    direction = torch.tensor([1.0, 0.0, 0.0])

    metric.update(predicted_ddg, label_type_id, bound_kcal_mol, direction)
    assert metric.compute().item() == 1.0
    assert metric.total.item() == 1


def test_nb_metric_only_scores_nb_entries_and_uses_anchor():
    metric = NBCorrectSideAccuracy()
    predicted_ddg = torch.tensor([metric.nb_anchor_kcal_mol + 1.0, metric.nb_anchor_kcal_mol - 1.0])
    label_type_id = torch.tensor([NB_ID, NB_ID])
    bound_kcal_mol = torch.tensor([float("nan"), float("nan")])
    direction = torch.tensor([0.0, 0.0])

    metric.update(predicted_ddg, label_type_id, bound_kcal_mol, direction)
    assert metric.compute().item() == 0.5


def test_metrics_accumulate_across_updates():
    metric = IneqCorrectSideAccuracy()
    label_type_id = torch.tensor([INEQ_ID])
    direction = torch.tensor([1.0])

    metric.update(torch.tensor([3.0]), label_type_id, torch.tensor([2.0]), direction)
    metric.update(torch.tensor([1.0]), label_type_id, torch.tensor([2.0]), direction)
    assert metric.compute().item() == 0.5
    assert metric.total.item() == 2
