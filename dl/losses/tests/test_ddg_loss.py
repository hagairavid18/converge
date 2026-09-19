import torch

from dl.losses.config import DDGLossConfig
from dl.losses.ddg_loss import DDGLoss
from dl.utils.label_codes import BOUNDED_ID, INEQ_ID, NB_ID
from shared.constants import LossIteration


def make_batch(config: DDGLossConfig):
    logits = torch.randn(6, config.num_bins, requires_grad=True)
    label_type_id = torch.tensor([BOUNDED_ID, BOUNDED_ID, BOUNDED_ID, INEQ_ID, INEQ_ID, NB_ID])
    target_bin = torch.tensor([0, 4, 9, 0, 0, 0])
    bound_kcal_mol = torch.tensor([float("nan"), float("nan"), float("nan"), 1.5, float("nan"), float("nan")])
    direction = torch.tensor([0.0, 0.0, 0.0, 1.0, -1.0, 0.0])
    return logits, label_type_id, target_bin, bound_kcal_mol, direction


def test_iteration_1_requires_no_bound_or_direction():
    config = DDGLossConfig(iteration=LossIteration.ITERATION_1_BOUNDED_ONLY)
    loss_fn = DDGLoss(config)
    logits, label_type_id, target_bin, _, _ = make_batch(config)
    output = loss_fn(logits, label_type_id, target_bin)
    assert torch.isfinite(output.total_loss)


def test_nb_anchor_is_further_out_than_ineq_bound():
    config = DDGLossConfig()
    loss_fn = DDGLoss(config)
    label_type_id = torch.tensor([INEQ_ID, NB_ID])
    bound_kcal_mol = torch.tensor([1.5, float("nan")])
    direction = torch.tensor([1.0, 0.0])

    resolved_bound, resolved_direction = loss_fn.resolve_bounds(label_type_id, bound_kcal_mol, direction)
    assert resolved_bound[1].item() > resolved_bound[0].item()
    assert resolved_direction[1].item() == 1.0


def test_ineq_missing_bound_falls_back_to_default():
    config = DDGLossConfig()
    loss_fn = DDGLoss(config)
    label_type_id = torch.tensor([INEQ_ID, INEQ_ID])
    bound_kcal_mol = torch.tensor([float("nan"), float("nan")])
    direction = torch.tensor([1.0, -1.0])

    resolved_bound, _ = loss_fn.resolve_bounds(label_type_id, bound_kcal_mol, direction)
    assert resolved_bound[0].item() == config.ineq_default_bound_kcal_mol
    assert resolved_bound[1].item() == -config.ineq_default_bound_kcal_mol


def test_tail_reweight_requires_target_ddg():
    config = DDGLossConfig(iteration=LossIteration.ITERATION_1_BOUNDED_ONLY, tail_reweight_enabled=True)
    loss_fn = DDGLoss(config)
    logits, label_type_id, target_bin, _, _ = make_batch(config)
    try:
        loss_fn(logits, label_type_id, target_bin)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_batch_imbalance_reweight_upweights_minority_category():
    config_reweighted = DDGLossConfig(
        iteration=LossIteration.ITERATION_2_WITH_HINGE, apply_batch_imbalance_reweight=True
    )
    config_unweighted = DDGLossConfig(
        iteration=LossIteration.ITERATION_2_WITH_HINGE, apply_batch_imbalance_reweight=False
    )

    logits = torch.randn(10, config_reweighted.num_bins)
    label_type_id = torch.tensor([BOUNDED_ID] * 9 + [INEQ_ID])
    target_bin = torch.randint(0, config_reweighted.num_bins, (10,))
    bound_kcal_mol = torch.full((10,), float("nan"))
    bound_kcal_mol[9] = 1.0
    direction = torch.zeros(10)
    direction[9] = 1.0

    reweighted = DDGLoss(config_reweighted)(logits, label_type_id, target_bin, bound_kcal_mol, direction)
    unweighted = DDGLoss(config_unweighted)(logits, label_type_id, target_bin, bound_kcal_mol, direction)

    assert reweighted.hinge_weight > unweighted.hinge_weight
    assert reweighted.bounded_weight < unweighted.bounded_weight
