import torch

from dl.losses.config import DDGLossConfig
from dl.losses.ddg_regression_loss import DDGRegressionLoss
from dl.utils.label_codes import BOUNDED_ID, INEQ_ID, NB_ID
from shared.constants import LossIteration


def make_batch():
    predicted_ddg = torch.tensor([0.0, 2.0, -3.0, 5.0, -6.0, 10.0])
    label_type_id = torch.tensor([BOUNDED_ID, BOUNDED_ID, BOUNDED_ID, INEQ_ID, INEQ_ID, NB_ID])
    target_ddg = torch.tensor([0.5, 2.5, -2.5, float("nan"), float("nan"), float("nan")])
    bound_kcal_mol = torch.tensor([float("nan"), float("nan"), float("nan"), 1.5, float("nan"), float("nan")])
    direction = torch.tensor([0.0, 0.0, 0.0, 1.0, -1.0, 0.0])
    return predicted_ddg, label_type_id, target_ddg, bound_kcal_mol, direction


def test_iteration_1_ignores_ineq_and_nb_entries():
    config = DDGLossConfig(iteration=LossIteration.ITERATION_1_BOUNDED_ONLY)
    loss_fn = DDGRegressionLoss(config)
    predicted_ddg, label_type_id, target_ddg, bound_kcal_mol, direction = make_batch()

    with_hinge_inputs = loss_fn(predicted_ddg, label_type_id, target_ddg, bound_kcal_mol, direction)
    without_hinge_inputs = loss_fn(predicted_ddg, label_type_id, target_ddg)

    assert with_hinge_inputs.hinge_loss.item() == 0.0
    assert torch.allclose(with_hinge_inputs.total_loss, without_hinge_inputs.total_loss)
    assert with_hinge_inputs.n_ineq == 2
    assert with_hinge_inputs.n_nb == 1


def test_iteration_2_requires_bounds_when_hinge_entries_present():
    config = DDGLossConfig(iteration=LossIteration.ITERATION_2_WITH_HINGE)
    loss_fn = DDGRegressionLoss(config)
    predicted_ddg, label_type_id, target_ddg, _, _ = make_batch()
    try:
        loss_fn(predicted_ddg, label_type_id, target_ddg)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_iteration_2_hinge_penalizes_wrong_side():
    config = DDGLossConfig(iteration=LossIteration.ITERATION_2_WITH_HINGE, hinge_scale=1.0)
    loss_fn = DDGRegressionLoss(config)
    label_type_id = torch.tensor([INEQ_ID])
    target_ddg = torch.tensor([float("nan")])
    bound_kcal_mol = torch.tensor([1.0])
    direction = torch.tensor([1.0])

    correct_side = loss_fn(torch.tensor([5.0]), label_type_id, target_ddg, bound_kcal_mol, direction)
    assert correct_side.hinge_loss.item() == 0.0

    wrong_side = loss_fn(torch.tensor([-5.0]), label_type_id, target_ddg, bound_kcal_mol, direction)
    assert wrong_side.hinge_loss.item() > 0.0


def test_accepts_bx1_shaped_predictions():
    config = DDGLossConfig(iteration=LossIteration.ITERATION_1_BOUNDED_ONLY)
    loss_fn = DDGRegressionLoss(config)
    predicted_ddg_flat = torch.tensor([0.5, -0.5, 3.0])
    predicted_ddg_column = predicted_ddg_flat.unsqueeze(-1)
    label_type_id = torch.tensor([BOUNDED_ID, BOUNDED_ID, BOUNDED_ID])
    target_ddg = torch.tensor([0.5, -0.5, 3.0])

    flat_output = loss_fn(predicted_ddg_flat, label_type_id, target_ddg)
    column_output = loss_fn(predicted_ddg_column, label_type_id, target_ddg)
    assert torch.allclose(flat_output.total_loss, column_output.total_loss)


def test_bounded_loss_ignores_nan_targets_on_non_bounded_rows():
    config = DDGLossConfig(iteration=LossIteration.ITERATION_1_BOUNDED_ONLY)
    loss_fn = DDGRegressionLoss(config)
    predicted_ddg, label_type_id, target_ddg, _, _ = make_batch()
    output = loss_fn(predicted_ddg, label_type_id, target_ddg)
    assert torch.isfinite(output.total_loss)


def test_tail_reweight_disabled_by_default_matches_explicit_disable():
    config_default = DDGLossConfig(iteration=LossIteration.ITERATION_1_BOUNDED_ONLY)
    config_explicit_off = DDGLossConfig(iteration=LossIteration.ITERATION_1_BOUNDED_ONLY, tail_reweight_enabled=False)
    predicted_ddg, label_type_id, target_ddg, _, _ = make_batch()

    default_output = DDGRegressionLoss(config_default)(predicted_ddg, label_type_id, target_ddg)
    explicit_off_output = DDGRegressionLoss(config_explicit_off)(predicted_ddg, label_type_id, target_ddg)

    assert torch.allclose(default_output.total_loss, explicit_off_output.total_loss)


def test_tail_reweight_enabled_changes_bounded_loss_when_targets_nonzero():
    config_disabled = DDGLossConfig(iteration=LossIteration.ITERATION_1_BOUNDED_ONLY, tail_reweight_enabled=False)
    config_enabled = DDGLossConfig(
        iteration=LossIteration.ITERATION_1_BOUNDED_ONLY,
        tail_reweight_enabled=True,
        tail_reweight_alpha=1.0,
        tail_reweight_reference_kcal_mol=2.0,
        tail_reweight_cap=2.0,
    )
    predicted_ddg = torch.tensor([5.0, 5.0, -5.0])
    label_type_id = torch.tensor([BOUNDED_ID, BOUNDED_ID, BOUNDED_ID])
    target_ddg = torch.tensor([0.0, 2.0, -2.0])

    disabled_output = DDGRegressionLoss(config_disabled)(predicted_ddg, label_type_id, target_ddg)
    enabled_output = DDGRegressionLoss(config_enabled)(predicted_ddg, label_type_id, target_ddg)

    assert not torch.allclose(disabled_output.total_loss, enabled_output.total_loss)


def test_tail_reweight_ignores_nan_targets_on_non_bounded_rows():
    config = DDGLossConfig(iteration=LossIteration.ITERATION_1_BOUNDED_ONLY, tail_reweight_enabled=True)
    loss_fn = DDGRegressionLoss(config)
    predicted_ddg, label_type_id, target_ddg, _, _ = make_batch()

    output = loss_fn(predicted_ddg, label_type_id, target_ddg)
    assert torch.isfinite(output.total_loss)


def test_backward_pass_runs_through_total_loss():
    config = DDGLossConfig(iteration=LossIteration.ITERATION_2_WITH_HINGE)
    loss_fn = DDGRegressionLoss(config)
    predicted_ddg, label_type_id, target_ddg, bound_kcal_mol, direction = make_batch()
    predicted_ddg = predicted_ddg.clone().requires_grad_(True)

    output = loss_fn(predicted_ddg, label_type_id, target_ddg, bound_kcal_mol, direction)
    output.total_loss.backward()
    assert predicted_ddg.grad is not None
    assert torch.isfinite(predicted_ddg.grad).all()
