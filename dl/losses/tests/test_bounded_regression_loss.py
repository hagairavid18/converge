import torch

from dl.losses.bounded_regression_loss import BoundedRegressionLoss
from dl.losses.config import DDGLossConfig


def test_zero_penalty_within_dead_zone():
    loss_fn = BoundedRegressionLoss(DDGLossConfig())
    predicted = torch.tensor([2.5, 1.0])
    target = torch.tensor([2.0, 1.0])
    loss = loss_fn(predicted, target)
    assert torch.allclose(loss, torch.zeros(2))


def test_penalty_grows_beyond_dead_zone():
    config = DDGLossConfig(regression_scale=1.0)
    loss_fn = BoundedRegressionLoss(config)
    target = torch.tensor([0.0, 0.0])

    small_error = loss_fn(torch.tensor([1.5, 0.0]), target)[0]
    large_error = loss_fn(torch.tensor([4.0, 0.0]), target)[0]

    assert small_error.item() > 0
    assert large_error.item() > small_error.item()


def test_squared_growth_matches_expected_value():
    config = DDGLossConfig(regression_penalty_growth="squared", regression_scale=1.0, regression_dead_zone_kcal_mol=1.0)
    loss_fn = BoundedRegressionLoss(config)
    predicted = torch.tensor([4.0])
    target = torch.tensor([0.0])
    loss = loss_fn(predicted, target)
    assert loss.item() == (4.0 - 1.0) ** 2


def test_linear_growth_matches_expected_value():
    config = DDGLossConfig(regression_penalty_growth="linear", regression_scale=1.0, regression_dead_zone_kcal_mol=1.0)
    loss_fn = BoundedRegressionLoss(config)
    predicted = torch.tensor([4.0])
    target = torch.tensor([0.0])
    loss = loss_fn(predicted, target)
    assert loss.item() == 3.0


def test_symmetric_around_target():
    loss_fn = BoundedRegressionLoss(DDGLossConfig())
    target = torch.tensor([0.0])
    above = loss_fn(torch.tensor([3.0]), target)
    below = loss_fn(torch.tensor([-3.0]), target)
    assert torch.allclose(above, below)


def test_gradient_flows_to_predicted_ddg():
    loss_fn = BoundedRegressionLoss(DDGLossConfig())
    predicted = torch.tensor([5.0], requires_grad=True)
    target = torch.tensor([0.0])
    loss_fn(predicted, target).sum().backward()
    assert predicted.grad is not None
    assert torch.isfinite(predicted.grad).all()
