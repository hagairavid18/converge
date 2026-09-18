import torch

from dl.losses.config import DDGLossConfig
from dl.losses.hinge_loss import HingeLoss


def test_zero_penalty_on_correct_side():
    hinge = HingeLoss(DDGLossConfig())
    predicted_ddg = torch.tensor([3.0, -3.0])
    bound = torch.tensor([2.0, -2.0])
    direction = torch.tensor([1.0, -1.0])
    loss = hinge(predicted_ddg, bound, direction)
    assert torch.allclose(loss, torch.zeros(2))


def test_penalty_grows_linearly_past_bound():
    config = DDGLossConfig()
    hinge = HingeLoss(config)
    bound = torch.tensor([2.0, 2.0])
    direction = torch.tensor([1.0, 1.0])

    small_violation = hinge(torch.tensor([1.5, 2.0]), bound, direction)[0]
    large_violation = hinge(torch.tensor([0.0, 2.0]), bound, direction)[0]

    assert small_violation.item() > 0
    assert large_violation.item() > small_violation.item()
    assert large_violation.item() == config.hinge_scale * 2.0


def test_direction_sign_is_respected():
    hinge = HingeLoss(DDGLossConfig())
    bound = torch.tensor([2.0])
    below_bound_pred = torch.tensor([1.0])

    greater_than_direction_loss = hinge(below_bound_pred, bound, torch.tensor([1.0]))
    less_than_direction_loss = hinge(below_bound_pred, bound, torch.tensor([-1.0]))

    assert greater_than_direction_loss.item() > 0
    assert less_than_direction_loss.item() == 0
