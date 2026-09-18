import torch

from dl.losses.bounded_classification_loss import (
    BinDistanceWeightedLoss,
    build_bin_distance_cost_matrix,
)
from dl.losses.config import DDGLossConfig


def test_cost_matrix_zero_on_diagonal():
    cost = build_bin_distance_cost_matrix(num_bins=10, noise_floor_agreement_fraction=0.84)
    assert torch.allclose(torch.diag(cost), torch.zeros(10))


def test_adjacent_cost_much_smaller_than_distant_cost():
    cost = build_bin_distance_cost_matrix(num_bins=10, noise_floor_agreement_fraction=0.84)
    adjacent = cost[5, 6].item()
    two_away = cost[5, 7].item()
    far_away = cost[5, 9].item()
    assert 0 < adjacent < two_away < far_away
    assert abs(adjacent - (1.0 - 0.84)) < 1e-5
    assert far_away > adjacent * 10


def test_loss_lower_when_probability_mass_on_true_bin():
    config = DDGLossConfig()
    loss_fn = BinDistanceWeightedLoss(config)
    target_bin = torch.tensor([5])

    confident_correct_logits = torch.zeros(1, config.num_bins)
    confident_correct_logits[0, 5] = 20.0

    confident_wrong_logits = torch.zeros(1, config.num_bins)
    confident_wrong_logits[0, 9] = 20.0

    correct_loss = loss_fn(confident_correct_logits, target_bin)
    wrong_loss = loss_fn(confident_wrong_logits, target_bin)
    assert correct_loss.item() < wrong_loss.item()


def test_adjacent_miss_costs_less_than_distant_miss_in_loss():
    config = DDGLossConfig()
    loss_fn = BinDistanceWeightedLoss(config)
    target_bin = torch.tensor([5])

    adjacent_miss_logits = torch.zeros(1, config.num_bins)
    adjacent_miss_logits[0, 6] = 20.0

    distant_miss_logits = torch.zeros(1, config.num_bins)
    distant_miss_logits[0, 9] = 20.0

    adjacent_loss = loss_fn(adjacent_miss_logits, target_bin)
    distant_loss = loss_fn(distant_miss_logits, target_bin)
    assert adjacent_loss.item() < distant_loss.item()


def test_gradient_flows_to_logits():
    config = DDGLossConfig()
    loss_fn = BinDistanceWeightedLoss(config)
    logits = torch.randn(4, config.num_bins, requires_grad=True)
    target_bin = torch.tensor([0, 3, 5, 9])
    loss = loss_fn(logits, target_bin).sum()
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
