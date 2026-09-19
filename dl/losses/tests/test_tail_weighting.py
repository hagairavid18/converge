import torch

from dl.losses.tail_weighting import tail_weight


def test_weight_is_one_at_zero_ddg():
    weight = tail_weight(torch.tensor([0.0]), alpha=1.0, reference_kcal_mol=2.0, cap=2.0)
    assert torch.allclose(weight, torch.tensor([1.0]))


def test_weight_scales_linearly_below_cap():
    target_ddg = torch.tensor([1.0, 2.0])
    weight = tail_weight(target_ddg, alpha=1.0, reference_kcal_mol=2.0, cap=2.0)
    assert torch.allclose(weight, torch.tensor([1.5, 2.0]))


def test_weight_saturates_at_cap():
    target_ddg = torch.tensor([4.0, 10.0])
    weight = tail_weight(target_ddg, alpha=1.0, reference_kcal_mol=2.0, cap=2.0)
    assert torch.allclose(weight, torch.tensor([3.0, 3.0]))


def test_weight_is_symmetric_in_sign():
    target_ddg = torch.tensor([-3.0, 3.0])
    weight = tail_weight(target_ddg, alpha=0.5, reference_kcal_mol=2.0, cap=5.0)
    assert torch.allclose(weight[0], weight[1])


def test_alpha_scales_the_up_weighting_amount():
    target_ddg = torch.tensor([2.0])
    weight_small_alpha = tail_weight(target_ddg, alpha=0.5, reference_kcal_mol=2.0, cap=2.0)
    weight_large_alpha = tail_weight(target_ddg, alpha=2.0, reference_kcal_mol=2.0, cap=2.0)
    assert weight_large_alpha.item() > weight_small_alpha.item()
