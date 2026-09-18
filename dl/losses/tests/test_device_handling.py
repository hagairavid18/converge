import torch

from dl.losses.config import DDGLossConfig
from dl.losses.ddg_loss import DDGLoss
from shared.constants import LossIteration


def test_ddg_loss_buffers_follow_module_to_device():
    loss_fn = DDGLoss(DDGLossConfig(iteration=LossIteration.ITERATION_1_BOUNDED_ONLY)).to("cpu")
    assert loss_fn.bin_centers.device.type == "cpu"
    assert loss_fn.bounded_loss_fn.cost_matrix.device.type == "cpu"


def test_output_total_loss_matches_input_device():
    loss_fn = DDGLoss(DDGLossConfig(iteration=LossIteration.ITERATION_1_BOUNDED_ONLY)).to("cpu")
    logits = torch.randn(3, loss_fn.config.num_bins, device="cpu")
    label_type_id = torch.tensor([0, 0, 0], device="cpu")
    target_bin = torch.tensor([0, 1, 2], device="cpu")
    output = loss_fn(logits, label_type_id, target_bin)
    assert output.total_loss.device.type == "cpu"
