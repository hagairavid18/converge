import torch

from dl.metrics.analyses import alanine_scanning_vs_other, ddg_by_interface_region
from dl.utils.region_codes import REGION_TO_ID, UNKNOWN_REGION_ID
from shared.constants import InterfaceRegion


def test_ddg_by_interface_region_groups_correctly():
    interface_region_id = torch.tensor(
        [REGION_TO_ID[InterfaceRegion.CORE], REGION_TO_ID[InterfaceRegion.CORE], REGION_TO_ID[InterfaceRegion.SURFACE]]
    )
    ddg = torch.tensor([2.0, 4.0, 0.5])
    result = ddg_by_interface_region(interface_region_id, ddg)
    assert result["core"]["n"] == 2
    assert result["core"]["mean"] == 3.0
    assert result["surface"]["n"] == 1


def test_ddg_by_interface_region_unknown_bucket():
    interface_region_id = torch.tensor([UNKNOWN_REGION_ID, REGION_TO_ID[InterfaceRegion.RIM]])
    ddg = torch.tensor([1.0, 2.0])
    result = ddg_by_interface_region(interface_region_id, ddg)
    assert "unknown" in result
    assert result["unknown"]["n"] == 1


def test_alanine_scanning_vs_other_separates_groups():
    is_alanine = torch.tensor([True, True, False, False])
    ddg = torch.tensor([1.0, 1.5, 3.0, 3.5])
    result = alanine_scanning_vs_other(is_alanine, ddg)
    assert result["alanine_scanning"]["n"] == 2
    assert result["other_substitutions"]["n"] == 2
    assert result["alanine_scanning"]["mean"] == 1.25
    assert result["other_substitutions"]["mean"] == 3.25
    assert result["mannwhitney"]["p_value"] is not None


def test_alanine_scanning_vs_other_drops_nan_values():
    is_alanine = torch.tensor([True, True, False])
    ddg = torch.tensor([1.0, float("nan"), 3.0])
    result = alanine_scanning_vs_other(is_alanine, ddg)
    assert result["alanine_scanning"]["n"] == 1
    assert result["alanine_scanning"]["mean"] == 1.0


def test_alanine_scanning_vs_other_empty_group_returns_nan_test():
    is_alanine = torch.tensor([True, True])
    ddg = torch.tensor([1.0, 2.0])
    result = alanine_scanning_vs_other(is_alanine, ddg)
    assert result["other_substitutions"]["n"] == 0
    import math

    assert math.isnan(result["mannwhitney"]["p_value"])
