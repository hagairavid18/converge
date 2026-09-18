"""Generic per-residue confidence-weighting layer (Implementation Spec §3,
weighting mechanism 1): multiplies a per-residue representation by a
per-residue confidence score broadcast over the feature dimension.

Which confidence tensor applies to which branch is model-specific business
logic, not part of this layer -- see `dl.models.confidence_selection`.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class ConfidenceWeighting(nn.Module):
    def forward(self, residue_features: torch.Tensor, confidence: Optional[torch.Tensor]) -> torch.Tensor:
        if confidence is None:
            return residue_features
        return residue_features * confidence.unsqueeze(-1)
