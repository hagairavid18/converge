"""Single source of truth for which `torch.device` the embedding backends
run on. CLAUDE.md: never hardcode CUDA, but also never leave a module/tensor
implicitly on CPU when a GPU is actually available -- every backend must
route through `resolve_device()` and move both its model and its inputs
there explicitly, so the exact same code uses a Colab GPU later with zero
changes beyond what this function resolves to.
"""

from __future__ import annotations

import torch


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
