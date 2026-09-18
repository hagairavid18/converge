"""Optional Comet logger construction for Lightning.

Returns `None` (skip logging) when no `COMET_API_KEY` is set, so local
CPU-only sanity runs work without credentials, per CLAUDE.md.
"""

from __future__ import annotations

import os
from typing import Optional

from pytorch_lightning.loggers import Logger


def comet_api_key_available() -> bool:
    return bool(os.environ.get("COMET_API_KEY"))


def build_comet_logger(experiment_name: str, project_name: str) -> Optional[Logger]:
    if not comet_api_key_available():
        return None
    from pytorch_lightning.loggers import CometLogger

    return CometLogger(
        api_key=os.environ["COMET_API_KEY"],
        project_name=project_name,
        experiment_name=experiment_name,
    )
