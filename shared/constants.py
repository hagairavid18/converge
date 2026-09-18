"""Shared constants and schemas for the SKEMPI antibody-antigen ddG predictor.

This is the single cross-cutting contract between the pre-processing, model/training,
losses/metrics, and data-exploration workstreams. Every path, enum, and schema that more
than one workstream depends on lives here. Nothing structural (label schema, split names,
metadata fields, embedding backend selection) should be redefined elsewhere -- import it
from here instead.

Paths are resolved relative to an overridable root so this file behaves the same on the
local CPU-only dev machine and on a cloud GPU environment (e.g. Colab) -- never hardcode a
local absolute path.
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(os.environ.get("SKEMPI_PROJECT_ROOT", Path(__file__).resolve().parent.parent))
DATA_ROOT = Path(os.environ.get("SKEMPI_DATA_ROOT", str(PROJECT_ROOT / "data")))

RAW_DATA_DIR = DATA_ROOT / "raw"
PROCESSED_DATA_DIR = DATA_ROOT / "processed"
EMBEDDING_CACHE_DIR = DATA_ROOT / "embeddings"
SPLITS_DIR = DATA_ROOT / "splits"

# ---------------------------------------------------------------------------
# SKEMPI 2.0 source
# ---------------------------------------------------------------------------

SKEMPI_DOWNLOAD_URL = "https://life.bsc.es/pid/skempi2/database/download/skempi_v2.csv"
SKEMPI_STRUCTURES_URL = "https://life.bsc.es/pid/skempi2/database/download/SKEMPI2_PDBs.tgz"
SKEMPI_RAW_CSV = RAW_DATA_DIR / "skempi_v2.csv"
SKEMPI_RAW_STRUCTURES_DIR = RAW_DATA_DIR / "pdbs"

# SKEMPI's own column marking complex/hold-out type; used to filter to the
# antibody-antigen subset. Verify against the actual downloaded header before
# relying on this in the pre-processing pipeline -- column names have drifted
# across SKEMPI 2.0 CSV releases.
ANTIBODY_ANTIGEN_HALLMARK_COLUMN = "Hold_out_type"
ANTIBODY_ANTIGEN_HALLMARK_VALUE = "AB/AG"

# ---------------------------------------------------------------------------
# Splits -- report all downstream metrics separately per split, never pooled.
# ---------------------------------------------------------------------------


class SplitName(str, Enum):
    HELD_OUT_PDB = "held_out_pdb"
    SAME_PDB_ALLOWED = "same_pdb_allowed"


SPLIT_SUBSETS = ("train", "val")

SPLIT_FILES = {
    split: {subset: SPLITS_DIR / f"{split.value}_{subset}.csv" for subset in SPLIT_SUBSETS}
    for split in SplitName
}

# ---------------------------------------------------------------------------
# Label schema
# ---------------------------------------------------------------------------


class LabelType(str, Enum):
    BOUNDED = "bounded"
    INEQ = "ineq"
    NB = "n.b."


# Empirical noise floor: 84% of independently-remeasured same-mutation pairs
# agree within this margin. This sizes the no-penalty margin (regression) /
# near-zero adjacent-bin cost (classification) and the bin width below.
NOISE_FLOOR_KCAL_MOL = 1.0
NOISE_FLOOR_AGREEMENT_FRACTION = 0.84

# Locked-in starting formulation for the bounded-ddG head: classification
# (bin-distance-weighted), not regression. Raw ddG is retained per-sample
# alongside the assigned bin so a regression variant can be tried later
# without reprocessing. Bin width matches the noise floor, so adjacent bins
# are exactly the ones the noise floor actually confuses.
DDG_BIN_EDGES_KCAL_MOL: list = [
    -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0,
]
DDG_NUM_BINS = len(DDG_BIN_EDGES_KCAL_MOL) - 1


def ddg_to_bin(ddg: float) -> int:
    """Map a raw ddG value (kcal/mol) to a classification bin index, clipping
    out-of-range values to the outermost bin."""
    if ddg < DDG_BIN_EDGES_KCAL_MOL[0]:
        return 0
    if ddg >= DDG_BIN_EDGES_KCAL_MOL[-1]:
        return DDG_NUM_BINS - 1
    for i in range(DDG_NUM_BINS):
        if DDG_BIN_EDGES_KCAL_MOL[i] <= ddg < DDG_BIN_EDGES_KCAL_MOL[i + 1]:
            return i
    raise AssertionError("unreachable")  # pragma: no cover


class InterfaceRegion(str, Enum):
    SUPPORT = "support"
    CORE = "core"
    RIM = "rim"
    SURFACE = "surface"
    INTERIOR = "interior"


class ChainRole(str, Enum):
    HEAVY = "heavy"
    LIGHT = "light"
    ANTIGEN = "antigen"


# ---------------------------------------------------------------------------
# Embedding backends -- pluggable so a backend can be swapped, or the
# structure-only vs. structure+sequence ablation toggled, without touching
# model or pre-processing code beyond reading these constants.
# ---------------------------------------------------------------------------


class StructureBackend(str, Enum):
    ESMFOLD = "esmfold"
    IGFOLD = "igfold"  # reserved for a later swap; not implemented yet


class SequenceBackend(str, Enum):
    ESM2 = "esm2"
    ABLANG2 = "ablang2"  # reserved for a later swap; not implemented yet


# Locked-in starting choice: the simplest, most general pair (ESMFold +
# ESM-2), applied uniformly to every chain (heavy, light, antigen) rather
# than antibody-specialized models. Change these two lines to swap backends.
ACTIVE_STRUCTURE_BACKEND = StructureBackend.ESMFOLD
ACTIVE_SEQUENCE_BACKEND = SequenceBackend.ESM2

# Small checkpoint by default so pre-processing and sanity checks run on the
# CPU-only dev machine; set SKEMPI_USE_SMALL_CHECKPOINTS=0 on a GPU box
# (e.g. Colab) to use the full-size checkpoint instead.
ESM2_CHECKPOINT_CPU_DEV = "facebook/esm2_t12_35M_UR50D"
ESM2_CHECKPOINT_FULL = "facebook/esm2_t33_650M_UR50D"
ESMFOLD_CHECKPOINT = "facebook/esmfold_v1"

USE_SMALL_CHECKPOINTS = os.environ.get("SKEMPI_USE_SMALL_CHECKPOINTS", "1") == "1"
ACTIVE_ESM2_CHECKPOINT = ESM2_CHECKPOINT_CPU_DEV if USE_SMALL_CHECKPOINTS else ESM2_CHECKPOINT_FULL


class EmbeddingSourceMode(str, Enum):
    """Which pre-computed embedding source(s) feed the model -- the staged
    ablation switch from the Implementation Spec (start structure-only, add
    sequence back in later and compare)."""

    STRUCTURE_ONLY = "structure_only"
    SEQUENCE_ONLY = "sequence_only"
    STRUCTURE_AND_SEQUENCE = "structure_and_sequence"


ACTIVE_EMBEDDING_SOURCE_MODE = EmbeddingSourceMode.STRUCTURE_ONLY

# ---------------------------------------------------------------------------
# Loss / training staging
# ---------------------------------------------------------------------------


class LossIteration(str, Enum):
    # Bounded-only, ineq/n.b. masked out of loss and batches.
    ITERATION_1_BOUNDED_ONLY = "iteration_1_bounded_only"
    # Adds the hinge term and includes ineq/n.b. entries.
    ITERATION_2_WITH_HINGE = "iteration_2_with_hinge"


ACTIVE_LOSS_ITERATION = LossIteration.ITERATION_1_BOUNDED_ONLY

RANDOM_SEED = 42

# Reweight bounded vs. ineq/n.b. terms if their batch counts are highly
# imbalanced. Only consulted once ACTIVE_LOSS_ITERATION includes both terms.
BATCH_IMBALANCE_REWEIGHT = True

# ---------------------------------------------------------------------------
# Per-sample metadata contract
#
# This is the record pre-processing writes and every downstream workstream
# reads. Keep it as the single source of truth for what a "sample" is --
# extend it here, not with ad hoc columns elsewhere.
# ---------------------------------------------------------------------------


class MutationRecord(BaseModel):
    sample_id: str
    pdb_id: str
    complex_name: Optional[str] = None

    chain_id: str
    chain_role: ChainRole

    wt_residue: str
    mutant_residue: str
    residue_position: int  # raw numbering, as given by SKEMPI
    aligned_interface_position: Optional[int] = None  # structural alignment position, used for homology dedup -- not raw numbering

    label_type: LabelType
    # Raw value: present for BOUNDED entries, and best-effort for INEQ
    # entries where a numeric bound is also reported. Always kept alongside
    # ddg_bin so a regression variant can be tried later without reprocessing.
    ddg_kcal_mol: Optional[float] = None
    ddg_bin: Optional[int] = None  # populated for BOUNDED entries under the classification formulation
    ineq_direction: Optional[str] = None  # e.g. ">" or "<", for INEQ entries

    interface_region: Optional[InterfaceRegion] = None
    is_alanine_scanning: bool = False  # X -> A, for the alanine-scanning distribution analysis

    split_membership: dict = Field(default_factory=dict)  # {SplitName.value: "train" | "val"}

    source_publication: Optional[str] = None
    notes: Optional[str] = None
