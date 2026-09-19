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

from pydantic import BaseModel, Field, field_validator

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
    IGFOLD = "igfold"  # live antibody structure backend; selected via hardcoded ChainRole routing in entry_embeddings.py, not through this enum
    SAPROT = "saprot"  # see data.embedding_pipeline.saprot_backend; live via USE_SAPROT_STRUCTURE below, not this enum


class SequenceBackend(str, Enum):
    ESM2 = "esm2"
    ABLANG2 = "ablang2"  # reserved for a later swap; not implemented yet


ACTIVE_SEQUENCE_BACKEND = SequenceBackend.ESM2

# Off by default (keeps existing IgFold/zero-placeholder behavior); set
# SKEMPI_USE_SAPROT_STRUCTURE=1 to route wt/mut_structure_embedding through
# SaProt instead, for every chain role including the antigen (see
# data.embedding_pipeline.entry_embeddings.apply_saprot_structure_override).
USE_SAPROT_STRUCTURE = os.environ.get("SKEMPI_USE_SAPROT_STRUCTURE", "0") == "1"

# Small checkpoint by default so pre-processing and sanity checks run on the
# CPU-only dev machine; set SKEMPI_USE_SMALL_CHECKPOINTS=0 on a GPU box
# (e.g. Colab) to use the full-size checkpoint instead.
ESM2_CHECKPOINT_CPU_DEV = "facebook/esm2_t12_35M_UR50D"
ESM2_CHECKPOINT_FULL = "facebook/esm2_t33_650M_UR50D"

# Hidden size of each ESM2 checkpoint above -- the model's sequence_embed_dim
# must track whichever checkpoint actually produced the cached embeddings.
ESM2_HIDDEN_DIM_CPU_DEV = 480
ESM2_HIDDEN_DIM_FULL = 1280

USE_SMALL_CHECKPOINTS = os.environ.get("SKEMPI_USE_SMALL_CHECKPOINTS", "1") == "1"
ACTIVE_ESM2_CHECKPOINT = ESM2_CHECKPOINT_CPU_DEV if USE_SMALL_CHECKPOINTS else ESM2_CHECKPOINT_FULL
ACTIVE_ESM2_HIDDEN_DIM = ESM2_HIDDEN_DIM_CPU_DEV if USE_SMALL_CHECKPOINTS else ESM2_HIDDEN_DIM_FULL

# SaProt checkpoints share ESM2's exact backbone sizes/hidden dims (35M/650M,
# 480/1280) since SaProt is initialized from ESM2 with an expanded
# amino-acid x 3Di-structure-token vocabulary -- see docs/future_work.md for
# why this is being prototyped (structure signal for the antigen chain,
# which IgFold cannot provide and ESMFold OOM'd on).
SAPROT_CHECKPOINT_CPU_DEV = "westlake-repl/SaProt_35M_AF2"
SAPROT_CHECKPOINT_FULL = "westlake-repl/SaProt_650M_AF2"
SAPROT_HIDDEN_DIM_CPU_DEV = 480
SAPROT_HIDDEN_DIM_FULL = 1280

ACTIVE_SAPROT_CHECKPOINT = SAPROT_CHECKPOINT_CPU_DEV if USE_SMALL_CHECKPOINTS else SAPROT_CHECKPOINT_FULL
ACTIVE_SAPROT_HIDDEN_DIM = SAPROT_HIDDEN_DIM_CPU_DEV if USE_SMALL_CHECKPOINTS else SAPROT_HIDDEN_DIM_FULL


class EmbeddingSourceMode(str, Enum):
    """Which pre-computed embedding source(s) feed the model -- the staged
    ablation switch from the Implementation Spec (start structure-only, add
    sequence back in later and compare)."""

    STRUCTURE_ONLY = "structure_only"
    SEQUENCE_ONLY = "sequence_only"
    STRUCTURE_AND_SEQUENCE = "structure_and_sequence"


ACTIVE_EMBEDDING_SOURCE_MODE = EmbeddingSourceMode.STRUCTURE_AND_SEQUENCE

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


class PointMutation(BaseModel):
    chain_id: str
    chain_role: ChainRole

    wt_residue: str
    mutant_residue: str
    residue_position: int  # raw numbering, as given by SKEMPI
    insertion_code: Optional[str] = None  # PDB insertion code, when the raw numbering alone is ambiguous
    aligned_interface_position: Optional[int] = None  # structural alignment position, used for homology dedup -- not raw numbering
    flat_residue_index: Optional[int] = None  # 0-based index into this sample's concatenated (heavy+light+antigen) cached per-residue tensors -- see data.chain_roles.ordered_chain_ids_for_sample

    interface_region: Optional[InterfaceRegion] = None
    is_alanine_scanning: bool = False  # X -> A, for the alanine-scanning distribution analysis


class MutationRecord(BaseModel):
    sample_id: str
    pdb_id: str
    complex_name: Optional[str] = None

    mutations: list[PointMutation]  # one or more point mutations measured together as a single ddG (a SKEMPI multi-mutant is one MutationRecord, not one per residue)

    label_type: LabelType
    # Raw value: present for BOUNDED entries, and best-effort for INEQ
    # entries where a numeric bound is also reported. Always kept alongside
    # ddg_bin so a regression variant can be tried later without reprocessing.
    ddg_kcal_mol: Optional[float] = None
    ddg_bin: Optional[int] = None  # populated for BOUNDED entries under the classification formulation
    ineq_direction: Optional[str] = None  # e.g. ">" or "<", for INEQ entries

    temperature_kelvin: Optional[float] = None  # from SKEMPI's Temperature column, same 298K fallback as data.affinity.parse_temperature_kelvin; not a model input yet, captured so it isn't lost

    split_membership: dict = Field(default_factory=dict)  # {SplitName.value: "train" | "val"}

    source_publication: Optional[str] = None
    notes: Optional[str] = None

    @field_validator("mutations")
    @classmethod
    def _require_at_least_one_mutation(cls, mutations: list[PointMutation]) -> list[PointMutation]:
        if not mutations:
            raise ValueError("MutationRecord.mutations must contain at least one PointMutation")
        return mutations
