"""Domain-internal constants for the `data/` pre-processing pipeline.

Anything hardcoded here is only ever consumed inside `data/`. A value only
belongs in `shared/constants.py` instead if the modeling side also needs
that exact value (see CLAUDE.md).
"""

from __future__ import annotations

import os
from pathlib import Path

from shared.constants import PROJECT_ROOT

SKEMPI_CSV_DELIMITER = ";"

COLUMN_PDB = "#Pdb"
COLUMN_MUTATIONS_PDB_NUMBERING = "Mutation(s)_PDB"
COLUMN_MUTATIONS_CLEANED_NUMBERING = "Mutation(s)_cleaned"
COLUMN_INTERFACE_LOCATIONS = "iMutation_Location(s)"
COLUMN_HOLD_OUT_TYPE = "Hold_out_type"
COLUMN_PROTEIN_1 = "Protein 1"
COLUMN_PROTEIN_2 = "Protein 2"
COLUMN_AFFINITY_MUT = "Affinity_mut (M)"
COLUMN_AFFINITY_WT = "Affinity_wt (M)"
COLUMN_TEMPERATURE = "Temperature"
COLUMN_REFERENCE = "Reference"
COLUMN_NOTES = "Notes"

MUTATION_LIST_SEPARATOR = ","
HOLD_OUT_TYPE_SEPARATOR = ","

MUTATION_TOKEN_PATTERN = r"^([A-Za-z])([A-Za-z])(-?\d+)([A-Za-z]?)([A-Za-z])$"

PDB_FIELD_SEPARATOR = "_"

ANTIBODY_NAME_KEYWORDS = (
    "fab",
    "fv",
    "igg",
    "iga",
    "igm",
    "mab",
    "scfv",
    "immunoglobulin",
    "nanobody",
    "vhh",
    "antibody",
)

ANTIBODY_NAME_OVERRIDES = frozenset({
    "Herceptin",
    "HyHEL-10",
    "VRC-PG04",
})

DEFAULT_ANTIBODY_SIDE_IS_PROTEIN_1 = True

HEAVY_CHAIN_LETTER = "H"
LIGHT_CHAIN_LETTER = "L"

HEAVY_CHAIN_FR4_MOTIF = "WG.GT"
LIGHT_CHAIN_FR4_MOTIF = "FG.GT"
CHAIN_CLASSIFICATION_VARIABLE_DOMAIN_MAX_LENGTH = 130

GAS_CONSTANT_KCAL_PER_MOL_K = 1.987204e-3

NO_BINDING_TOKENS = frozenset({"n.b", "n.b.", "nb"})
INEQUALITY_PREFIXES = (">", "<")

TEMPERATURE_ASSUMED_SUFFIX = "(assumed)"
DEFAULT_TEMPERATURE_KELVIN = 298.0

INTERFACE_REGION_CODE_TO_ENUM_VALUE = {
    "COR": "core",
    "RIM": "rim",
    "SUP": "support",
    "SUR": "surface",
    "INT": "interior",
}

DEFAULT_VAL_FRACTION = 0.2

# same_pdb_allowed's two-tier split: a subset of complexes is accumulated
# (by sample count) into an entirely held-out bucket until it reaches this
# fraction of the total sample count, then the remaining ("shared") complexes'
# samples are individually split at SHARED_SAMPLE_VAL_FRACTION. Against the
# current 1057-record dataset (43 complexes, RANDOM_SEED=42) this lands at
# 115 held-out-complex + 87 shared-sample val rows (202/1057, 19.1% total).
SAME_PDB_ALLOWED_NEW_COMPLEX_FRACTION = 0.1
SAME_PDB_ALLOWED_SHARED_SAMPLE_FRACTION = 0.1

RCSB_PDB_DOWNLOAD_URL_TEMPLATE = "https://files.rcsb.org/download/{pdb_id}.pdb"
HTTP_DOWNLOAD_CHUNK_BYTES = 1 << 16
HTTP_DOWNLOAD_TIMEOUT_SECONDS = 60

ESM2_MAX_RECOMMENDED_SEQUENCE_LENGTH_FOR_CPU_DEV = 1024

# compute_sequence_pseudo_log_likelihood scores every residue position by
# masking it and running a forward pass -- naively batching all positions
# for a chain of length L at once means a batch of size L through a model
# whose own per-layer cost is also ~O(L) in sequence length, so peak memory
# scales with L^2. That OOM'd real training (tried to allocate 6.46 GiB with
# ~27 GiB already in use scoring one ~225-residue chain on a 32GB V100).
# Chunking the masked positions keeps peak memory roughly flat in L.
ESM2_PSEUDO_LL_MASK_BATCH_SIZE = 32

# foldseek is a standalone C++ binary (not a pip package): download the
# static build into .tools/ (see docs/future_work.md for the setup command);
# override with SKEMPI_FOLDSEEK_BIN for a system-wide/conda install instead.
FOLDSEEK_BINARY_PATH = Path(os.environ.get("SKEMPI_FOLDSEEK_BIN", str(PROJECT_ROOT / ".tools" / "foldseek" / "bin" / "foldseek")))

# SaProt's own convention for "sequence known, structure unknown at this
# residue" -- combined with the amino acid letter (e.g. "A#"), one of the
# model's trained vocabulary tokens, not an arbitrary placeholder. Used to
# mask the mutated residue's 3Di token in the mutant branch, since SKEMPI
# never provides a separately-solved mutant structure (see
# entry_embeddings.py's module docstring on the WT-geometry-only assumption).
SAPROT_STRUCTURE_UNKNOWN_TOKEN = "#"
