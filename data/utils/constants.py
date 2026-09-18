"""Domain-internal constants for the `data/` pre-processing pipeline.

Anything hardcoded here is only ever consumed inside `data/`. A value only
belongs in `shared/constants.py` instead if the modeling side also needs
that exact value (see CLAUDE.md).
"""

from __future__ import annotations

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

RCSB_PDB_DOWNLOAD_URL_TEMPLATE = "https://files.rcsb.org/download/{pdb_id}.pdb"
HTTP_DOWNLOAD_CHUNK_BYTES = 1 << 16
HTTP_DOWNLOAD_TIMEOUT_SECONDS = 60

ESM2_MAX_RECOMMENDED_SEQUENCE_LENGTH_FOR_CPU_DEV = 1024
