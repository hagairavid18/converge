"""PDB structure loading and per-chain extraction.

Structures are fetched from RCSB by PDB id (see `data.download`) rather than
taken from the SKEMPI2_PDBs.tgz bundle: spot-checking against the bundle
during pipeline development turned up residue-numbering mismatches against
SKEMPI's own CSV mutation numbering for a large fraction of the
antibody-antigen PDBs (e.g. 1AHW chain C position 122 is LYS in both the CSV
and the official RCSB deposition, but LEU in the bundled structure; 1JRH
chain I position 98 is missing from the bundled structure entirely). RCSB's
depositions matched the CSV numbering in every case checked. See the
pipeline report for details -- this is a deviation from "download SKEMPI 2.0
from its public source" worth flagging, since it means per-PDB downloads
from RCSB rather than the one bundled tarball.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.Chain import Chain
from Bio.PDB.Structure import Structure

from shared.constants import SKEMPI_RAW_STRUCTURES_DIR
from data.utils.residue_codes import three_letter_to_one_letter

_PDB_PARSER = PDBParser(QUIET=True)


def skempi_structure_path(pdb_id: str):
    return SKEMPI_RAW_STRUCTURES_DIR / f"{pdb_id.upper()}.pdb"


@lru_cache(maxsize=64)
def load_structure(pdb_id: str) -> Structure:
    path = skempi_structure_path(pdb_id)
    if not path.exists():
        raise FileNotFoundError(f"No structure file for PDB id {pdb_id!r} at {path}")
    return _PDB_PARSER.get_structure(pdb_id.upper(), str(path))


def get_chain(structure: Structure, chain_id: str) -> Chain:
    model = next(structure.get_models())
    if chain_id not in model:
        raise KeyError(f"Chain {chain_id!r} not found in structure {structure.id}")
    return model[chain_id]


def standard_residues(chain: Chain) -> list:
    return [residue for residue in chain if residue.id[0] == " "]


def chain_residue_sequence(chain: Chain) -> list[tuple]:
    residues = standard_residues(chain)
    sequence_and_ids = []
    for residue in residues:
        one_letter = three_letter_to_one_letter(residue.get_resname())
        if one_letter is None:
            continue
        _, raw_position, insertion_code = residue.id
        sequence_and_ids.append((raw_position, insertion_code.strip(), one_letter, residue))
    return sequence_and_ids


def chain_sequence(chain: Chain) -> str:
    return "".join(one_letter for _, _, one_letter, _ in chain_residue_sequence(chain))


def chain_residue_position_index(chain: Chain) -> dict[tuple[int, str], int]:
    return {
        (raw_position, insertion_code): index
        for index, (raw_position, insertion_code, _, _) in enumerate(chain_residue_sequence(chain))
    }


def chain_ca_coordinates(chain: Chain) -> np.ndarray:
    coordinates = []
    for _, _, _, residue in chain_residue_sequence(chain):
        if "CA" in residue:
            coordinates.append(residue["CA"].get_coord())
        else:
            coordinates.append(np.full(3, np.nan, dtype=np.float32))
    return np.asarray(coordinates, dtype=np.float32)


def chain_ids_in_structure(structure: Structure) -> list[str]:
    model = next(structure.get_models())
    return [chain.id for chain in model]


def residue_one_letter_at_position(chain: Chain, raw_position: int, insertion_code: str = "") -> str | None:
    for position, code, one_letter, _ in chain_residue_sequence(chain):
        if position == raw_position and code.upper() == insertion_code.upper():
            return one_letter
    return None


def residue_identity_matches(
    chain: Chain, raw_position: int, expected_one_letter: str, insertion_code: str = ""
) -> bool:
    """Guards against a mutation position from the CSV not lining up with the
    residue numbering actually present in the fetched structure for a given
    entry. Records that fail this check are excluded rather than silently
    mis-indexed; see the pipeline report for observed mismatch sources.
    """
    actual = residue_one_letter_at_position(chain, raw_position, insertion_code)
    return actual is not None and actual.upper() == expected_one_letter.upper()
