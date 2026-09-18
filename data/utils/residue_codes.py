"""Amino acid residue code conversions shared across `data/` modules."""

from __future__ import annotations

THREE_TO_ONE_LETTER_AMINO_ACID = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U", "PYL": "O",
}

VALID_ONE_LETTER_AMINO_ACIDS = frozenset(THREE_TO_ONE_LETTER_AMINO_ACID.values())


def three_letter_to_one_letter(residue_name: str) -> str | None:
    return THREE_TO_ONE_LETTER_AMINO_ACID.get(residue_name.upper())


def is_standard_amino_acid_one_letter(code: str) -> bool:
    return code.upper() in VALID_ONE_LETTER_AMINO_ACIDS
