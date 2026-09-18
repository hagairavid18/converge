"""Parsing of SKEMPI's per-entry mutation list columns into structured tokens.

A single SKEMPI row can carry multiple point mutations at once (e.g.
"TC121A,KC122A" for a double mutant); `Mutation(s)_PDB`,
`Mutation(s)_cleaned`, and `iMutation_Location(s)` are parallel comma-joined
lists, one element per point mutation, in the same order.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

from data.utils.constants import (
    INTERFACE_REGION_CODE_TO_ENUM_VALUE,
    MUTATION_LIST_SEPARATOR,
    MUTATION_TOKEN_PATTERN,
)
from shared.constants import InterfaceRegion

_MUTATION_TOKEN_REGEX = re.compile(MUTATION_TOKEN_PATTERN)


class MutationToken(BaseModel):
    wt_residue: str
    chain_id: str
    residue_position: int
    insertion_code: str = ""
    mutant_residue: str
    aligned_interface_position: int | None = None
    interface_region: InterfaceRegion | None = None


def parse_mutation_string(token: str) -> tuple[str, str, int, str, str]:
    match = _MUTATION_TOKEN_REGEX.match(token.strip())
    if match is None:
        raise ValueError(f"Unrecognized mutation token format: {token!r}")
    wt_residue, chain_id, position, insertion_code, mutant_residue = match.groups()
    return wt_residue.upper(), chain_id, int(position), insertion_code, mutant_residue.upper()


def interface_region_from_code(code: str) -> InterfaceRegion | None:
    enum_value = INTERFACE_REGION_CODE_TO_ENUM_VALUE.get(code.strip().upper())
    return InterfaceRegion(enum_value) if enum_value else None


def split_mutation_list(raw_value: str) -> list[str]:
    return [item.strip() for item in raw_value.split(MUTATION_LIST_SEPARATOR) if item.strip()]


def parse_mutation_tokens(
    mutations_pdb_numbering: str,
    mutations_cleaned_numbering: str,
    interface_locations: str,
) -> list[MutationToken]:
    pdb_tokens = split_mutation_list(mutations_pdb_numbering)
    cleaned_tokens = split_mutation_list(mutations_cleaned_numbering)
    location_tokens = split_mutation_list(interface_locations)

    tokens: list[MutationToken] = []
    for i, pdb_token in enumerate(pdb_tokens):
        wt_residue, chain_id, position, insertion_code, mutant_residue = parse_mutation_string(pdb_token)

        aligned_position = None
        if i < len(cleaned_tokens):
            _, _, aligned_position, _, _ = parse_mutation_string(cleaned_tokens[i])

        interface_region = None
        if i < len(location_tokens):
            interface_region = interface_region_from_code(location_tokens[i])

        tokens.append(
            MutationToken(
                wt_residue=wt_residue,
                chain_id=chain_id,
                residue_position=position,
                insertion_code=insertion_code,
                mutant_residue=mutant_residue,
                aligned_interface_position=aligned_position,
                interface_region=interface_region,
            )
        )
    return tokens
