"""Antibody-vs-antigen side detection and heavy/light chain assignment for
SKEMPI antibody-antigen entries.

SKEMPI's `Protein 1` / `Protein 2` free-text names are not consistently
ordered: either can be the antibody. This module resolves, for a given row,
which physical PDB chains are heavy, light, and antigen.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

from data.utils.constants import (
    ANTIBODY_NAME_KEYWORDS,
    ANTIBODY_NAME_OVERRIDES,
    CHAIN_CLASSIFICATION_VARIABLE_DOMAIN_MAX_LENGTH,
    DEFAULT_ANTIBODY_SIDE_IS_PROTEIN_1,
    HEAVY_CHAIN_FR4_MOTIF,
    HEAVY_CHAIN_LETTER,
    LIGHT_CHAIN_FR4_MOTIF,
    LIGHT_CHAIN_LETTER,
    PDB_FIELD_SEPARATOR,
)
from shared.constants import ChainRole

_HEAVY_FR4_REGEX = re.compile(HEAVY_CHAIN_FR4_MOTIF)
_LIGHT_FR4_REGEX = re.compile(LIGHT_CHAIN_FR4_MOTIF)


class PdbFieldSpec(BaseModel):
    pdb_id: str
    protein_1_chains: str
    protein_2_chains: str


def parse_pdb_field(pdb_field: str) -> PdbFieldSpec:
    pdb_id, protein_1_chains, protein_2_chains = pdb_field.split(PDB_FIELD_SEPARATOR)
    return PdbFieldSpec(pdb_id=pdb_id, protein_1_chains=protein_1_chains, protein_2_chains=protein_2_chains)


def is_antibody_protein_name(name: str) -> bool:
    """True if `name` (a SKEMPI `Protein 1`/`Protein 2` value) names an
    antibody. Keyword-based, plus a small curated override set
    (`ANTIBODY_NAME_OVERRIDES`) for the handful of names in the current
    SKEMPI antibody-antigen subset that are trade names or bare clone names
    carrying no antibody-indicating keyword (e.g. "Herceptin", "HyHEL-10").
    """
    if name in ANTIBODY_NAME_OVERRIDES:
        return True
    lowered = name.lower()
    return any(keyword in lowered for keyword in ANTIBODY_NAME_KEYWORDS)


def antibody_side_is_protein_1(protein_1_name: str, protein_2_name: str) -> bool:
    protein_1_is_antibody = is_antibody_protein_name(protein_1_name)
    protein_2_is_antibody = is_antibody_protein_name(protein_2_name)
    if protein_1_is_antibody != protein_2_is_antibody:
        return protein_1_is_antibody
    return DEFAULT_ANTIBODY_SIDE_IS_PROTEIN_1


def antibody_and_antigen_chain_letters(
    pdb_field_spec: PdbFieldSpec, protein_1_name: str, protein_2_name: str
) -> tuple[str, str]:
    if antibody_side_is_protein_1(protein_1_name, protein_2_name):
        return pdb_field_spec.protein_1_chains, pdb_field_spec.protein_2_chains
    return pdb_field_spec.protein_2_chains, pdb_field_spec.protein_1_chains


def classify_heavy_light_by_chain_letter(antibody_chain_letters: str) -> dict[str, ChainRole]:
    roles: dict[str, ChainRole] = {}
    for letter in antibody_chain_letters:
        if letter.upper() == HEAVY_CHAIN_LETTER:
            roles[letter] = ChainRole.HEAVY
        elif letter.upper() == LIGHT_CHAIN_LETTER:
            roles[letter] = ChainRole.LIGHT
    return roles


def classify_heavy_light_by_sequence(chain_letter_to_sequence: dict[str, str]) -> dict[str, ChainRole]:
    """Best-effort heavy/light assignment from sequence alone, for antibody
    chains whose PDB chain letters are not literally "H"/"L". Looks for the
    IMGT J-segment framework-4 signature marking the end of the variable
    domain (heavy: "WGQGT..."; light: "FGQGT..." kappa or lambda). Searched
    within each chain's first `CHAIN_CLASSIFICATION_VARIABLE_DOMAIN_MAX_LENGTH`
    residues rather than at the literal chain C-terminus, since a Fab chain's
    variable domain (where FR4 sits) ends well before the chain does -- the
    constant domain follows it. Falls back to "the longer chain is heavy"
    when neither chain matches either motif. This is a heuristic, not
    ANARCI/IMGT numbering.
    """
    variable_domain_regions = {
        letter: sequence[:CHAIN_CLASSIFICATION_VARIABLE_DOMAIN_MAX_LENGTH]
        for letter, sequence in chain_letter_to_sequence.items()
    }
    heavy_matches = {letter for letter, region in variable_domain_regions.items() if _HEAVY_FR4_REGEX.search(region)}
    light_matches = {letter for letter, region in variable_domain_regions.items() if _LIGHT_FR4_REGEX.search(region)}

    roles: dict[str, ChainRole] = {}
    for letter in chain_letter_to_sequence:
        if letter in heavy_matches and letter not in light_matches:
            roles[letter] = ChainRole.HEAVY
        elif letter in light_matches and letter not in heavy_matches:
            roles[letter] = ChainRole.LIGHT

    unresolved = [letter for letter in chain_letter_to_sequence if letter not in roles]
    if len(unresolved) == 2:
        longer, shorter = sorted(unresolved, key=lambda letter: len(chain_letter_to_sequence[letter]), reverse=True)
        roles[longer] = ChainRole.HEAVY
        roles[shorter] = ChainRole.LIGHT
    elif len(unresolved) == 1:
        roles[unresolved[0]] = ChainRole.HEAVY

    return roles


def ordered_chain_ids_for_sample(chain_map: dict[str, str]) -> list[str]:
    """Canonical heavy-then-light-then-antigen chain order used both when
    concatenating per-chain arrays into one per-sample cached tensor
    (`data.embedding_pipeline.entry_embeddings`) and when translating a
    mutation's chain-local residue position into a flat index into that
    tensor (`data.flat_indexing`) -- the two must stay in lockstep.
    """
    ordered_roles = (ChainRole.HEAVY, ChainRole.LIGHT, ChainRole.ANTIGEN)
    chain_ids: list[str] = []
    for role in ordered_roles:
        chain_ids.extend(chain_map.get(role.value, ""))
    return chain_ids


def resolve_antibody_chain_roles(
    antibody_chain_letters: str, chain_letter_to_sequence: dict[str, str]
) -> dict[str, ChainRole]:
    roles = classify_heavy_light_by_chain_letter(antibody_chain_letters)
    unresolved_letters = [letter for letter in antibody_chain_letters if letter not in roles]
    if unresolved_letters:
        sequences_for_unresolved = {
            letter: chain_letter_to_sequence[letter]
            for letter in unresolved_letters
            if letter in chain_letter_to_sequence
        }
        roles.update(classify_heavy_light_by_sequence(sequences_for_unresolved))
    return roles
