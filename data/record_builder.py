"""Assembly of `MutationRecord`s from a single raw SKEMPI antibody-antigen row."""

from __future__ import annotations

from pydantic import BaseModel

from data.affinity import compute_ddg_label, parse_temperature_kelvin
from data.chain_roles import (
    antibody_and_antigen_chain_letters,
    antibody_side_is_protein_1,
    parse_pdb_field,
    resolve_antibody_chain_roles,
)
from data.mutation_tokens import MutationToken, parse_mutation_tokens
from data.structures import get_chain, chain_sequence, load_structure, residue_identity_matches
from data.utils.constants import (
    COLUMN_AFFINITY_MUT,
    COLUMN_AFFINITY_WT,
    COLUMN_MUTATIONS_CLEANED_NUMBERING,
    COLUMN_MUTATIONS_PDB_NUMBERING,
    COLUMN_NOTES,
    COLUMN_PDB,
    COLUMN_PROTEIN_1,
    COLUMN_PROTEIN_2,
    COLUMN_REFERENCE,
    COLUMN_TEMPERATURE,
    COLUMN_INTERFACE_LOCATIONS,
)
from shared.constants import ChainRole, MutationRecord, PointMutation


class RowBuildOutcome(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    records: list[MutationRecord] = []
    skipped_mutation_tokens: list[tuple[str, str]] = []
    row_error: str | None = None
    sample_id: str | None = None
    chain_map: dict[str, str] = {}


def is_alanine_scanning_mutation(wt_residue: str, mutant_residue: str) -> bool:
    return mutant_residue.upper() == "A" and wt_residue.upper() != "A"


def build_complex_name(antibody_name: str, antigen_name: str) -> str:
    return f"{antibody_name} :: {antigen_name}"


def build_chain_map(antibody_chain_roles: dict[str, ChainRole], antigen_chain_letters: str) -> dict[str, str]:
    chain_map = {role.value: chain_id for chain_id, role in antibody_chain_roles.items()}
    if antigen_chain_letters:
        chain_map[ChainRole.ANTIGEN.value] = antigen_chain_letters
    return chain_map


def resolve_chain_role_for_token(
    token: MutationToken,
    antibody_chain_roles: dict[str, ChainRole],
    antigen_chain_letters: str,
) -> ChainRole | None:
    if token.chain_id in antibody_chain_roles:
        return antibody_chain_roles[token.chain_id]
    if token.chain_id in antigen_chain_letters:
        return ChainRole.ANTIGEN
    return None


def build_point_mutation_for_token(
    token: MutationToken,
    structure,
    antibody_chain_roles: dict[str, ChainRole],
    antigen_chain_letters: str,
    sample_id: str,
) -> tuple[PointMutation | None, str | None]:
    chain_role = resolve_chain_role_for_token(token, antibody_chain_roles, antigen_chain_letters)
    if chain_role is None:
        return None, f"unresolved chain role for chain {token.chain_id!r}"

    try:
        chain = get_chain(structure, token.chain_id)
    except KeyError:
        return None, f"chain {token.chain_id!r} not found in structure"

    if not residue_identity_matches(chain, token.residue_position, token.wt_residue, token.insertion_code):
        return None, f"residue identity mismatch at chain {token.chain_id!r} position {token.residue_position}"

    point_mutation = PointMutation(
        chain_id=token.chain_id,
        chain_role=chain_role,
        wt_residue=token.wt_residue,
        mutant_residue=token.mutant_residue,
        residue_position=token.residue_position,
        insertion_code=token.insertion_code or None,
        aligned_interface_position=token.aligned_interface_position,
        interface_region=token.interface_region,
        is_alanine_scanning=is_alanine_scanning_mutation(token.wt_residue, token.mutant_residue),
    )
    return point_mutation, None


def build_mutation_records_for_row(row: dict, row_index: int) -> RowBuildOutcome:
    """One SKEMPI row -> zero or one `MutationRecord`. A row is skipped in
    full (not just the offending residue) when any of its point mutations
    fails to resolve: the row's ddG was measured for the complete mutant
    with every listed point mutation applied, so a partial `PointMutation`
    list would misrepresent what was actually measured.
    """
    outcome = RowBuildOutcome()

    pdb_field_spec = parse_pdb_field(row[COLUMN_PDB])
    antibody_chain_letters, antigen_chain_letters = antibody_and_antigen_chain_letters(
        pdb_field_spec, row[COLUMN_PROTEIN_1], row[COLUMN_PROTEIN_2]
    )

    try:
        structure = load_structure(pdb_field_spec.pdb_id)
    except FileNotFoundError as exc:
        outcome.row_error = str(exc)
        return outcome

    antibody_chain_sequences = {}
    for letter in antibody_chain_letters:
        try:
            antibody_chain_sequences[letter] = chain_sequence(get_chain(structure, letter))
        except KeyError:
            continue
    antibody_chain_roles = resolve_antibody_chain_roles(antibody_chain_letters, antibody_chain_sequences)

    is_antibody_side_protein_1 = antibody_side_is_protein_1(row[COLUMN_PROTEIN_1], row[COLUMN_PROTEIN_2])
    antibody_name = row[COLUMN_PROTEIN_1] if is_antibody_side_protein_1 else row[COLUMN_PROTEIN_2]
    antigen_name = row[COLUMN_PROTEIN_2] if is_antibody_side_protein_1 else row[COLUMN_PROTEIN_1]
    complex_name = build_complex_name(antibody_name, antigen_name)

    ddg_label = compute_ddg_label(row[COLUMN_AFFINITY_MUT], row[COLUMN_AFFINITY_WT], row[COLUMN_TEMPERATURE])

    mutation_tokens = parse_mutation_tokens(
        row[COLUMN_MUTATIONS_PDB_NUMBERING], row[COLUMN_MUTATIONS_CLEANED_NUMBERING], row[COLUMN_INTERFACE_LOCATIONS]
    )

    sample_id = f"{row[COLUMN_PDB]}_{row[COLUMN_MUTATIONS_PDB_NUMBERING]}_{row_index}"
    outcome.sample_id = sample_id
    outcome.chain_map = build_chain_map(antibody_chain_roles, antigen_chain_letters)

    point_mutations: list[PointMutation] = []
    for token in mutation_tokens:
        point_mutation, failure_reason = build_point_mutation_for_token(
            token, structure, antibody_chain_roles, antigen_chain_letters, sample_id
        )
        if failure_reason is not None:
            outcome.skipped_mutation_tokens.append((sample_id, failure_reason))
            continue
        point_mutations.append(point_mutation)

    if len(point_mutations) != len(mutation_tokens):
        return outcome

    outcome.records.append(
        MutationRecord(
            sample_id=sample_id,
            pdb_id=pdb_field_spec.pdb_id,
            complex_name=complex_name,
            mutations=point_mutations,
            label_type=ddg_label.label_type,
            ddg_kcal_mol=ddg_label.ddg_kcal_mol,
            ddg_bin=ddg_label.ddg_bin,
            ineq_direction=ddg_label.ineq_direction,
            temperature_kelvin=parse_temperature_kelvin(row[COLUMN_TEMPERATURE]),
            source_publication=row.get(COLUMN_REFERENCE) or None,
            notes=row.get(COLUMN_NOTES) or None,
        )
    )

    return outcome


class AllRowsBuildResult(BaseModel):
    records: list[MutationRecord]
    skipped_mutation_tokens: list[tuple[str, str]]
    row_errors: list[str]
    sample_chain_maps: dict[str, dict[str, str]]


def build_all_mutation_records(rows: list[dict]) -> AllRowsBuildResult:
    all_records: list[MutationRecord] = []
    all_skipped: list[tuple[str, str]] = []
    row_errors: list[str] = []
    sample_chain_maps: dict[str, dict[str, str]] = {}
    for row_index, row in enumerate(rows):
        outcome = build_mutation_records_for_row(row, row_index)
        if outcome.row_error is not None:
            row_errors.append(outcome.row_error)
            continue
        all_records.extend(outcome.records)
        all_skipped.extend(outcome.skipped_mutation_tokens)
        if outcome.sample_id is not None:
            sample_chain_maps[outcome.sample_id] = outcome.chain_map
    return AllRowsBuildResult(
        records=all_records, skipped_mutation_tokens=all_skipped, row_errors=row_errors,
        sample_chain_maps=sample_chain_maps,
    )
