"""Identification of homologous SKEMPI entries that measure the same set of
point mutations (matched by aligned interface position, not raw residue
number) on the same antibody-antigen pair, re-crystallized under a different
PDB code.

`data.pipeline` discards every non-representative "sibling" in a homology
group before splitting (`discard_homology_siblings`): a duplicate deposition
of a mutation already represented elsewhere in the dataset isn't new
information for either training or validation. Records for the same
mutation(s) on genuinely different antibody/antigen partners are never
grouped this way -- a different binding partner is a different measurement.
A multi-point mutant is matched as a whole: two records only group together
if they share the exact same set of point mutations (by chain role, wt/mutant
residue, and aligned position) on the same pair, not just an overlapping
subset.
"""

from __future__ import annotations

from collections import defaultdict

from shared.constants import MutationRecord, PointMutation


def normalize_protein_name(name: str) -> str:
    return " ".join(name.strip().lower().split())


def point_mutation_key(mutation: PointMutation) -> tuple:
    return (mutation.chain_role, mutation.wt_residue, mutation.mutant_residue, mutation.aligned_interface_position)


def mutation_set_key(record: MutationRecord) -> tuple:
    return tuple(sorted(point_mutation_key(mutation) for mutation in record.mutations))


def homology_group_key(record: MutationRecord) -> tuple:
    return (normalize_protein_name(record.complex_name or ""), mutation_set_key(record))


def record_is_groupable(record: MutationRecord) -> bool:
    return all(mutation.aligned_interface_position is not None for mutation in record.mutations)


def group_homologous_records(records: list[MutationRecord]) -> dict[tuple, list[MutationRecord]]:
    """Groups records sharing the same antibody+antigen pair and the same
    whole set of point mutations. Ungroupable records (missing an aligned
    position somewhere in their mutation set) and singleton groups are
    omitted -- only genuine multi-member homology groups are returned.
    """
    groups: dict[tuple, list[MutationRecord]] = defaultdict(list)
    for record in records:
        if record_is_groupable(record):
            groups[homology_group_key(record)].append(record)
    return {key: group for key, group in groups.items() if len(group) > 1}


def homology_sibling_sample_ids(records: list[MutationRecord]) -> set[str]:
    """Sample ids that are not the (first-encountered, deterministic)
    representative of their homology group -- these are the ones
    `discard_homology_siblings` drops.
    """
    siblings: set[str] = set()
    for group in group_homologous_records(records).values():
        siblings.update(record.sample_id for record in group[1:])
    return siblings


def discard_homology_siblings(records: list[MutationRecord]) -> list[MutationRecord]:
    """Drops every non-representative sibling, keeping one record per
    homology group (plus every ungroupable/singleton record untouched).
    """
    sibling_ids = homology_sibling_sample_ids(records)
    return [record for record in records if record.sample_id not in sibling_ids]
