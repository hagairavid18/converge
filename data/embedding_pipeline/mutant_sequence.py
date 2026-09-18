"""Construction of a chain's mutant sequence from its wild-type sequence and
the `PointMutation`s that fall on that chain.
"""

from __future__ import annotations

from Bio.PDB.Chain import Chain

from data.structures import chain_residue_position_index, chain_sequence
from shared.constants import PointMutation


def mutant_sequence_for_chain(chain: Chain, mutations: list[PointMutation]) -> str:
    wt_sequence = list(chain_sequence(chain))
    position_index = chain_residue_position_index(chain)
    for mutation in mutations:
        key = (mutation.residue_position, (mutation.insertion_code or "").upper())
        index = position_index[key]
        wt_sequence[index] = mutation.mutant_residue
    return "".join(wt_sequence)
