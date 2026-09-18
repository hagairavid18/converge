"""Wild-type-branch "structural embedding": derived directly from the real
PDB structure's own coordinates (backbone dihedrals, local CA-CA distances),
rather than a structure-model prediction -- per the Implementation Spec's
"Wild-type structure: use the real PDB structure directly."

Open question for the modeling side (flagged in the pipeline report): this
per-residue feature vector has a different, much smaller dimensionality than
the mutant branch's learned ESMFold `s_s` embedding, so the two branches
need a per-branch input projection to a shared dimension before the
spec's subtraction step -- this module does not attempt to reconcile that,
it only produces the wild-type-side features.
"""

from __future__ import annotations

import numpy as np
from Bio.PDB.Chain import Chain
from Bio.PDB.vectors import calc_dihedral
from Bio.PDB.Structure import Structure

from data.structures import chain_residue_sequence

REAL_STRUCTURE_FEATURE_DIM = 8


def _atom_or_none(residue, atom_name: str):
    return residue[atom_name] if atom_name in residue else None


def _dihedral_or_zero(atoms: list) -> float:
    if any(atom is None for atom in atoms):
        return 0.0
    return calc_dihedral(*(atom.get_vector() for atom in atoms))


def _ca_distance_or_zero(residue_a, residue_b) -> float:
    if residue_a is None or residue_b is None or "CA" not in residue_a or "CA" not in residue_b:
        return 0.0
    return float(np.linalg.norm(residue_a["CA"].get_coord() - residue_b["CA"].get_coord()))


def compute_backbone_dihedrals(residues: list) -> np.ndarray:
    num_residues = len(residues)
    dihedrals = np.zeros((num_residues, 3), dtype=np.float32)
    for i, residue in enumerate(residues):
        previous_residue = residues[i - 1] if i > 0 else None
        next_residue = residues[i + 1] if i < num_residues - 1 else None

        if previous_residue is not None:
            phi_atoms = [_atom_or_none(previous_residue, "C"), _atom_or_none(residue, "N"),
                         _atom_or_none(residue, "CA"), _atom_or_none(residue, "C")]
            omega_atoms = [_atom_or_none(previous_residue, "CA"), _atom_or_none(previous_residue, "C"),
                           _atom_or_none(residue, "N"), _atom_or_none(residue, "CA")]
            dihedrals[i, 0] = _dihedral_or_zero(phi_atoms)
            dihedrals[i, 2] = _dihedral_or_zero(omega_atoms)
        if next_residue is not None:
            psi_atoms = [_atom_or_none(residue, "N"), _atom_or_none(residue, "CA"),
                         _atom_or_none(residue, "C"), _atom_or_none(next_residue, "N")]
            dihedrals[i, 1] = _dihedral_or_zero(psi_atoms)
    return dihedrals


def compute_neighbor_ca_distances(residues: list) -> np.ndarray:
    num_residues = len(residues)
    distances = np.zeros((num_residues, 2), dtype=np.float32)
    for i, residue in enumerate(residues):
        previous_residue = residues[i - 1] if i > 0 else None
        next_residue = residues[i + 1] if i < num_residues - 1 else None
        distances[i, 0] = _ca_distance_or_zero(previous_residue, residue)
        distances[i, 1] = _ca_distance_or_zero(residue, next_residue)
    return distances


def compute_real_structure_features(structure: Structure, chain_id: str) -> np.ndarray:
    chain: Chain = next(structure.get_models())[chain_id]
    residues = [residue for _, _, _, residue in chain_residue_sequence(chain)]

    dihedrals = compute_backbone_dihedrals(residues)
    dihedral_trig_features = np.concatenate([np.sin(dihedrals), np.cos(dihedrals)], axis=1)
    neighbor_distances = compute_neighbor_ca_distances(residues)

    return np.concatenate([dihedral_trig_features, neighbor_distances], axis=1).astype(np.float32)
