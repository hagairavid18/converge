"""Foldseek wrapper: converts a solved PDB structure into per-chain 3Di
structure tokens for SaProt.

`structureto3didescriptor` is a fast geometric encoding (not a neural fold),
so it sidesteps the ESMFold OOM history documented in `docs/future_work.md`
entirely, and runs directly off the real, already-downloaded wild-type
structure (`data.structures.skempi_structure_path`) -- one call per PDB
covers every chain at once, mirroring the per-`pdb_id` wild-type caching
already used for the structure/sequence backends.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from data.utils.constants import FOLDSEEK_BINARY_PATH


def _run_structureto3didescriptor(pdb_path: Path, output_path: Path) -> None:
    subprocess.run(
        [
            str(FOLDSEEK_BINARY_PATH),
            "structureto3didescriptor",
            "-v",
            "1",
            "--threads",
            "1",
            "--chain-name-mode",
            "1",
            str(pdb_path),
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def _chain_id_from_name_field(name_field: str) -> str:
    return name_field.split(" ", 1)[0].rsplit("_", 1)[-1]


def _parse_descriptor_line(line: str) -> tuple[str, str, str]:
    name_field, aa_sequence, di_sequence, _per_residue_confidence = line.split("\t")
    return _chain_id_from_name_field(name_field), aa_sequence, di_sequence


def compute_3di_sequences_by_chain(pdb_path: Path) -> dict[str, tuple[str, str]]:
    """Per chain id in `pdb_path`: (amino-acid sequence, 3Di structure
    sequence), each the same length and residue order as
    `data.structures.chain_sequence` would return for that chain.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir) / "descriptor.tsv"
        _run_structureto3didescriptor(pdb_path, output_path)
        lines = output_path.read_text().splitlines()
    return {chain_id: (aa_sequence, di_sequence) for chain_id, aa_sequence, di_sequence in map(_parse_descriptor_line, lines)}
