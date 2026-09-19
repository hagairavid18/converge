"""SaProt embedding-extraction probe (validation experiment, not production code).

Question: does `data.embedding_pipeline.saprot_backend` actually work end to
end on our real SKEMPI structures, and how long does it take -- concretely,
is it a viable fix for the antigen chain's zero-structure-signal gap (see
`docs/future_work.md`) without repeating the ESMFold OOM history?

For a handful of real PDB files: runs `foldseek_util.compute_3di_sequences_by_chain`
once (whole complex, all chains), sanity-checks its amino-acid sequence
against `data.structures.chain_sequence` (same residue order/length), then
runs `saprot_backend.compute_saprot_embeddings` per chain -- once as the
wild-type token string, once with every residue's 3Di token masked (worst
case for the mutant branch: every position looks "structure unknown") -- and
times both steps.

Run: uv run python notebooks/saprot_extraction_probe.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.embedding_pipeline.foldseek_util import compute_3di_sequences_by_chain  # noqa: E402
from data.embedding_pipeline.saprot_backend import compute_saprot_embeddings  # noqa: E402
from data.structures import chain_sequence, get_chain, load_structure, skempi_structure_path  # noqa: E402

PROBE_PDB_IDS = ["1AHW", "1BJ1", "1DVF"]


def probe_pdb(pdb_id: str) -> None:
    print(f"\n=== {pdb_id} ===")
    structure = load_structure(pdb_id)

    foldseek_start = time.perf_counter()
    di_sequences_by_chain = compute_3di_sequences_by_chain(skempi_structure_path(pdb_id))
    foldseek_seconds = time.perf_counter() - foldseek_start
    print(f"foldseek (all {len(di_sequences_by_chain)} chains): {foldseek_seconds:.2f}s")

    for chain_id, (foldseek_aa_sequence, di_sequence) in di_sequences_by_chain.items():
        expected_aa_sequence = chain_sequence(get_chain(structure, chain_id))
        if foldseek_aa_sequence != expected_aa_sequence:
            print(f"  chain {chain_id}: SEQUENCE MISMATCH (foldseek {len(foldseek_aa_sequence)} vs biopython {len(expected_aa_sequence)})")
            continue

        wt_start = time.perf_counter()
        wt_embedding = compute_saprot_embeddings(foldseek_aa_sequence, di_sequence)
        wt_seconds = time.perf_counter() - wt_start

        mut_start = time.perf_counter()
        mut_embedding = compute_saprot_embeddings(foldseek_aa_sequence, di_sequence, masked_positions=frozenset(range(len(di_sequence))))
        mut_seconds = time.perf_counter() - mut_start

        print(
            f"  chain {chain_id}: L={len(foldseek_aa_sequence)}, "
            f"wt_embedding={wt_embedding.shape} ({wt_seconds:.2f}s), "
            f"fully-masked_embedding={mut_embedding.shape} ({mut_seconds:.2f}s)"
        )


if __name__ == "__main__":
    overall_start = time.perf_counter()
    for pdb_id in PROBE_PDB_IDS:
        probe_pdb(pdb_id)
    print(f"\ntotal: {time.perf_counter() - overall_start:.2f}s for {len(PROBE_PDB_IDS)} PDBs")
