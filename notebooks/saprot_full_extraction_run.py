"""Full-dataset SaProt embedding extraction run (validation experiment, not
production code).

Runs `data.embedding_pipeline.saprot_entry_embeddings.compute_and_cache_all_saprot_embeddings`
over every one of the 1211 real SKEMPI AB/AG samples (loaded once from the
`held_out_pdb` split files, which partition -- not duplicate -- the full
sample set), writing to the `structure_saprot/` cache namespace
(`data.embedding_pipeline.cache.saprot_wt_cache_path` / `saprot_mut_cache_path`)
under `EMBEDDING_CACHE_DIR`, entirely separate from the production
`structure/`+`sequence/` cache this run does not touch.

Reports wall-clock time and basic sanity stats (embedding shapes, any
failures) -- this is the real-scale confirmation of the timing estimated
from the 3-PDB probe in `saprot_extraction_probe.py`.

Run: uv run python notebooks/saprot_full_extraction_run.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.embedding_pipeline.saprot_entry_embeddings import compute_and_cache_saprot_entry_embeddings  # noqa: E402
from data.processed_io import load_sample_chain_maps  # noqa: E402
from dl.datasets.mutation_csv_dataset import load_records  # noqa: E402
from shared.constants import SPLIT_FILES, SplitName  # noqa: E402


def load_all_unique_records() -> list:
    records_by_sample_id = {}
    for path in SPLIT_FILES[SplitName.HELD_OUT_PDB].values():
        for record in load_records(path):
            records_by_sample_id[record.sample_id] = record
    return list(records_by_sample_id.values())


def main() -> None:
    records = load_all_unique_records()
    chain_maps = load_sample_chain_maps()
    print(f"extracting SaProt embeddings for {len(records)} samples...")

    start = time.perf_counter()
    failures = []
    for i, record in enumerate(records):
        try:
            bundle = compute_and_cache_saprot_entry_embeddings(record, chain_maps[record.sample_id])
        except Exception as exc:  # noqa: BLE001 -- report every failure, don't stop the run
            failures.append((record.sample_id, repr(exc)))
            continue
        if i == 0:
            print(f"  first sample shapes: {[(k, tuple(v.shape)) for k, v in bundle.items()]}")
        if (i + 1) % 100 == 0:
            elapsed = time.perf_counter() - start
            print(f"  {i + 1}/{len(records)} done, {elapsed:.1f}s elapsed")

    elapsed = time.perf_counter() - start
    print(f"\ndone: {len(records) - len(failures)}/{len(records)} succeeded in {elapsed:.1f}s ({elapsed / 60:.2f} min)")
    if failures:
        print(f"{len(failures)} failures:")
        for sample_id, error in failures[:20]:
            print(f"  {sample_id}: {error}")


if __name__ == "__main__":
    main()
