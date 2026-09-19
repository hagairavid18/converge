# Wrap-up status (2026-09-19)

Tracks what this documentation/cleanup pass completed vs. what's still open,
so nothing is silently lost.

## Done
- `docs/01_data_exploration.md`, `docs/02_data_preprocessing.md`,
  `docs/03_model_and_training.md` — concise per-stage write-ups.
- Conservative cleanup: stale docstrings/comments fixed, confirmed-dead code
  removed (`chain_ids_in_structure`, `ACTIVE_STRUCTURE_BACKEND`,
  `ESMFOLD_CHECKPOINT`). `uv run pytest` passes (99/99).
- `dl/configs/` trimmed: 22 hyperparameter-sweep configs moved to
  `dl/configs/archive/`, 11 representative configs + `dl/configs/README.md`
  kept at top level.
- `predict.py` — inference CLI, tested end-to-end against a real checkpoint,
  reports seen-vs-unseen-complex regime per input row.
- `README.md`.

## Still open
- **`notebooks/03_mutation_localization_rationale.ipynb` and
  `notebooks/visualize_pooling_norms_3d.py` (the py3Dmol structure
  visualization) were in progress in a background agent when this session
  ran low on budget.** The agent may still complete and land on its own; if
  not, resume it (see the plan file's execution-plan section 4) or redo it
  from scratch — it's independent of everything else here.
- The two existing EDA notebooks (`01_mutation_location_vs_ddg.ipynb`,
  `02_alanine_scanning_comparison.ipynb`) are still hardcoded to
  `USE_SYNTHETIC_DATA = True`, and their shared loader
  (`notebooks/processed_data.py`) expects a flat per-mutation-row CSV schema
  that predates the current one (mutations are now JSON-encoded into one
  `mutations` column in `data/splitting.py`'s output). They currently cannot
  produce real-data plots without a loader fix.
- `docs/03_model_and_training.md` has two `RESULTS TODO` placeholders
  (metadata-feature ablation, hinge/inequality loss results) awaiting numbers
  only the user has — see that doc for exactly what's needed.
- Everything else listed in `docs/future_work.md`'s "Gaps" section is
  unchanged by this pass (n.b. entries, kon/koff, ΔH/ΔS, alanine
  stratification, homology-dedup field confirmation, outlier removal,
  multi-measurement noise, hard-example reweighting, loss-balance testing).
