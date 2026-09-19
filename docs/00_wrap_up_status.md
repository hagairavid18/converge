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
- `notebooks/03_mutation_localization_rationale.ipynb` — consolidates the
  raw-diff-norm and pooling-weight localization probes into one narrated,
  executed notebook.
- `notebooks/visualize_pooling_norms_3d.py` — renders real per-residue
  `norm_softmax_by_chain_role` pooling weights (temperature 5.0, matching the
  current-best config) on the actual WT 3D structure via py3Dmol, with the
  true mutated residue highlighted in magenta. Run against two real cached
  examples (`1AHW_AB_C_DC178A_10`, `1BJ1_HL_VW_HH101Y_647`); in both, the
  argmax weight lands exactly on the true mutation site while the softmax
  still spreads meaningful mass elsewhere (weight-at-site 0.38 and 0.21
  respectively, not ~1.0) — visual confirmation of the "not
  over-concentrated" property `future_work.md` describes numerically. Output
  HTML artifacts are committed alongside the script; open them directly in a
  browser (needs internet, for the 3Dmol.js CDN script). Requires
  `SKEMPI_USE_SAPROT_STRUCTURE=1 SKEMPI_USE_SMALL_CHECKPOINTS=0` (the cache
  was populated with the full-size checkpoint) to rerun on other samples.
- The two original EDA notebooks (`01_mutation_location_vs_ddg.ipynb`,
  `02_alanine_scanning_comparison.ipynb`) were legacy-blocked on a fabricated
  placeholder dataset (`notebooks/synthetic_data.py`, since deleted): it
  existed only because the real pre-processing pipeline hadn't produced
  output yet when the notebooks were first written, and it predated the
  current `MutationRecord` schema (it called the pydantic model with
  fields that now live nested under `mutations`, so it would have raised a
  validation error if actually run). Fixed: `notebooks/processed_data.py` now
  explodes each sample's JSON `mutations` column into one row per point
  mutation, both notebooks load real data unconditionally, and both are
  executed against the real 1057-sample dataset with real findings in
  `docs/01_data_exploration.md`.
- `docs/03_model_and_training.md`'s metadata-feature and hinge-loss sections
  now state plainly that both were tried and showed no measurable
  improvement, per the user (no numbers needed).

## Still open
- Everything listed in `docs/future_work.md`'s "Gaps" section is
  unchanged by this pass (n.b. entries, kon/koff, ΔH/ΔS, alanine
  stratification, homology-dedup field confirmation, outlier removal,
  multi-measurement noise, hard-example reweighting, loss-balance testing).
