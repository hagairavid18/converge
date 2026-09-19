# Model & Training

Final architecture, loss, and metrics. Grounded in `dl/models/`, `dl/layers/`,
`dl/losses/`, `dl/metrics/`; numbers cited from `docs/future_work.md`
(primary source) and `notebooks/final_sota_vs_baseline_histogram_probe.py`,
not re-derived here.

## Architecture (`dl/models/ddg_model.py`)

1. **Fuse**: concatenate each branch's active embedding source(s), no
   learned projection.
2. **Confidence-weight** (`dl/models/confidence_selection.py`): mutant
   branch by structural x sequence confidence; WT branch by sequence
   confidence only (it's the real deposited structure, not a prediction).
   Under `structure_only` mode WT has no confidence signal and is left
   unweighted — intentional, not an open question.
3. **Subtract** mutant-weighted minus WT-weighted, per residue. Mutation
   identity is inferred purely from this difference — CLAUDE.md's documented
   deviation from spec §3: no separate mutation-query encoder.
4. **Pool** with `ChainRoleSplitPooling` (`norm_softmax_by_chain_role`,
   `dl/layers/pooling.py`): temperature-softened softmax over each residue's
   difference-vector L2 norm, computed *separately* for antibody- and
   antigen-chain residues, then concatenated. Replaces the spec's Gaussian-
   distance pooling entirely — the diff-norm self-locates the mutation by
   construction (99.71%-100% top-1 localization accuracy, per
   `docs/future_work.md`). A single global softmax was tried first and
   always concentrated on whichever chain held the mutation, starving the
   other side. `gaussian_distance`/plain `norm_softmax` remain available as
   intentional alternatives (CLAUDE.md), not replaced.
5. **Concatenate metadata** (`extra_metadata_fields`, e.g.
   `temperature_kelvin`) onto the pooled vector, when configured.
6. **Head**: linear or MLP regression head — the *only* trained component in
   the pipeline; everything upstream is parameter-free.

## Loss functions (`dl/losses/`)

Two parallel framings, both dispatching every sample by `label_type` via a
mask: **`DDGRegressionLoss`** (active default, `DEFAULT_ACTIVE_HEADS =
("regression",)`) uses a dead-zone regression loss for bounded entries;
**`DDGLoss`** (classification, implemented+tested, unused by any current
config) uses a bin-distance-weighted expected-cost loss instead. Both use a
one-sided hinge loss for `ineq` entries (zero penalty on the correct side).

`n.b.` entries are **not yet included** in either loss (see doc 02's gap).
`tail_reweight_enabled`/`apply_batch_imbalance_reweight` are configurable
hard-example/term-balance knobs but **unverified** — `docs/future_work.md`
still lists "no hard-example weighting" and "loss balance untested" as open.

## Metrics tied to stratification

Three axes (`dl/training/lightning_module.py`,
`dl/metrics/regression_metrics.py`): (1) seen-complex-new-mutation vs.
unseen-complex (`complex_seen_in_train`), (2) tail/high-|ΔΔG|
(`HIGH_ABS_DDG_THRESHOLD_KCAL_MOL` breakout), (3) binned error —
`BucketedMAE`, 5 named magnitude buckets.

`notebooks/final_sota_vs_baseline_histogram_probe.py` is the "pair histogram
MAE" comparison: current-best checkpoint
(`structure_saprot_only_linear_splitpool_temp5_same_pdb_allowed_20ep`,
`4411e9ff...`) bucketed MAE vs. a naive per-complex-mean baseline (mean
bounded ΔΔG of that complex's train rows, global mean if new), same
`same_pdb_allowed` val rows, bucket-for-bucket:

| bucket | n | model MAE | baseline MAE |
|---|---|---|---|
| large_destabilizing | 12 | 1.29 | 3.22 |
| moderate_destabilizing | 16 | 0.56 | 2.52 |
| near_zero | 103 | 0.75 | 0.98 |
| moderate_stabilizing | 27 | 0.74 | 0.50 |
| large_stabilizing | 23 | 1.34 | 1.68 |
| **overall** | 181 | **0.84** | **1.28** |

The model beats the baseline overall and by the widest margin on both
large-magnitude buckets — exactly where a naive per-complex average is
weakest. One nuance: on `moderate_stabilizing` (n=27) the baseline is
actually better; treat as a small-sample observation.

## Baseline comparison

The classical-regressors table in `docs/future_work.md`
(`notebooks/classical_regressors_probe.py`, fit on the same frozen pooled
representation with no learned head at all) has one headline finding:
**`random_forest` ties the trained neural linear head** (0.843 vs. 0.841 on
`complex_seen_in_train`; 1.152 vs. the neural model's best 1.052 on
`new_complex`). Since everything upstream of the head is parameter-free, most
of the real signal already lives in the SaProt embeddings + pooling, not
clearly in the learned head itself.

## Metadata features and hinge/inequality loss

`extra_metadata_fields` concatenates z-scored per-sample metadata (currently
just `temperature_kelvin`) onto the pooled representation, right before the
head — additive only, `()` (default) is a byte-identical no-op.

Tried on top of the current-best config: no measurable MAE improvement, so it
isn't part of the current-best config.

`LossIteration.ITERATION_2_WITH_HINGE` turns on the hinge term for
`ineq`/`n.b.` entries alongside the bounded term; configs exist for it and
its variants (`..._hinge.yaml`, `..._hinge_reweighted.yaml`,
`..._tailweighted.yaml`) but the active default stays `ITERATION_1_BOUNDED_ONLY`.

Tried, including the reweighted and tail-weighted variants: no measurable
improvement over the plain bounded-only baseline, which is why
`ITERATION_1_BOUNDED_ONLY` stays the active default.

## Gaps

- Hard-example/tail reweighting and loss-balance between bounded/hinge terms
  are implemented but unverified end-to-end.
- Alanine-scanning stratification is not implemented as a metric breakout
  (see `docs/01_data_exploration.md`).
