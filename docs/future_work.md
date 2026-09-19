# Gaps & Future Work

Tracks known gaps, open questions, and deliberate scope cuts across the project, so
they're not lost between sessions. Not a task list with owners/dates — just what's
worth revisiting and why.

## Current status (2026-09-19)

**Confirmed architecture**: SaProt structure-only embeddings
(`model.params.embedding_source_mode: structure_only`), `ChainRoleSplitPooling`
(`pooling_strategy: norm_softmax_by_chain_role`) at `pooling_temperature: 5.0`, plain
linear regression head (`regression_hidden_dims: []`, no weight decay) — see
`dl/configs/structure_saprot_only_linear_splitpool_temp5_same_pdb_allowed_20ep.yaml`.
This beat every alternative tested on `same_pdb_allowed`, 20 epochs each:

| variant | val_metric |
|---|---|
| `structure_and_sequence`, linear (5120-dim concat after split-pool) | 1.150 |
| `structure_and_sequence`, linear + `weight_decay=1e-3` | 1.166 |
| `structure_and_sequence`, MLP head (hidden=64, dropout=0.3) | 1.133 |
| **`structure_only`, linear (2560-dim after split-pool)** | **1.046 by epoch 4** |

Neither weight decay nor an MLP head fixes the "concat dim large relative to ~600-900
training rows" concern for the combined-modality variant — dropping the sequence
branch (which also happens to shrink the dimension) is what actually works, not adding
regularization/capacity to the linear head on top of the bigger representation.

**Splits** (regrouped by `complex_name`, see the "Splits are grouped by complex_name"
bullet under Data scope for why): `held_out_pdb` 872 train / 185 val (zero complex
overlap between sides); `same_pdb_allowed` 855 train / 202 val — 4 complexes entirely
new to val (109 bounded rows), 22 shared complexes present in val with a different
mutation than any train row on that complex (72 bounded rows).

**Baselines on `same_pdb_allowed`** (train-set mean ddG MAE against val; per-complex
mean is a harder floor, only definable for the seen-complex bucket since new-complex
rows have no train rows of their own to average):

| bucket | n | global-mean MAE | per-complex-mean MAE |
|---|---|---|---|
| overall | 181 | 1.389 | n/a |
| new complex | 109 | 1.499 | n/a |
| complex seen in train, different mutation | 72 | 1.222 | 0.948 |

**Latest confirmed-architecture run** (job 910265, on the regrouped split):

| epoch | val_metric (overall) | new_complex | complex_seen_in_train |
|---|---|---|---|
| 5 (best overall) | 0.984 | 1.052 | 0.882 |
| 9 | 1.034 | 1.141 | 0.873 |
| 20 (final) | 1.113 | 1.293 | 0.841 (best) |

Key finding: the two buckets overfit in **opposite directions** over training.
`complex_seen_in_train` improves monotonically through epoch 20 — it beats even the
harder per-complex-mean baseline (0.841 vs. 0.948), suggesting real per-mutation signal
on known complexes, not just memorized per-complex averages. `new_complex` peaks around
epoch 5 (1.052, still well under its 1.499 floor) then steadily worsens to 1.293 by
epoch 20 as the model increasingly specializes to known complexes. If genuine
new-complex generalization is the goal, an early-stopping checkpoint (~epoch 5) beats
training the full 20 epochs.

**Classical regressors on the same frozen pooled representation**
(`notebooks/classical_regressors_probe.py`, `structure_only` + split-pool T=5, no
learned head at all upstream of the regressor):

| regressor | val_new_complex | val_complex_seen_in_train |
|---|---|---|
| dummy_mean (floor) | 1.499 | 1.222 |
| linear (unregularized) | 104.4 (blown up) | 36.1 (blown up) |
| ridge (alpha=10) | 2.202 (worse than dummy) | 1.154 |
| knn (k=10) | 1.131 | 0.954 |
| random_forest | 1.152 | **0.843** |
| hist_gradient_boosting | 1.205 | 0.878 |
| svr (rbf) | 1.204 | 0.860 |

Two takeaways: (1) plain unregularized `LinearRegression` catastrophically explodes
(2560-dim features, 738 train rows — the same p>>n danger noted earlier this session
with unscaled sklearn baselines), and ridge alone is still worse than the dummy
baseline on `new_complex` (2.202 vs. 1.499) — an unlearned linear map on this feature
space doesn't generalize to unseen complexes at all. (2) `random_forest` on the frozen
features is essentially tied with the *trained* neural linear head (0.843 vs. 0.841 on
`complex_seen_in_train`; 1.152 vs. the neural model's best 1.052, and better than its
overfit epoch-20 value of 1.293, on `new_complex`). Since everything upstream of the
head is parameter-free, this means most of the real signal already lives in the SaProt
embeddings + pooling, not clearly in the learned head itself — worth keeping in mind
before crediting the neural head with more than it's actually contributing.

**Pooling temperature (`pooling_temperature: 5.0`) re-verified** on the actual
full-checkpoint model (`SKEMPI_USE_SMALL_CHECKPOINTS=0`, what real GPU training uses —
not the small CPU-dev checkpoint the original temperature sweep used):
antigen-side participation ratio 80–141 and antibody-side 228–399 across T=1..10 (total
complex length ~628 residues), with the true mutation site getting the top weight
100% of the time on both sides at every tested temperature. Correction to the original
rationale: the full checkpoint's split-pool softmax is *not* one-hot even at T=1 (that
collapse was specific to the small dev checkpoint used for offline exploration before
GPU access), so T=5 isn't fixing a collapse that doesn't actually exist on the real
model — but it tests out fine regardless (meaningfully non-uniform, correct
localization preserved) and matches the empirically best training results, so it's
being kept as-is rather than re-swept.

**Open question: adding real metadata as model inputs (not yet implemented).**
`MutationRecord` already captures `temperature_kelvin` and `source_publication` but
neither is fed to the model yet. A quick leakage/quality check before doing so:
- `temperature_kelvin` is 83% exactly 298.0K (763/919 bounded rows) — that's
  `data.affinity.parse_temperature_kelvin`'s missing-value fallback, not a real
  measurement, so most of the field's variance is really "was temperature reported at
  all," not true thermodynamic temperature. Correlation with `ddg_kcal_mol` is weak
  (-0.10). Using it as-is risks the model learning "which publications omit
  temperature" rather than a real temperature effect.
- `source_publication` is nearly a proxy for `complex_name` (median 1 distinct
  publication per complex, max 5) — using it as a categorical feature would let the
  model shortcut the `complex_seen_in_train` bucket by recognizing the publication
  rather than learning mutation-level signal, without helping (and arguably random,
  since an unseen publication ID carries no information) `new_complex` generalization.
  A likely-spurious win on one bucket, not a real capability improvement.
- `PointMutation.interface_region` (COR/RIM/SUP/SUR/INT, from SKEMPI's
  `iMutation_Location(s)` column) is already parsed by `data.mutation_tokens` and
  stored on every point mutation, but currently unused as a model input (see
  `data.embedding_pipeline.entry_embeddings`'s note that it was dropped in favor of
  `mutation_distances`). This is the more promising candidate to revisit: it is a
  structural property of the mutation site itself (independent of assay/publication,
  not derived from the ddG value), so it carries much lower leakage risk than the other
  two fields above.

## Data scope

- **Other modalities.** The model is trained only on the antibody-antigen (`AB/AG`)
  subset of SKEMPI 2.0 (1211 raw rows -> 1211 `MutationRecord` samples, one per row;
  a multi-point SKEMPI mutant is one sample with several `PointMutation`s, not one
  sample per residue). SKEMPI also tags `Pr/PI` (general protein-protein, 1349 rows)
  and `TCR/pMHC` (751 rows) subsets. Worth testing whether pre-training or joint
  training on these broader modalities helps the antibody-antigen problem
  specifically versus training on AB/AG alone.
- **Homology-group siblings are discarded entirely (2026-09-19), reversing an
  earlier design.** Records measuring the same point-mutation set on the same
  antibody+antigen pair, re-crystallized under a different PDB code (154 of
  1211 samples, 15 of 55 PDBs), were previously kept and forced into `val` on
  both splits as a deliberate consistency check (compare predictions on
  nominally-the-same mutation across different depositions;
  `val_metric_homology_sibling` / `val_metric_genuine_held_out` in
  `dl.training.lightning_module`). Across a full ablation round, that check
  didn't turn out to be the comparison that mattered -- the user's call was
  that val should represent only genuinely new information (a new PDB, or the
  same protein pair with a *different* mutation), never a duplicate
  measurement of a mutation already represented elsewhere in the dataset. One
  representative per homology group is kept (wherever the normal split hash
  places it); every other member is now dropped from the dataset outright by
  `data.homology_dedup.discard_homology_siblings`, before split assignment
  ever runs (`data.pipeline.run_preprocessing_pipeline`). New split sizes:
  `held_out_pdb` 677 train / 380 val (was 677/534); `same_pdb_allowed` 844
  train / 213 val (was 844/367). The `is_homology_sibling` field/metric
  breakout was removed end to end since it would be `False` for every
  remaining record by construction.
- **Splits are grouped by `complex_name`, not `pdb_id` (2026-09-19).** Both
  splits previously grouped on `MutationRecord.pdb_id`, treating "same PDB
  code" as a proxy for "same antibody-antigen complex" -- wrong, since SKEMPI
  frequently re-crystallizes the same complex under a different PDB code with
  a different mutation. Verified: 18% of `held_out_pdb`'s old val rows shared
  a `complex_name` with a train row under a different `pdb_id` (e.g. `1VFB`
  in train, `1KIQ`/`1XGR` in val -- all "IgG1-kappa D1.3 Fv :: HEW
  lysozyme"), so "held out" only guaranteed a new PDB file, not a new
  complex. Both splits now group by `complex_name` (normalized via
  `data.homology_dedup.normalize_protein_name`, reused from the dedup logic
  above -- `data/splitting.py` no longer treats `pdb_id` as an identity key
  at all): `held_out_pdb` keeps every record of a complex on one side (872
  train / 185 val, down from 677/380 under the old grouping); `same_pdb_allowed`
  is now a two-tier assignment -- complexes are accumulated (by sample count)
  into an entirely held-out bucket up to `SAME_PDB_ALLOWED_NEW_COMPLEX_FRACTION`
  of total samples (guaranteeing genuinely new complexes in val, which the old
  per-sample-only split never did -- verified zero such rows existed before
  this fix), then the remaining ("shared") complexes' samples are individually
  split at `SAME_PDB_ALLOWED_SHARED_SAMPLE_FRACTION` (855 train / 202 val: 4
  complexes entirely new, 22 shared complexes present with different
  mutations on both sides). `dl.datasets.schemas.DDGLabels.complex_seen_in_train`
  and `dl.training.lightning_module`'s `val_metric_new_complex` /
  `val_metric_complex_seen_in_train` breakout (renamed from the `..._pdb`
  names used in an earlier iteration of this same session, which keyed on the
  wrong identity) now report against this corrected grouping.
- **Cross-complex homology matching was explicitly scoped out.** Only literal
  re-measurements of the same {antibody, antigen, mutation-set} are grouped;
  different antibody-antigen pairs hitting the same physical spot on the antigen
  are always kept as distinct measurements. Revisit only if a concrete homology
  criterion (e.g. antigen sequence identity threshold) becomes worth the added
  complexity.
- **Whole-chain interface-region classification (Levy-scheme, SASA-based) was
  built, then removed.** `data/interface_classification.py` computed
  support/core/rim/surface/interior for *every* residue via Biopython
  Shrake-Rupley SASA, for a per-residue pooling mask. Removed once the model's
  pooling settled on pure distance-based Gaussian falloff (no consumer of a
  whole-chain classification remained). `PointMutation.interface_region` (for
  mutated residues only) still comes directly from SKEMPI's own
  `iMutation_Location(s)` column, unaffected by the removal. If a future model
  variant wants interface-region weighting again, it needs re-implementing (the
  git history/old code is gone, not just disconnected).

## Model scope

- **No learned embedding-fusion layer.** Structure and sequence embeddings (when
  both active) are combined by plain concatenation, not a learned projection
  into a shared hidden dimension. This makes the head the model's only learned
  component (fusion, confidence weighting, subtraction, and Gaussian pooling
  are all parameter-free). Revisit only if concatenation-only underperforms.
- **Padding/masking kept, not revisited yet.** Batching variable-length
  per-residue sequences currently uses padding + a padding mask. An
  alternative was considered: process each sample at its own natural length
  and only stack results into a true batch after pooling (the first point
  every sample is fixed-size), avoiding padding/masking entirely at the cost
  of losing batched-matmul parallelism for the per-residue steps (matters more
  on GPU than on this CPU-only dev machine). Not pursued now; revisit if the
  padding/masking code becomes a maintenance burden or a real bottleneck.
- **Wild-type structural confidence: captured but unused, needs validation.**
  Pre-processing caches structural confidence (pLDDT) for *both* WT and mutant
  ESMFold runs now (previously only mutant) -- cached as
  `wt_structure_plddt_diagnostic` alongside `mut_structure_confidence` in each
  sample's `.pt` bundle. The model's confidence-weighting still follows the
  spec as decided (WT weighted by sequence confidence only, never structural)
  — WT's structural confidence isn't fed into the model as an input; the field
  name says "diagnostic" specifically so it isn't picked up by accident.
  Open question raised during design: since WT and mutant sequences differ at
  only one (or a few) residues, do two independent ESMFold runs agree closely
  at residues far from the mutation site? If not, the mutant-minus-WT
  subtraction could carry prediction noise unrelated to the actual mutation.
  Needs a validation analysis (compare WT vs. mutant confidence away from the
  mutation site, similar pattern to the EDA notebooks) before deciding whether
  WT structural confidence should factor into the model after all.
- **Mutation-to-residue distances precomputed raw, not sigma-weighted.** Each
  sample's cache also has `mutation_distances`, shape `[num_mutations, L]`:
  the raw Euclidean distance (Angstroms) from each point mutation's wild-type
  CA coordinate to every residue in the concatenated complex. The Gaussian
  pooling falloff (with a configurable sigma) is applied live at train time
  from this, not precomputed, so sigma can be tuned without reprocessing.
- **No explicit mutation-query encoder.** Deliberately dropped in favor of
  inferring mutation identity purely from the embedding-difference pathway (see
  CLAUDE.md's Code Style section). Not revisited unless the embedding-only
  approach underperforms.
- **Mutation-localization hypothesis validated empirically (2026-09-18/19),
  confirmed separately for antibody- and antigen-chain mutations.**
  `notebooks/mutation_site_probe.py` and two ad hoc follow-ups (not checked
  in) confirmed the mutant-minus-wild-type embedding-difference signal is
  sharply, reliably concentrated at the true mutation site:
  - 32 real antibody-chain single-point-mutation samples: untrained
    argmax-of-difference-norm baseline located the exact mutated residue
    100% of the time (train and held-out val alike); top-10 most-changed
    residues sit ~4x closer to the mutation site than a random residue
    (8.36A mean vs. 35.17A baseline; 95% within 2 sigma of the model's 8A
    Gaussian pooling window).
  - Antigen-chain single-point-mutation samples: first checked on a 32-sample
    subset (2026-09-19, after fixing the ESM-2 checkpoint-dimension cache bug
    -- see cache.py's `expected_last_dim_by_key`), then re-verified on the
    FULL population once the embedding cache finished warming (345 unique
    antigen-chain single-mutation samples, all four split files, purely from
    cache): mean diff-norm at the true mutation site 3.73 vs. 0.087 elsewhere
    (~43x contrast); untrained argmax-of-difference-norm localization
    accuracy 99.71% (344/345, vs. ~0.17% random-guess baseline); top-10
    most-changed residues 6.73A mean distance from the mutation site vs.
    38.40A baseline (~5.7x tighter), 97.2% within 2 sigma of the model's 8A
    Gaussian pooling window. Matches the earlier 32-sample numbers closely
    (100%->99.71%, 6.34A->6.73A) -- confirmed as a real, robust effect at full
    scale, not a small-sample artifact. Notable since the structure branch is
    a zero placeholder for the antigen chain (see below) -- the signal here
    comes entirely from the ESM-2 sequence branch, and it's comparable to (if
    anything slightly tighter than) the antibody-chain result.
  Worth rerunning both checks again once the softmax pooling idea below, or
  the interface-distance feature, actually change the embeddings this signal
  is measured on.
- **No antibody-antigen interface/complex structural awareness at all,
  currently.** IgFold folds the antibody chains in complete isolation (no
  knowledge the antigen exists), the antigen structure branch is a
  zero-placeholder, and critically: the *real*, solved wild-type complex
  coordinates are already loaded and cached (`residue_coordinates`) but are
  discarded before reaching the model (`select_model_fields` drops them) --
  currently used only to derive `mutation_distances` for the pooling
  window, never as an actual structural feature. So the model cannot
  currently learn anything about real binding-interface geometry or
  proximity, for either branch. A genuine joint complex-structure predictor
  (AlphaFold-Multimer-style) was considered and rejected as impractical at
  this scale -- MSA-based approaches need per-target genetic-database
  searches (minutes-to-hours each, not feasible for ~1211 samples), and
  even an MSA-free joint model would need triangular attention over the
  *combined* antibody+antigen length, which is worse than the antigen-alone
  case that already OOM'd with ESMFold (see the structure-backbone history
  entry above). Two candidate fixes discussed, not yet prototyped or
  implemented, deliberately deferred:
  - **Interface-distance feature (cheap, no new model)**: for each sample's
    mutation, the minimum real 3D distance (from the already-cached WT
    coordinates) from the mutated residue to the nearest residue on the
    *other* chain -- a direct proxy for "is this mutation at the binding
    interface," which should matter a lot for ΔΔG-binding specifically.
    Proposed integration point: append as a single extra scalar to the
    pooled representation, *after* subtraction and Gaussian pooling, right
    before the head (`fused_embedding_dim() + 1`) -- keeps the
    confidence-weighting/subtraction/pooling pipeline fully parameter-free,
    only the head (already the model's one learned component) grows by one
    input dim. A richer, more invasive variant would fold it in per-residue
    before pooling instead of as one post-pooling scalar; start with the
    cheap version first. Before touching the real model/pipeline: validate
    with a standalone check (e.g. does this scalar correlate with `|ddG|`?)
    the same way the localization hypothesis above was validated first.
  - **Softmax (attention-style) pooling over the difference-norm, instead
    of/alongside the fixed Gaussian-distance pooling**: since the raw
    per-residue difference-norm already localizes the mutation site with
    100% accuracy (see above), a parameter-free `softmax(‖diff_i‖ /
    temperature)` pooling weight could work as well as or better than the
    fixed-sigma Gaussian kernel, and would remove the dependency on
    precomputed 3D distances entirely. Caveat: self-referential (the same
    difference vectors decide the pooling weights and get pooled) -- attention
    pooling is a standard, sound pattern, but this needs testing specifically
    on antigen-mutation samples too (structure branch is zero there, so the
    signal is sequence-only) before assuming it behaves the same as the
    antibody case it was originally motivated by. Would replace a currently
    deliberate, documented design choice (Gaussian pooling) the in-flight
    training run depends on, so treat as a new variant to test alongside the
    current default, not a straight swap.
- **SaProt (structure-aware PLM) extraction, validated and wired in
  (2026-09-19).** Direct candidate fix for the antigen-chain zero-structure-signal
  gap above: `foldseek structureto3didescriptor` converts the *already-cached real
  WT PDB* (no folding, no OOM risk -- a fast geometric algorithm, unlike ESMFold)
  into a per-chain 3Di structure-token sequence; SaProt itself is an
  `EsmForMaskedLM` checkpoint with an expanded amino-acid x 3Di vocabulary, so it
  loads/runs through the exact same `AutoModelForMaskedLM`/`AutoTokenizer`
  machinery `sequence_backend.py` already uses for ESM2 (`SaProt_35M_AF2`,
  hidden dim 480, mirrors `ESM2_CHECKPOINT_CPU_DEV`'s small/full split -- see
  `shared.constants.ACTIVE_SAPROT_CHECKPOINT`). New modules:
  `data.embedding_pipeline.foldseek_util` (the foldseek subprocess wrapper) and
  `data.embedding_pipeline.saprot_backend` (token building + embedding
  extraction); probed end-to-end on 3 real PDBs (`notebooks/saprot_extraction_probe.py`)
  -- foldseek's amino-acid output matched `data.structures.chain_sequence` exactly
  for every chain (antibody and antigen alike), confirming residue alignment.
  Timing on this CPU-only machine (warm model, per PDB averaging ~5 chains of
  ~100-220 residues): foldseek ~0.01-0.04s per PDB (all chains in one call), SaProt
  forward pass ~0.04-0.07s per chain -- both negligible next to IgFold's 0.5-2s/call
  and worlds away from the ESMFold OOM/73-min-per-sample history, since SaProt
  needs no folding at all, just the one real structure already on disk. Mutant-
  branch handling: SaProt's own trained vocabulary includes an amino-acid + "#"
  ("structure unknown at this residue") token, used to mask the mutated
  position(s)' 3Di state -- `build_saprot_tokens`'s `masked_positions` arg, no
  hack needed. Foldseek is a standalone binary (not pip-installable); a static
  build was downloaded to `.tools/foldseek/` (gitignored, `SKEMPI_FOLDSEEK_BIN`
  env var overrides) -- needs the same one-time download step on the eventual
  Colab machine. Full-1211-sample extraction is done (`structure_saprot/wt/`: 55
  PDBs; `structure_saprot/mut/`: 1211 samples), and wired into
  `entry_embeddings.py` as an opt-in override
  (`apply_saprot_structure_override`), gated on `shared.constants.
  USE_SAPROT_STRUCTURE` (env var `SKEMPI_USE_SAPROT_STRUCTURE=1`) -- compared
  alongside the IgFold-only+zero-placeholder default, not replacing it, same
  spirit as the pooling-strategy comparison in CLAUDE.md.
  Mutation-localization check (`notebooks/saprot_diff_localization_probe.py`,
  100 real single-mutation samples, `USE_SAPROT_STRUCTURE=1`, full checkpoint):
  the SaProt-backed `mut_structure_embedding - wt_structure_embedding`
  localizes the mutated residue *at least* as cleanly as the ESM-2 sequence
  signal did -- mean diff-norm 3.385 at the true site vs. 0.051 elsewhere
  (66x), 100/100 (100%) argmax-of-diff-norm accuracy, top-10 spatial
  clustering 5.88A vs. a 32.74A random baseline. This confirms the SaProt
  structural signal is real and well-localized before trusting it in a real
  training run (`structure_saprot_and_sequence_linear_normsoftmax_7ep`).
  **Not done yet**: a residue-offset check for the *mutated* position
  specifically (the extraction probe only checked whole-chain sequence
  equality, not that `PointMutation.flat_residue_index` lines up with
  `di_sequence`'s indexing -- though the localization result above is strong
  indirect evidence the offsets are in fact correct), and any confidence
  signal for the SaProt-backed branch (currently left unweighted, same as
  any other `None` confidence).
- **Structure backbone history (2026-09-18, on moving to the GPU machine):**
  mixed IgFold (antibody)/ESMFold (antigen) → reverted to uniform ESMFold →
  reverted again to IgFold-only with a zero-placeholder for the antigen.
  The uniform-ESMFold attempt OOM'd folding a real (long) chain even on a
  32GB V100 (its trunk's triangular attention tried to allocate 30GB in one
  matmul; lowering `chunk_size` and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments`
  were tried but not enough to make it reliable at full dataset scale), and
  was judged too complex/fragile relative to its benefit. The final choice:
  IgFold still handles heavy/light chains (real embeddings, ~0.5-2s per
  call, zero-padded 64→1024-dim); the antigen chain gets no structure-model
  call at all -- `data.embedding_pipeline.structure_backend.compute_zero_structure_placeholder`
  returns an all-zero `[L, 1024]` embedding and zero confidence instead, so
  the antigen chain currently contributes **no structural signal** to the
  model, only sequence (ESM-2). A sharper consequence, not just "the
  antigen has no structure": for a sample whose mutation falls on the
  *antigen*, the structure branch is functionally inert end to end -- the
  antibody chains get IgFold called on their identical, unmutated sequence
  in both the wild-type and mutant branch (`mutant_sequence_for_chain`
  returns the wild-type sequence unchanged when a chain has no mutations on
  it), so their wild-type/mutant IgFold outputs are the same and the
  embedding-difference signal there is ~zero, while the antigen chain
  itself is the zero-placeholder on both sides too. So antigen-mutation
  samples learn ΔΔG only from the sequence (ESM-2) branch, which does see
  the real mutant antigen sequence. This is a real, deliberate capability
  gap, not an oversight -- revisit if antigen structure turns out to matter
  (a lighter/faster general structure model would be the natural fix, not
  re-attempting ESMFold at this scale without a more robust chunking/tiling
  strategy). `StructureBackend.IGFOLD`/`ACTIVE_STRUCTURE_BACKEND` in
  `shared/constants.py` are now stale relative to this (still say
  ESMFold-only, "applied uniformly to every chain") -- not wired into the
  real routing (`data.embedding_pipeline.entry_embeddings` routes by
  `ChainRole` directly, mirroring the earlier note about this same drift).
- **Regression loss variant not implemented.** The bounded-ΔΔG head is
  classification-only for now; raw `ddg_kcal_mol` is retained per-sample
  specifically so a regression head/loss can be added later without
  reprocessing.
- **Iteration 2 (hinge loss for ineq/n.b. entries) is implemented but gated off**
  (`ACTIVE_LOSS_ITERATION = ITERATION_1_BOUNDED_ONLY`). Not yet exercised
  end-to-end against real ineq/n.b. data — only iteration 1 has a real-data
  verification pass so far.
- **The softmax-weighted bin-center point estimate** (used by the hinge loss and
  the post-training analyses to get a continuous ΔΔG from classification logits)
  is a placeholder, deferred as low-priority since iteration 2 isn't active yet.
  Revisit if/when iteration 2 is turned on.

## Compute / scale

- **Full-dataset embedding computation hasn't been run.** Verified end-to-end on
  one real sample (ESMFold + ESM-2, real weights); a single 2-chain, 327-residue
  sample took ~73 minutes on CPU. Running all 1211 samples locally isn't practical
  — this is expected to happen on a cloud GPU (Colab), per CLAUDE.md's Compute
  Environment section.
- **No real training run yet.** Only the mandatory overfitting sanity check has
  been run (on synthetic and small real-CSV-derived data).
- **On-the-fly embedding extraction, not mandatory offline precompute — a deliberate
  deviation from the spec.** The Implementation Spec says embeddings are "precomputed
  once per entry, offline, before training." Since training isn't happening on this
  CPU-only machine, the dataset instead checks for a cached `.pt` file and computes +
  caches it on a miss, rather than requiring a separate full precompute pass first.
  This still produces the same cache format either way, so a real training run on a
  GPU can either rely on this lazy path or run the full offline pipeline first —
  both work.
- **Checked (2026-09-19): no good precomputed-structure shortcut for the antigen
  branch, but the real gap may not need one.** Looked for a bulk-downloadable
  database of ESMFold-predicted structures/embeddings to avoid running ESMFold
  ourselves (dropped 2026-09-18 for OOM/complexity, see "Structure backbone
  history" above). Two candidates, neither a clean fit:
  - ESM Metagenomic Atlas (617M structures, ESMFold-predicted, embeddings
    included) — built from metagenomic/environmental sequences; SKEMPI
    antigens are known, characterized proteins unlikely to have exact hits.
  - AlphaFold Protein Structure Database (214M structures, all UniProt) —
    much better coverage odds for SKEMPI antigens, but not ESMFold output.
  More relevant, and confirmed by reading the code (`data/structures.py`,
  `entry_embeddings.py`): SKEMPI provides one real, solved PDB complex per
  `pdb_id` -- the wild-type. There is no separately-solved mutant complex
  anywhere in this codebase or its data sourcing (standard for ddG datasets
  like SKEMPI, where mutations are assayed, not re-crystallized);
  `mutation_distances` is already computed from the WT structure's CA
  coordinates only, and both the WT and mutant structure-bundle functions
  load and reuse that same single WT structure object. So "real coordinates"
  means real *wild-type* geometry, assumed to also hold for the mutant
  backbone -- no external download needed to get that much, but any
  antigen-embedding-from-coordinates idea below inherits this same
  WT-geometry-only assumption, not true per-mutant structure.
  Not implementing anything here yet — deferred alongside the interface-distance
  feature idea above.
## Heuristics worth revisiting

- **Antibody-vs-antigen side detection** and **heavy/light chain assignment** in
  `data/chain_roles.py` are keyword- and motif-based heuristics with a small
  curated override list (verified by manual inspection of the current 55 unique
  AB/AG PDBs). A newly downloaded/updated SKEMPI version could introduce protein
  names or chain layouts the current heuristics don't cover.
