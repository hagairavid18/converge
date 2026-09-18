# Gaps & Future Work

Tracks known gaps, open questions, and deliberate scope cuts across the project, so
they're not lost between sessions. Not a task list with owners/dates — just what's
worth revisiting and why.

## Data scope

- **Other modalities.** The model is trained only on the antibody-antigen (`AB/AG`)
  subset of SKEMPI 2.0 (1211 raw rows -> 1211 `MutationRecord` samples, one per row;
  a multi-point SKEMPI mutant is one sample with several `PointMutation`s, not one
  sample per residue). SKEMPI also tags `Pr/PI` (general protein-protein, 1349 rows)
  and `TCR/pMHC` (751 rows) subsets. Worth testing whether pre-training or joint
  training on these broader modalities helps the antibody-antigen problem
  specifically versus training on AB/AG alone.
- **Homology groups are routed to validation, not discarded.** Records measuring
  the same point-mutation set on the same antibody+antigen pair, re-crystallized
  under a different PDB code (154 of 1211 samples, 15 of 55 PDBs), are no longer
  dropped: one representative per group is left to the normal split assignment
  and every sibling is forced into `val` at the single-sample level on *both*
  splits, as a deliberate consistency check (compare predictions on nominally
  the same mutation across different depositions). For `held_out_pdb` this is a
  deliberate, scoped exception to that split's "no PDB in both" invariant: an
  earlier version forced the sibling's *whole PDB* to val instead, to keep the
  invariant strictly intact, but that solved the wrong problem (these records
  don't serve the generalization-to-an-unseen-PDB purpose the invariant exists
  for) and cascaded badly whenever a sibling's PDB was also a "hub" of unrelated
  mutations (834 val / 377 train, 69% val). Overriding only the 154 sibling
  samples themselves brings it back to 534 val / 677 train (44% val) -- still
  higher than the ~20% target because held-out-PDB hashes at the PDB level on
  only 55 unevenly-sized PDBs (pre-existing sampling variance, not an artifact
  of sibling routing; `same_pdb_allowed`, which hashes per-sample, lands at a
  more on-target 367 val / 844 train, 30% val).
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
- **Structure/sequence backbones are ESMFold + ESM-2 only.** `IgFold`/`AbLang2`
  are reserved enum values in `shared/constants.py` but not implemented. Worth
  trying once the pipeline is validated, per the spec's staged embedding-ablation
  plan.
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
  been run (on synthetic and small real-CSV-derived data). A first real training
  run needs the full embedding cache populated first.

## Heuristics worth revisiting

- **Antibody-vs-antigen side detection** and **heavy/light chain assignment** in
  `data/chain_roles.py` are keyword- and motif-based heuristics with a small
  curated override list (verified by manual inspection of the current 55 unique
  AB/AG PDBs). A newly downloaded/updated SKEMPI version could introduce protein
  names or chain layouts the current heuristics don't cover.
