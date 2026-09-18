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
- **Structure backbone is now mixed: IgFold for antibody chains, ESMFold for
  the antigen** (`AbLang2` remains a reserved, unimplemented sequence-backend
  enum value). Chosen for speed -- IgFold is purpose-built for antibody
  variable-domain prediction from sequence alone and is dramatically lighter
  than ESMFold (real CPU runs: ~0.5-2s for a single IgFold call covering both
  heavy+light jointly, vs. ~700-1200s for one ESMFold call on a similarly
  sized chain) -- but it does not handle arbitrary sequences, so the antigen
  chain still needs ESMFold. This is a deliberate reintroduction of
  mixed-model complexity, not an oversight. Two design choices made while
  implementing it, flagged for review rather than treated as settled:
  - **Dimension mismatch (64 vs. 1024)**: IgFold's per-residue `structure_embs`
    is 64-dim; ESMFold's `s_s` is 1024-dim. Since both feed the same
    per-sample concatenated tensor, IgFold's output is zero-padded up to 1024
    (`data.embedding_pipeline.structure_backend.pad_structure_embedding_to_common_dim`)
    before concatenation -- the simplest resolution, but the 960 padded
    zero-columns are inert signal for every antibody residue; a learned
    projection instead of padding is an alternative if this turns out to
    matter.
  - **Confidence scale mismatch**: ESMFold reports pLDDT (0-1, higher is more
    confident); IgFold reports prmsd (predicted RMSD in Angstroms, unbounded,
    *lower* is more confident). Converted prmsd to a bounded,
    higher-is-better pseudo-confidence via `exp(-prmsd)` so the two chain
    groups' confidence values are at least on a comparable footing within the
    same per-sample confidence tensor the model consumes -- an unvalidated
    choice of transform, not a principled calibration between the two
    models' uncertainty estimates.
  - IgFold's license (JHU Academic Software License Agreement) permits
    non-commercial use "including at commercial entities"; commercial use
    needs a separate license through Johns Hopkins Technology Ventures. Not
    a blocker for this research pipeline, but worth knowing before any
    production/commercial use of the resulting cached embeddings.
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
- **Structure backbone is per-chain-role, not uniform** — a further deliberate
  deviation from the original "simplest, one model for every chain" decision.
  IgFold now handles heavy/light (antibody) chains; ESMFold handles the antigen
  chain. Chosen for speed (IgFold is much lighter/faster, purpose-built for
  antibody variable-domain prediction) at the cost of the mixed-model complexity
  we'd originally avoided. Revisit if this ever becomes a real maintenance burden.

## Heuristics worth revisiting

- **Antibody-vs-antigen side detection** and **heavy/light chain assignment** in
  `data/chain_roles.py` are keyword- and motif-based heuristics with a small
  curated override list (verified by manual inspection of the current 55 unique
  AB/AG PDBs). A newly downloaded/updated SKEMPI version could introduce protein
  names or chain layouts the current heuristics don't cover.
