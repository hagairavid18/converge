# SKEMPI ΔΔG Predictor — Summary

*One-page summary of the approach and results. Full detail in `docs/01-03_*.md`; content here is written to be pasted into a one-page LaTeX writeup.*

## Data

SKEMPI antibody-antigen mutation dataset: 1057 samples after homology
deduplication (919 with an exact ΔΔG, 59 as a one-sided inequality bound, 79
as no-detectable-binding). Split two ways, both grouped by complex (never raw
PDB id, since the same complex is often redeposited under different codes):
`held_out_pdb` (whole complexes held out) and `same_pdb_allowed` (some
complexes held out entirely, others split by mutation). Not yet done:
stratifying by alanine-scanning or antibody/antigen mutation side, outlier
removal, and exploiting multi-measurement entries as a noise signal.

## Preprocessing

A single embedding source is used: **SaProt**, which fuses sequence and
structure internally via its AA+3Di vocabulary. ESM-2 sequence embeddings
were tried concatenated alongside SaProt and dropped — no measurable gain,
and too large an input for a light head at this sample size.

## Model

1. Fuse each branch's active embedding.
2. Confidence-weight (structural × sequence for the mutant; sequence-only for
   the wild type, since it's the real deposited structure).
3. **Subtract** mutant − wild-type, per residue — mutation identity is
   inferred purely from this difference, with no explicit mutation-location
   input.
4. **Pool**: a temperature-softened softmax over each residue's difference
   norm, computed separately per chain (antibody/antigen) and concatenated.
   The diff-norm localizes the true mutation site on its own (~99.7–100%
   top-1 accuracy); temperature (T=5.0) keeps the softmax from over-
   concentrating on just that one residue at the cost of surrounding context.
5. Concatenate optional metadata, then a linear/MLP regression head — the
   only trained component in the pipeline.

Loss: a dead-zone regression term for exact ΔΔG values, plus a one-sided
hinge term for inequality bounds. Metadata features (e.g. temperature) and
the hinge-loss variants were both tried and showed no measurable improvement
over the simpler defaults.

## Results

Bucketed MAE vs. a naive per-complex-mean baseline (`same_pdb_allowed` val):
**0.84 vs. 1.28 kcal/mol overall**, with the model's largest margin on
large-magnitude mutations — where the naive baseline is weakest. A classical
random-forest regressor fit on the model's frozen (untrained) pooled
representation ties the trained linear head (0.843 vs. 0.841 MAE),
suggesting most of the predictive signal already lives in the SaProt
embeddings and pooling mechanism, not the learned head itself.
