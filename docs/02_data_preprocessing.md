# Data Preprocessing

Two separate pipelines, easy to confuse by name alone:

- **`data/pipeline.py`** — fast, non-ML. Download SKEMPI CSV + PDB structures
  → filter to antibody-antigen rows → build `MutationRecord`s → assign flat
  per-residue indices → discard homology-duplicate records → assign
  train/val split membership → write per-split CSVs (`shared.constants.SPLIT_FILES`)
  and the full `mutation_records.jsonl`.
- **`data/embedding_pipeline/pipeline.py`** — slow, ML. For each record,
  computes (or loads from cache) the per-sample structure/sequence embedding
  tensors under `EMBEDDING_CACHE_DIR`, keyed by `pdb_id` for wild-type and
  `sample_id` for mutant. Runs independently of and after the fast pipeline.

## Embedding backends

Three were tried:

- **ESM-2** — sequence embeddings, both chains.
- **IgFold** — structure embeddings, antibody (heavy/light) chains only; the
  antigen chain gets an all-zero placeholder. ESMFold was tried for the
  antigen side and dropped after OOM'ing on a real chain even on a 32GB
  V100, plus added complexity not worth the benefit.
- **SaProt** — structure-aware PLM; fuses amino-acid identity and Foldseek
  3Di structural tokens into one vocabulary, needing only the real WT PDB
  already on disk (no folding, no OOM risk).

**Final decision: SaProt embeddings alone are the model's input**
(`embedding_source_mode: structure_only`). ESM-2 concatenated alongside
SaProt (`structure_and_sequence`) was tried and dropped: at this sample count
(~600-900 training rows), the larger concatenated input (5120-dim after
pooling vs. 2560-dim for SaProt alone) was too big for a light head to learn
well, and gave no measurable improvement (`structure_only` hit 1.046 val
metric by epoch 4 vs. 1.150-1.166 for `structure_and_sequence` variants — see
`docs/future_work.md`). This still satisfies a "multimodal sequence+structure"
design even without ESM-2: SaProt fuses sequence and structure internally via
its AA+3Di vocabulary, so both signals flow through one embedding rather than
two concatenated backbones.

## Splits

Two independent strategies, both keyed on `complex_name` (normalized) — never
raw `pdb_id`, since SKEMPI frequently re-deposits the same complex under a
different PDB code with a different mutation:

- **`held_out_pdb`** — every record of a complex goes entirely to one side
  (872 train / 185 val).
- **`same_pdb_allowed`** — a fraction of complexes is held out entirely
  (guaranteeing genuinely new complexes in val); the remaining ("shared")
  complexes' samples are split individually (855 train / 202 val).

Leakage key = complex identity + mutation identity (aligned interface
position, not raw residue number or PDB id) — the same complex reappearing
with a *different* mutation is valid new signal, not leakage. Validation
therefore mixes held-out-complex rows and held-out-mutation-on-a-seen-complex
rows in one set (`dl.training.lightning_module` reports these as separate
`val_metric_new_complex` / `val_metric_complex_seen_in_train` breakouts).

## Gaps

- **`n.b.` entries not yet in training.** The hinge-style loss only covers
  `ineq` one-sided bounds (`ACTIVE_LOSS_ITERATION = ITERATION_1_BOUNDED_ONLY`);
  `n.b.` was planned as an extreme-threshold variant of the same idea, not
  implemented/exercised.
- **kon/koff** as an auxiliary multi-task target — agreed in principle, not
  implemented. **ΔH/ΔS subset** — not used at all. **Experimental method**
  (SPR/ITC/etc.) as an input feature — flagged, not implemented.
- **Homology-dedup-uses-SKEMPI's-own-field** — unconfirmed, see
  `docs/01_data_exploration.md`.
- **`wt_real_structure_features`** (`dl/datasets/mutation_csv_dataset.py`) is
  computed and cached per sample but dropped by `select_model_fields` before
  reaching the model — a leftover fallback for an alternative
  wild-type-encoding design that wasn't chosen. Unused, not removed (touches
  the embedding cache schema).
