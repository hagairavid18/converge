# SKEMPI ΔΔG Predictor

Predicts the change in antibody-antigen binding free energy (ΔΔG) caused by a
point mutation, from SKEMPI structures/sequences. See `docs/summary.md` for a
one-page overview, `docs/` for full design rationale and results per stage,
and `docs/future_work.md` for open gaps.

## Setup

```bash
uv sync
```

Python >=3.11, PyTorch 2.6. Device selection is automatic (CPU here, CUDA on
Colab, no code changes needed). SaProt embeddings additionally need a local
`foldseek` binary and `SKEMPI_USE_SAPROT_STRUCTURE=1`.

## 1. Data pipeline

```bash
uv run python -c "from data.pipeline import run_preprocessing_pipeline; run_preprocessing_pipeline()"
```

Builds `MutationRecord`s from raw SKEMPI + PDB downloads (filtering, homology
dedup, train/val splits) into `data/processed/` and `data/splits/`.

## 2. Embedding pipeline

```bash
uv run python -c "from data.embedding_pipeline.pipeline import run_embedding_pipeline; run_embedding_pipeline()"
```

Computes and caches per-sample embeddings (ESM-2, IgFold, SaProt) into
`data/embeddings/`. Separate from step 1 so it can be skipped on rerun.

## 3. Training

```bash
uv run python train.py --config dl/configs/structure_saprot_only_linear_splitpool_temp5_same_pdb_allowed_20ep.yaml
```

See `dl/configs/README.md` for what each config varies. Comet tracking is
automatic; set your own API key via Comet's standard env vars.

## 4. Evaluation & inference

A trained checkpoint is committed at
`checkpoints/structure_saprot_only_linear_splitpool_temp5_same_pdb_allowed_20ep.ckpt`
(current-best config, tiny since the only trained component is the regression
head) so evaluation/inference work on a fresh clone without training first —
also needs `data/splits/`, `data/processed/`, and `data/raw/skempi_v2.csv`
(all committed) plus `SKEMPI_USE_SAPROT_STRUCTURE=1 SKEMPI_USE_SMALL_CHECKPOINTS=0`:

```bash
uv run python train.py --config dl/configs/<name>.yaml --mode eval --ckpt <path>
uv run python predict.py --config dl/configs/<name>.yaml --ckpt <path> --input <mutations.csv>
```

`predict.py` prints a predicted ΔΔG per row plus whether that row's complex
was seen during the checkpoint's training (see its module docstring). Any
sample not already in the embedding cache is computed on the fly from
`data/raw/pdbs/*.pdb` (auto-fetched from RCSB if missing) — not committed,
since the full cache is 11GB.

## Notebooks

Four, backed by `notebooks/utils.py`:
- `01_data_exploration.ipynb` — dataset description, dedup, stratification, gaps.
- `02_data_preprocessing.ipynb` — embedding model choice and pipeline stats.
- `03_model.ipynb` — architecture stages and proof the mut−wt diff localizes
  the mutation (incl. inline 3D structure view; `pooling_norms_3d_*.html` are
  the standalone interactive versions).
- `04_results.ipynb` — bucketed MAE vs. baseline, classical-regressor check.
