# SKEMPI ΔΔG Predictor

Predicts the change in antibody-antigen binding free energy (ΔΔG) caused by a
point mutation, from SKEMPI structures/sequences. See `docs/` for the design
rationale and results per pipeline stage, and `docs/future_work.md` for open
gaps.

## Setup

```bash
uv sync
```

Python >=3.11, PyTorch 2.6. No local GPU is required — device selection is
automatic (CUDA if available, else CPU); the same code runs unchanged on a
Colab GPU. SaProt structure embeddings additionally require a local
`foldseek` binary (`shared.constants.FOLDSEEK_BINARY_PATH`) and are gated
behind the `SKEMPI_USE_SAPROT_STRUCTURE=1` environment variable.

## 1. Data pipeline

Builds per-sample metadata (`MutationRecord`s) from raw SKEMPI + PDB
downloads: antibody/antigen filtering, homology deduplication, and the two
train/val split strategies. Output: `data/processed/mutation_records.jsonl`
and the four CSVs under `data/splits/` (`shared.constants.SPLIT_FILES`).

```bash
uv run python -c "from data.pipeline import run_preprocessing_pipeline; run_preprocessing_pipeline()"
```

## 2. Embedding pipeline

Computes and caches per-sample sequence/structure embeddings (ESM-2, IgFold,
and SaProt when enabled) into `data/embeddings/`, reading the output of step 1.

```bash
uv run python -c "from data.embedding_pipeline.pipeline import run_embedding_pipeline; run_embedding_pipeline()"
```

This is the slow, ML-heavy stage — it's separate from step 1 on purpose so the
fast metadata pipeline can be rerun without recomputing embeddings.

## 3. Training

```bash
uv run python train.py --config dl/configs/structure_saprot_only_linear_splitpool_temp5_same_pdb_allowed_20ep.yaml
```

See `dl/configs/README.md` for what each kept config varies (pooling
strategy, embedding source, loss iteration, etc.) and `dl/configs/archive/`
for the full hyperparameter sweep behind those choices. Comet experiment
tracking is used automatically; set your own API key via the standard Comet
environment variables (never hard-coded).

## 4. Evaluation

```bash
uv run python train.py --config dl/configs/<name>.yaml --mode eval --ckpt <path-to-checkpoint>
```

## 5. Inference

```bash
uv run python predict.py --config dl/configs/<name>.yaml --ckpt <path-to-checkpoint> --input <mutations.csv>
```

Prints a predicted ΔΔG per row, plus whether that row's complex was seen
during training for the given checkpoint (seen-complex/new-mutation vs.
unseen-complex regime) — see `predict.py`'s module docstring for the expected
input CSV shape.

## Notebooks

- `notebooks/01_mutation_location_vs_ddg.ipynb`,
  `notebooks/02_alanine_scanning_comparison.ipynb` — EDA (see
  `docs/01_data_exploration.md`).
- `notebooks/03_mutation_localization_rationale.ipynb` — proves the
  mut−wt embedding-difference pooling mechanism localizes the true mutation
  site (see `docs/03_model_and_training.md`).
- `notebooks/visualize_pooling_norms_3d.py` — renders per-residue pooling
  norms on the real 3D structure as an interactive HTML file.
- `notebooks/*_probe.py` — supporting one-off validation scripts referenced
  by the docs above.
