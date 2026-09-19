# CLAUDE.md

This is a draft CLAUDE.md — general coding standards and repo conventions for this project. It is separate from the modeling spec (see the "Implementation Spec" tab) — this file governs *how* code is written, not *what* to build.

## Environment & Tooling

- Use UV for dependency and environment management (not pip/conda directly).
- Use PyTorch as the deep learning framework.
- Use PyTorch Lightning for the training loop, not raw PyTorch training loops.
- Use Pydantic models for config and data schemas where reasonable — avoid hard-coded strings and untyped dicts for structural things (config objects, entry schemas, label-type enums). Not required for every internal function, but the default for anything crossing a module boundary.
- Use Comet for experiment tracking. API key will be provided separately; do not hard-code it.
- Implement metrics as `torchmetrics.Metric` subclasses (state + `update()`/`compute()`), not stateless functions — the Lightning module accumulates via `update()` each step and reads results via `compute()`, one metric instance per split so they're never pooled.
- The Lightning module builds its own model, loss, and metrics from config, not from pre-built instances passed in by `train.py`. `train.py` passes the raw `model`/`loss`/`metrics` config sections through to the Lightning module's `__init__`; the Lightning module resolves each one with the generic `build_object(module, class_name, **params)` helper in `dl/utils/factory.py` (`getattr(module, class_name)(**params)`). Every class built this way must be exported from its package's `__init__.py` (e.g. `dl.models.<ModelClass>`, `dl.losses.DDGLoss`, `dl.metrics.<MetricClass>`) so `build_object` can find it by name.
- Use numpy for data preparation / preprocessing work (array manipulation, splitting, dedup logic) rather than hand-rolled loops.

## Code Style

- Stick to the Implementation Spec as written. Don't add extra learned parameters, layers, mechanisms, or knobs beyond what it specifies — no gold-plating, no "might be useful later." If the spec doesn't call for it, leave it out.
- One deliberate, user-approved deviation from Implementation Spec §3: there is no explicit mutation-query encoder (no separate feature vector for 3D location / one-hot WT→mutant residue, concatenated before the head). Mutation identity is inferred purely from the embedding-difference pathway (confidence-weighted subtraction). Mutation-window pooling (`dl/layers/pooling.py`) offers two selectable strategies via `ModelConfig.pooling_strategy` (2026-09-19): the original `gaussian_distance` (fixed-sigma Gaussian falloff over real 3D distance to the mutated residue(s), combining multi-point mutations via elementwise max), and `norm_softmax` (parameter-free softmax over each residue's difference-vector norm — empirically validated to localize the mutated site on its own, see `docs/future_work.md`, and works under `sequence_only` where no 3D distance exists at all; multi-point mutations are handled for free since the softmax naturally weights every residue with high difference norm). Both are being compared empirically rather than one replacing the other.
- Prefer many short, single-purpose functions with meaningful names over long ones — a function name plus its signature should make the docstring unnecessary most of the time.
- Add a docstring only when the function's behavior isn't obvious from its name and signature (a non-obvious edge case, a formula, a subtle contract).
- No inline comments in code. If a line needs explaining, that's a signal to extract it into a well-named function instead.
- Put shared helper functions in a `utils/` module (`data/utils/` for preprocessing helpers, the DL folder's `utils/` for modeling helpers) rather than duplicating them or burying them inline.
- Domain-internal constants (magic numbers, defaults, internal file/column names) that only one side needs go in `data/utils/constants.py` or `dl/utils/constants.py`, not scattered inline. Only promote a constant to `shared/constants.py` once a second workstream actually needs the same value.

## Repository Structure

- `data/` — preprocessing, dataset construction, splitting, homology dedup logic. Anything that turns raw SKEMPI into model-ready inputs. Its output is per-sample metadata written as csv (the `MutationRecord` fields), one file per split-name/subset pair — the four files at `shared.constants.SPLIT_FILES` (`held_out_pdb_train.csv`, `held_out_pdb_val.csv`, `same_pdb_allowed_train.csv`, `same_pdb_allowed_val.csv`). `dl/datasets/` builds its `Dataset`s from these csv files, joining in embeddings from `EMBEDDING_CACHE_DIR` by sample id.
  - `data/utils/` — shared helper functions for preprocessing (not one-off scripts).
- A DL folder (e.g. `dl/` or `src/`) — the modeling codebase, containing:
- `models/` — architecture code (full models, assembled from `layers/`).
- `layers/` — reusable building blocks the models need (custom modules: pooling, masking, weighting, encoders, etc.), kept separate from the full architectures that assemble them.
- `datasets/` — PyTorch `Dataset` classes only, reading the processed output `data/` produces (per `shared/constants.py`) into samples. `DataLoader` construction (batching, sampling, etc.) happens only in `train.py`, driven by the config's `dataset` section — not inside `dl/datasets/` or the Lightning module.
- `losses/` — loss function implementations.
- `metrics/` — evaluation metric implementations.
- `utils/` — shared helper functions used across models/losses/metrics/training.
- Training logic (PyTorch Lightning modules/trainers).
- `notebooks/` — exploration, analysis, visualizations, one-off checks. Kept outside the DL folder; not production code, not imported by it.
- `dl/configs/` — one yaml file per full run configuration, each with a section per submodule (`dataset`, `model`, `loss`, `trainer`, ...). Name each file for what it configures, e.g. `dl/configs/only_esm2.yaml` for a structure-only-embedding run.
- `docs/` — project-level documentation (e.g. `docs/future_work.md` for tracked gaps/open questions), not implementation code.
- `train.py` — entry point that reads a config yaml from `dl/configs/`, builds the dataset/dataloaders, model, loss, and Lightning module from it, and runs training or evaluation.

## Compute Environment

- No local GPU / CUDA is available during development — this machine is CPU-only. All code must run correctly on CPU, even if slow, so development and sanity checks happen locally.
- Do not assume CUDA is present. Device selection must be automatic (fall back to CPU when CUDA is unavailable) — never hard-code a CUDA-only device.
- All code must also be fully CUDA-compatible, not just CPU-runnable: every tensor/module must end up on the same, consistently-resolved device (no hardcoded `.cpu()`/`.cuda()`, no device mismatches between a model and its inputs/buffers/masks), and nothing should assume CPU-only behavior (e.g. silent host-device syncs, CPU-only ops). It must run correctly under either device with no code changes — only the device selection differs.
- Design for portability to a cloud GPU environment later, most likely Google Colab. Keep dependencies and paths Colab-compatible (avoid assumptions about local absolute paths or local-only file layouts) so the project can be picked up and run there with minimal changes.
