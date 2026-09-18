# CLAUDE.md

This is a draft CLAUDE.md — general coding standards and repo conventions for this project. It is separate from the modeling spec (see the "Implementation Spec" tab) — this file governs *how* code is written, not *what* to build.

## Environment & Tooling

- Use UV for dependency and environment management (not pip/conda directly).
- Use PyTorch as the deep learning framework.
- Use PyTorch Lightning for the training loop, not raw PyTorch training loops.
- Use Pydantic models for config and data schemas where reasonable — avoid hard-coded strings and untyped dicts for structural things (config objects, entry schemas, label-type enums). Not required for every internal function, but the default for anything crossing a module boundary.
- Use Comet for experiment tracking. API key will be provided separately; do not hard-code it.

## Repository Structure

- `data/` — preprocessing, dataset construction, splitting, homology dedup logic. Anything that turns raw SKEMPI into model-ready inputs.
- A DL folder (e.g. `dl/` or `src/`) — the modeling codebase, containing:
- `models/` — architecture code.
- `losses/` — loss function implementations.
- `metrics/` — evaluation metric implementations.
- Training logic (PyTorch Lightning modules/trainers).
- `notebooks/` — exploration, analysis, visualizations, one-off checks. Kept outside the DL folder; not production code, not imported by it.

## Compute Environment

- No local GPU / CUDA is available during development — this machine is CPU-only. All code must run correctly on CPU, even if slow, so development and sanity checks happen locally.
- Do not assume CUDA is present. Device selection must be automatic (fall back to CPU when CUDA is unavailable) — never hard-code a CUDA-only device.
- Design for portability to a cloud GPU environment later, most likely Google Colab. Keep dependencies and paths Colab-compatible (avoid assumptions about local absolute paths or local-only file layouts) so the project can be picked up and run there with minimal changes.
