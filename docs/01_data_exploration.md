# Data Exploration

Two dataset-level EDA notebooks, run on raw labels before any model exists
(Implementation Spec §5). Details and numbers: `docs/future_work.md`.

## What we did

- **`notebooks/01_mutation_location_vs_ddg.ipynb`** — ΔΔG distribution per
  interface region (`COR`/`RIM`/`SUP`/`SUR`/`INT`, from SKEMPI's own
  `iMutation_Location(s)` column) via box + violin plots and a Kruskal-Wallis
  test, plus `label_type` (bounded/ineq/n.b.) composition per region.
- **`notebooks/02_alanine_scanning_comparison.ipynb`** — ΔΔG for
  alanine-scanning (X→A) mutations vs. all other single-point substitutions,
  via KDE + box plot and a Mann-Whitney U test.

## Data coverage

Every antibody-antigen SKEMPI entry carries a ΔΔG value in some form: an
exact measurement (`bounded`, 919 samples), a one-sided inequality bound
(`ineq`, 59), or a no-detectable-binding flag (`n.b.`, 79) — 1057 samples
total, nothing discarded for missingness.

## Caveat: real numbers not run yet

As committed, both notebooks still execute against a synthetic placeholder
dataset (`USE_SYNTHETIC_DATA = True`) with assumed effects baked into the
generator (region-dependent means, an alanine-scanning shift) — illustrative,
not a finding. `notebooks/processed_data.py`'s loader also predates the
current split-CSV schema: it expects one flat column per `MutationRecord`
field, but real split CSVs now nest all of a sample's point mutations into
one JSON `mutations` column. The loader needs a small update before either
notebook can run against real data.

## Gaps

- **Alanine-scanning stratification not implemented.** It's the dominant
  single-point-mutation category (52.5% of 709 single-point samples), so the
  model risks shortcutting on "alanine-ness" instead of learning real
  structure/sequence signal. Not split-stratified or otherwise handled yet.
- **Homology dedup vs. SKEMPI's own field, unconfirmed.** `data/homology_dedup.py`
  groups records by a self-normalized protein name + mutation set, not any
  SKEMPI-native homology annotation — never checked whether SKEMPI ships one
  that should be used instead/in addition.
- **Mutation-location (antibody-side vs. antigen-side) split stratification**
  undecided.
- **Outlier removal approach** undecided.
- **Multi-measurement entries** (same complex + mutation, several different
  measured ΔΔG values from different labs/methods) are currently just
  deduplicated down to one representative, not exploited as a
  measurement-noise signal.
