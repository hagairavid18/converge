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

## Findings (real data, `notebooks/processed_data.py`, 2083 point-mutation rows
from 1057 samples)

- **ΔΔG varies significantly by interface region** (Kruskal-Wallis H=237.98,
  p=2.5e-50): `support` is the most destabilizing on average (mean 1.56
  kcal/mol, n=262), `surface` the least (mean -0.19, n=450) — consistent with
  the usual interface-hot-spot intuition (buried/packed positions matter more
  than solvent-exposed ones).
- **Alanine-scanning mutations are significantly more destabilizing** than
  other single-point substitutions (mean 1.45 vs. 0.51 kcal/mol,
  Mann-Whitney p=6.4e-14) — one more reason the alanine-scanning
  stratification gap below matters: the model could shortcut on this signal
  instead of learning real structure/sequence effects.

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
