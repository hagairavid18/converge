"""Collates per-sample `(ModelInputs-field-dict, DDGLabels-field-dict)` pairs
(from `dl.datasets.mutation_csv_dataset.MutationCsvDataset`) into a batched
`(ModelInputs, DDGLabels)`.

`mutation_distances` carries both variable-length axes at once (`[num_sites,
L]` per sample) since pre-processing now precomputes it directly (see
`dl.models.schemas`), so its own shape is the single source of truth for
both target lengths: the residue axis (-> `padding_mask`, also used to pad
the ordinary per-residue fields) and the mutation-site axis (->
`mutation_site_mask`).
"""

from __future__ import annotations

from dl.datasets.schemas import DDGLabels
from dl.models.schemas import PER_RESIDUE_FIELDS, ModelInputs
from dl.utils.tensor_batching import (
    build_padding_mask,
    stack_field_values,
    stack_field_values_with_padding,
    stack_fields,
    stack_matrix_field_with_padding,
)


def residue_and_site_counts(input_dicts: list[dict]) -> tuple[list[int], list[int]]:
    site_counts = [fields["mutation_distances"].shape[0] for fields in input_dicts]
    residue_counts = [fields["mutation_distances"].shape[1] for fields in input_dicts]
    return site_counts, residue_counts


def collate_ddg_batch(batch: list[tuple[dict, dict]]) -> tuple[ModelInputs, DDGLabels]:
    input_dicts, label_dicts = zip(*batch)
    input_dicts = list(input_dicts)
    site_counts, residue_counts = residue_and_site_counts(input_dicts)
    target_sites, target_residues = max(site_counts), max(residue_counts)

    stacked_inputs = {
        key: stack_field_values_with_padding([fields[key] for fields in input_dicts], target_residues)
        for key in PER_RESIDUE_FIELDS
    }
    other_keys = [key for key in input_dicts[0].keys() if key not in PER_RESIDUE_FIELDS and key != "mutation_distances"]
    for key in other_keys:
        stacked_inputs[key] = stack_field_values([fields[key] for fields in input_dicts])

    stacked_inputs["mutation_distances"] = stack_matrix_field_with_padding(
        [fields["mutation_distances"] for fields in input_dicts], target_sites, target_residues
    )
    stacked_inputs["padding_mask"] = build_padding_mask(residue_counts, target_residues)
    stacked_inputs["mutation_site_mask"] = build_padding_mask(site_counts, target_sites)

    stacked_labels = stack_fields(list(label_dicts))
    return ModelInputs(**stacked_inputs), DDGLabels(**stacked_labels)
