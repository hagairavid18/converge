"""A `torch.utils.data.Dataset` over a pre-generated synthetic
(`ModelInputs`, `DDGLabels`) batch, indexed per-sample and re-collated by
`dl.datasets.collation.collate_ddg_batch` -- the same collate function the
real `dl.datasets.mutation_csv_dataset.MutationCsvDataset` uses, so swapping
between them is a `Dataset` swap, not a format change. Used by
`dl.training.sanity_check` and as the `dl.datasets.dataset_builder`
fallback when no real processed data is present yet.

`padding_mask` and `mutation_site_mask` are collation-derived (see
`dl.models.schemas`), so per-sample storage keeps only each sample's true,
unpadded `mutation_distances` (sliced by its own real site count) rather
than the fixed-`max_mutation_sites` batch tensor -- otherwise every sample
would look like it has the maximum site count once re-collated.
"""

from __future__ import annotations

from torch.utils.data import Dataset

from dl.datasets.synthetic_labels import make_synthetic_labels
from dl.models.config import ModelConfig
from dl.models.synthetic import make_synthetic_inputs
from dl.utils.tensor_batching import index_fields, to_field_dict


class SyntheticDDGDataset(Dataset):
    def __init__(self, model_config: ModelConfig, num_bins: int, num_samples: int, sequence_length: int, seed: int):
        inputs = make_synthetic_inputs(model_config, num_samples, sequence_length, seed)
        labels = make_synthetic_labels(num_samples, num_bins, seed)
        self._input_fields = to_field_dict(inputs)
        self._input_fields.pop("padding_mask", None)
        self._mutation_distances = self._input_fields.pop("mutation_distances")
        self._mutation_site_mask = self._input_fields.pop("mutation_site_mask")
        self._label_fields = to_field_dict(labels)
        self._num_samples = num_samples

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, index: int) -> tuple[dict, dict]:
        input_fields = index_fields(self._input_fields, index)
        site_count = int(self._mutation_site_mask[index].sum())
        input_fields["mutation_distances"] = self._mutation_distances[index, :site_count]
        return input_fields, index_fields(self._label_fields, index)
