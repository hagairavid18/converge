"""Generic per-sample indexing/stacking/padding for Pydantic schemas whose
fields are tensors with a leading batch dimension
(`dl.models.schemas.ModelInputs`, `dl.datasets.schemas.DDGLabels`). Used to
build a `torch.utils.data.Dataset` over a pre-generated batch and to collate
per-sample dicts back into a batched schema instance (see
`dl.datasets.collation`, which owns the specific padding shapes).
"""

from __future__ import annotations

import torch
from pydantic import BaseModel


def to_field_dict(model: BaseModel) -> dict:
    return dict(model.__dict__)


def index_field_value(value, index: int):
    return value[index] if isinstance(value, torch.Tensor) else value


def index_fields(fields: dict, index: int) -> dict:
    return {name: index_field_value(value, index) for name, value in fields.items()}


def stack_field_values(values: list):
    return torch.stack(values, dim=0) if isinstance(values[0], torch.Tensor) else values[0]


def stack_fields(field_dicts: list[dict]) -> dict:
    keys = field_dicts[0].keys()
    return {key: stack_field_values([fields[key] for fields in field_dicts]) for key in keys}


def pad_tensor_to_length(tensor: torch.Tensor, target_length: int) -> torch.Tensor:
    missing = target_length - tensor.shape[0]
    if missing <= 0:
        return tensor
    pad_shape = (missing, *tensor.shape[1:])
    return torch.cat([tensor, tensor.new_zeros(pad_shape)], dim=0)


def stack_field_values_with_padding(values: list, target_length: int):
    if values[0] is None:
        return None
    return torch.stack([pad_tensor_to_length(value, target_length) for value in values], dim=0)


def build_padding_mask(lengths: list[int], target_length: int) -> torch.Tensor:
    mask = torch.zeros(len(lengths), target_length, dtype=torch.bool)
    for row, length in enumerate(lengths):
        mask[row, :length] = True
    return mask


def pad_matrix_to_shape(tensor: torch.Tensor, target_rows: int, target_cols: int) -> torch.Tensor:
    rows_missing = target_rows - tensor.shape[0]
    cols_missing = target_cols - tensor.shape[1]
    padded = tensor
    if cols_missing > 0:
        padded = torch.cat([padded, padded.new_zeros(padded.shape[0], cols_missing)], dim=1)
    if rows_missing > 0:
        padded = torch.cat([padded, padded.new_zeros(rows_missing, padded.shape[1])], dim=0)
    return padded


def stack_matrix_field_with_padding(values: list[torch.Tensor], target_rows: int, target_cols: int) -> torch.Tensor:
    return torch.stack([pad_matrix_to_shape(value, target_rows, target_cols) for value in values], dim=0)
