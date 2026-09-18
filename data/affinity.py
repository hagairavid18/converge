"""Conversion of SKEMPI's raw affinity (Kd) columns into a ddG label:
bounded value, one-sided inequality bound, or no-binding.
"""

from __future__ import annotations

import math

from pydantic import BaseModel

from data.utils.constants import (
    DEFAULT_TEMPERATURE_KELVIN,
    GAS_CONSTANT_KCAL_PER_MOL_K,
    INEQUALITY_PREFIXES,
    NO_BINDING_TOKENS,
    TEMPERATURE_ASSUMED_SUFFIX,
)
from shared.constants import LabelType, ddg_to_bin


class ParsedAffinity(BaseModel):
    is_no_binding: bool
    bound_direction: str | None = None
    value_molar: float | None = None


def parse_affinity_value(raw_value: str) -> ParsedAffinity:
    cleaned = raw_value.strip()
    if cleaned.lower() in NO_BINDING_TOKENS:
        return ParsedAffinity(is_no_binding=True)
    for prefix in INEQUALITY_PREFIXES:
        if cleaned.startswith(prefix):
            return ParsedAffinity(is_no_binding=False, bound_direction=prefix, value_molar=float(cleaned[len(prefix):]))
    return ParsedAffinity(is_no_binding=False, value_molar=float(cleaned))


def parse_temperature_kelvin(raw_value: str) -> float:
    cleaned = raw_value.strip()
    if not cleaned:
        return DEFAULT_TEMPERATURE_KELVIN
    if cleaned.endswith(TEMPERATURE_ASSUMED_SUFFIX):
        cleaned = cleaned[: -len(TEMPERATURE_ASSUMED_SUFFIX)]
    return float(cleaned)


def ddg_kcal_mol_from_affinities(kd_mut_molar: float, kd_wt_molar: float, temperature_kelvin: float) -> float:
    return GAS_CONSTANT_KCAL_PER_MOL_K * temperature_kelvin * math.log(kd_mut_molar / kd_wt_molar)


def _flip_direction(direction: str) -> str:
    return "<" if direction == ">" else ">"


class DdgLabel(BaseModel):
    label_type: LabelType
    ddg_kcal_mol: float | None = None
    ddg_bin: int | None = None
    ineq_direction: str | None = None


def compute_ddg_label(affinity_mut_raw: str, affinity_wt_raw: str, temperature_raw: str) -> DdgLabel:
    mut = parse_affinity_value(affinity_mut_raw)
    if mut.is_no_binding:
        return DdgLabel(label_type=LabelType.NB)

    wt = parse_affinity_value(affinity_wt_raw)
    if wt.is_no_binding:
        return DdgLabel(label_type=LabelType.NB)

    temperature_kelvin = parse_temperature_kelvin(temperature_raw)
    ddg_estimate = ddg_kcal_mol_from_affinities(mut.value_molar, wt.value_molar, temperature_kelvin)

    if mut.bound_direction is None and wt.bound_direction is None:
        return DdgLabel(label_type=LabelType.BOUNDED, ddg_kcal_mol=ddg_estimate, ddg_bin=ddg_to_bin(ddg_estimate))

    if mut.bound_direction is not None:
        direction = mut.bound_direction
    else:
        direction = _flip_direction(wt.bound_direction)

    return DdgLabel(label_type=LabelType.INEQ, ddg_kcal_mol=ddg_estimate, ineq_direction=direction)
