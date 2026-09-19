"""SaProt (structure-aware protein language model) embeddings: same ESM2
tokenizer/model mechanics as `sequence_backend.py` (SaProt is an
`EsmForMaskedLM` checkpoint), but each token encodes a residue's amino acid
*and* 3Di structure state (from `foldseek_util`) rather than amino acid
alone. Prototyped as a fix for the antigen chain's current zero-structure-
signal gap -- see `docs/future_work.md`. Wired into
`data.embedding_pipeline.entry_embeddings`'s routing via
`apply_saprot_structure_override`, gated behind
`shared.constants.USE_SAPROT_STRUCTURE`.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from data.embedding_pipeline.device import resolve_device
from data.utils.constants import SAPROT_STRUCTURE_UNKNOWN_TOKEN
from shared.constants import ACTIVE_SAPROT_CHECKPOINT


@lru_cache(maxsize=1)
def _load_saprot_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(ACTIVE_SAPROT_CHECKPOINT)
    model = AutoModelForMaskedLM.from_pretrained(ACTIVE_SAPROT_CHECKPOINT)
    model.to(resolve_device())
    model.eval()
    return tokenizer, model


def build_saprot_tokens(aa_sequence: str, di_sequence: str, masked_positions: frozenset[int] = frozenset()) -> str:
    """SaProt's input format: each residue is one two-character token,
    amino acid letter followed by its lowercased 3Di letter (e.g. "Ad"), or
    by `SAPROT_STRUCTURE_UNKNOWN_TOKEN` where the structure state at that
    position is unknown (the mutant branch's mutated residue).
    """
    structure_chars = [
        SAPROT_STRUCTURE_UNKNOWN_TOKEN if position in masked_positions else di_char.lower()
        for position, di_char in enumerate(di_sequence)
    ]
    return "".join(aa_char + structure_char for aa_char, structure_char in zip(aa_sequence, structure_chars))


def compute_saprot_embeddings(aa_sequence: str, di_sequence: str, masked_positions: frozenset[int] = frozenset()) -> np.ndarray:
    tokenizer, model = _load_saprot_model_and_tokenizer()
    device = resolve_device()
    tokens = build_saprot_tokens(aa_sequence, di_sequence, masked_positions)
    input_ids = tokenizer(tokens, return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        hidden_states = model(input_ids, output_hidden_states=True).hidden_states[-1]
    return hidden_states[0, 1:-1, :].cpu().numpy()
