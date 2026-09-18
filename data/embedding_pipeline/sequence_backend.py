"""Sequence-model (ESM-2) embeddings and per-residue pseudo-log-likelihood,
branching on `shared.constants.ACTIVE_SEQUENCE_BACKEND` so a later swap to
AbLang2 requires no pipeline rework -- only a new branch here.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from data.embedding_pipeline.device import resolve_device
from shared.constants import ACTIVE_ESM2_CHECKPOINT, ACTIVE_SEQUENCE_BACKEND, SequenceBackend


@lru_cache(maxsize=1)
def _load_esm2_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(ACTIVE_ESM2_CHECKPOINT)
    model = AutoModelForMaskedLM.from_pretrained(ACTIVE_ESM2_CHECKPOINT)
    model.to(resolve_device())
    model.eval()
    return tokenizer, model


def _require_esm2_backend() -> None:
    if ACTIVE_SEQUENCE_BACKEND != SequenceBackend.ESM2:
        raise NotImplementedError(f"Sequence backend {ACTIVE_SEQUENCE_BACKEND} is not implemented yet")


def _tokenize_sequence(sequence: str, tokenizer) -> torch.Tensor:
    return tokenizer(sequence, return_tensors="pt")["input_ids"]


def compute_sequence_embeddings(sequence: str) -> np.ndarray:
    _require_esm2_backend()
    tokenizer, model = _load_esm2_model_and_tokenizer()
    device = resolve_device()
    input_ids = _tokenize_sequence(sequence, tokenizer).to(device)
    with torch.no_grad():
        hidden_states = model(input_ids, output_hidden_states=True).hidden_states[-1]
    per_residue = hidden_states[0, 1:-1, :]
    return per_residue.cpu().numpy()


def compute_sequence_pseudo_log_likelihood(sequence: str) -> np.ndarray:
    """Per-residue pseudo-log-likelihood: for each position, mask it and
    score the log-probability the model assigns to the true residue there.
    """
    _require_esm2_backend()
    tokenizer, model = _load_esm2_model_and_tokenizer()
    device = resolve_device()

    base_input_ids = _tokenize_sequence(sequence, tokenizer).to(device)
    sequence_length = base_input_ids.shape[1] - 2

    batched_input_ids = base_input_ids.repeat(sequence_length, 1)
    residue_positions = torch.arange(1, sequence_length + 1, device=device)
    batched_input_ids[torch.arange(sequence_length, device=device), residue_positions] = tokenizer.mask_token_id

    true_residue_ids = base_input_ids[0, 1 : sequence_length + 1]

    with torch.no_grad():
        logits = model(batched_input_ids).logits
    log_probs = torch.log_softmax(logits, dim=-1)
    masked_position_log_probs = log_probs[torch.arange(sequence_length, device=device), residue_positions, :]
    true_residue_log_probs = masked_position_log_probs.gather(1, true_residue_ids.unsqueeze(1)).squeeze(1)

    return true_residue_log_probs.cpu().numpy()
