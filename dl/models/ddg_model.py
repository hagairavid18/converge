"""Full two-branch ΔΔG predictor (Implementation Spec §3).

Forward pass, in order:

1. `fuse_embeddings` concatenates each branch's active embedding source(s)
   -- selected by `ModelConfig.embedding_source_mode`, see
   `shared.constants.EmbeddingSourceMode` -- into a per-residue vector. No
   learned transform here (per CLAUDE.md's simplicity rule); the
   classification head is the model's only learned component.
2. `ConfidenceWeighting`, driven by `dl.models.confidence_selection`, scales
   each branch's per-residue representation by its own precomputed
   confidence, before subtraction.
3. The wild-type branch is subtracted from the mutant branch to form a
   per-residue difference representation. Mutation identity (which
   residue, wild-type -> mutant) is inferred purely from this
   embedding-difference pathway -- there is deliberately no separate
   explicit mutation-query feature vector (a user-directed simplification
   from the Implementation Spec's literal §3 wording).
4. `ModelConfig.pooling_strategy` selects the pooling mechanism that
   reduces the difference representation to one pooled vector -- see
   `dl.layers.pooling` for the options (`gaussian_distance`; `norm_softmax`;
   `norm_softmax_by_chain_role`, which pools the antigen and antibody
   residues separately and concatenates the two, doubling the head's input
   width -- see `dl.layers.pooling.pooling_output_dim_multiplier`, used
   below to size `PredictionHeads` correctly for whichever strategy is
   active).
5. The pooled vector, optionally standardized first
   (`ModelConfig.normalize_pooled_representation`, parameter-free,
   per-sample z-score), is exposed as a named intermediate. When
   `ModelConfig.extra_metadata_fields` is non-empty, z-scored per-sample
   metadata (e.g. `temperature_kelvin`, see
   `dl.layers.metadata_features.build_metadata_tensor`) is concatenated onto
   it here, widening `PredictionHeads`' input by
   `ModelConfig.metadata_feature_dim()`. Default `()` means this is a no-op
   -- purely additive, byte-identical to the pre-existing forward pass.

The spec states the wild-type branch is weighted by sequence confidence
only and never receives structural confidence. Under
`EmbeddingSourceMode.STRUCTURE_ONLY` (the current
`ACTIVE_EMBEDDING_SOURCE_MODE`), no sequence confidence is consumed at all,
so `wildtype_branch_confidence` returns `None` and the wild-type branch is
intentionally left unweighted (multiplied by 1) in that mode.
"""

from __future__ import annotations

import torch
from torch import nn

from dl.layers.confidence_weighting import ConfidenceWeighting
from dl.layers.embedding_fusion import fuse_embeddings, standardize_last_dim
from dl.layers.heads import PredictionHeads
from dl.layers.metadata_features import build_metadata_tensor
from dl.layers.pooling import build_pooling, pooling_output_dim_multiplier
from dl.models.confidence_selection import mutant_branch_confidence, wildtype_branch_confidence
from dl.models.config import ModelConfig
from dl.models.schemas import ModelInputs


class DDGPredictor(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.confidence_weighting = ConfidenceWeighting()
        self.pooling = build_pooling(config.pooling_strategy, config)
        head_input_dim = (
            config.fused_embedding_dim() * pooling_output_dim_multiplier(config.pooling_strategy)
            + config.metadata_feature_dim()
        )
        self.heads = PredictionHeads(config, head_input_dim)

    def forward(self, inputs: ModelInputs) -> dict:
        mode = self.config.embedding_source_mode
        normalize_before_fusion = self.config.normalize_before_fusion

        wt_features = fuse_embeddings(inputs.wt_structure_embedding, inputs.wt_sequence_embedding, mode, normalize_before_fusion)
        mut_features = fuse_embeddings(inputs.mut_structure_embedding, inputs.mut_sequence_embedding, mode, normalize_before_fusion)

        wt_weighted = self.confidence_weighting(wt_features, wildtype_branch_confidence(inputs, mode))
        mut_weighted = self.confidence_weighting(mut_features, mutant_branch_confidence(inputs, mode))

        difference = mut_weighted - wt_weighted
        pooled_representation, pooling_weights = self.pooling(difference, inputs)
        if self.config.normalize_pooled_representation:
            pooled_representation = standardize_last_dim(pooled_representation)
        if self.config.extra_metadata_fields:
            metadata_features = build_metadata_tensor(inputs, self.config.extra_metadata_fields)
            pooled_representation = torch.cat([pooled_representation, metadata_features], dim=-1)

        return {
            "head_outputs": self.heads(pooled_representation),
            "pooled_representation": pooled_representation,
            "pooling_weights": pooling_weights,
        }
