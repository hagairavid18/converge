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
4. `MutationWindowPooling` reduces the difference representation to one
   pooled vector using a fixed-sigma Gaussian falloff over pre-processing's
   precomputed per-mutation-site distances (`ModelInputs.mutation_distances`).
5. The pooled vector, exposed as a named intermediate, is fed to
   `PredictionHeads`.

Open design question for review: the spec states the wild-type branch is
weighted by sequence confidence only and never receives structural
confidence. Under `EmbeddingSourceMode.STRUCTURE_ONLY` (the current
`ACTIVE_EMBEDDING_SOURCE_MODE`), no sequence confidence is consumed at all,
so `wildtype_branch_confidence` returns `None` and the wild-type branch is
left unweighted (multiplied by 1) in that mode. Confirmed correct as-is.
"""

from __future__ import annotations

from torch import nn

from dl.layers.confidence_weighting import ConfidenceWeighting
from dl.layers.embedding_fusion import fuse_embeddings
from dl.layers.heads import PredictionHeads
from dl.layers.pooling import MutationWindowPooling
from dl.models.confidence_selection import mutant_branch_confidence, wildtype_branch_confidence
from dl.models.config import ModelConfig
from dl.models.schemas import ModelInputs


class DDGPredictor(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.confidence_weighting = ConfidenceWeighting()
        self.pooling = MutationWindowPooling(config)
        self.heads = PredictionHeads(config, config.fused_embedding_dim())

    def forward(self, inputs: ModelInputs) -> dict:
        mode = self.config.embedding_source_mode

        wt_features = fuse_embeddings(inputs.wt_structure_embedding, inputs.wt_sequence_embedding, mode)
        mut_features = fuse_embeddings(inputs.mut_structure_embedding, inputs.mut_sequence_embedding, mode)

        wt_weighted = self.confidence_weighting(wt_features, wildtype_branch_confidence(inputs, mode))
        mut_weighted = self.confidence_weighting(mut_features, mutant_branch_confidence(inputs, mode))

        difference = mut_weighted - wt_weighted
        pooled_representation, pooling_weights = self.pooling(difference, inputs)

        return {
            "head_outputs": self.heads(pooled_representation),
            "pooled_representation": pooled_representation,
            "pooling_weights": pooling_weights,
        }
