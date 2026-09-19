# Configs

Current best: `structure_saprot_only_linear_splitpool_temp5_same_pdb_allowed_20ep.yaml`
(SaProt structure-only embeddings, linear head, `norm_softmax_by_chain_role`
pooling at temperature 5.0, `same_pdb_allowed` split, 20 epochs).

| Filename | Axis varied vs. current-best |
|---|---|
| `regression.yaml` | baseline: default embedding source, no pooling/head tuning |
| `sequence_only.yaml` | embedding source: sequence only (ESM-2), `norm_softmax` pooling |
| `structure_only.yaml` | embedding source: structure only, no SaProt/pooling tuning |
| `structure_and_sequence.yaml` | embedding source: sequence + structure, `norm_softmax` pooling |
| `sequence_only_linear_gaussian.yaml` | pooling strategy: `gaussian_distance` (paired with the row below) |
| `sequence_only_linear_normsoftmax.yaml` | pooling strategy: plain `norm_softmax`, not split by chain role |
| `structure_saprot_only_linear_splitpool_temp5_same_pdb_allowed_20ep.yaml` | current best |
| `structure_saprot_only_linear_splitpool_temp5_same_pdb_allowed_20ep_temperature.yaml` | metadata feature: adds `temperature_kelvin` |
| `structure_saprot_only_linear_splitpool_temp5_same_pdb_allowed_20ep_hinge.yaml` | loss iteration: `iteration_2_with_hinge` |
| `structure_saprot_only_linear_splitpool_temp5_same_pdb_allowed_20ep_hinge_reweighted.yaml` | loss iteration: hinge + batch-imbalance reweighting |
| `structure_saprot_only_linear_splitpool_temp5_same_pdb_allowed_20ep_tailweighted.yaml` | loss iteration: tail reweighting |

`archive/` contains pure hyperparameter-sweep variants (epoch count / batch
size / weight decay / fusion-normalization suffixes such as `7ep`, `bs4`,
`wd`, `normfusion`) of the configs above.
