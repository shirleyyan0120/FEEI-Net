# Model index

## Baselines

| Public ID | Description | Preprocessing |
| --- | --- | --- |
| `stanet` | spatiotemporal attention baseline | channel z-score |
| `xanet` | bilateral cross-attention baseline | channel z-score and fixed bilateral groups |
| `darnet` | dual attention refinement baseline | full-rank CSP |
| `dbpnet` | temporal-frequency dual-branch baseline | channel z-score plus full-rank CSP |
| `listennet` | lightweight nested spatiotemporal baseline | channel z-score and train-fitted Euclidean alignment |
| `mhanet` | multi-scale hybrid attention baseline | full-rank CSP |
| `hcan` | hybrid channel attention baseline | full-rank CSP |

The adapters preserve the principal model stages while accepting 8, 20, or 28 input channels. The shared runner controls data splits, preprocessing fit scope, optimization, and output format.

DBPNet receives two equal channel blocks: channel-z-scored sensor signals and full-rank CSP components. Its five-band frequency branch projects sensor features to fixed 32 x 32 maps built from the custom montage geometry. The benchmark's frozen 1--32 Hz filter means that the nominal 31--50 Hz band contains only the available 31--32 Hz content.

ListenNet uses one Euclidean-alignment matrix estimated from the inner-training windows after channel z-scoring. The same matrix is applied to validation and test windows. XANet splits the physical montage into participant-right and participant-left groups: frontal 1--4 versus 5--8, and the first versus second sets of ten periauricular channels. The frontal-ear view joins channels from the corresponding side before cross-attention.

DBPNet and ListenNet are adapted from the author repositories. XANet is marked as a paper-faithful reproduction because no public author implementation was located when this release was prepared.

All seven baselines use AdamW with a learning rate of `3e-4`, weight decay of `3e-4`, and a batch size of 32. The same settings are used by FEEI-Net and every progressive ablation level.

## Progressive ablation

| Level | Public ID | Added design element | Accuracy (%) |
| ---: | --- | --- | ---: |
| 1 | `unified_carrier` | single 28-channel carrier | 89.06 +/- 12.66 |
| 2 | `dual_branch` | region-specific frontal and ear encoding | 90.02 +/- 11.52 |
| 3 | `dual_branch_cd` | frontal common/difference summaries | 90.18 +/- 11.59 |
| 4 | `dual_branch_cd_interaction` | bidirectional token interaction | 90.28 +/- 11.37 |
| 5 | `feei_full` | regional evidence interaction and gated refinement | **90.76 +/- 10.92** |

All levels use 28-channel frontal-ear input and the same seed, data splits, optimizer family, learning rate, weight decay, batch size, and early-stopping rule. Levels 3 and 4 have an identical parameterization; the interaction residual is disabled at Level 3. Level 2 retains the same module tree as Level 3 but removes the frontal common/difference contribution and disables token interaction.

The class and state-dict names in `models/feei.py` retain internal experiment identifiers so that archived checkpoints remain loadable. In those identifiers, `scan` denotes the eight frontal sensors. The legacy word `ocular` labels the frontal evidence path and does not assign a physiological source. Use the public IDs above for new runs.
