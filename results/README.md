# Result files

All files report held-out 1 s window accuracy for model seed 20261010 and split seed 20260807. The seven baseline summaries use the shared AdamW configuration (`lr=3e-4`, `weight_decay=3e-4`, `batch_size=32`). The progressive ablation uses the same configuration.

| File | Contents |
| --- | --- |
| `baselines_seed20261010.csv` | mean and sample SD for seven models and three input views |
| `baseline_participant_accuracy_seed20261010.csv` | five-fold mean for every baseline, view, and subject |
| `progressive_ablation_seed20261010.csv` | mean and sample SD for the five ablation levels |
| `ablation_participant_accuracy_seed20261010.csv` | five-fold mean for every ablation level and subject |

The participant is the statistical unit. Summary statistics are calculated from 15 participant values, not from pooled windows or 75 fold values.

The DBPNet, ListenNet, and XANet rows were produced by Slurm array `3666603`. All 45 model-view-fold tasks completed with exit code 0 under the shared configuration, yielding 675 subject-fold result files. The remaining four baseline rows and the progressive ablation are retained from the matching seed-20261010 release runs.
