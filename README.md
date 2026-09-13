# FEEI-Net

This repository contains the code used for the 1 s frontal/periauricular EEG auditory spatial attention experiments. The release covers:

- seven baseline models evaluated with frontal-only, ear-only, and frontal-ear input;
- the five-level progressive ablation leading to the full FEEI-Net model;
- grouped cross-validation, fold-local preprocessing, training, and subject-level result aggregation.

The EEG recordings are not included.

## Repository map

```text
FEEI-Net/
├── configs/                 subject order and trial-pair groups
├── docs/                    protocol and model notes
├── results/                 reported seed-20261010 summaries
├── scripts/                 local experiment launchers
├── slurm/                   GPUA40 array jobs
├── src/feei_aad/
│   ├── data.py              filtering and 1 s windowing
│   ├── preprocessing.py     grouped CV, normalization, and CSP
│   ├── models/
│   │   ├── baselines.py     STAnet, DARNet, MHANet, and HCAN adapters
│   │   ├── recent_baselines.py  XANet, DBPNet, and ListenNet adapters
│   │   ├── feei.py          FEEI-Net and ablation modules
│   │   └── registry.py      public model names and training settings
│   ├── train.py             training entry point
│   └── summarize.py         subject-level aggregation
└── tests/                   protocol and model checks
```

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
```

## Data layout

Each MAT file must contain one EEGLAB-style variable whose name starts with `EEG_`. Its `data` field must be arranged as channels by samples. Every trial contains 60 s at 128 Hz.

```text
DATA_ROOT/
└── <subject>/
    ├── scan/128/1.mat ... 40.mat
    └── ear/128/1.mat  ... 40.mat
```

`scan` is retained only as the legacy folder name for the eight-channel frontal acquisition system. The paper and public API use `frontal`. The `ear` folder contains the 20-channel periauricular cEEGrid recording. Edit `configs/subjects.txt` if the released subject identifiers differ from your local names.

The trial labels and stimulus-pair groups are stored in `configs/trial_groups.csv`. Label 0 denotes attended-left and label 1 denotes attended-right.

## Reproduce the experiments

Run all seven baselines and three input views:

```bash
bash scripts/run_baselines.sh /path/to/data
```

Run the progressive ablation with 28-channel frontal-ear input:

```bash
bash scripts/run_ablation.sh /path/to/data
```

The launchers use model seed `20261010`, split seed `20260807`, five outer folds, four inner folds, 100 maximum epochs, and early stopping with patience 10. Every released model uses AdamW with a learning rate of `3e-4`, weight decay of `3e-4`, and a batch size of 32. For a single model, subject, or fold, call the module directly:

```bash
python -m feei_aad.train \
  --data-root /path/to/data \
  --run-id example \
  --models feei_full \
  --views frontal_ear \
  --subjects clt \
  --folds 1
```

On Slurm, export the data path and submit an array:

```bash
export DATA_ROOT=/path/to/data
export PYTHON_BIN=/path/to/python
sbatch slurm/baselines.sbatch
sbatch slurm/ablation.sbatch
```

Both job files request the standard `GPUA40` partition.

## Evaluation protocol

Continuous trials are band-pass filtered at 1--32 Hz with an eighth-order zero-phase Butterworth filter. Signals are then divided into 1 s windows with a 0.5 s step. The same windows, labels, and trial groups are used for all input views.

Stimulus pairs are kept intact in a 5-fold outer and 4-fold inner `StratifiedGroupKFold`. No trial or overlapping window crosses train, validation, and test splits. Preprocessing is fitted on the inner-training split only:

| Model | Network input |
| --- | --- |
| STAnet | Filtered EEG with per-channel z-score |
| XANet | Filtered EEG with per-channel z-score and fixed bilateral grouping |
| DARNet | Full-rank CSP time series with component z-score |
| DBPNet | Concatenated channel-z-scored EEG and fold-local full-rank CSP time series |
| ListenNet | Channel-z-scored EEG followed by train-fitted Euclidean alignment |
| MHANet | Full-rank CSP time series with component z-score |
| HCAN | Full-rank CSP time series with component z-score |
| FEEI-Net and ablations | Filtered EEG with per-channel z-score |

CSP replaces the sensor input for DARNet, MHANet, and HCAN. DBPNet instead receives both the normalized sensor signals and a full-rank CSP block, matching its dual-input adaptation. ListenNet's Euclidean alignment and every CSP transform are fitted only on the inner-training split and then applied unchanged to validation and test data.

XANet uses the physical montage rather than acquisition-software channel labels. Frontal channels 1--4 and 5--8 form the two sides; the first and second sets of ten periauricular channels form the participant-right and participant-left groups. For frontal-ear input, frontal and periauricular channels are joined within the corresponding side.

The primary endpoint is held-out 1 s window accuracy. Each subject is first averaged across five outer folds. Reported mean and sample standard deviation are then computed across subjects.

## Outputs

Every subject-fold directory contains:

- `result.json`: seeds, parameter count, selected epoch, and test accuracy;
- `predictions.csv`: held-out window probabilities and immutable identifiers;
- `split_audit.json`: trial and stimulus-pair allocation;
- `normalization.json`: training-fitted preprocessing parameters;
- `history.json`: epoch-level losses and validation accuracy;
- `best_model.pt`: selected checkpoint, unless `--no-checkpoint` is set.

Aggregate one or more run directories with:

```bash
python -m feei_aad.summarize runs/* --output-dir results/generated
```

## Reported results

The frozen summaries and participant-level values are in `results/`. Values are mean +/- sample SD across 15 subjects after averaging each subject over five folds.

| Model | Frontal only | Ear only | Frontal-ear |
| --- | ---: | ---: | ---: |
| STAnet | 67.83 +/- 12.98 | 57.17 +/- 10.01 | 70.71 +/- 14.44 |
| XANet | 84.07 +/- 15.74 | 79.66 +/- 14.96 | 85.12 +/- 15.30 |
| DARNet | 87.94 +/- 12.51 | 87.81 +/- 12.31 | 89.98 +/- 12.08 |
| DBPNet | 86.92 +/- 14.13 | 87.01 +/- 12.21 | 89.68 +/- 12.19 |
| ListenNet | 86.90 +/- 14.65 | 86.79 +/- 13.44 | 89.84 +/- 12.96 |
| MHANet | 79.69 +/- 15.92 | 87.97 +/- 12.36 | 90.07 +/- 11.71 |
| HCAN | 87.57 +/- 13.11 | 86.94 +/- 13.04 | 89.02 +/- 12.69 |

These baseline results were obtained with the shared training configuration stated above. The full progressive ablation is described in `docs/MODELS.md`; its predeclared FEEI-Net result is retained rather than replaced by an independent rerun.

## Baseline attribution

The baseline modules are compact benchmark adapters for a common input and training pipeline, not verbatim mirrors of every upstream repository. See `THIRD_PARTY_NOTICES.md` for papers, source links, and adaptation notes.

## Citation

Citation information will be added after publication. Until then, cite the baseline papers when using their adapters.
