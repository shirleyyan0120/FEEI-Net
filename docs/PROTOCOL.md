# Experimental protocol

## Fixed settings

| Item | Value |
| --- | --- |
| Sampling rate | 128 Hz |
| Trial duration | 60 s |
| Passband | 1--32 Hz |
| Filter | eighth-order zero-phase Butterworth |
| Decision window | 1 s, 128 samples |
| Step | 0.5 s, 64 samples |
| Windows per trial | 119 |
| Trials per subject | 40 |
| Windows per subject | 4,760 |
| Outer CV | 5-fold StratifiedGroupKFold |
| Inner CV | 4-fold StratifiedGroupKFold |
| Group | stimulus pair |
| Model seed | 20261010 |
| Split seed | 20260807 |
| Maximum epochs | 100 |
| Early-stopping patience | 10 |
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Weight decay | 3e-4 |
| Batch size | 32 |
| Loss | cross-entropy |
| Primary endpoint | held-out 1 s window accuracy |
| Statistical unit | subject |

Each subject-fold seed is `20261010 + 100 * subject_index + fold_index`, where both indices are zero-based. `configs/subjects.txt` fixes the subject order.

## Input views

- `frontal_only`: the eight frontal channels.
- `ear_only`: the 20 periauricular channels.
- `frontal_ear`: frontal channels followed by periauricular channels, for 28 channels in total.

The views differ only in selected channels. Trials, labels, windows, split groups, and optimization settings remain fixed. All released models use the same learning rate, weight decay, and batch size shown above.

## Leakage controls

The two trials belonging to one stimulus pair always remain in the same split. Filtering is performed within each continuous 60 s trial before windowing. Normalization, CSP, and Euclidean alignment are fitted on the inner-training windows, then applied unchanged to validation and test windows. The saved split audit verifies that trial IDs, group IDs, array indices, and window IDs do not overlap across splits.

## Aggregation

For subject `s` and model `m`, the reported accuracy is the arithmetic mean of the five outer-fold window accuracies. Group summaries use the arithmetic mean and sample standard deviation of the 15 subject values.
