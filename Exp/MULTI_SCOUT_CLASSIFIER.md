# Multi-scout classifier

`multi_scout_classifier.py` classifies subjects (default: control vs depressed +
suicidal) from several DKT scouts at once. The scouts' PC time courses are stacked as
input channels of one network (early fusion), the way an EEG model can be given 3 or
64 channels. It asks whether combining the scouts that classify well on their own
beats the best single scout, random sets of the same size, all 62 scouts, and the
sensor channels.

## Commands

```bash
python3 multi_scout_classifier.py --smoke-test                 # synthetic check, ~15 s
python3 multi_scout_classifier.py --device cpu --jobs 12       # full study
python3 multi_scout_classifier.py --arms selected all sensor --models grouped
python3 multi_scout_classifier.py --compare-only               # statistics and figure only
```

## Arms

- `selected`: scouts whose single-scout soft-vote AUC in `--screen-csv` (the
  all-scout run of `scout_subject_classifier.py`) exceeds `--threshold` (0.60; 17
  scouts). They were picked using every subject's label, so this arm is optimistic
  relative to `random`, `all`, and `sensor`.
- `random`: `--n-random` (30) random sets of the same size from all scouts.
- `all`: all 62 scouts.
- `sensor`: the 64 sensor channels, each divided by the subject's baseline SD.
- The best single scout (highest AUC in `--screen-csv`) is compared from its saved
  `scout_subject_classifier.py` predictions; it is not retrained.

Every scout is a linear map of the same 64 channels, so a stacked input of k x 32 PCs
has rank at most 64, and all 62 scouts carry roughly the sensor-space information.

## Pipeline

Each scout is preprocessed as in `scout_subject_classifier.py`, fitted on the outer
training subjects: one orientation per vertex, 32 shared PCs, each PC scaled by the
subject's pooled baseline SD. The per-subject PC x channel matrices of every scout and
fold are cached in `projections/`. Bootstrap averages of the sensor trials are
projected into the scouts per batch; this equals the single-scout pipeline exactly
(checked) and keeps memory independent of the number of scouts. A scout with fewer
than 32 vertices-worth of rank keeps zero rows.

Input: k scouts x 32 PCs x 200 samples of incongruent - congruent, 0 to <800 ms.

## Models

- `grouped`: 1x1 convolution per scout (32 PCs -> 4 virtual sources, weights not
  shared between scouts); depthwise temporal convolution (kernel 31), GroupNorm with
  one group per scout, ELU, average pool 4, dropout 0.25; depthwise temporal
  convolution (kernel 15) and a 1x1 convolution that mixes all scouts into 16
  features, GroupNorm, ELU, average pool 8, dropout 0.25; LayerNorm, dropout 0.3,
  linear. About 6.9 k parameters for 17 scouts and 24 k for 62.
- `flat`: the single-scout EEGNet with its first 1x1 convolution over all stacked PCs.
- The sensor arm uses the single-scout EEGNet on the 64 channels.

Training matches the single-scout model (bootstrap averages of 8-24 trials, 16 per
subject per epoch, time shift, amplitude jitter, row dropout 0.1, class-weighted
cross-entropy with label smoothing, AdamW, one-cycle learning rate, 40 epochs with
the weights of epochs 31-40 averaged, 32-average soft vote at test). Scout inputs also
drop whole scouts with probability 0.2. Nothing is tuned on this data.

## Evaluation and output

The outer splits are those of `scout_subject_classifier.py` (3 repeats x 5 folds,
checked against its saved folds), and every arm uses the same seed per fold, so all
comparisons pair subject by subject. AUC is pooled over each repeat's out-of-fold
predictions and averaged over repeats. Per model: label permutation against chance;
the selected arm's percentile among the random sets; paired subject bootstraps
(stratified by the three original groups) against the best single scout, all
scouts, and the sensor arm; Holm correction across those comparisons.

Output in `Results/multi_scout_classifier/`: `arms/<arm>/` per-fold JSON and
histories, `arms.json`, `arms_summary.csv`, `comparisons.csv`,
`comparison_summary.json`, and `multi_scout_comparison.pdf`. Finished folds are
reused when their settings and inputs match.
