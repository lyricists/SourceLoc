# Scout subject classifier

`scout_subject_classifier.py` classifies subjects from one DKT scout's source
responses to congruent and incongruent intention sentences (`senIdx_TOI.pkl`,
category `Intention`; the same trials as `senIdx_congruency_Intention.pkl`). The
default comparisons are control vs depressed + suicidal, then depressed vs suicidal.

## Commands

```bash
python3 scout_subject_classifier.py --smoke-test                      # synthetic check
python3 scout_subject_classifier.py --scout 13                        # one scout, both pairs
python3 scout_subject_classifier.py --scout 13 48 --pair control-vs-depressed+suicidal
python3 scout_subject_classifier.py --scout all --device cpu --jobs 12
python3 scout_subject_classifier.py --scout 13 --representation fusion # shared encoder on both conditions
python3 scout_subject_classifier.py --scout 13 --permute-seed 1       # one label-permutation run
```

`--scout` takes zero-based indices separated by spaces or commas, or `all`.
`--device auto` uses CUDA, then Apple MPS, then CPU. With one job the terminal shows
one line per epoch; `--jobs N` runs folds in parallel on CPU and prints one line per
fold, with the per-epoch values in the history files. On this Mac a single fit is
about 1.5x faster on MPS, while 12 CPU jobs are much faster for many scouts. CPU
runs are deterministic; MPS and CUDA give close but not identical numbers.

The first run builds `Results/subject_sensor_cache/` (per-subject sensor trials and
inverse kernels, about 4.6 GB) from `Data_sen_bl.pkl` and `tMatrix.pkl`. Finished
folds are reused when every parameter and input-file signature matches; otherwise
they are recomputed.

## Pipeline

Everything that pools subjects is fitted on the training subjects of each outer fold.

1. Each sensor trial is baseline-corrected per channel (-200 to 0 ms).
2. One XYZ orientation per vertex: the leading eigenvector of the training-average,
   trace-normalized 3x3 source covariance. The three kernel rows collapse to one.
3. The scout's vertices are reduced to `--n-components` (32) shared PCs from the
   training-average vertex covariance. A scout's source activity has rank at most
   64 (channels), so 32 PCs keep nearly all of its variance.
4. Each PC is divided by the subject's pooled baseline SD (scale-free across subjects).
5. Input (`--representation contrast`, default): incongruent - congruent, 0 to <800 ms
   (32 PCs x 200 samples). With `fusion`, each condition goes through the same
   encoder and the latent features are joined as [congruent, incongruent, difference].

## Model and training

EEGNet-style network (~2.7 k parameters): 1x1 spatial convolution (32 -> 16),
depthwise temporal convolution (kernel 31), GroupNorm, ELU, average pool 4,
dropout 0.25; separable convolution (kernel 15), GroupNorm, ELU, average pool 8,
dropout 0.25; LayerNorm, dropout 0.3, linear output. It differs from the original
EEGNet-8,2 in block order, kernel lengths, GroupNorm instead of BatchNorm, and weight
decay instead of max-norm constraints.

- Every epoch draws 16 new bootstrap averages per training subject (8-24 trials each),
  with +-3 sample time shifts, amplitude jitter, and random PC dropout.
- Class-weighted cross-entropy with label smoothing 0.1; AdamW (weight decay 1e-2);
  one-cycle learning rate peaking at 1e-3.
- 40 epochs; the final model averages the weights of epochs 31-40. No validation
  set or early stopping.
- Test subjects: soft vote over 32 bootstrap averages of 16 trials (`soft vote`);
  one average of all valid trials is reported as well (`all-trial`).
- The per-epoch test metrics come from a separate set of bootstrap draws and are
  only a monitor; nothing is selected from them.

## Evaluation and output

`--repeats` (3) x `--folds` (5) CV, stratified by the original three groups, with
the same splits for every scout. Each repeat's AUC uses one out-of-fold probability
per subject. Output goes to `Results/scout_subject_classifier/scout_XX/<pair>/`:
per-fold JSON and `_history.csv`, `subject_predictions.csv`, `summary.json`, and one
`<pair>_scouts.csv` across scouts. Label-permutation runs go to
`permutation_XXXX/`.

## Results so far (control vs depressed + suicidal)

From the October 2026 screen (`Results/scout_subject_screen/`, an earlier
multi-model version of this script):

- EEGNet on the contrast beat a compact CNN + Transformer, the large CNN +
  Transformer from `scout_vertex_group_classifier.py`, spectral models, and
  logistic regression.
- Across all 62 scouts the median soft-vote AUC was 0.56. Right inferior temporal
  (13) reached 0.727 ± 0.035, right superior temporal (57) 0.69, right middle
  temporal (27) 0.67, and left rostral ACC (48) 0.66. Visual scouts scored 0.38-0.42.
  These scouts were picked after seeing all 62 results, so their AUCs are optimistic.
- Label-permutation null: single scout mean 0.48, SD 0.07, 95th percentile ~0.58.
- Prespecified literature five (48, 35, 43, 15, 25), mean of their probabilities:
  0.64 ± 0.05; its best repeat (0.67) beat all 20 null ensembles. Fold-nested top-10
  selection: 0.65.
- Stopping when the training loss plateaued (1%, 5 epochs) stopped at ~53 epochs
  and lowered AUCs (ensemble 0.61); a stricter rule (2%, 3 epochs) stopped at ~24
  epochs and matched the fixed 40 epochs (ensemble 0.64).
- Scout 13, same splits: 32 PCs + contrast 0.727 ± 0.035; 64 PCs + contrast
  0.686 ± 0.040; fusion 0.569 with 32 or 64 PCs. The group difference at this scout
  is carried by the congruency contrast rather than the response to each condition.
- Scout 13, same splits: all four TOI categories (160 trials per condition) 0.552;
  five-seed ensemble (`--n-models 5`) 0.730; full -200 to 1000 ms window 0.694. The
  effect does not carry over to the other sentence categories.
- Scout 13, rank average of EEGNet and logistic regression (evoked + spectral, 0.643
  alone): 0.712 ± 0.002, more stable across repeats but not higher than EEGNet alone.
- Scout 13, examples per subject per epoch x epochs: 16 x 40 0.727; 64 x 40 0.686;
  64 x 10 0.699; 8 x 40 0.705. Four times more training lowered the AUC.
- Scout 13, 128 instead of 32 test averages (identical networks): 0.716 ± 0.032.
  Per-subject soft-vote probabilities moved by 0.03 on average between the two, so
  32 averages leave about ±0.01-0.02 of sampling noise in the AUC.
- Depressed vs suicidal stayed at chance for every scout and model.
