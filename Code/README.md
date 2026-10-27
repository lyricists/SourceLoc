# SourceLoc analysis code

Source-space EEG analyses for DARPA-NEAT: sentence decoding within each DKT scout,
cluster-based permutation statistics, source-response maps, and group (Control vs
Depressed + Suicidal) classification and time-resolved group decoding.

Run everything from this directory (`cd Code`); scripts import `decoders/`, `models/`,
and `utils/` relative to it. Results go to `../Results/`. Pre-restructure scripts are
archived unchanged in `../Exp/`.

## Layout

```
main_sentence_decoding.py   time-resolved sentence decoding per scout (SVM per time point)
main_sentence_stats.py      scout x time CBP of sentence decoding vs chance, per group
main_decoding_plot.py       group decoding time courses with a time-series CBP (decodingPlot)
main_source_response.py     source-response maps per condition and group
main_source_significant.py  those maps kept only where scout decoding is significant
main_scout_screen.py        single-scout EEGNet group classification (selects scouts)
main_group_classifier.py    flat multi-scout EEGNet on one named scout set + ablation (random sets, all scouts, sensors)
main_ablation_summary.py    one table/figure across the scout sets of main_group_classifier.py
main_group_decoding.py      time-resolved group decoding per scout (ridge, label permutations)
main_group_decoding_plot.py chosen group-decoding scouts in the decodingPlot layout (± SD across CV repeats)
main_network_timeline.py    network timing of sentence decoding (by group) and group-decoding clusters
main_model_figures.py       method figures: pipeline flowchart, EEGNet architecture, flat multi-scout schematic
run_sentence_decoding.sh    both sentence contrasts: decode (opt-in) -> stats -> plot
run_source_response.sh      response maps -> masking
run_group_classification.sh screen -> time-resolved group decoding -> six scout sets + ablation -> summary
decoders/
  sentence_decoder.py       fold-PCA SVM cross-validation per subject/scout (checkpointed)
  subject_classifier.py     subject-level EEGNet cross-validation, single and multi scout
  group_time_decoder.py     ridge decoder that scores all label permutations at once
models/
  eegnet.py                 EEGNet (single scout, stacked scouts = flat model, sensors)
  classifiers.py            linear SVM; closed-form ridge
utils/
  paths.py                  every default data/results path
  io.py                     atomic writes, signatures, atlas/subject/trial loaders, groupings
  source.py                 subject cache, scout orientation + PCA, projection cache
  cv.py                     shared outer splits, bootstrap averaging
  stats.py                  CBP (time series and scout x time), permutation clusters, AUC, bootstrap, Holm
  source_response.py        response estimation and significance masking
  plotting.py               colors, significance bars, multipage PDFs
```

Every entry point has `--help`; the classifier and group-decoding scripts also have
`--smoke-test` (synthetic data, seconds).

## Conventions

- Results are written atomically and keyed by a signature of their settings and input
  files (size and mtime). Matching results are reused; changed settings recompute. There
  are no version constants: after changing a computation, use `--no-resume` (decoders,
  classifiers), delete the cache, or pick a new output directory. Plots are redrawn only
  after new statistics or with `--rebuild-figure`.
- Results from before the restructure stay reusable: signature fields the code no longer
  records (`cache_version`, `group_codes`, `representation`, `n_models`, `per_scout`) are
  ignored when comparing.
- The subject-level analyses (screen, multi-scout classifier, group decoding) share one
  set of outer splits (3 repeats x 5 folds, stratified by the three groups), so their
  predictions pair subject by subject.
- Memory: the EEG and kernel pickles are large; work is done one subject (and scout or
  vertex chunk) at a time, BLAS threads capped by `--blas-threads`.
- Baselines: the group analyses and source responses use -200 to < 0 ms. Sentence
  decoding z-scores over -200 to 0 ms inclusive (one extra sample); it is kept so the
  cluster results stay valid.

## Analyses

**Sentence decoding** (`main_sentence_decoding.py`). Per subject and scout: inverse-kernel
projection, PCA fitted on training trials of each fold (3 PCs), per-trial baseline
z-score, separate train/test pseudotrials (`--trial-num` from `--avg-num`-trial
averages), train-derived scaling, linear SVM per time point. `--condition con_Int`
(congruent vs incongruent intention) or `sent_noBio` (sentiment). Output
`svmDecoding_scout_<condition>.pkl` with `decodeScore` (scout x time x subject). The
existing results were computed on the cluster; `run_sentence_decoding.sh` only reruns
decoding with `RUN_DECODING=1`.

Accuracies are for 12-trial pseudotrial averages, not single trials. With 40 trials per
intention condition, each test fold has 8 trials per class, so its 25 test pseudotrials
are resamples of the same 8 trials. Cross-validation is also slightly pessimistic:
before stimulus onset, con_Int averages 49.6% (sent_noBio 50.1%). Tests against 50% are
therefore conservative; dips below 50% carry no meaning.

**Sentence statistics** (`main_sentence_stats.py`). Per group, a one-tailed one-sample
spatiotemporal CBP on accuracy - 0.5 (10,000 sign-flip permutations; FWE across scouts x
time), then Bonferroni across groups: `--grouping three` (x3) or `control-patients` (x2).
Writes `CBP_Chance_scout_<condition>_3group_corrected` / `_control_patients`.

**Decoding plot** (`main_decoding_plot.py`, formerly decodingPlot.py). Mean ± SEM per
group, averaged over scouts or for `--scout`, with a one-sample time-series cluster test
vs 50% (sign-flip; Bonferroni across groups by default). The old script compared each
group with a constant 50% "group", which is not a valid permutation null.

**Source response** (`main_source_response.py`, `main_source_significant.py`). Group-mean
XYZ-norm source magnitude per vertex for four conditions (default: evoked, components
baseline z-scored); masking keeps each scout's vertices only at its significant decoding
times (three-group CBP). The maps show response magnitude, not the pattern that
separates the conditions, and each group is masked with its own test against chance (not
a group difference). A norm's noise floor depends on the trial count, which differs
between conditions (sentiment 108 vs 132).

**Single-scout screen** (`main_scout_screen.py`). Per scout, fitted on the outer training
subjects: one orientation per vertex, 32 shared PCs, each PC scaled by the subject's
baseline SD. Input: incongruent - congruent, 0 to <800 ms (32 PCs x 200 samples).
EEGNet (about 2 k parameters); every epoch draws 16 bootstrap averages per subject (8-24
trials) with time shift, amplitude jitter, and PC dropout; class-weighted cross-entropy
with label smoothing 0.1, AdamW, one-cycle learning rate, 40 epochs with the weights of
epochs 31-40 averaged; test subjects scored by a soft vote over 32 bootstrap averages.

**Multi-scout classifier and ablation** (`main_group_classifier.py`). The selected scouts'
PCs are stacked as input rows of the same EEGNet (flat: its first 1x1 convolution mixes
all scouts). Bootstrap averages are formed on the sensors and projected per batch (exactly
equal to the screen's preprocessing). A set is the screen threshold (AUC > 0.60, set
`selected17`) or `--selected ... --set-name NAME`; arms: the set, `--n-random` random sets
of the same size, all 62 scouts, the 64 sensors (the last two are shared by every set);
the screen's best scout is compared from its saved predictions. Statistics: label
permutation vs chance, percentile among random sets, paired subject bootstraps, Holm;
written to `Results/group_classifier/comparison_<set>/`. `main_ablation_summary.py` puts
all sets in one table and figure and adds between-set paired bootstraps.
`--fast-eegnet` (default) runs the 1x1 convolutions as matrix multiplies (4-16x faster,
equal up to rounding; tagged in each fold's signature). The first run, with the
convolution implementation and 30 random sets, is kept in `Results/multi_scout_classifier/`
(`--no-fast-eegnet --output-dir ../Results/multi_scout_classifier` reproduces it).

**Time-resolved group decoding** (`main_group_decoding.py`). Per scout and time point a
ridge classifier (penalty = 1 x training subjects) on the subject's congruency contrast in
the 32 fold-fitted PCs. 1,000 label permutations (same permutation for every scout and
time) give a null per point; points above their null 95th percentile form clusters
(contiguous times within a scout), cluster mass = sum of null z-scores, and cluster p comes
from the maximum mass over all scouts x times per permutation (FWE).

## Results so far (Control vs Depressed + Suicidal, 137 subjects)

Every scout set below was chosen with the labels of the same 137 subjects: from the
screen's AUCs (top10, selected17), from the time-resolved group-decoding clusters
(timeres3, timeres6, star5), or both (network10). Their AUCs and their comparisons with
random sets, all scouts, and sensors are therefore optimistic; an unbiased estimate needs
selection nested inside the outer folds (earlier honest estimates were about 0.64-0.65).
The best single scout is likewise the maximum over 62 scouts.

- Single-scout screen: median AUC 0.56; best inferiortemporal R 0.727, superiortemporal R
  0.694, middletemporal R 0.667. These are maxima over 62 scouts; label-permutation null
  per scout: mean 0.48, 95th percentile ~0.58. Depressed vs Suicidal is at chance.
- Flat multi-scout EEGNet on the 17 scouts with screen AUC > 0.60 (first run,
  `Results/multi_scout_classifier/`, 30 random sets): 0.760 ± 0.029 (balanced accuracy
  0.685). Versus the best single scout +0.034 (95% CI -0.04 to +0.10, n.s.); above all 30
  random sets (median 0.651; p = 1/31); above all 62 scouts (0.641) and the sensors
  (0.601), p < 0.01 (all optimistic, see above).
- The per-scout-constrained ("grouped") variant was worse than flat in 25 of 30 random
  sets; it is archived in `../Exp/multi_scout_classifier.py`, with its results in
  `Results/multi_scout_classifier/summary_grouped_and_flat_2026-10-17/`.
- Scout-set ablation (flat EEGNet, 0-800 ms, 100 random sets per size; `Results/group_classifier/ablation_summary/`):
  timeres3 0.697, timeres6 0.703, star5 (5 scouts with literature + time-resolved p < 0.10) 0.729, top10 (10 best
  screen AUCs) 0.760, network10 (top10 minus paracentral R and transversetemporal R, plus medial and lateral
  OFC R; grouped by network, but built from the data-driven top10 and group-decoding results) 0.756, selected17
  0.760. Versus random sets of the same size (median +0.10 to +0.15): top10, network10, and selected17 are above
  all 100 (Holm p = 0.02); timeres3 (99th percentile, Holm p = 0.08) and timeres6 (97th, Holm p = 0.11) are not
  significant after Holm; star5 was not compared with random sets. No set beats the best single scout
  significantly; differences between sets are not significant after Holm. The 0-984 ms window does not help
  (selected sets -0.005 to -0.023, random sets +0.003 to +0.008; `Results/group_classifier_window0_988/`).
- Time-resolved group decoding: significant clusters in middletemporal R (304-996 ms,
  p = 0.004), superiortemporal R (312-860 ms, p = 0.018), medialorbitofrontal R
  (596-976 ms, p = 0.042); nothing before ~300 ms.
