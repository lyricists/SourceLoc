# Scout group classifier

`scout_group_classifier.py` trains separate binary group classifiers for one or
more zero-based DKT scouts. It uses the 40 congruent and 40 incongruent intention trials per
subject. Each example is a pair of 12-trial bootstrap averages and their exact
incongruent-minus-congruent contrast. Three small temporal encoders feed a
late-fusion binary classifier. The `conditions` ablation uses only the two
condition encoders.

## Run

The synthetic smoke test uses no study EEG:

```bash
python3 scout_group_classifier.py --smoke-test
```

To run scouts 35 and 45 for Control versus pooled Depressed + Suicidal,
followed by Depressed versus Suicidal:

```bash
python3 scout_group_classifier.py --scout 35 45
```

Those two comparisons are the default, in that order. To request a different
comparison or sequence, use `--pair`:

```bash
python3 scout_group_classifier.py --scout 35 45 --pair control-depressed control-suicidal
```

Available names are `control-vs-depressed+suicidal`, `depressed-suicidal`,
`control-depressed`, and `control-suicidal`. The pooled comparison assigns
Control to class 0 and both Depressed and Suicidal to class 1. Outer test folds
and validation splits preserve the three original groups.

Comma-separated scout indices also work, for example `--scout 0,1,2`. Uncached
scouts share one input read and subject preprocessing pass. The models train
sequentially, first completing every requested scout for one comparison, with
terminal progress for each comparison, scout, fold, and epoch. Each
scout/comparison gets its own result directory. CSV summaries for each
comparison and all requested comparisons are saved in
`Results/scout_group_classifier_v2/` by default.
This keeps its checkpoints separate from runs made before the MPS pooling fix.

To compare the contrast branch on the same split and selected scout:

```bash
python3 scout_group_classifier.py --scout 0 --pair control-depressed --branches conditions
```

`--device auto` selects CUDA, then Apple MPS, then CPU. Use `--device mps` or
`--device cuda` to require that device. `--threads` controls CPU threads.
The encoder uses fixed temporal bins instead of the adaptive pooling operation
that fails on Apple MPS for non-divisible widths, including 300- and 301-sample
epochs.

The first real run reads the large EEG and inverse-kernel pickles and caches
the requested scouts; later group-pair and branch runs reuse those caches.
An interrupted run resumes completed folds by default. Use `--no-resume` to
retrain folds with the same settings. For different training settings, choose
a different `--output-dir` so results remain distinct. Use `--help` for all
settings.

## Evaluation and outputs

The script uses ten stratified subject folds. Within each outer training split,
it reserves one ninth of subjects for validation, giving approximately
80% training, 10% validation, and 10% test subjects. PCA is fitted separately
for each subject using that subject's intention trials. In each fold, the
cross-subject PC alignment reference is fitted using training subjects only.
Each trial is baseline normalized after alignment. Training bootstrap draws
are regenerated every epoch; validation and test draws are fixed for that fold.

At each epoch the terminal shows training loss, subject-level validation
negative log likelihood, ROC-AUC, balanced accuracy, and elapsed time. Early
stopping uses validation negative log likelihood. Subject predictions average
the probabilities of 100 test bootstrap examples by default.

The output directory contains `subject_predictions.csv`, `fold_metrics.csv`,
one epoch-history CSV and best model checkpoint per fold, the fold alignment
references, `run_config.json`, and `summary.json`. It also reports a regularized
linear baseline fitted to the same two condition responses. A completed fold
can be reused after an interruption. The selected-scout cache contains raw
PC scores, PCA loadings, valid-trial masks, and explained variance.

Bootstrap examples from a subject are correlated; performance and group
splits are always evaluated at the subject level. The script reports scores
for prespecified scouts. If many scouts are tried and the highest scorer is
selected, that selection needs another held-out evaluation for an unbiased
estimate of the selected-scout procedure.
