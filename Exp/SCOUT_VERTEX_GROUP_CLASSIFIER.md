# Vertex-level scout group classifier

`scout_vertex_group_classifier.py` trains a binary classifier within each requested,
zero-based DKT scout. It accepts the same scout and comparison style as the PC
classifier. The default comparisons run in order: Control vs Depressed + Suicidal,
then Depressed vs Suicidal. The default representation is **incongruent minus
congruent** intention response.

## Commands

Synthetic end-to-end smoke test (does not open study EEG):

```bash
python3 scout_vertex_group_classifier.py --smoke-test --scout 35
```

Full run for two scouts, which you start yourself:

```bash
python3 scout_vertex_group_classifier.py --scout 35 52
```

Comma-separated indices also work: `--scout 35,52`. Other inputs are selected with
`--representation congruent`, `incongruent`, `contrast`, `fusion`, or `all`.
`fusion` applies one shared encoder to congruent and incongruent inputs and fuses
their embeddings and difference. `all` uses three separate encoders for congruent,
incongruent, and their exact difference. For example:

```bash
python3 scout_vertex_group_classifier.py --scout 35 52 --representation fusion
```

The default `--pair` runs both binary comparisons. `--pair depressed-suicidal`
runs only that one; the other available names match `scout_group_classifier.py`.
`--device auto` selects CUDA, then Apple MPS, then CPU. `--help` lists all options.

## Data and model

The intention trial index file uses generic `Sentiment` keys: `positive` means
congruent and `negative` means incongruent in this file. For each subject, the
script selects 40 trials per condition, removes invalid trials, and keeps
-200 to just before 800 ms from the original 250-Hz EEG. Each original trial
is Z-scored using its -200 to <0 ms baseline, then cropped to 0 to <800 ms
before bootstrap averaging and model input. The default model input is thus
`vertices × 200 time samples` at 250 Hz. A lower `--resample-rate` remains
available if needed.

Within each outer fold, the 3×3 source covariance at each vertex is averaged
across **training subjects only**, after normalizing each subject's covariance by
its trace. Its leading eigenvector is the fold's X/Y/Z projection for that vertex.
The direction has a fixed sign convention and is reused for validation and test.
The three source-kernel rows are combined into one row before EEG projection.
After baseline normalization and cropping, each **training** pseudotrial averages
16 original trials sampled with replacement. There are 200 training pseudotrials
per subject by default. The contrast uses the same congruent and incongruent
pseudotrial draws as the condition branches. Validation and test each use one
deterministic average of all available valid trials per condition per subject.
For these subjects, the contrast is the difference between the two condition
averages.

The model accepts `vertices × time`, uses a learned `vertices → 64` spatial
bottleneck, two temporal residual CNN blocks, a two-layer/four-head Transformer,
temporal average pooling, and a binary head. Each scout has its own model.

## Evaluation and output

Five outer folds are stratified by the original three diagnostic groups. Within
each training split, 12.5% of subjects are reserved for validation (about 10% of
the full dataset). Subjects never cross training, validation, and test splits.
The 200 training pseudotrials are generated once per fold and reused across
epochs; this avoids repeated large source projection. Early stopping selects
the epoch with lowest **subject-level validation log loss**, computed from one
full-average prediction per validation subject. The terminal reports training
pseudotrial loss, accuracy and balanced accuracy; subject-level validation log
loss, AUC and balanced accuracy; epoch time; and fold test metrics. Each test
subject receives one full-average prediction. The final AUC is calculated from
one out-of-fold score per subject.

Results and per-fold checkpoints are saved under
`Results/scout_vertex_classifier_v3/scout_<index>/<pair>/<representation>/`.
Per-fold X/Y/Z PCA directions and explained-variance ratios are saved. A matching
completed fold resumes by default. Different settings require a different
`--output-dir`.

The selected-scout input cache is stored under `Results/scout_vertex_cache_v2/`.
Temporary pseudotrials are written on disk in float16 **after** baseline
normalization and deleted after each fold. A large scout such as scout 52 needs
roughly 8 GiB of temporary space for the default single-branch setting, and
roughly three times that for `--representation all`. Use `--scratch-dtype float32`
to avoid half-precision storage at approximately twice the disk cost.

The model assumes the same atlas vertex index refers to the same anatomical
location across subjects, as specified for this analysis. Source vertices are
correlated reconstructions of sensor EEG, not independent recordings. Trying many
scouts and reporting only the highest test AUC requires a separate held-out
evaluation of the selected-scout procedure.
