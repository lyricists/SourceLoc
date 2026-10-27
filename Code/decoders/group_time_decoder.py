"""Time-resolved group decoding within one scout: a ridge classifier per time point.

Subjects are the examples. A subject's feature at time t is its incongruent - congruent
average projected to the scout's PCs (the fold-specific projection cache). Features are
standardized on the training subjects. Ridge predictions are linear in the labels, so
the observed labels and every label permutation are scored with one matrix product.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

from models.classifiers import ridge_scores
from utils import source


def subject_contrasts(cfg: dict, meta: dict, selected: np.ndarray) -> np.ndarray:
    """Subject x channel x time: incongruent minus congruent average of baseline-corrected trials."""
    sets = source.load_trial_sets(Path(cfg["trial_index"]), cfg["category"], len(meta["groups"]))
    baseline = np.asarray(meta["times_ms"]) < source.BASELINE_END_MS
    out = []
    for s in selected:
        trials = source.load_subject(Path(cfg["cache_dir"]), s, sets[s], np.array([0]), baseline)["trials"]
        out.append(trials[1].mean(0, dtype=np.float64) - trials[0].mean(0, dtype=np.float64))
    return np.stack(out)


def decode_scout(cfg: dict, scout: int, contrasts: np.ndarray, labels: np.ndarray, splits: list,
                 alpha: float, blas_threads: int) -> tuple[int, np.ndarray]:
    """Repeat x time x labeling AUC; labeling 0 is the observed labels, the rest permutations.

    AUC is pooled over each repeat's out-of-fold scores.
    """
    with threadpool_limits(blas_threads):
        n, n_times = contrasts.shape[0], contrasts.shape[2]
        n1 = int(labels[:, 0].sum())
        out = np.empty((len(splits), n_times, labels.shape[1]), dtype=np.float32)
        for r, folds in enumerate(splits):
            scores = np.empty((n, n_times, labels.shape[1]), dtype=np.float32)
            for f, (train, test) in enumerate(folds):
                m = np.load(source.projection_path(cfg, scout, r, f)).astype(np.float64)
                x = np.einsum("npc,nct->tnp", m, contrasts, optimize=True)  # time x subject x PC
                mean, sd = x[:, train].mean(1, keepdims=True), x[:, train].std(1, keepdims=True)
                sd[sd == 0] = 1  # zero-padded PCs stay zero
                x_train, x_test = (x[:, train] - mean) / sd, (x[:, test] - mean) / sd
                scores[test] = ridge_scores(x_train, x_test, labels[train], alpha).transpose(1, 0, 2)
            ranks = rankdata(scores, axis=0)
            out[r] = (np.einsum("itk,ik->tk", ranks, labels) - n1 * (n1 + 1) / 2) / (n1 * (n - n1))
    return scout, out
