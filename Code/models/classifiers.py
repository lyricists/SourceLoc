"""Linear classifiers: the SVM of sentence decoding and the closed-form ridge of group decoding."""

from __future__ import annotations

import numpy as np
from sklearn.svm import SVC


def linear_svm() -> SVC:
    """Linear SVM used at every time point of sentence decoding."""
    return SVC(kernel="linear", C=1, tol=1e-3, class_weight=None, cache_size=50)


def ridge_scores(x_train: np.ndarray, x_test: np.ndarray, labels_train: np.ndarray, alpha: float) -> np.ndarray:
    """Ridge-regression scores for many labelings at once, at many time points.

    x_train / x_test: time x subject x feature (standardized); labels_train:
    subject x labeling. The penalty is ``alpha`` x number of training subjects.
    Predictions are linear in the labels, so every labeling (e.g. label
    permutations) costs one extra matrix column. Returns time x test x labeling.
    """
    gram = x_train.transpose(0, 2, 1) @ x_train + alpha * x_train.shape[1] * np.eye(x_train.shape[2])
    hat = x_test @ np.linalg.solve(gram, x_train.transpose(0, 2, 1))  # time x test x train
    mean = labels_train.mean(0)
    return hat @ (labels_train - mean) + mean
