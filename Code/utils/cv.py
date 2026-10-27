"""Cross-validation splits and bootstrap trial averaging shared by the subject-level analyses."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold


def outer_splits(originals: np.ndarray, folds: int, repeats: int, seed: int) -> list[list]:
    """Repeated stratified subject splits (stratified by the original group codes).

    The same splits are used by every subject-level analysis (single-scout screen,
    multi-scout classifier, time-resolved group decoding), so their predictions pair
    subject by subject.
    """
    return [list(StratifiedKFold(folds, shuffle=True, random_state=seed + 1009 * r)
                 .split(originals, originals)) for r in range(repeats)]


def bootstrap_weights(rng: np.random.Generator, n: int, sizes: np.ndarray) -> np.ndarray:
    """One row of trial weights per bootstrap average; row k averages ``sizes[k]`` draws."""
    counts = np.stack([rng.multinomial(m, np.full(n, 1 / n)) for m in sizes]).astype(np.float32)
    return counts / sizes[:, None].astype(np.float32)
