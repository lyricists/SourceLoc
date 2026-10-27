"""Source-space response maps and their masking by significant scout decoding.

Response maps: per subject and condition, the XYZ-norm source magnitude per vertex
(evoked: signed trial mean, inverse projection, optional baseline z-score, XYZ norm;
trial-magnitude: XYZ norm per trial, then trial mean). Masking keeps a scout's vertices
only at that scout's significant decoding time points (main_sentence_stats.py output).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from utils import io

CONDITION_NAMES = (
    "congruent_intention",
    "incongruent_intention",
    "positive_neutral_noBio",
    "negative_noBio",
)
GROUP_CODES = {"c": 1, "d": 2, "s": 3}
RESPONSE_MODES = ("evoked", "trial-magnitude")
NORMALIZATIONS = ("kernel", "baseline-zscore")
# Source-response group prefix -> CBP group key.
GROUPS = {"c": "control", "d": "depressed", "s": "suicidal"}
# CBP task -> source-response conditions decoded by that task.
TASK_CONDITIONS = {
    "intention": ("congruent_intention", "incongruent_intention"),
    "sentiment": ("positive_neutral_noBio", "negative_noBio"),
}


def read_conditions(
    intention_path: Path, sentiment_path: Path, n_subjects: int, n_trials: int
):
    intention = io.load_pickle(intention_path)["Sentiment"]
    sentiment = io.load_pickle(sentiment_path)["Sentiment"]
    if len(intention) != n_subjects or len(sentiment) != n_subjects:
        raise ValueError("Condition index files do not match the EEG subject count.")

    result = []
    for subject in range(n_subjects):
        source = (
            intention[subject]["positive"],
            intention[subject]["negative"],
            sentiment[subject]["positive"],
            sentiment[subject]["negative"],
        )
        indices = []
        for name, values in zip(CONDITION_NAMES, source):
            trial_indices = np.asarray(values, dtype=np.intp).reshape(-1)
            if trial_indices.size == 0:
                raise ValueError(f"Subject {subject}: {name} has no trials.")
            if np.any((trial_indices < 0) | (trial_indices >= n_trials)):
                raise IndexError(
                    f"Subject {subject}: {name} has an out-of-range trial index."
                )
            if np.unique(trial_indices).size != trial_indices.size:
                raise ValueError(
                    f"Subject {subject}: {name} contains duplicate trials."
                )
            indices.append(trial_indices)
        if np.intersect1d(indices[0], indices[1]).size:
            raise ValueError(f"Subject {subject}: intention conditions overlap.")
        if np.intersect1d(indices[2], indices[3]).size:
            raise ValueError(f"Subject {subject}: sentiment conditions overlap.")
        result.append(tuple(indices))
    return result


def estimate_subject(
    eeg_subject: np.ndarray,
    kernel: np.ndarray,
    good_channels_matlab: np.ndarray,
    time_indices: np.ndarray,
    condition_indices: tuple[np.ndarray, ...],
    vertex_chunk: int,
    blas_threads: int,
) -> np.ndarray:
    """Return condition x vertex x time; norm is taken before trial means."""
    if eeg_subject.ndim != 3 or kernel.ndim != 2:
        raise ValueError("Expected channel x time x trial EEG and a 2D inverse kernel.")
    if kernel.shape[0] % 3:
        raise ValueError("Inverse-kernel rows must contain three XYZ rows per vertex.")
    n_channels, _, n_trials = eeg_subject.shape
    good = np.asarray(good_channels_matlab, dtype=np.intp).reshape(-1) - 1
    if good.size == 0 or np.unique(good).size != good.size:
        raise ValueError("GoodChannel must contain unique MATLAB channel indices.")
    if np.any((good < 0) | (good >= n_channels)):
        raise IndexError("GoodChannel contains an out-of-range channel index.")

    selected = eeg_subject[:, time_indices, :]
    if kernel.shape[1] == good.size:
        selected = selected[good, :, :]
    elif kernel.shape[1] != n_channels:
        raise ValueError(
            "Inverse-kernel columns match neither GoodChannel nor all EEG channels."
        )
    n_times = time_indices.size
    sensor_2d = np.ascontiguousarray(selected, dtype=np.float32).reshape(
        -1, n_times * n_trials
    )
    del selected

    n_vertices = kernel.shape[0] // 3
    result = np.empty((len(CONDITION_NAMES), n_vertices, n_times), dtype=np.float32)
    with threadpool_limits(limits=blas_threads):
        for start in range(0, n_vertices, vertex_chunk):
            stop = min(start + vertex_chunk, n_vertices)
            xyz = (
                np.asarray(kernel[3 * start : 3 * stop, :], dtype=np.float32)
                @ sensor_2d
            )
            xyz = xyz.reshape(stop - start, 3, n_times, n_trials)
            magnitude = np.einsum("vctn,vctn->vtn", xyz, xyz, optimize=True)
            np.sqrt(magnitude, out=magnitude)
            if not np.isfinite(magnitude).all():
                raise ValueError(
                    f"Non-finite source magnitude at vertices {start}:{stop}."
                )
            for condition, trials in enumerate(condition_indices):
                result[condition, start:stop] = np.take(magnitude, trials, axis=2).mean(
                    axis=2, dtype=np.float64
                )
    return result


def estimate_evoked_subject(
    eeg_subject: np.ndarray,
    kernel: np.ndarray,
    good_channels_matlab: np.ndarray,
    time_indices: np.ndarray,
    baseline_indices: np.ndarray,
    condition_indices: tuple[np.ndarray, ...],
    normalization: str,
    vertex_chunk: int,
    blas_threads: int,
) -> np.ndarray:
    """Return condition x vertex x time after signed trial averaging."""
    if eeg_subject.ndim != 3 or kernel.ndim != 2:
        raise ValueError("Expected channel x time x trial EEG and a 2D inverse kernel.")
    if kernel.shape[0] % 3:
        raise ValueError("Inverse-kernel rows must contain three XYZ rows per vertex.")
    if normalization not in NORMALIZATIONS:
        raise ValueError(f"Unknown normalization: {normalization}.")

    n_channels = eeg_subject.shape[0]
    good = np.asarray(good_channels_matlab, dtype=np.intp).reshape(-1) - 1
    if good.size == 0 or np.unique(good).size != good.size:
        raise ValueError("GoodChannel must contain unique MATLAB channel indices.")
    if np.any((good < 0) | (good >= n_channels)):
        raise IndexError("GoodChannel contains an out-of-range channel index.")

    selected = eeg_subject[:, time_indices, :]
    if kernel.shape[1] == good.size:
        selected = selected[good, :, :]
    elif kernel.shape[1] != n_channels:
        raise ValueError(
            "Inverse-kernel columns match neither GoodChannel nor all EEG channels."
        )

    n_times = time_indices.size
    n_conditions = len(CONDITION_NAMES)
    sensor_means = np.stack(
        [
            np.take(selected, trials, axis=2).mean(axis=2, dtype=np.float64)
            for trials in condition_indices
        ]
    )
    sensor_2d = np.ascontiguousarray(sensor_means.transpose(1, 0, 2)).reshape(
        selected.shape[0], n_conditions * n_times
    )
    del selected, sensor_means

    n_vertices = kernel.shape[0] // 3
    result = np.empty((n_conditions, n_vertices, n_times), dtype=np.float32)
    with threadpool_limits(limits=blas_threads):
        for start in range(0, n_vertices, vertex_chunk):
            stop = min(start + vertex_chunk, n_vertices)
            xyz = (
                np.asarray(kernel[3 * start : 3 * stop, :], dtype=np.float64)
                @ sensor_2d
            ).reshape(stop - start, 3, n_conditions, n_times)
            if normalization == "baseline-zscore":
                baseline = xyz[..., baseline_indices]
                baseline_mean = baseline.mean(axis=-1, keepdims=True)
                baseline_std = baseline.std(axis=-1, keepdims=True)
                if not np.isfinite(baseline_std).all() or np.any(baseline_std <= 0):
                    raise ValueError(
                        f"Zero or non-finite source baseline SD at vertices {start}:{stop}."
                    )
                xyz -= baseline_mean
                xyz /= baseline_std
            magnitude = np.linalg.norm(xyz, axis=1)
            if not np.isfinite(magnitude).all():
                raise ValueError(
                    f"Non-finite source magnitude at vertices {start}:{stop}."
                )
            result[:, start:stop] = magnitude.transpose(1, 0, 2)
    return result


def find_group_statistics(cbp_dir: Path, group_key: str) -> Path:
    candidates = sorted(
        (cbp_dir / "spatiotemporal_statistics").glob(f"group_{group_key}_scouts_*.pkl")
    )
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one '{group_key}' statistics pickle in "
            f"{cbp_dir / 'spatiotemporal_statistics'}; found {len(candidates)}."
        )
    return candidates[0]


def load_scout_mask(
    path: Path, times_ms: np.ndarray, atlas_labels: list[str]
) -> tuple[np.ndarray, dict]:
    """Return a time x atlas-scout significance mask from one CBP result."""
    statistics = io.load_pickle(path)
    test = statistics.get("test", {})
    if (tuple(test.get("group_family", ())) != ("control", "depressed", "suicidal")
            or test.get("group_correction_factor") != 3):
        raise ValueError(f"{path.name}: expected significance corrected across all three diagnostic groups.")
    cbp_times = np.asarray(statistics["times_ms"], dtype=np.float64)
    if cbp_times.shape != times_ms.shape or not np.allclose(cbp_times, times_ms):
        raise ValueError(f"{path.name}: CBP time vector differs from the source maps.")

    scout_indices = np.asarray(statistics["scout_indices"], dtype=np.intp)
    significant = np.asarray(statistics["significant_mask"], dtype=bool)
    if significant.shape != (times_ms.size, scout_indices.size):
        raise ValueError(
            f"{path.name}: significant_mask has shape {significant.shape}; expected "
            f"{(times_ms.size, scout_indices.size)}."
        )
    if np.any((scout_indices < 0) | (scout_indices >= len(atlas_labels))):
        raise IndexError(f"{path.name}: scout index outside the atlas.")
    cbp_labels = [
        io.clean_label(label, int(scout))
        for label, scout in zip(statistics["scout_labels"], scout_indices)
    ]
    expected_labels = [atlas_labels[scout] for scout in scout_indices]
    if cbp_labels != expected_labels:
        raise ValueError(f"{path.name}: scout labels do not match the atlas.")

    # Scouts the CBP run did not test stay non-significant.
    mask = np.zeros((times_ms.size, len(atlas_labels)), dtype=bool)
    mask[:, scout_indices] = significant
    info = {
        "statistics": io.file_signature(path),
        "cluster_alpha": float(test["cluster_alpha"]),
        "group_correction_factor": int(test["group_correction_factor"]),
        "n_subjects": int(statistics["n_subjects"]),
        "tested_scout_indices": scout_indices.tolist(),
        "n_significant_scouts": int(mask.any(axis=0).sum()),
        "n_significant_scout_time_points": int(mask.sum()),
    }
    return mask, info


def mask_response(
    response: np.ndarray, scout_vertices: list[np.ndarray], scout_mask: np.ndarray
) -> np.ndarray:
    masked = np.zeros_like(response)
    for scout, vertices in enumerate(scout_vertices):
        times = np.flatnonzero(scout_mask[:, scout])
        if times.size:
            masked[np.ix_(vertices, times)] = response[np.ix_(vertices, times)]
    return masked
