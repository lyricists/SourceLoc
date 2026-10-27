"""Estimate group XYZ source-response maps for four sentence conditions.

The evoked method averages signed EEG trials within each
subject and condition, projects through the inverse kernel, then takes the XYZ
norm before an equal-weight group average. The optional baseline-zscore method
is an alternative for an unstandardized inverse kernel: it z-scores signed
source components over the prestimulus baseline before the XYZ norm.

"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import mat73
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent.parent  # SourceLoc; this script lives in Exp/
DATA = ROOT.parent.parent / "Data"
PREPROCESSED = DATA / "Preprocessed data"
BEHAVIOR = DATA / "Behavior"

CONDITION_NAMES = (
    "congruent_intention",
    "incongruent_intention",
    "positive_neutral_noBio",
    "negative_noBio",
)
GROUP_CODES = {"c": 1, "d": 2, "s": 3}
VERSION = 5
RESPONSE_MODES = ("evoked", "trial-magnitude")
NORMALIZATIONS = ("kernel", "baseline-zscore")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eeg", type=Path, default=PREPROCESSED / "Data_sen_bl.pkl")
    parser.add_argument("--tmatrix", type=Path, default=PREPROCESSED / "tMatrix.pkl")
    parser.add_argument("--atlas", type=Path, default=PREPROCESSED / "DKT_scout.mat")
    parser.add_argument(
        "--subject-index", type=Path, default=BEHAVIOR / "subject_index.mat"
    )
    parser.add_argument(
        "--intention-indices",
        type=Path,
        default=BEHAVIOR / "senIdx_congruency_Intention.pkl",
    )
    parser.add_argument(
        "--sentiment-indices",
        type=Path,
        default=BEHAVIOR / "senIdx_sentiment_noBio.pkl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination; default separates output by response mode and normalization.",
    )
    parser.add_argument("--response-mode", choices=RESPONSE_MODES, default="evoked")
    parser.add_argument(
        "--normalization",
        choices=NORMALIZATIONS,
        default="kernel",
        help="'kernel' keeps the inverse kernel's scaling (recommended for sLORETA); 'baseline-zscore' is an alternative for unstandardized kernels.",
    )
    parser.add_argument(
        "--save-subject-maps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save each subject's four maps for quality checks and statistics (about 9.2 GiB with default data).",
    )
    parser.add_argument("--sampling-rate", type=float, default=250.0)
    parser.add_argument("--epoch-start-ms", type=float, default=-200.0)
    parser.add_argument("--baseline-start-ms", type=float, default=-200.0)
    parser.add_argument("--baseline-end-ms", type=float, default=-4.0)
    parser.add_argument(
        "--analysis-end-ms",
        type=float,
        default=1000.0,
        help="Upper time boundary in ms; the last sample may precede it by one sample period.",
    )
    parser.add_argument("--vertex-chunk", type=int, default=128)
    parser.add_argument("--blas-threads", type=int, default=1)
    return parser.parse_args()


def file_signature(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def load_pickle(path: Path):
    with path.open("rb") as file:
        return pickle.load(file)


def load_eeg(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path, mmap_mode="r")
    if path.suffix == ".pkl":
        return np.asarray(load_pickle(path))
    raise ValueError("EEG must be a .pkl array or a .npy array.")


def atomic_save_array(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as file:
        np.save(file, value)
    os.replace(temporary, path)


def atomic_save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2)
        file.write("\n")
    os.replace(temporary, path)


def read_conditions(
    intention_path: Path, sentiment_path: Path, n_subjects: int, n_trials: int
):
    intention = load_pickle(intention_path)["Sentiment"]
    sentiment = load_pickle(sentiment_path)["Sentiment"]
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


def run(args: argparse.Namespace) -> None:
    if args.vertex_chunk < 1 or args.blas_threads < 1 or args.sampling_rate <= 0:
        raise ValueError(
            "vertex-chunk, blas-threads, and sampling-rate must be positive."
        )
    if args.response_mode not in RESPONSE_MODES:
        raise ValueError(f"Unknown response mode: {args.response_mode}.")
    if args.normalization not in NORMALIZATIONS:
        raise ValueError(f"Unknown normalization: {args.normalization}.")
    if args.response_mode == "trial-magnitude" and args.normalization != "kernel":
        raise ValueError("Baseline z-scoring is only defined for evoked responses.")
    raw_groups = np.asarray(
        mat73.loadmat(str(args.subject_index))["subject_index"]
    ).reshape(-1)
    if (
        not np.isfinite(raw_groups).all()
        or not np.equal(raw_groups, np.floor(raw_groups)).all()
    ):
        raise ValueError("subject_index must contain finite integer group codes.")
    groups = raw_groups.astype(int)
    if not np.isin(groups, list(GROUP_CODES.values())).all():
        raise ValueError("subject_index must contain only group codes 1, 2, and 3.")
    atlas = mat73.loadmat(str(args.atlas))["DKT_scout"]
    atlas_max_vertex = max(int(np.max(vertices)) for vertices in atlas["Vertices"])

    print(
        "Loading EEG and inverse kernels; this may take several minutes...", flush=True
    )
    eeg = load_eeg(args.eeg)
    t_matrix = load_pickle(args.tmatrix)
    kernels = t_matrix["kernel"]
    good_channels = t_matrix["GoodChannel"]
    if eeg.ndim != 4:
        raise ValueError("EEG must have channel x time x trial x subject axes.")
    n_channels, n_original_times, n_trials, n_subjects = eeg.shape
    if not (len(kernels) == len(good_channels) == groups.size == n_subjects):
        raise ValueError("EEG, kernel, GoodChannel, and subject_index counts differ.")
    n_kernel_rows = np.asarray(kernels[0]).shape[0]
    if n_kernel_rows % 3 or n_kernel_rows // 3 < atlas_max_vertex:
        raise ValueError("Kernel XYZ rows do not cover the atlas vertex indices.")
    n_vertices = n_kernel_rows // 3
    sample_period_ms = 1000 / args.sampling_rate
    times_ms = args.epoch_start_ms + np.arange(n_original_times) * sample_period_ms
    epoch_end_exclusive_ms = float(times_ms[-1] + sample_period_ms)
    time_indices = np.flatnonzero(times_ms <= args.analysis_end_ms + 1e-9)
    # An epoch sampled on [-200, 1000) at 250 Hz ends at 996 ms. Accept
    # 1000 ms as its exclusive boundary, but reject requests beyond it.
    if (
        time_indices.size == 0
        or args.analysis_end_ms > times_ms[-1] + sample_period_ms + 1e-9
    ):
        raise ValueError("Analysis end time lies outside the EEG epoch.")
    times_ms = times_ms[time_indices]
    baseline_indices = np.flatnonzero(
        (times_ms >= args.baseline_start_ms - 1e-9)
        & (times_ms <= args.baseline_end_ms + 1e-9)
    )
    poststimulus_indices = np.flatnonzero(times_ms >= -1e-9)
    if (
        args.baseline_start_ms < times_ms[0] - 1e-9
        or args.baseline_end_ms >= 0
        or baseline_indices.size < 1
        or poststimulus_indices.size < 1
    ):
        raise ValueError(
            "The EEG epoch must contain the requested prestimulus baseline "
            "and at least one poststimulus sample."
        )
    if args.normalization == "baseline-zscore" and baseline_indices.size < 2:
        raise ValueError("Baseline z-scoring requires at least two samples.")
    conditions = read_conditions(
        args.intention_indices, args.sentiment_indices, n_subjects, n_trials
    )

    metadata = {
        "version": VERSION,
        "inputs": {
            name: file_signature(getattr(args, name))
            for name in (
                "eeg",
                "tmatrix",
                "atlas",
                "subject_index",
                "intention_indices",
                "sentiment_indices",
            )
        },
        "sampling_rate": args.sampling_rate,
        "epoch_start_ms": args.epoch_start_ms,
        "epoch_end_exclusive_ms": epoch_end_exclusive_ms,
        "analysis_end_ms": args.analysis_end_ms,
        "analysis_end_ms_actual": float(times_ms[-1]),
        "response_mode": args.response_mode,
        "normalization": args.normalization,
        "baseline_window_ms": (
            [args.baseline_start_ms, args.baseline_end_ms]
            if args.normalization == "baseline-zscore"
            else None
        ),
        "qc_baseline_window_ms": [args.baseline_start_ms, args.baseline_end_ms],
        "save_subject_maps": args.save_subject_maps,
        "condition_names": CONDITION_NAMES,
        "n_subjects": n_subjects,
        "n_vertices": n_vertices,
        "n_times": time_indices.size,
    }
    output_dir = args.output_dir or (
        ROOT
        / "Results"
        / "source_response_norm"
        / f"{args.response_mode}_{args.normalization}"
    )
    print(
        f"EEG {eeg.shape}; {n_vertices} vertices; {time_indices.size} time points; "
        f"groups C={np.sum(groups == 1)}, D={np.sum(groups == 2)}, S={np.sum(groups == 3)}",
        flush=True,
    )
    # Float64 group sums preserve precision while keeping only one subject map
    # and the three group accumulators in memory at any time.
    sums = {
        name: np.zeros(
            (len(CONDITION_NAMES), n_vertices, time_indices.size), dtype=np.float64
        )
        for name in GROUP_CODES
    }
    counts = {name: int(np.sum(groups == code)) for name, code in GROUP_CODES.items()}
    if any(count == 0 for count in counts.values()):
        raise ValueError("Every C, D, and S group must contain at least one subject.")
    code_to_name = {code: name for name, code in GROUP_CODES.items()}
    subject_qc = []
    for subject in range(n_subjects):
        kernel = np.asarray(kernels[subject])
        if kernel.shape[0] != n_kernel_rows:
            raise ValueError(
                f"Subject {subject}: inverse kernel has a different vertex count."
            )
        if args.response_mode == "evoked":
            subject_map = estimate_evoked_subject(
                eeg[:, :, :, subject],
                kernel,
                good_channels[subject],
                time_indices,
                baseline_indices,
                conditions[subject],
                args.normalization,
                args.vertex_chunk,
                args.blas_threads,
            )
        else:
            subject_map = estimate_subject(
                eeg[:, :, :, subject],
                kernel,
                good_channels[subject],
                time_indices,
                conditions[subject],
                args.vertex_chunk,
                args.blas_threads,
            )
        sums[code_to_name[groups[subject]]] += subject_map
        subject_file = None
        if args.save_subject_maps:
            subject_file = f"subjects/subject_{subject:03d}.npy"
            atomic_save_array(output_dir / subject_file, subject_map)
        baseline_means = np.mean(
            subject_map[:, :, baseline_indices], axis=(1, 2), dtype=np.float64
        )
        poststimulus_means = np.mean(
            subject_map[:, :, poststimulus_indices], axis=(1, 2), dtype=np.float64
        )
        subject_qc.append(
            {
                "subject_index_0based": subject,
                "group": code_to_name[groups[subject]],
                "subject_file": subject_file,
                "baseline_mean_by_condition": {
                    name: float(value)
                    for name, value in zip(CONDITION_NAMES, baseline_means)
                },
                "poststimulus_mean_by_condition": {
                    name: float(value)
                    for name, value in zip(CONDITION_NAMES, poststimulus_means)
                },
            }
        )
        print(f"Subject {subject + 1}/{n_subjects}: accumulated", flush=True)
        del subject_map

    del eeg, t_matrix, kernels, good_channels
    filenames = {}
    for name in ("c", "d", "s", "d+s"):
        numerator = sums["d"] + sums["s"] if name == "d+s" else sums[name]
        denominator = counts["d"] + counts["s"] if name == "d+s" else counts[name]
        filenames[name] = {}
        for condition, condition_name in enumerate(CONDITION_NAMES):
            filename = f"{name}_{condition_name}.npy"
            atomic_save_array(
                output_dir / filename,
                (numerator[condition] / denominator).astype(np.float32),
            )
            filenames[name][condition_name] = filename

    atomic_save_array(output_dir / "times_ms.npy", times_ms)
    atomic_save_array(
        output_dir / "vertex_indices_1based.npy",
        np.arange(1, n_vertices + 1, dtype=np.int32),
    )
    atomic_save_json(output_dir / "subject_qc.json", {"subjects": subject_qc})
    atomic_save_json(
        output_dir / "metadata.json",
        {
            **metadata,
            "method": (
                "Signed trial mean, inverse projection, optional XYZ-component baseline "
                "z-score, XYZ norm, then equal-weight subject mean"
                if args.response_mode == "evoked"
                else "XYZ norm per trial, then trial mean per subject, then equal-weight subject mean"
            ),
            "source_baseline_correction": (
                "componentwise z-score before XYZ norm"
                if args.normalization == "baseline-zscore"
                else "none"
            ),
            "condition_files": filenames,
            "group_subject_counts": {**counts, "d+s": counts["d"] + counts["s"]},
            "condition_trial_counts_by_subject": [
                {
                    name: int(indices.size)
                    for name, indices in zip(CONDITION_NAMES, subject_conditions)
                }
                for subject_conditions in conditions
            ],
            "times_file": "times_ms.npy",
            "vertex_indices_file": "vertex_indices_1based.npy",
            "subject_qc_file": "subject_qc.json",
            "subject_file_axes": "condition x vertex x time",
            "source_space_alignment_note": (
                "Vertex count and atlas range were checked; common anatomical "
                "vertex correspondence must be verified from the source models."
            ),
            "inverse_method_note": (
                "The pickle contains kernels and GoodChannel but no inverse-method metadata."
            ),
        },
    )
    print(f"Saved 16 vertex x time group maps to {output_dir}", flush=True)


if __name__ == "__main__":
    run(parse_args())
