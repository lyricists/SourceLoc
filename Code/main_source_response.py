"""Estimate group XYZ source-response maps for four sentence conditions.

The evoked method averages signed EEG trials within each subject and condition,
projects through the inverse kernel, then takes the XYZ norm before an equal-weight
group average. By default (baseline-zscore) the signed source components are z-scored
over the prestimulus baseline before the XYZ norm; --normalization kernel keeps the
inverse kernel's own scaling. Writes group maps (c, d, s, d+s) per condition, times,
metadata.json, and optionally every subject's maps; the masking step is
main_source_significant.py.

The maps are response magnitudes (a norm, so never below zero): they show where and
when the brain responds, not what separates the conditions. The noise floor of a norm
depends on the number of averaged trials, which differs between conditions (e.g.
sentiment 108 vs 132), so compare conditions with care.

Example (run from Code/):
    python3 main_source_response.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mat73
import numpy as np

from utils import io, paths
from utils import source_response as sr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eeg", type=Path, default=paths.EEG)
    parser.add_argument("--tmatrix", type=Path, default=paths.TMATRIX)
    parser.add_argument("--atlas", type=Path, default=paths.ATLAS)
    parser.add_argument(
        "--subject-index", type=Path, default=paths.SUBJECT_INDEX
    )
    parser.add_argument(
        "--intention-indices",
        type=Path,
        default=paths.INTENTION_INDEX,
    )
    parser.add_argument(
        "--sentiment-indices",
        type=Path,
        default=paths.SENTIMENT_INDEX,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination; default separates output by response mode and normalization.",
    )
    parser.add_argument("--response-mode", choices=sr.RESPONSE_MODES, default="evoked")
    parser.add_argument(
        "--normalization",
        choices=sr.NORMALIZATIONS,
        default=None,
        help="'baseline-zscore' (default for evoked, as in run_source_response.sh and the default input of "
             "main_source_significant.py) z-scores each XYZ component over the baseline; 'kernel' keeps the inverse "
             "kernel's own scaling (suited to noise-normalized kernels such as sLORETA; the only option for "
             "trial-magnitude).",
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
    args = parser.parse_args()
    if args.normalization is None:
        args.normalization = "baseline-zscore" if args.response_mode == "evoked" else "kernel"
    return args


def load_eeg(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path, mmap_mode="r")
    if path.suffix == ".pkl":
        return np.asarray(io.load_pickle(path))
    raise ValueError("EEG must be a .pkl array or a .npy array.")


def run(args: argparse.Namespace) -> None:
    if args.vertex_chunk < 1 or args.blas_threads < 1 or args.sampling_rate <= 0:
        raise ValueError(
            "vertex-chunk, blas-threads, and sampling-rate must be positive."
        )
    if args.response_mode not in sr.RESPONSE_MODES:
        raise ValueError(f"Unknown response mode: {args.response_mode}.")
    if args.normalization not in sr.NORMALIZATIONS:
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
    if not np.isin(groups, list(sr.GROUP_CODES.values())).all():
        raise ValueError("subject_index must contain only group codes 1, 2, and 3.")
    atlas = mat73.loadmat(str(args.atlas))["DKT_scout"]
    atlas_max_vertex = max(int(np.max(vertices)) for vertices in atlas["Vertices"])

    print(
        "Loading EEG and inverse kernels; this may take several minutes...", flush=True
    )
    eeg = load_eeg(args.eeg)
    t_matrix = io.load_pickle(args.tmatrix)
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
    conditions = sr.read_conditions(
        args.intention_indices, args.sentiment_indices, n_subjects, n_trials
    )

    metadata = {
        "inputs": {
            name: io.file_signature(getattr(args, name))
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
        "condition_names": sr.CONDITION_NAMES,
        "n_subjects": n_subjects,
        "n_vertices": n_vertices,
        "n_times": time_indices.size,
    }
    output_dir = args.output_dir or (
        paths.RESULTS / "source_response_norm" / f"{args.response_mode}_{args.normalization}"
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
            (len(sr.CONDITION_NAMES), n_vertices, time_indices.size), dtype=np.float64
        )
        for name in sr.GROUP_CODES
    }
    counts = {name: int(np.sum(groups == code)) for name, code in sr.GROUP_CODES.items()}
    if any(count == 0 for count in counts.values()):
        raise ValueError("Every C, D, and S group must contain at least one subject.")
    code_to_name = {code: name for name, code in sr.GROUP_CODES.items()}
    subject_qc = []
    for subject in range(n_subjects):
        kernel = np.asarray(kernels[subject])
        if kernel.shape[0] != n_kernel_rows:
            raise ValueError(
                f"Subject {subject}: inverse kernel has a different vertex count."
            )
        if args.response_mode == "evoked":
            subject_map = sr.estimate_evoked_subject(
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
            subject_map = sr.estimate_subject(
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
            io.save_array(output_dir / subject_file, subject_map)
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
                    for name, value in zip(sr.CONDITION_NAMES, baseline_means)
                },
                "poststimulus_mean_by_condition": {
                    name: float(value)
                    for name, value in zip(sr.CONDITION_NAMES, poststimulus_means)
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
        for condition, condition_name in enumerate(sr.CONDITION_NAMES):
            filename = f"{name}_{condition_name}.npy"
            io.save_array(
                output_dir / filename,
                (numerator[condition] / denominator).astype(np.float32),
            )
            filenames[name][condition_name] = filename

    io.save_array(output_dir / "times_ms.npy", times_ms)
    io.save_array(
        output_dir / "vertex_indices_1based.npy",
        np.arange(1, n_vertices + 1, dtype=np.int32),
    )
    io.save_json(output_dir / "subject_qc.json", {"subjects": subject_qc})
    io.save_json(
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
                    for name, indices in zip(sr.CONDITION_NAMES, subject_conditions)
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
