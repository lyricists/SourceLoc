"""Plot scout-wise decoding with joint scout-by-time cluster inference.

Each diagnostic group is analyzed with one one-tailed, one-sample
spatiotemporal cluster permutation test on accuracy minus 50% chance. The
maximum-cluster null distribution controls family-wise error jointly across
all selected scouts and time points within that group. Cluster p-values are
then Bonferroni-corrected across the three diagnostic groups (Control,
Depressed, and Suicidal). The combined-subject All panel is not tested.

"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import pickle
import re
import tempfile
from pathlib import Path

import mat73
import matplotlib
import numpy as np
from pypdf import PdfReader, PdfWriter

# Keep MNE's optional JIT cache in a writable location during permutation tests.
NUMBA_CACHE_DIRECTORY = Path(tempfile.gettempdir()) / "mne_numba_cache"
NUMBA_CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", str(NUMBA_CACHE_DIRECTORY))

from mne.stats import spatio_temporal_cluster_1samp_test
from scipy import sparse
from scipy.stats import t as t_distribution

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIRECTORY = Path("/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Code/SourceLoc")
DEFAULT_INPUT = BASE_DIRECTORY / "Results" / "svmDecoding_scout_sent_noBio.pkl"
DEFAULT_ATLAS = (
    BASE_DIRECTORY.parent.parent / "Data" / "Preprocessed data" / "DKT_scout.mat"
)
DEFAULT_OUTPUT = BASE_DIRECTORY / "Results" / "CBP_Chance_scout_sent_noBio_3group_corrected"

T_START_MS = -200
T_EPOCH_END_EXCLUSIVE_MS = 1000
SAMPLING_RATE_HZ = 250
T_END_MS = T_EPOCH_END_EXCLUSIVE_MS - 1000 / SAMPLING_RATE_HZ
CHANCE_LEVEL = 0.5
CACHE_VERSION = 5

GROUP_DEFINITIONS = (
    ("control", "Control", 1),
    ("depressed", "Depressed", 2),
    ("suicidal", "Suicidal", 3),
)
GROUP_KEYS = tuple(definition[0] for definition in GROUP_DEFINITIONS)
GROUP_CORRECTION_FACTOR = len(GROUP_DEFINITIONS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--atlas", type=Path, default=DEFAULT_ATLAS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--reuse-uncorrected-dir", type=Path, default=None,
        help="Reuse matching version-4 permutation caches and apply the three-group correction without rerunning permutations.",
    )
    parser.add_argument(
        "--reuse-only", action="store_true",
        help="Require matching existing permutations; never start a new permutation test.",
    )
    parser.add_argument("--n-permutations", type=int, default=10_000)
    parser.add_argument("--threshold-p", type=float, default=0.05)
    parser.add_argument("--cluster-alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel permutation workers; -1 uses every available CPU core.",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=1000,
        help="Number of scout-time statistics computed per worker block.",
    )
    parser.add_argument("--rows-per-page", type=int, default=4)
    parser.add_argument(
        "--rebuild-figure",
        action="store_true",
        help="Rebuild the PDF even when cached statistics are unchanged.",
    )
    parser.add_argument(
        "--scouts",
        type=int,
        nargs="*",
        default=None,
        help="Optional zero-based scout indices. The default processes all scouts.",
    )
    parser.add_argument(
        "--groups",
        choices=GROUP_KEYS,
        nargs="+",
        default=list(GROUP_KEYS),
        help="Diagnostic groups to analyze. Correction always covers all three groups.",
    )
    parser.add_argument(
        "--scout-adjacency",
        type=Path,
        default=None,
        help=(
            "Optional anatomical scout adjacency (.npz sparse matrix, .npy dense "
            "matrix, or pickle). Default: no between-scout edges."
        ),
    )
    return parser.parse_args()


def atomic_pickle_dump(value: object, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with open(temporary, "wb") as file:
        pickle.dump(value, file, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, destination)


def load_decoder_result(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    with open(path, "rb") as file:
        result = pickle.load(file)

    scores = np.asarray(result["decodeScore"], dtype=np.float64)
    subject_groups = np.asarray(result["subIdx"]).squeeze()

    if scores.ndim != 3:
        raise ValueError(
            "decodeScore must be scout x time x subject or "
            f"subject x scout x time; received {scores.shape}."
        )

    n_subjects = subject_groups.size
    if scores.shape[-1] == n_subjects:
        pass
    elif scores.shape[0] == n_subjects:
        scores = scores.transpose(1, 2, 0)
    else:
        raise ValueError(
            f"No decodeScore axis matches the {n_subjects} subject labels: "
            f"{scores.shape}."
        )

    if not np.isfinite(scores).all():
        raise ValueError("decodeScore contains NaN or infinite values.")
    if scores.min() < 0 or scores.max() > 1:
        raise ValueError("decodeScore values must be proportions between 0 and 1.")

    return scores, subject_groups, result


def load_scout_labels(path: Path, n_scouts: int | None = None) -> list[str]:
    atlas = mat73.loadmat(str(path))["DKT_scout"]
    raw_labels = atlas.get("Label", atlas.get("Region"))
    if raw_labels is None:
        raise ValueError("Atlas contains no scout labels.")
    if n_scouts is not None and len(raw_labels) != n_scouts:
        raise ValueError(f"Atlas must contain {n_scouts} scout labels.")

    labels = []
    for index, label in enumerate(raw_labels):
        clean = re.sub(r"\s+", " ", str(label)).strip()
        labels.append(clean or f"Scout {index + 1}")
    return labels


def make_time_vector(
    n_times: int, recorded_times: np.ndarray | None = None
) -> np.ndarray:
    step_ms = 1000 / SAMPLING_RATE_HZ
    times = np.arange(T_START_MS, T_EPOCH_END_EXCLUSIVE_MS, step_ms)
    if times.size != n_times:
        raise ValueError(
            f"Expected {times.size} samples in [{T_START_MS}, "
            f"{T_EPOCH_END_EXCLUSIVE_MS}) ms (last sample {times[-1]:g} ms), "
            f"but decodeScore contains {n_times}."
        )
    if recorded_times is not None:
        recorded = np.asarray(recorded_times, dtype=np.float64)
        if (
            recorded.shape != times.shape
            or not np.isfinite(recorded).all()
            or not np.allclose(recorded, times, rtol=0, atol=1e-6)
        ):
            raise ValueError(
                "Decoder times_ms does not match the 250 Hz " "[-200, 1000) ms epoch."
            )
        return recorded
    return times


def _read_adjacency(path: Path) -> sparse.csr_matrix:
    if path.suffix.lower() == ".npz":
        try:
            return sparse.csr_matrix(sparse.load_npz(path))
        except (KeyError, ValueError):
            # Also accept an NPZ containing a conventional dense ``adjacency``
            # array instead of SciPy's sparse-matrix serialization.
            with np.load(path) as archive:
                if "adjacency" not in archive:
                    raise ValueError(
                        f"{path} is neither a SciPy sparse matrix nor an NPZ "
                        "containing an 'adjacency' array."
                    )
                return sparse.csr_matrix(archive["adjacency"])
    if path.suffix.lower() == ".npy":
        return sparse.csr_matrix(np.load(path))
    if path.suffix.lower() in {".pkl", ".pickle"}:
        with open(path, "rb") as file:
            value = pickle.load(file)
        if isinstance(value, dict) and "adjacency" in value:
            value = value["adjacency"]
        return sparse.csr_matrix(value)
    raise ValueError("Scout adjacency must be a .npz, .npy, .pkl, or .pickle file.")


def load_scout_adjacency(
    path: Path | None,
    n_scouts: int,
    scout_indices: list[int],
) -> tuple[sparse.csr_matrix, str]:
    n_selected = len(scout_indices)
    if path is None:
        return (
            sparse.csr_matrix((n_selected, n_selected), dtype=np.int8),
            "disconnected scouts (temporal clusters only)",
        )

    adjacency = _read_adjacency(path)
    if adjacency.shape == (n_scouts, n_scouts):
        adjacency = adjacency[scout_indices][:, scout_indices]
    elif adjacency.shape != (n_selected, n_selected):
        raise ValueError(
            "Scout adjacency shape must match either all scouts "
            f"({n_scouts}, {n_scouts}) or selected scouts "
            f"({n_selected}, {n_selected}); received {adjacency.shape}."
        )

    if adjacency.data.size and not np.isfinite(adjacency.data).all():
        raise ValueError("Scout adjacency contains NaN or infinite values.")

    # Interpret every nonzero entry as an undirected edge and remove self-edges.
    adjacency = sparse.csr_matrix((adjacency + adjacency.T) != 0, dtype=np.int8)
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    return adjacency, f"anatomical adjacency from {path.resolve()}"


def adjacency_digest(adjacency: sparse.csr_matrix) -> str:
    adjacency = adjacency.tocsr()
    digest = hashlib.sha256()
    digest.update(np.asarray(adjacency.shape, dtype=np.int64).tobytes())
    digest.update(adjacency.indptr.astype(np.int64, copy=False).tobytes())
    digest.update(adjacency.indices.astype(np.int64, copy=False).tobytes())
    digest.update(adjacency.data.astype(np.int8, copy=False).tobytes())
    return digest.hexdigest()


def cluster_to_mask(cluster: object, shape: tuple[int, int]) -> np.ndarray:
    if isinstance(cluster, tuple):
        mask = np.zeros(shape, dtype=bool)
        mask[cluster] = True
        return mask

    values = np.asarray(cluster)
    if values.shape != shape:
        raise ValueError(f"Unexpected cluster shape {values.shape}; expected {shape}.")
    return values.astype(bool, copy=False)


def contiguous_runs(mask: np.ndarray) -> list[np.ndarray]:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return []
    return list(np.split(indices, np.where(np.diff(indices) > 1)[0] + 1))


def make_cache_signature(
    args: argparse.Namespace,
    group_key: str,
    subject_indices: np.ndarray,
    scout_indices: list[int],
    scores_shape: tuple[int, ...],
    times: np.ndarray,
    adjacency: sparse.csr_matrix,
) -> dict:
    input_stat = args.input.stat()
    return {
        "cache_version": CACHE_VERSION,
        "input_path": str(args.input.resolve()),
        "input_size": int(input_stat.st_size),
        "input_mtime_ns": int(input_stat.st_mtime_ns),
        "scores_shape": tuple(int(value) for value in scores_shape),
        "group_key": group_key,
        "subject_indices": subject_indices.astype(int).tolist(),
        "scout_indices": list(scout_indices),
        "n_times": int(times.size),
        "first_time_ms": float(times[0]),
        "last_time_ms": float(times[-1]),
        "chance_level": CHANCE_LEVEL,
        "group_family": GROUP_KEYS,
        "group_correction_factor": GROUP_CORRECTION_FACTOR,
        "n_permutations": int(args.n_permutations),
        "threshold_p": float(args.threshold_p),
        "cluster_alpha": float(args.cluster_alpha),
        "tail": 1,
        "max_step": 1,
        "t_power": 1,
        "seed": int(args.seed),
        "adjacency_sha256": adjacency_digest(adjacency),
    }


def compute_group_statistics(
    selected_scores: np.ndarray,
    times: np.ndarray,
    scout_indices: list[int],
    selected_scout_labels: list[str],
    group_key: str,
    group_label: str,
    subject_indices: np.ndarray,
    adjacency: sparse.csr_matrix,
    adjacency_description: str,
    cache_signature: dict,
    args: argparse.Namespace,
) -> dict:
    # selected_scores: selected scout x time x all subjects. Keeping subject
    # selection separate avoids NumPy advanced-index axis reordering surprises.
    group_scores = np.take(selected_scores, subject_indices, axis=2)
    centered_scores = np.ascontiguousarray(
        group_scores.transpose(2, 1, 0) - CHANCE_LEVEL,
        dtype=np.float64,
    )
    degrees_of_freedom = centered_scores.shape[0] - 1
    threshold = float(t_distribution.ppf(1.0 - args.threshold_p, degrees_of_freedom))

    t_observed, clusters, cluster_p_values, null_distribution = (
        spatio_temporal_cluster_1samp_test(
            centered_scores,
            threshold=threshold,
            n_permutations=args.n_permutations,
            tail=1,
            adjacency=adjacency,
            max_step=1,
            t_power=1,
            # Index tuples are substantially smaller than one full boolean
            # time-by-scout array for every candidate cluster.
            out_type="indices",
            check_disjoint=True,
            buffer_size=args.buffer_size,
            rng=args.seed,
            n_jobs=args.n_jobs,
            verbose=False,
        )
    )

    expected_shape = (times.size, len(scout_indices))
    significant_mask = np.zeros(expected_shape, dtype=bool)
    significant_clusters = []

    adjusted_cluster_p_values = np.minimum(
        1.0, np.asarray(cluster_p_values, dtype=np.float64) * GROUP_CORRECTION_FACTOR
    )
    for cluster_index, (cluster, raw_p_value, adjusted_p_value) in enumerate(
        zip(clusters, cluster_p_values, adjusted_cluster_p_values)
    ):
        if adjusted_p_value >= args.cluster_alpha:
            continue
        mask = cluster_to_mask(cluster, expected_shape)
        significant_mask |= mask
        local_scout_positions = np.flatnonzero(mask.any(axis=0))
        time_positions = np.flatnonzero(mask.any(axis=1))
        significant_clusters.append(
            {
                "cluster_index": int(cluster_index),
                "cluster_number": int(cluster_index + 1),
                "mask": mask,
                "raw_p_value": float(raw_p_value),
                "adjusted_p_value": float(adjusted_p_value),
                "cluster_mass": float(np.asarray(t_observed)[mask].sum()),
                "n_points": int(mask.sum()),
                "n_scouts": int(local_scout_positions.size),
                "scout_indices": [
                    int(scout_indices[position]) for position in local_scout_positions
                ],
                "start_ms": float(times[time_positions[0]]),
                "end_ms": float(times[time_positions[-1]]),
            }
        )

    mean_accuracy = group_scores.mean(axis=2)
    sem_accuracy = group_scores.std(axis=2, ddof=1) / np.sqrt(group_scores.shape[2])

    return {
        "cache_signature": cache_signature,
        "group_key": group_key,
        "group_label": group_label,
        "subject_indices": subject_indices.astype(int),
        "n_subjects": int(subject_indices.size),
        "scout_indices": np.asarray(scout_indices, dtype=int),
        "scout_labels": list(selected_scout_labels),
        "times_ms": times,
        "chance_level": CHANCE_LEVEL,
        "mean_accuracy": mean_accuracy,
        "sem_accuracy": sem_accuracy,
        "t_observed": np.asarray(t_observed),
        "n_clusters_tested": int(len(clusters)),
        "cluster_p_values": np.asarray(cluster_p_values),
        "adjusted_cluster_p_values": adjusted_cluster_p_values,
        "significant_mask": significant_mask,
        "significant_clusters": significant_clusters,
        "null_distribution": np.asarray(null_distribution),
        "test": {
            "name": "one-sample scout-by-time cluster permutation test",
            "contrast": "cross-validated balanced accuracy minus 0.5",
            "tail": 1,
            "n_permutations": args.n_permutations,
            "threshold_p": args.threshold_p,
            "cluster_forming_t": threshold,
            "cluster_alpha": args.cluster_alpha,
            "group_family": GROUP_KEYS,
            "group_correction": "Bonferroni across Control, Depressed, and Suicidal",
            "group_correction_factor": GROUP_CORRECTION_FACTOR,
            "seed": args.seed,
            "n_jobs": args.n_jobs,
            "buffer_size": args.buffer_size,
            "max_step": 1,
            "adjacency": adjacency_description,
            "spatial_edge_count": int(adjacency.nnz // 2),
            "correction_scope": (
                "all selected scouts and time points within each group; "
                "Bonferroni across the three diagnostic groups"
            ),
        },
    }


def reuse_uncorrected_group(
    path: Path, cache_signature: dict, cluster_alpha: float
) -> dict | None:
    """Apply the stricter group correction to matching legacy permutation results."""
    if not path.exists():
        return None
    with open(path, "rb") as file:
        statistics = pickle.load(file)
    legacy_signature = {
        key: value for key, value in cache_signature.items()
        if key not in ("group_family", "group_correction_factor")
    }
    legacy_signature["cache_version"] = 4
    if statistics.get("cache_signature") != legacy_signature:
        print(f"Cannot reuse incompatible permutation cache: {path}")
        return None

    raw_p_values = np.asarray(statistics["cluster_p_values"], dtype=np.float64)
    adjusted_p_values = np.minimum(1.0, raw_p_values * GROUP_CORRECTION_FACTOR)
    significant_mask = np.zeros_like(statistics["significant_mask"], dtype=bool)
    significant_clusters = []
    for old_cluster in statistics["significant_clusters"]:
        cluster_index = old_cluster["cluster_index"]
        raw_p_value = float(raw_p_values[cluster_index])
        if not np.isclose(raw_p_value, old_cluster["p_value"]):
            raise ValueError(f"Inconsistent cached cluster p-value in {path}")
        adjusted_p_value = float(adjusted_p_values[cluster_index])
        if adjusted_p_value >= cluster_alpha:
            continue
        cluster = dict(old_cluster)
        cluster.pop("p_value")
        cluster["raw_p_value"] = raw_p_value
        cluster["adjusted_p_value"] = adjusted_p_value
        significant_clusters.append(cluster)
        significant_mask |= cluster["mask"]
    eligible_indices = set(np.flatnonzero(adjusted_p_values < cluster_alpha))
    cached_indices = {cluster["cluster_index"] for cluster in significant_clusters}
    if cached_indices != eligible_indices:
        raise ValueError(f"Legacy cache omits a cluster needed after correction: {path}")

    statistics["cache_signature"] = cache_signature
    statistics["adjusted_cluster_p_values"] = adjusted_p_values
    statistics["significant_mask"] = significant_mask
    statistics["significant_clusters"] = significant_clusters
    test = dict(statistics["test"])
    test.update({
        "group_family": GROUP_KEYS,
        "group_correction": "Bonferroni across Control, Depressed, and Suicidal",
        "group_correction_factor": GROUP_CORRECTION_FACTOR,
        "correction_scope": (
            "all selected scouts and time points within each group; "
            "Bonferroni across the three diagnostic groups"
        ),
        "raw_statistics_reused_from": str(path.resolve()),
    })
    statistics["test"] = test
    return statistics


def load_or_compute_group(
    cache_path: Path,
    cache_signature: dict,
    uncorrected_cache_path: Path | None = None,
    reuse_only: bool = False,
    **compute_kwargs: object,
) -> tuple[dict, str]:
    if cache_path.exists():
        with open(cache_path, "rb") as file:
            cached = pickle.load(file)
        if cached.get("cache_signature") == cache_signature:
            return cached, "cache"
        print(f"Ignoring incompatible statistics cache: {cache_path}")

    if uncorrected_cache_path is not None:
        reused = reuse_uncorrected_group(
            uncorrected_cache_path, cache_signature,
            float(compute_kwargs["args"].cluster_alpha),
        )
        if reused is not None:
            atomic_pickle_dump(reused, cache_path)
            return reused, "reused existing permutations"
    if reuse_only:
        raise ValueError(f"No matching permutation cache for {cache_path.name}")

    statistics = compute_group_statistics(
        cache_signature=cache_signature,
        **compute_kwargs,
    )
    atomic_pickle_dump(statistics, cache_path)
    return statistics, "computed"


def write_cluster_summary(
    statistics_by_group: dict[str, dict], destination: Path
) -> None:
    fieldnames = (
        "group",
        "n_subjects",
        "cluster_number",
        "raw_cluster_p_value",
        "group_adjusted_p_value",
        "cluster_mass",
        "cluster_n_points",
        "cluster_n_scouts",
        "scout_index",
        "scout_number",
        "scout_label",
        "interval_number",
        "start_ms",
        "end_ms",
        "n_time_samples",
    )
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with open(temporary, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for statistics in statistics_by_group.values():
            times = statistics["times_ms"]
            for cluster in statistics["significant_clusters"]:
                mask = cluster["mask"]
                for local_scout, scout_index in enumerate(statistics["scout_indices"]):
                    for interval_number, run in enumerate(
                        contiguous_runs(mask[:, local_scout]), start=1
                    ):
                        writer.writerow(
                            {
                                "group": statistics["group_label"],
                                "n_subjects": statistics["n_subjects"],
                                "cluster_number": cluster["cluster_number"],
                                "raw_cluster_p_value": cluster["raw_p_value"],
                                "group_adjusted_p_value": cluster["adjusted_p_value"],
                                "cluster_mass": cluster["cluster_mass"],
                                "cluster_n_points": cluster["n_points"],
                                "cluster_n_scouts": cluster["n_scouts"],
                                "scout_index": int(scout_index),
                                "scout_number": int(scout_index) + 1,
                                "scout_label": statistics["scout_labels"][local_scout],
                                "interval_number": interval_number,
                                "start_ms": float(times[run[0]]),
                                "end_ms": float(times[run[-1]]),
                                "n_time_samples": int(run.size),
                            }
                        )
    os.replace(temporary, destination)


def plot_significant_intervals(
    axis: plt.Axes,
    times: np.ndarray,
    significant_mask: np.ndarray,
    y_position: float,
) -> None:
    for run in contiguous_runs(significant_mask):
        axis.plot(
            times[run],
            np.full(run.size, y_position),
            color="red",
            linewidth=4,
            solid_capstyle="butt",
            zorder=5,
        )


def create_multipage_figure(
    statistics_by_group: dict[str, dict],
    group_definitions: list[tuple[str, str, int]],
    scout_indices: list[int],
    scout_labels: list[str],
    destination: Path,
    rows_per_page: int,
    rebuild: bool,
) -> None:
    if destination.exists() and not rebuild:
        print(f"Figure already exists; not rebuilding: {destination}")
        return

    all_means = (
        np.concatenate(
            [
                statistics["mean_accuracy"].ravel()
                for statistics in statistics_by_group.values()
            ]
        )
        * 100
    )
    all_sems = (
        np.concatenate(
            [
                statistics["sem_accuracy"].ravel()
                for statistics in statistics_by_group.values()
            ]
        )
        * 100
    )
    lower_limit = min(45.0, float(np.floor(np.min(all_means - all_sems) - 1)))
    upper_limit = max(65.0, float(np.ceil(np.max(all_means + all_sems) + 1)))

    pdf_scratch_directory = destination.parent / ".pdf_scratch"
    pdf_scratch_directory.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.pdf")
    page_paths = []

    # Saving pages independently avoids a Matplotlib PdfPages resource-reuse
    # bug observed with data-dense middle pages. Merge rendered pages afterward.
    with tempfile.TemporaryDirectory(
        prefix="scout_decoding_pages_", dir=pdf_scratch_directory
    ) as temporary_directory:
        for page_start in range(0, len(scout_indices), rows_per_page):
            page_scouts = scout_indices[page_start : page_start + rows_per_page]
            figure, axes = plt.subplots(
                len(page_scouts),
                len(group_definitions),
                figsize=(6 * len(group_definitions), 4.2 * len(page_scouts)),
                sharex=True,
                sharey=True,
                squeeze=False,
            )

            for row, scout_index in enumerate(page_scouts):
                local_scout = page_start + row
                for column, (group_key, group_label, _) in enumerate(group_definitions):
                    statistics = statistics_by_group[group_key]
                    axis = axes[row, column]
                    times = statistics["times_ms"]
                    mean = statistics["mean_accuracy"][local_scout] * 100
                    sem = statistics["sem_accuracy"][local_scout] * 100

                    axis.plot(times, mean, color="#1f77b4", linewidth=2.2)
                    axis.fill_between(
                        times,
                        mean - sem,
                        mean + sem,
                        color="#1f77b4",
                        alpha=0.28,
                        linewidth=0,
                    )
                    axis.axhline(50, color="black", linestyle="--", linewidth=1.5)
                    axis.axvline(0, color="0.55", linestyle=":", linewidth=1)
                    plot_significant_intervals(
                        axis,
                        times,
                        statistics["significant_mask"][:, local_scout],
                        y_position=lower_limit + 0.8,
                    )
                    axis.set_xlim(T_START_MS, T_END_MS)
                    axis.set_ylim(lower_limit, upper_limit)
                    axis.grid(axis="y", color="0.9", linewidth=0.7)
                    axis.tick_params(labelsize=10)

                    if row == 0:
                        axis.set_title(
                            f"{group_label} (n={statistics['n_subjects']})",
                            fontsize=14,
                            fontweight="bold",
                        )
                    if column == 0:
                        axis.set_ylabel(
                            f"{scout_index + 1:02d}. {scout_labels[scout_index]}\n"
                            "Balanced accuracy (%)",
                            fontsize=11,
                            fontweight="bold",
                        )
                    if row == len(page_scouts) - 1:
                        axis.set_xlabel("Time from stimulus onset (ms)", fontsize=11)

            figure.suptitle(
                "Scout-wise decoding by diagnostic group\n"
                "Mean ± SEM; red bars: group-adjusted cluster p < 0.05 vs 50%",
                fontsize=17,
                fontweight="bold",
                y=0.998,
            )
            figure.text(
                0.5,
                0.004,
                "Cluster permutation correction spans selected scouts and time; "
                "Bonferroni correction spans Control, Depressed, and Suicidal (3 tests).",
                ha="center",
                fontsize=9,
            )
            figure.tight_layout(rect=(0.02, 0.025, 1, 0.965))
            page_path = (
                Path(temporary_directory) / f"page_{len(page_paths) + 1:03d}.pdf"
            )
            figure.savefig(page_path, format="pdf")
            page_paths.append(page_path)
            plt.close(figure)

        writer = PdfWriter()
        for page_path in page_paths:
            reader = PdfReader(page_path)
            if len(reader.pages) != 1:
                raise ValueError(f"Expected one page in {page_path}.")
            writer.add_page(reader.pages[0])
        with open(temporary, "wb") as file:
            writer.write(file)

    os.replace(temporary, destination)


def validate_args(args: argparse.Namespace) -> None:
    if args.reuse_only and args.reuse_uncorrected_dir is None:
        raise ValueError("--reuse-only requires --reuse-uncorrected-dir.")
    if args.n_permutations < 2:
        raise ValueError("n-permutations must be at least 2.")
    if not 0 < args.threshold_p < 0.5:
        raise ValueError("threshold-p must be between 0 and 0.5.")
    if not 0 < args.cluster_alpha < 1:
        raise ValueError("cluster-alpha must be between 0 and 1.")
    if args.n_jobs == 0:
        raise ValueError("n-jobs cannot be zero.")
    if args.buffer_size < 1:
        raise ValueError("buffer-size must be at least 1.")
    if args.rows_per_page < 1:
        raise ValueError("rows-per-page must be at least 1.")
    if len(set(args.groups)) != len(args.groups):
        raise ValueError("groups must not contain duplicates.")


def main() -> None:
    args = parse_args()
    validate_args(args)

    scores, subject_groups, decoder_result = load_decoder_result(args.input)
    n_scouts, n_times, n_subjects = scores.shape
    times = make_time_vector(n_times, decoder_result.get("times_ms"))
    scout_labels = load_scout_labels(args.atlas, n_scouts)

    scout_indices = list(range(n_scouts)) if args.scouts is None else args.scouts
    if not scout_indices or len(set(scout_indices)) != len(scout_indices):
        raise ValueError("scouts must contain unique indices and cannot be empty.")
    if min(scout_indices) < 0 or max(scout_indices) >= n_scouts:
        raise IndexError(f"Scout indices must be between 0 and {n_scouts - 1}.")

    group_definitions = [
        definition for definition in GROUP_DEFINITIONS if definition[0] in args.groups
    ]
    group_indices = {}
    for group_key, _, group_code in group_definitions:
        indices = np.flatnonzero(subject_groups == group_code)
        if indices.size < 2:
            raise ValueError(f"Group {group_key} has fewer than two subjects.")
        group_indices[group_key] = indices

    adjacency, adjacency_description = load_scout_adjacency(
        args.scout_adjacency, n_scouts, scout_indices
    )
    selected_scores = np.ascontiguousarray(scores[scout_indices, :, :])
    selected_scout_labels = [scout_labels[index] for index in scout_indices]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    statistics_directory = args.output_dir / "spatiotemporal_statistics"
    statistics_directory.mkdir(parents=True, exist_ok=True)

    print(f"Input: {args.input}")
    print(f"decodeScore: {scores.shape} (scout x time x subject)")
    print(
        "Groups: "
        + ", ".join(
            f"{label}={group_indices[key].size}" for key, label, _ in group_definitions
        )
    )
    print(
        f"Testing {len(scout_indices)} scouts jointly with "
        f"{args.n_permutations} permutations per group"
    )
    print("Cluster p-values are Bonferroni-corrected across the three diagnostic groups")
    print(f"Scout adjacency: {adjacency_description}")
    print(f"Spatial edges: {adjacency.nnz // 2}; permutation workers: {args.n_jobs}")

    statistics_by_group = {}
    loaded_count = 0
    computed_count = 0
    updated_count = 0
    selection_digest = hashlib.sha256(
        np.asarray(scout_indices, dtype=np.int64).tobytes()
    ).hexdigest()[:12]

    for group_key, group_label, _ in group_definitions:
        subject_indices = group_indices[group_key]
        cache_signature = make_cache_signature(
            args=args,
            group_key=group_key,
            subject_indices=subject_indices,
            scout_indices=scout_indices,
            scores_shape=scores.shape,
            times=times,
            adjacency=adjacency,
        )
        cache_path = (
            statistics_directory / f"group_{group_key}_scouts_{selection_digest}.pkl"
        )
        print(f"Analyzing {group_label} (n={subject_indices.size})...")
        uncorrected_cache_path = (
            args.reuse_uncorrected_dir / "spatiotemporal_statistics" /
            f"group_{group_key}_scouts_{selection_digest}.pkl"
            if args.reuse_uncorrected_dir is not None else None
        )
        statistics, source = load_or_compute_group(
            cache_path=cache_path,
            cache_signature=cache_signature,
            uncorrected_cache_path=uncorrected_cache_path,
            reuse_only=args.reuse_only,
            selected_scores=selected_scores,
            times=times,
            scout_indices=scout_indices,
            selected_scout_labels=selected_scout_labels,
            group_key=group_key,
            group_label=group_label,
            subject_indices=subject_indices,
            adjacency=adjacency,
            adjacency_description=adjacency_description,
            args=args,
        )
        statistics_by_group[group_key] = statistics
        loaded_count += int(source != "computed")
        computed_count += int(source == "computed")
        updated_count += int(source != "cache")
        print(
            f"  {source}; significant clusters: "
            f"{len(statistics['significant_clusters'])}"
        )

    summary_path = args.output_dir / "cluster_summary.csv"
    write_cluster_summary(statistics_by_group, summary_path)

    figure_path = args.output_dir / "scout_decoding_three_groups.pdf"
    create_multipage_figure(
        statistics_by_group=statistics_by_group,
        group_definitions=group_definitions,
        scout_indices=scout_indices,
        scout_labels=scout_labels,
        destination=figure_path,
        rows_per_page=args.rows_per_page,
        rebuild=args.rebuild_figure or updated_count > 0,
    )

    metadata = {
        "input_path": str(args.input.resolve()),
        "input_axis_order": decoder_result.get("axis_order"),
        "decode_parameters": decoder_result.get("parameters"),
        "score_shape": scores.shape,
        "time_range_ms": (float(times[0]), float(times[-1])),
        "epoch_end_exclusive_ms": T_EPOCH_END_EXCLUSIVE_MS,
        "sampling_rate_hz": SAMPLING_RATE_HZ,
        "groups_analyzed": [definition[0] for definition in group_definitions],
        "group_sizes": {
            label: int(group_indices[key].size) for key, label, _ in group_definitions
        },
        "scout_indices": scout_indices,
        "n_scouts_processed": len(scout_indices),
        "test_parameters": {
            "name": "one-sample scout-by-time cluster permutation test",
            "n_permutations": args.n_permutations,
            "threshold_p": args.threshold_p,
            "cluster_alpha": args.cluster_alpha,
            "group_family": GROUP_KEYS,
            "group_correction": "Bonferroni across Control, Depressed, and Suicidal",
            "group_correction_factor": GROUP_CORRECTION_FACTOR,
            "tail": 1,
            "seed": args.seed,
            "n_jobs": args.n_jobs,
            "buffer_size": args.buffer_size,
            "adjacency": adjacency_description,
            "spatial_edge_count": int(adjacency.nnz // 2),
            "correction_scope": (
                "all selected scouts and time points within each group; "
                "Bonferroni across the three diagnostic groups"
            ),
        },
    }
    atomic_pickle_dump(metadata, args.output_dir / "run_metadata.pkl")

    significant_scout_group_panels = sum(
        int(statistics["significant_mask"].any(axis=0).sum())
        for statistics in statistics_by_group.values()
    )
    significant_clusters = sum(
        len(statistics["significant_clusters"])
        for statistics in statistics_by_group.values()
    )
    print(f"Group caches loaded or reused: {loaded_count}")
    print(f"Group tests computed: {computed_count}")
    print(
        "Scout/group panels containing significance: "
        f"{significant_scout_group_panels}"
    )
    print(f"Significant joint clusters: {significant_clusters}")
    print(f"Figure: {figure_path}")
    print(f"Cluster summary: {summary_path}")


if __name__ == "__main__":
    main()
