"""Plot direct group differences with joint scout-by-time cluster inference.

Three two-sided pairwise contrasts are tested: Control - Depressed,
Control - Suicidal, and Depressed - Suicidal. Each contrast uses an
independent-samples Welch t statistic and subject-label permutations. The
maximum-cluster null distribution controls family-wise error jointly across
all selected scouts and time points within a contrast.

By default, the already scout-by-time-corrected cluster p-values are multiplied
by the number of requested contrasts (three for the full analysis), providing
Bonferroni control across pairwise contrast families. Pass
``--no-correct-across-contrasts`` only when each contrast is intentionally
treated as a separate inferential family.

Scouts have no spatial edges by default because the atlas contains parcel
memberships but no cortical mesh faces. Clusters therefore extend through time
within a scout, while the permutation maximum still corrects across scouts.
An anatomically valid adjacency matrix can be supplied with
``--scout-adjacency``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import pickle
import tempfile
from pathlib import Path

import matplotlib
import numpy as np

# Some managed Python installations make MNE's package-level Numba cache
# undiscoverable. A writable explicit cache keeps JIT acceleration available
# in both the main process and permutation workers.
NUMBA_CACHE_DIRECTORY = Path(tempfile.gettempdir()) / "mne_numba_cache"
NUMBA_CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", str(NUMBA_CACHE_DIRECTORY))

from mne.stats import spatio_temporal_cluster_test
from scipy import sparse
from scipy.stats import t as t_distribution
from scipy.stats import ttest_ind

import plot_scout_decoding as common

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_OUTPUT = common.BASE_DIRECTORY / "Results" / "CBP_GroupDifference_scout_3pc"
CACHE_VERSION = 2

CONTRAST_DEFINITIONS = (
    ("control-depressed", "Control - Depressed", 1, 2, "Control", "Depressed"),
    ("control-suicidal", "Control - Suicidal", 1, 3, "Control", "Suicidal"),
    ("depressed-suicidal", "Depressed - Suicidal", 2, 3, "Depressed", "Suicidal"),
)
CONTRAST_KEYS = tuple(definition[0] for definition in CONTRAST_DEFINITIONS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=common.DEFAULT_INPUT)
    parser.add_argument("--atlas", type=Path, default=common.DEFAULT_ATLAS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
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
        "--contrasts",
        choices=CONTRAST_KEYS,
        nargs="+",
        default=list(CONTRAST_KEYS),
        help="Pairwise contrasts to analyze. The default runs all three.",
    )
    parser.add_argument(
        "--no-correct-across-contrasts",
        action="store_true",
        help=(
            "Do not multiply cluster p-values by the number of requested "
            "contrasts. Scout-by-time correction remains active."
        ),
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


def welch_t_statistic(group1: np.ndarray, group2: np.ndarray) -> np.ndarray:
    """Return group1-minus-group2 Welch t values for every tested feature."""
    statistic = ttest_ind(
        group1,
        group2,
        axis=0,
        equal_var=False,
        nan_policy="raise",
    ).statistic
    return np.asarray(statistic, dtype=np.float64)


def make_cache_signature(
    args: argparse.Namespace,
    contrast_key: str,
    group1_indices: np.ndarray,
    group2_indices: np.ndarray,
    scout_indices: list[int],
    scores_shape: tuple[int, ...],
    times: np.ndarray,
    adjacency: sparse.csr_matrix,
    contrast_correction_factor: int,
) -> dict:
    input_stat = args.input.stat()
    return {
        "cache_version": CACHE_VERSION,
        "input_path": str(args.input.resolve()),
        "input_size": int(input_stat.st_size),
        "input_mtime_ns": int(input_stat.st_mtime_ns),
        "scores_shape": tuple(int(value) for value in scores_shape),
        "contrast_key": contrast_key,
        "group1_indices": group1_indices.astype(int).tolist(),
        "group2_indices": group2_indices.astype(int).tolist(),
        "scout_indices": list(scout_indices),
        "n_times": int(times.size),
        "first_time_ms": float(times[0]),
        "last_time_ms": float(times[-1]),
        "n_permutations": int(args.n_permutations),
        "threshold_p": float(args.threshold_p),
        "cluster_alpha": float(args.cluster_alpha),
        "tail": 0,
        "statistic": "Welch independent-samples t",
        "max_step": 1,
        "t_power": 1,
        "seed": int(args.seed),
        "contrast_correction_factor": contrast_correction_factor,
        "adjacency_sha256": common.adjacency_digest(adjacency),
    }


def compute_contrast_statistics(
    selected_scores: np.ndarray,
    times: np.ndarray,
    scout_indices: list[int],
    selected_scout_labels: list[str],
    contrast_key: str,
    contrast_label: str,
    group1_label: str,
    group2_label: str,
    group1_indices: np.ndarray,
    group2_indices: np.ndarray,
    adjacency: sparse.csr_matrix,
    adjacency_description: str,
    contrast_correction_factor: int,
    cache_signature: dict,
    args: argparse.Namespace,
) -> dict:
    # MNE expects observations x time x space. Subject labels, rather than
    # individual scout-time values, are shuffled by permutation_cluster_test.
    group1_scores = np.take(selected_scores, group1_indices, axis=2)
    group2_scores = np.take(selected_scores, group2_indices, axis=2)
    group1_data = np.ascontiguousarray(
        group1_scores.transpose(2, 1, 0), dtype=np.float64
    )
    group2_data = np.ascontiguousarray(
        group2_scores.transpose(2, 1, 0), dtype=np.float64
    )

    # Welch degrees of freedom vary by feature. A fixed pooled-DF critical t is
    # used only as the cluster-forming threshold; permutation cluster p-values,
    # not this nominal pointwise threshold, determine final significance.
    threshold_reference_df = group1_data.shape[0] + group2_data.shape[0] - 2
    threshold = float(
        t_distribution.ppf(
            1.0 - args.threshold_p / 2.0,
            threshold_reference_df,
        )
    )

    t_observed, clusters, cluster_p_values, null_distribution = (
        spatio_temporal_cluster_test(
            [group1_data, group2_data],
            stat_fun=welch_t_statistic,
            threshold=threshold,
            n_permutations=args.n_permutations,
            tail=0,
            adjacency=adjacency,
            max_step=1,
            t_power=1,
            out_type="indices",
            # Avoid MNE's optional disjoint-partition prepass. On some MNE /
            # Numba installations that prepass attempts to create a package
            # cache in a non-writable location; it does not change inference.
            check_disjoint=False,
            buffer_size=args.buffer_size,
            rng=args.seed,
            n_jobs=args.n_jobs,
            verbose=False,
        )
    )

    expected_shape = (times.size, len(scout_indices))
    significant_mask = np.zeros(expected_shape, dtype=bool)
    positive_mask = np.zeros(expected_shape, dtype=bool)
    negative_mask = np.zeros(expected_shape, dtype=bool)
    significant_clusters = []

    for cluster_index, (cluster, raw_p_value) in enumerate(
        zip(clusters, cluster_p_values)
    ):
        adjusted_p_value = min(
            1.0, float(raw_p_value) * contrast_correction_factor
        )
        if adjusted_p_value >= args.cluster_alpha:
            continue

        mask = common.cluster_to_mask(cluster, expected_shape)
        cluster_mass = float(np.asarray(t_observed)[mask].sum())
        direction = (
            f"{group1_label} > {group2_label}"
            if cluster_mass > 0
            else f"{group1_label} < {group2_label}"
        )
        significant_mask |= mask
        if cluster_mass > 0:
            positive_mask |= mask
        else:
            negative_mask |= mask

        local_scout_positions = np.flatnonzero(mask.any(axis=0))
        time_positions = np.flatnonzero(mask.any(axis=1))
        significant_clusters.append(
            {
                "cluster_index": int(cluster_index),
                "cluster_number": int(cluster_index + 1),
                "mask": mask,
                "raw_p_value": float(raw_p_value),
                "adjusted_p_value": adjusted_p_value,
                "cluster_mass": cluster_mass,
                "direction": direction,
                "n_points": int(mask.sum()),
                "n_scouts": int(local_scout_positions.size),
                "scout_indices": [
                    int(scout_indices[position]) for position in local_scout_positions
                ],
                "start_ms": float(times[time_positions[0]]),
                "end_ms": float(times[time_positions[-1]]),
            }
        )

    mean1 = group1_scores.mean(axis=2)
    mean2 = group2_scores.mean(axis=2)
    mean_difference = mean1 - mean2
    difference_sem = np.sqrt(
        group1_scores.var(axis=2, ddof=1) / group1_scores.shape[2]
        + group2_scores.var(axis=2, ddof=1) / group2_scores.shape[2]
    )

    return {
        "cache_signature": cache_signature,
        "contrast_key": contrast_key,
        "contrast_label": contrast_label,
        "group1_label": group1_label,
        "group2_label": group2_label,
        "group1_indices": group1_indices.astype(int),
        "group2_indices": group2_indices.astype(int),
        "n_group1": int(group1_indices.size),
        "n_group2": int(group2_indices.size),
        "scout_indices": np.asarray(scout_indices, dtype=int),
        "scout_labels": list(selected_scout_labels),
        "times_ms": times,
        "mean_group1": mean1,
        "mean_group2": mean2,
        "mean_difference": mean_difference,
        "difference_sem": difference_sem,
        "t_observed": np.asarray(t_observed),
        "n_clusters_tested": int(len(clusters)),
        "cluster_p_values": np.asarray(cluster_p_values),
        "significant_mask": significant_mask,
        "positive_mask": positive_mask,
        "negative_mask": negative_mask,
        "significant_clusters": significant_clusters,
        "null_distribution": np.asarray(null_distribution),
        "test": {
            "name": "two-sample scout-by-time cluster permutation test",
            "statistic": "Welch independent-samples t",
            "contrast": contrast_label,
            "tail": 0,
            "n_permutations": args.n_permutations,
            "threshold_p": args.threshold_p,
            "threshold_reference_df": threshold_reference_df,
            "cluster_forming_abs_t": threshold,
            "cluster_alpha": args.cluster_alpha,
            "seed": args.seed,
            "n_jobs": args.n_jobs,
            "buffer_size": args.buffer_size,
            "max_step": 1,
            "adjacency": adjacency_description,
            "spatial_edge_count": int(adjacency.nnz // 2),
            "contrast_correction": (
                "Bonferroni across requested pairwise contrasts"
                if contrast_correction_factor > 1
                else "none; contrast treated as a separate inferential family"
            ),
            "contrast_correction_factor": contrast_correction_factor,
            "correction_scope": (
                "all selected scouts and time points within this contrast"
            ),
        },
    }


def load_or_compute_contrast(
    cache_path: Path,
    cache_signature: dict,
    **compute_kwargs: object,
) -> tuple[dict, bool]:
    if cache_path.exists():
        with open(cache_path, "rb") as file:
            cached = pickle.load(file)
        if cached.get("cache_signature") == cache_signature:
            return cached, True
        print(f"Ignoring incompatible statistics cache: {cache_path}")

    statistics = compute_contrast_statistics(
        cache_signature=cache_signature,
        **compute_kwargs,
    )
    common.atomic_pickle_dump(statistics, cache_path)
    return statistics, False


def write_cluster_summary(
    statistics_by_contrast: dict[str, dict], destination: Path
) -> None:
    fieldnames = (
        "contrast",
        "n_group1",
        "n_group2",
        "cluster_number",
        "direction",
        "raw_cluster_p_value",
        "contrast_adjusted_p_value",
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
        for statistics in statistics_by_contrast.values():
            times = statistics["times_ms"]
            for cluster in statistics["significant_clusters"]:
                mask = cluster["mask"]
                for local_scout, scout_index in enumerate(
                    statistics["scout_indices"]
                ):
                    for interval_number, run in enumerate(
                        common.contiguous_runs(mask[:, local_scout]), start=1
                    ):
                        writer.writerow(
                            {
                                "contrast": statistics["contrast_label"],
                                "n_group1": statistics["n_group1"],
                                "n_group2": statistics["n_group2"],
                                "cluster_number": cluster["cluster_number"],
                                "direction": cluster["direction"],
                                "raw_cluster_p_value": cluster["raw_p_value"],
                                "contrast_adjusted_p_value": cluster[
                                    "adjusted_p_value"
                                ],
                                "cluster_mass": cluster["cluster_mass"],
                                "cluster_n_points": cluster["n_points"],
                                "cluster_n_scouts": cluster["n_scouts"],
                                "scout_index": int(scout_index),
                                "scout_number": int(scout_index) + 1,
                                "scout_label": statistics["scout_labels"][
                                    local_scout
                                ],
                                "interval_number": interval_number,
                                "start_ms": float(times[run[0]]),
                                "end_ms": float(times[run[-1]]),
                                "n_time_samples": int(run.size),
                            }
                        )
    os.replace(temporary, destination)


def plot_cluster_intervals(
    axis: plt.Axes,
    times: np.ndarray,
    mask: np.ndarray,
    y_position: float,
    color: str,
) -> None:
    for run in common.contiguous_runs(mask):
        axis.plot(
            times[run],
            np.full(run.size, y_position),
            color=color,
            linewidth=4,
            solid_capstyle="butt",
            zorder=5,
        )


def create_multipage_figure(
    statistics_by_contrast: dict[str, dict],
    contrast_definitions: list[tuple[str, str, int, int, str, str]],
    scout_indices: list[int],
    scout_labels: list[str],
    destination: Path,
    rows_per_page: int,
    rebuild: bool,
    contrast_correction_factor: int,
) -> None:
    if destination.exists() and not rebuild:
        print(f"Figure already exists; not rebuilding: {destination}")
        return

    all_differences = np.concatenate(
        [
            statistics["mean_difference"].ravel()
            for statistics in statistics_by_contrast.values()
        ]
    ) * 100
    all_sems = np.concatenate(
        [
            statistics["difference_sem"].ravel()
            for statistics in statistics_by_contrast.values()
        ]
    ) * 100
    absolute_limit = max(
        5.0,
        float(np.ceil(np.max(np.abs(all_differences) + all_sems) + 1.0)),
    )

    pdf_scratch_directory = destination.parent / ".pdf_scratch"
    pdf_scratch_directory.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.pdf")
    page_paths = []

    with tempfile.TemporaryDirectory(
        prefix="scout_group_difference_pages_", dir=pdf_scratch_directory
    ) as temporary_directory:
        for page_start in range(0, len(scout_indices), rows_per_page):
            page_scouts = scout_indices[page_start : page_start + rows_per_page]
            figure, axes = plt.subplots(
                len(page_scouts),
                len(contrast_definitions),
                figsize=(7 * len(contrast_definitions), 4.2 * len(page_scouts)),
                sharex=True,
                sharey=True,
                squeeze=False,
            )

            for row, scout_index in enumerate(page_scouts):
                local_scout = page_start + row
                for column, definition in enumerate(contrast_definitions):
                    contrast_key, contrast_label, _, _, _, _ = definition
                    statistics = statistics_by_contrast[contrast_key]
                    axis = axes[row, column]
                    times = statistics["times_ms"]
                    difference = statistics["mean_difference"][local_scout] * 100
                    sem = statistics["difference_sem"][local_scout] * 100

                    axis.plot(times, difference, color="#1f77b4", linewidth=2.2)
                    axis.fill_between(
                        times,
                        difference - sem,
                        difference + sem,
                        color="#1f77b4",
                        alpha=0.28,
                        linewidth=0,
                    )
                    axis.axhline(0, color="black", linestyle="--", linewidth=1.5)
                    axis.axvline(0, color="0.55", linestyle=":", linewidth=1)
                    plot_cluster_intervals(
                        axis,
                        times,
                        statistics["positive_mask"][:, local_scout],
                        y_position=-absolute_limit + 0.6,
                        color="#d62728",
                    )
                    plot_cluster_intervals(
                        axis,
                        times,
                        statistics["negative_mask"][:, local_scout],
                        y_position=-absolute_limit + 1.2,
                        color="#9467bd",
                    )
                    axis.set_xlim(common.T_START_MS, common.T_END_MS)
                    axis.set_ylim(-absolute_limit, absolute_limit)
                    axis.grid(axis="y", color="0.9", linewidth=0.7)
                    axis.tick_params(labelsize=10)

                    if row == 0:
                        axis.set_title(
                            f"{contrast_label}\n"
                            f"(n={statistics['n_group1']} vs "
                            f"{statistics['n_group2']})",
                            fontsize=14,
                            fontweight="bold",
                        )
                    if column == 0:
                        axis.set_ylabel(
                            f"{scout_index + 1:02d}. {scout_labels[scout_index]}\n"
                            "Accuracy difference (percentage points)",
                            fontsize=11,
                            fontweight="bold",
                        )
                    if row == len(page_scouts) - 1:
                        axis.set_xlabel("Time from stimulus onset (ms)", fontsize=11)

            correction_text = (
                f"Bonferroni corrected across {contrast_correction_factor} contrasts"
                if contrast_correction_factor > 1
                else "No additional correction across contrasts"
            )
            figure.suptitle(
                "Direct scout-wise group differences\n"
                "Mean difference ± SEM; red: first > second; purple: first < second",
                fontsize=17,
                fontweight="bold",
                y=0.998,
            )
            figure.text(
                0.5,
                0.004,
                "Cluster FWER is joint across selected scouts and time within each "
                f"contrast; {correction_text}.",
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

        writer = common.PdfWriter()
        for page_path in page_paths:
            reader = common.PdfReader(page_path)
            if len(reader.pages) != 1:
                raise ValueError(f"Expected one page in {page_path}.")
            writer.add_page(reader.pages[0])
        with open(temporary, "wb") as file:
            writer.write(file)

    os.replace(temporary, destination)


def validate_args(args: argparse.Namespace) -> None:
    if args.n_permutations < 2:
        raise ValueError("n-permutations must be at least 2.")
    if not 0 < args.threshold_p < 1:
        raise ValueError("threshold-p must be between 0 and 1.")
    if not 0 < args.cluster_alpha < 1:
        raise ValueError("cluster-alpha must be between 0 and 1.")
    if args.n_jobs == 0:
        raise ValueError("n-jobs cannot be zero.")
    if args.buffer_size < 1:
        raise ValueError("buffer-size must be at least 1.")
    if args.rows_per_page < 1:
        raise ValueError("rows-per-page must be at least 1.")
    if len(set(args.contrasts)) != len(args.contrasts):
        raise ValueError("contrasts must not contain duplicates.")


def main() -> None:
    args = parse_args()
    validate_args(args)

    scores, subject_groups, decoder_result = common.load_decoder_result(args.input)
    n_scouts, n_times, _ = scores.shape
    times = common.make_time_vector(n_times)
    scout_labels = common.load_scout_labels(args.atlas, n_scouts)

    scout_indices = list(range(n_scouts)) if args.scouts is None else args.scouts
    if not scout_indices or len(set(scout_indices)) != len(scout_indices):
        raise ValueError("scouts must contain unique indices and cannot be empty.")
    if min(scout_indices) < 0 or max(scout_indices) >= n_scouts:
        raise IndexError(f"Scout indices must be between 0 and {n_scouts - 1}.")

    contrast_definitions = [
        definition
        for definition in CONTRAST_DEFINITIONS
        if definition[0] in args.contrasts
    ]
    contrast_correction_factor = (
        1 if args.no_correct_across_contrasts else len(contrast_definitions)
    )

    adjacency, adjacency_description = common.load_scout_adjacency(
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
        f"Testing {len(scout_indices)} scouts jointly with "
        f"{args.n_permutations} label permutations per contrast"
    )
    print("Statistic: two-sided independent-samples Welch t")
    print(f"Scout adjacency: {adjacency_description}")
    print(f"Spatial edges: {adjacency.nnz // 2}; permutation workers: {args.n_jobs}")
    print(f"Across-contrast correction factor: {contrast_correction_factor}")

    statistics_by_contrast = {}
    loaded_count = 0
    computed_count = 0
    selection_digest = hashlib.sha256(
        np.asarray(scout_indices, dtype=np.int64).tobytes()
    ).hexdigest()[:12]

    for definition in contrast_definitions:
        (
            contrast_key,
            contrast_label,
            group1_code,
            group2_code,
            group1_label,
            group2_label,
        ) = definition
        group1_indices = np.flatnonzero(subject_groups == group1_code)
        group2_indices = np.flatnonzero(subject_groups == group2_code)
        if group1_indices.size < 2 or group2_indices.size < 2:
            raise ValueError(f"Contrast {contrast_label} has fewer than two subjects.")

        cache_signature = make_cache_signature(
            args=args,
            contrast_key=contrast_key,
            group1_indices=group1_indices,
            group2_indices=group2_indices,
            scout_indices=scout_indices,
            scores_shape=scores.shape,
            times=times,
            adjacency=adjacency,
            contrast_correction_factor=contrast_correction_factor,
        )
        cache_path = (
            statistics_directory
            / f"contrast_{contrast_key}_scouts_{selection_digest}.pkl"
        )
        print(
            f"Analyzing {contrast_label} "
            f"(n={group1_indices.size} vs {group2_indices.size})..."
        )
        statistics, loaded = load_or_compute_contrast(
            cache_path=cache_path,
            cache_signature=cache_signature,
            selected_scores=selected_scores,
            times=times,
            scout_indices=scout_indices,
            selected_scout_labels=selected_scout_labels,
            contrast_key=contrast_key,
            contrast_label=contrast_label,
            group1_label=group1_label,
            group2_label=group2_label,
            group1_indices=group1_indices,
            group2_indices=group2_indices,
            adjacency=adjacency,
            adjacency_description=adjacency_description,
            contrast_correction_factor=contrast_correction_factor,
            args=args,
        )
        statistics_by_contrast[contrast_key] = statistics
        loaded_count += int(loaded)
        computed_count += int(not loaded)
        source = "cache" if loaded else "computed"
        print(
            f"  {source}; significant clusters: "
            f"{len(statistics['significant_clusters'])}"
        )

    summary_path = args.output_dir / "cluster_summary.csv"
    write_cluster_summary(statistics_by_contrast, summary_path)

    figure_path = args.output_dir / "scout_decoding_group_differences.pdf"
    create_multipage_figure(
        statistics_by_contrast=statistics_by_contrast,
        contrast_definitions=contrast_definitions,
        scout_indices=scout_indices,
        scout_labels=scout_labels,
        destination=figure_path,
        rows_per_page=args.rows_per_page,
        rebuild=args.rebuild_figure or computed_count > 0,
        contrast_correction_factor=contrast_correction_factor,
    )

    metadata = {
        "input_path": str(args.input.resolve()),
        "input_axis_order": decoder_result.get("axis_order"),
        "decode_parameters": decoder_result.get("parameters"),
        "score_shape": scores.shape,
        "time_range_ms": (float(times[0]), float(times[-1])),
        "sampling_rate_hz": common.SAMPLING_RATE_HZ,
        "contrasts_analyzed": [definition[0] for definition in contrast_definitions],
        "scout_indices": scout_indices,
        "n_scouts_processed": len(scout_indices),
        "test_parameters": {
            "name": "two-sample scout-by-time cluster permutation test",
            "statistic": "Welch independent-samples t",
            "n_permutations": args.n_permutations,
            "threshold_p": args.threshold_p,
            "cluster_alpha": args.cluster_alpha,
            "tail": 0,
            "seed": args.seed,
            "n_jobs": args.n_jobs,
            "buffer_size": args.buffer_size,
            "adjacency": adjacency_description,
            "spatial_edge_count": int(adjacency.nnz // 2),
            "contrast_correction_factor": contrast_correction_factor,
            "correction_scope": (
                "all selected scouts and time points within each contrast; "
                + (
                    "Bonferroni across requested contrasts"
                    if contrast_correction_factor > 1
                    else "no additional correction across contrasts"
                )
            ),
        },
    }
    common.atomic_pickle_dump(metadata, args.output_dir / "run_metadata.pkl")

    significant_scout_contrast_panels = sum(
        int(statistics["significant_mask"].any(axis=0).sum())
        for statistics in statistics_by_contrast.values()
    )
    significant_clusters = sum(
        len(statistics["significant_clusters"])
        for statistics in statistics_by_contrast.values()
    )
    print(f"Contrast caches loaded: {loaded_count}")
    print(f"Contrast tests computed: {computed_count}")
    print(
        "Scout/contrast panels containing significance: "
        f"{significant_scout_contrast_panels}"
    )
    print(f"Significant joint clusters: {significant_clusters}")
    print(f"Figure: {figure_path}")
    print(f"Cluster summary: {summary_path}")


if __name__ == "__main__":
    main()
