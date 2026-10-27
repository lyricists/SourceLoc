"""Plot scout-wise decoding for Control vs. Patients with scout-by-time CBP.

Same analysis as ``plot_scout_decoding.py`` but with two groups: Control
(group code 1) and Patients (Depressed + Suicidal, codes 2 and 3 pooled).
Each group is analyzed with one one-tailed, one-sample spatiotemporal cluster
permutation test on accuracy minus 50% chance. The maximum-cluster null
distribution controls family-wise error jointly across all selected scouts
and time points within that group. Cluster p-values are then
Bonferroni-corrected across the two groups.

"""

from __future__ import annotations

import argparse
import hashlib
import os
import pickle
import tempfile
from pathlib import Path

import numpy as np

import plot_scout_decoding as common

from mne.stats import spatio_temporal_cluster_1samp_test
from scipy import sparse
from scipy.stats import t as t_distribution

import matplotlib.pyplot as plt

DEFAULT_OUTPUT = (
    common.BASE_DIRECTORY / "Results" / "CBP_Chance_scout_sent_noBio_control_patients"
)

# (key, label, subject_index codes pooled into the group)
GROUP_DEFINITIONS = (
    ("control", "Control", (1,)),
    ("patients", "Patients (D+S)", (2, 3)),
)
GROUP_KEYS = tuple(definition[0] for definition in GROUP_DEFINITIONS)
GROUP_CORRECTION_FACTOR = len(GROUP_DEFINITIONS)
GROUP_CORRECTION = "Bonferroni across Control and Patients (Depressed + Suicidal)"
CORRECTION_SCOPE = (
    "all selected scouts and time points within each group; "
    "Bonferroni across the two groups"
)


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
        "--groups",
        choices=GROUP_KEYS,
        nargs="+",
        default=list(GROUP_KEYS),
        help="Groups to analyze. Correction always covers both groups.",
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


def make_cache_signature(
    args: argparse.Namespace,
    group_key: str,
    group_codes: tuple[int, ...],
    subject_indices: np.ndarray,
    scout_indices: list[int],
    scores_shape: tuple[int, ...],
    times: np.ndarray,
    adjacency: sparse.csr_matrix,
) -> dict:
    input_stat = args.input.stat()
    return {
        "input_path": str(args.input.resolve()),
        "input_size": int(input_stat.st_size),
        "input_mtime_ns": int(input_stat.st_mtime_ns),
        "scores_shape": tuple(int(value) for value in scores_shape),
        "group_key": group_key,
        "group_codes": tuple(int(code) for code in group_codes),
        "subject_indices": subject_indices.astype(int).tolist(),
        "scout_indices": list(scout_indices),
        "n_times": int(times.size),
        "first_time_ms": float(times[0]),
        "last_time_ms": float(times[-1]),
        "chance_level": common.CHANCE_LEVEL,
        "group_family": GROUP_KEYS,
        "group_correction_factor": GROUP_CORRECTION_FACTOR,
        "n_permutations": int(args.n_permutations),
        "threshold_p": float(args.threshold_p),
        "cluster_alpha": float(args.cluster_alpha),
        "tail": 1,
        "max_step": 1,
        "t_power": 1,
        "seed": int(args.seed),
        "adjacency_sha256": common.adjacency_digest(adjacency),
    }


def compute_group_statistics(
    selected_scores: np.ndarray,
    times: np.ndarray,
    scout_indices: list[int],
    selected_scout_labels: list[str],
    group_key: str,
    group_label: str,
    group_codes: tuple[int, ...],
    subject_indices: np.ndarray,
    adjacency: sparse.csr_matrix,
    adjacency_description: str,
    cache_signature: dict,
    args: argparse.Namespace,
) -> dict:
    # selected_scores: selected scout x time x all subjects.
    group_scores = np.take(selected_scores, subject_indices, axis=2)
    centered_scores = np.ascontiguousarray(
        group_scores.transpose(2, 1, 0) - common.CHANCE_LEVEL,
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
        mask = common.cluster_to_mask(cluster, expected_shape)
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
        "group_codes": tuple(group_codes),
        "subject_indices": subject_indices.astype(int),
        "n_subjects": int(subject_indices.size),
        "scout_indices": np.asarray(scout_indices, dtype=int),
        "scout_labels": list(selected_scout_labels),
        "times_ms": times,
        "chance_level": common.CHANCE_LEVEL,
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
            "group_correction": GROUP_CORRECTION,
            "group_correction_factor": GROUP_CORRECTION_FACTOR,
            "seed": args.seed,
            "n_jobs": args.n_jobs,
            "buffer_size": args.buffer_size,
            "max_step": 1,
            "adjacency": adjacency_description,
            "spatial_edge_count": int(adjacency.nnz // 2),
            "correction_scope": CORRECTION_SCOPE,
        },
    }


def load_or_compute_group(
    cache_path: Path, cache_signature: dict, **compute_kwargs: object
) -> tuple[dict, str]:
    if cache_path.exists():
        with open(cache_path, "rb") as file:
            cached = pickle.load(file)
        if cached.get("cache_signature") == cache_signature:
            return cached, "cache"
        print(f"Ignoring incompatible statistics cache: {cache_path}")

    statistics = compute_group_statistics(
        cache_signature=cache_signature, **compute_kwargs
    )
    common.atomic_pickle_dump(statistics, cache_path)
    return statistics, "computed"


def create_multipage_figure(
    statistics_by_group: dict[str, dict],
    group_definitions: list[tuple[str, str, tuple[int, ...]]],
    scout_indices: list[int],
    scout_labels: list[str],
    destination: Path,
    rows_per_page: int,
    rebuild: bool,
) -> None:
    if destination.exists() and not rebuild:
        print(f"Figure already exists; not rebuilding: {destination}")
        return

    all_means = np.concatenate(
        [statistics["mean_accuracy"].ravel() for statistics in statistics_by_group.values()]
    ) * 100
    all_sems = np.concatenate(
        [statistics["sem_accuracy"].ravel() for statistics in statistics_by_group.values()]
    ) * 100
    lower_limit = min(45.0, float(np.floor(np.min(all_means - all_sems) - 1)))
    upper_limit = max(65.0, float(np.ceil(np.max(all_means + all_sems) + 1)))

    pdf_scratch_directory = destination.parent / ".pdf_scratch"
    pdf_scratch_directory.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.pdf")
    page_paths = []

    # Pages are saved independently and merged, as in plot_scout_decoding.py.
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
                    common.plot_significant_intervals(
                        axis,
                        times,
                        statistics["significant_mask"][:, local_scout],
                        y_position=lower_limit + 0.8,
                    )
                    axis.set_xlim(common.T_START_MS, common.T_END_MS)
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
                "Scout-wise decoding: Control vs. Patients\n"
                "Mean ± SEM; red bars: group-adjusted cluster p < 0.05 vs 50%",
                fontsize=17,
                fontweight="bold",
                y=0.998,
            )
            figure.text(
                0.5,
                0.004,
                "Cluster permutation correction spans selected scouts and time; "
                "Bonferroni correction spans Control and Patients (2 tests).",
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

    scores, subject_groups, decoder_result = common.load_decoder_result(args.input)
    n_scouts, n_times, n_subjects = scores.shape
    times = common.make_time_vector(n_times, decoder_result.get("times_ms"))
    scout_labels = common.load_scout_labels(args.atlas, n_scouts)

    scout_indices = list(range(n_scouts)) if args.scouts is None else args.scouts
    if not scout_indices or len(set(scout_indices)) != len(scout_indices):
        raise ValueError("scouts must contain unique indices and cannot be empty.")
    if min(scout_indices) < 0 or max(scout_indices) >= n_scouts:
        raise IndexError(f"Scout indices must be between 0 and {n_scouts - 1}.")

    group_definitions = [
        definition for definition in GROUP_DEFINITIONS if definition[0] in args.groups
    ]
    group_indices = {}
    for group_key, _, group_codes in group_definitions:
        indices = np.flatnonzero(np.isin(subject_groups, group_codes))
        if indices.size < 2:
            raise ValueError(f"Group {group_key} has fewer than two subjects.")
        group_indices[group_key] = indices

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
        "Groups: "
        + ", ".join(
            f"{label}={group_indices[key].size}" for key, label, _ in group_definitions
        )
    )
    print(
        f"Testing {len(scout_indices)} scouts jointly with "
        f"{args.n_permutations} permutations per group"
    )
    print("Cluster p-values are Bonferroni-corrected across Control and Patients")
    print(f"Scout adjacency: {adjacency_description}")
    print(f"Spatial edges: {adjacency.nnz // 2}; permutation workers: {args.n_jobs}")

    statistics_by_group = {}
    computed_count = 0
    selection_digest = hashlib.sha256(
        np.asarray(scout_indices, dtype=np.int64).tobytes()
    ).hexdigest()[:12]

    for group_key, group_label, group_codes in group_definitions:
        subject_indices = group_indices[group_key]
        cache_signature = make_cache_signature(
            args=args,
            group_key=group_key,
            group_codes=group_codes,
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
        statistics, source = load_or_compute_group(
            cache_path=cache_path,
            cache_signature=cache_signature,
            selected_scores=selected_scores,
            times=times,
            scout_indices=scout_indices,
            selected_scout_labels=selected_scout_labels,
            group_key=group_key,
            group_label=group_label,
            group_codes=group_codes,
            subject_indices=subject_indices,
            adjacency=adjacency,
            adjacency_description=adjacency_description,
            args=args,
        )
        statistics_by_group[group_key] = statistics
        computed_count += int(source == "computed")
        print(
            f"  {source}; significant clusters: "
            f"{len(statistics['significant_clusters'])}"
        )

    summary_path = args.output_dir / "cluster_summary.csv"
    common.write_cluster_summary(statistics_by_group, summary_path)

    figure_path = args.output_dir / "scout_decoding_control_patients.pdf"
    create_multipage_figure(
        statistics_by_group=statistics_by_group,
        group_definitions=group_definitions,
        scout_indices=scout_indices,
        scout_labels=scout_labels,
        destination=figure_path,
        rows_per_page=args.rows_per_page,
        rebuild=args.rebuild_figure or computed_count > 0,
    )

    metadata = {
        "input_path": str(args.input.resolve()),
        "input_axis_order": decoder_result.get("axis_order"),
        "decode_parameters": decoder_result.get("parameters"),
        "score_shape": scores.shape,
        "time_range_ms": (float(times[0]), float(times[-1])),
        "epoch_end_exclusive_ms": common.T_EPOCH_END_EXCLUSIVE_MS,
        "sampling_rate_hz": common.SAMPLING_RATE_HZ,
        "groups_analyzed": [definition[0] for definition in group_definitions],
        "group_codes": {
            label: tuple(codes) for _, label, codes in group_definitions
        },
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
            "group_correction": GROUP_CORRECTION,
            "group_correction_factor": GROUP_CORRECTION_FACTOR,
            "tail": 1,
            "seed": args.seed,
            "n_jobs": args.n_jobs,
            "buffer_size": args.buffer_size,
            "adjacency": adjacency_description,
            "spatial_edge_count": int(adjacency.nnz // 2),
            "correction_scope": CORRECTION_SCOPE,
        },
    }
    common.atomic_pickle_dump(metadata, args.output_dir / "run_metadata.pkl")

    significant_scout_group_panels = sum(
        int(statistics["significant_mask"].any(axis=0).sum())
        for statistics in statistics_by_group.values()
    )
    significant_clusters = sum(
        len(statistics["significant_clusters"])
        for statistics in statistics_by_group.values()
    )
    print(f"Group caches loaded: {len(statistics_by_group) - computed_count}")
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
