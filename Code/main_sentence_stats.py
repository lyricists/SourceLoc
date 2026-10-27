"""Scout-wise sentence decoding vs chance with joint scout-by-time cluster inference.

Each group is analyzed with one one-tailed, one-sample spatiotemporal cluster
permutation test on accuracy minus 50% chance (input: main_sentence_decoding.py). The
maximum-cluster null distribution controls family-wise error jointly across all
selected scouts and time points within that group. Cluster p-values are then
Bonferroni-corrected across the groups of --grouping:
  three             Control, Depressed, Suicidal (x3)
  control-patients  Control and Patients = Depressed + Suicidal pooled (x2)

Statistics are cached per group and reused when the settings and input match; the
PDF is redrawn only after new statistics or with --rebuild-figure. The significance
masks of the three-group run feed main_source_significant.py.

Examples (run from Code/):
    python3 main_sentence_stats.py --input ../Results/svmDecoding_scout_con_Int.pkl
    python3 main_sentence_stats.py --input ../Results/svmDecoding_scout_con_Int.pkl --grouping control-patients
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse

from utils import io, paths, plotting, stats

CHANCE_LEVEL = 0.5
T_EPOCH_END_EXCLUSIVE_MS = 1000
T_END_MS = T_EPOCH_END_EXCLUSIVE_MS - 1000 / paths.FS
GROUPINGS = io.GROUPINGS
GROUPING_TEXT = {
    "three": ("Bonferroni across Control, Depressed, and Suicidal", "the three diagnostic groups",
              "_3group_corrected", "scout_decoding_three_groups.pdf"),
    "control-patients": ("Bonferroni across Control and Patients (Depressed + Suicidal)", "the two groups",
                         "_control_patients", "scout_decoding_control_patients.pdf"),
}
# Signature keys older runs stored that the current code no longer records.
LEGACY = {"cache_version": None, "group_codes": None}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=paths.RESULTS / "svmDecoding_scout_con_Int.pkl")
    parser.add_argument("--atlas", type=Path, default=paths.ATLAS)
    parser.add_argument("--grouping", choices=GROUPINGS, default="three")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Default: Results/CBP_Chance_scout_<condition>_3group_corrected (or _control_patients)")
    parser.add_argument("--n-permutations", type=int, default=10_000)
    parser.add_argument("--threshold-p", type=float, default=0.05)
    parser.add_argument("--cluster-alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel permutation workers; -1 uses every core.")
    parser.add_argument("--buffer-size", type=int, default=1000,
                        help="Number of scout-time statistics computed per worker block.")
    parser.add_argument("--rows-per-page", type=int, default=4)
    parser.add_argument("--rebuild-figure", action="store_true",
                        help="Rebuild the PDF even when cached statistics are unchanged.")
    parser.add_argument("--scouts", type=int, nargs="*", default=None,
                        help="Optional zero-based scout indices. The default processes all scouts.")
    parser.add_argument("--groups", nargs="+", default=None,
                        help="Group keys of --grouping to analyze (default all); correction always covers all.")
    parser.add_argument("--scout-adjacency", type=Path, default=None,
                        help="Optional anatomical scout adjacency (.npz, .npy, or pickle). Default: no scout edges.")
    args = parser.parse_args()
    keys = [g[0] for g in GROUPINGS[args.grouping]]
    args.groups = args.groups or keys
    if not set(args.groups) <= set(keys) or len(set(args.groups)) != len(args.groups):
        parser.error(f"--groups must be distinct keys of {args.grouping}: {keys}")
    if args.output_dir is None:
        condition = args.input.stem.removeprefix("svmDecoding_scout_")
        args.output_dir = paths.RESULTS / f"CBP_Chance_scout_{condition}{GROUPING_TEXT[args.grouping][2]}"
    if args.n_permutations < 2:
        parser.error("n-permutations must be at least 2.")
    if not 0 < args.threshold_p < 0.5 or not 0 < args.cluster_alpha < 1:
        parser.error("threshold-p must be in (0, 0.5) and cluster-alpha in (0, 1).")
    if args.n_jobs == 0 or args.buffer_size < 1 or args.rows_per_page < 1:
        parser.error("n-jobs cannot be zero; buffer-size and rows-per-page must be positive.")
    return args


def load_decoder_result(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """decodeScore as scout x time x subject, subject group codes, and the whole result."""
    result = io.load_pickle(path)
    scores = np.asarray(result["decodeScore"], dtype=np.float64)
    subject_groups = np.asarray(result["subIdx"]).squeeze()
    if scores.ndim != 3:
        raise ValueError(f"decodeScore must be scout x time x subject or subject x scout x time; got {scores.shape}.")
    if scores.shape[-1] != subject_groups.size:
        if scores.shape[0] != subject_groups.size:
            raise ValueError(f"No decodeScore axis matches the {subject_groups.size} subject labels: {scores.shape}.")
        scores = scores.transpose(1, 2, 0)
    if not np.isfinite(scores).all():
        raise ValueError("decodeScore contains NaN or infinite values.")
    if scores.min() < 0 or scores.max() > 1:
        raise ValueError("decodeScore values must be proportions between 0 and 1.")
    return scores, subject_groups, result


def make_time_vector(n_times: int, recorded_times: np.ndarray | None = None) -> np.ndarray:
    times = np.arange(paths.EPOCH_START_MS, T_EPOCH_END_EXCLUSIVE_MS, 1000 / paths.FS)
    if times.size != n_times:
        raise ValueError(f"Expected {times.size} samples in [{paths.EPOCH_START_MS}, {T_EPOCH_END_EXCLUSIVE_MS}) ms, "
                         f"but decodeScore contains {n_times}.")
    if recorded_times is not None:
        recorded = np.asarray(recorded_times, dtype=np.float64)
        if recorded.shape != times.shape or not np.allclose(recorded, times, rtol=0, atol=1e-6):
            raise ValueError("Decoder times_ms does not match the 250 Hz [-200, 1000) ms epoch.")
        return recorded
    return times


def cache_signature(args, grouping, group_key, subject_indices, scout_indices, scores_shape, times, adjacency) -> dict:
    input_stat = args.input.stat()
    return {
        "input_path": str(args.input.resolve()), "input_size": int(input_stat.st_size),
        "input_mtime_ns": int(input_stat.st_mtime_ns), "scores_shape": tuple(int(v) for v in scores_shape),
        "group_key": group_key, "subject_indices": subject_indices.astype(int).tolist(),
        "scout_indices": list(scout_indices), "n_times": int(times.size), "first_time_ms": float(times[0]),
        "last_time_ms": float(times[-1]), "chance_level": CHANCE_LEVEL,
        "group_family": tuple(g[0] for g in grouping), "group_correction_factor": len(grouping),
        "n_permutations": int(args.n_permutations), "threshold_p": float(args.threshold_p),
        "cluster_alpha": float(args.cluster_alpha), "tail": 1, "max_step": 1, "t_power": 1,
        "seed": int(args.seed), "adjacency_sha256": stats.adjacency_digest(adjacency),
    }


def compute_group_statistics(selected_scores, times, scout_indices, selected_scout_labels, grouping_name, group,
                             subject_indices, adjacency, adjacency_description, cache_signature, args) -> dict:
    grouping = GROUPINGS[grouping_name]
    correction, family_text = GROUPING_TEXT[grouping_name][:2]
    factor = len(grouping)
    group_key, group_label, group_codes = group
    # selected_scores: selected scout x time x all subjects.
    group_scores = np.take(selected_scores, subject_indices, axis=2)
    centered = np.ascontiguousarray(group_scores.transpose(2, 1, 0) - CHANCE_LEVEL, dtype=np.float64)
    t_observed, clusters, cluster_p_values, null_distribution, threshold = stats.scout_time_cluster_1samp(
        centered, args.threshold_p, args.n_permutations, adjacency, args.seed, args.n_jobs, args.buffer_size)

    expected_shape = (times.size, len(scout_indices))
    significant_mask = np.zeros(expected_shape, dtype=bool)
    significant_clusters = []
    adjusted = np.minimum(1.0, np.asarray(cluster_p_values, dtype=np.float64) * factor)
    for cluster_index, (cluster, raw_p, adjusted_p) in enumerate(zip(clusters, cluster_p_values, adjusted)):
        if adjusted_p >= args.cluster_alpha:
            continue
        mask = stats.cluster_to_mask(cluster, expected_shape)
        significant_mask |= mask
        local_scouts = np.flatnonzero(mask.any(axis=0))
        time_positions = np.flatnonzero(mask.any(axis=1))
        significant_clusters.append({
            "cluster_index": int(cluster_index), "cluster_number": int(cluster_index + 1), "mask": mask,
            "raw_p_value": float(raw_p), "adjusted_p_value": float(adjusted_p),
            "cluster_mass": float(t_observed[mask].sum()), "n_points": int(mask.sum()),
            "n_scouts": int(local_scouts.size), "scout_indices": [int(scout_indices[p]) for p in local_scouts],
            "start_ms": float(times[time_positions[0]]), "end_ms": float(times[time_positions[-1]]),
        })
    return {
        "cache_signature": cache_signature, "group_key": group_key, "group_label": group_label,
        "group_codes": tuple(group_codes), "subject_indices": subject_indices.astype(int),
        "n_subjects": int(subject_indices.size), "scout_indices": np.asarray(scout_indices, dtype=int),
        "scout_labels": list(selected_scout_labels), "times_ms": times, "chance_level": CHANCE_LEVEL,
        "mean_accuracy": group_scores.mean(axis=2),
        "sem_accuracy": group_scores.std(axis=2, ddof=1) / np.sqrt(group_scores.shape[2]),
        "t_observed": t_observed, "n_clusters_tested": int(len(clusters)),
        "cluster_p_values": np.asarray(cluster_p_values), "adjusted_cluster_p_values": adjusted,
        "significant_mask": significant_mask, "significant_clusters": significant_clusters,
        "null_distribution": null_distribution,
        "test": {
            "name": "one-sample scout-by-time cluster permutation test",
            "contrast": "cross-validated balanced accuracy minus 0.5", "tail": 1,
            "n_permutations": args.n_permutations, "threshold_p": args.threshold_p, "cluster_forming_t": threshold,
            "cluster_alpha": args.cluster_alpha, "group_family": tuple(g[0] for g in grouping),
            "group_correction": correction, "group_correction_factor": factor, "seed": args.seed,
            "n_jobs": args.n_jobs, "buffer_size": args.buffer_size, "max_step": 1,
            "adjacency": adjacency_description, "spatial_edge_count": int(adjacency.nnz // 2),
            "correction_scope": f"all selected scouts and time points within each group; Bonferroni across {family_text}",
        },
    }


def load_or_compute_group(cache_path: Path, signature: dict, **compute_kwargs) -> tuple[dict, str]:
    if cache_path.exists():
        cached = io.load_pickle(cache_path)
        if io.signatures_match(cached.get("cache_signature"), signature, LEGACY):
            return cached, "cache"
        print(f"Ignoring incompatible statistics cache: {cache_path}")
    statistics = compute_group_statistics(cache_signature=signature, **compute_kwargs)
    io.save_pickle(cache_path, statistics)
    return statistics, "computed"


def write_cluster_summary(statistics_by_group: dict[str, dict], destination: Path) -> None:
    fields = ["group", "n_subjects", "cluster_number", "raw_cluster_p_value", "group_adjusted_p_value",
              "cluster_mass", "cluster_n_points", "cluster_n_scouts", "scout_index", "scout_number", "scout_label",
              "interval_number", "start_ms", "end_ms", "n_time_samples"]
    rows = []
    for statistics in statistics_by_group.values():
        times = statistics["times_ms"]
        for cluster in statistics["significant_clusters"]:
            for local, scout in enumerate(statistics["scout_indices"]):
                for number, run in enumerate(stats.contiguous_runs(cluster["mask"][:, local]), start=1):
                    rows.append({
                        "group": statistics["group_label"], "n_subjects": statistics["n_subjects"],
                        "cluster_number": cluster["cluster_number"], "raw_cluster_p_value": cluster["raw_p_value"],
                        "group_adjusted_p_value": cluster["adjusted_p_value"],
                        "cluster_mass": cluster["cluster_mass"], "cluster_n_points": cluster["n_points"],
                        "cluster_n_scouts": cluster["n_scouts"], "scout_index": int(scout),
                        "scout_number": int(scout) + 1, "scout_label": statistics["scout_labels"][local],
                        "interval_number": number, "start_ms": float(times[run[0]]), "end_ms": float(times[run[-1]]),
                        "n_time_samples": int(run.size)})
    io.save_csv(destination, rows, fields)


def create_multipage_figure(statistics_by_group, groups, scout_indices, scout_labels, destination, rows_per_page,
                            grouping_name) -> None:
    means = np.concatenate([s["mean_accuracy"].ravel() for s in statistics_by_group.values()]) * 100
    sems = np.concatenate([s["sem_accuracy"].ravel() for s in statistics_by_group.values()]) * 100
    lower = min(45.0, float(np.floor(np.min(means - sems) - 1)))
    upper = max(65.0, float(np.ceil(np.max(means + sems) + 1)))
    family = ", ".join(g[1] for g in GROUPINGS[grouping_name])
    with plotting.MultipagePdf(destination) as pdf:
        for page_start in range(0, len(scout_indices), rows_per_page):
            page_scouts = scout_indices[page_start:page_start + rows_per_page]
            figure, axes = plt.subplots(len(page_scouts), len(groups), figsize=(6 * len(groups), 4.2 * len(page_scouts)),
                                        sharex=True, sharey=True, squeeze=False)
            for row, scout_index in enumerate(page_scouts):
                local = page_start + row
                for column, (group_key, group_label, _) in enumerate(groups):
                    statistics = statistics_by_group[group_key]
                    axis = axes[row, column]
                    times = statistics["times_ms"]
                    mean = statistics["mean_accuracy"][local] * 100
                    sem = statistics["sem_accuracy"][local] * 100
                    axis.plot(times, mean, color="#1f77b4", linewidth=2.2)
                    axis.fill_between(times, mean - sem, mean + sem, color="#1f77b4", alpha=0.28, linewidth=0)
                    axis.axhline(50, color="black", linestyle="--", linewidth=1.5)
                    axis.axvline(0, color="0.55", linestyle=":", linewidth=1)
                    plotting.plot_significant_intervals(axis, times, statistics["significant_mask"][:, local],
                                                        lower + 0.8)
                    axis.set_xlim(paths.EPOCH_START_MS, T_END_MS)
                    axis.set_ylim(lower, upper)
                    axis.grid(axis="y", color="0.9", linewidth=0.7)
                    axis.tick_params(labelsize=10)
                    if row == 0:
                        axis.set_title(f"{group_label} (n={statistics['n_subjects']})", fontsize=14, fontweight="bold")
                    if column == 0:
                        axis.set_ylabel(f"{io.pretty_scout_name(scout_labels[scout_index])}\nBalanced accuracy (%)",
                                        fontsize=11, fontweight="bold")
                    if row == len(page_scouts) - 1:
                        axis.set_xlabel("Time from stimulus onset (ms)", fontsize=11)
            figure.suptitle("Scout-wise sentence decoding by group\n"
                            "Mean ± SEM; red bars: group-adjusted cluster p < 0.05 vs 50%",
                            fontsize=17, fontweight="bold", y=0.998)
            figure.text(0.5, 0.004, "Cluster permutation correction spans selected scouts and time; Bonferroni "
                        f"correction spans {family} ({len(GROUPINGS[grouping_name])} tests).", ha="center", fontsize=9)
            figure.tight_layout(rect=(0.02, 0.025, 1, 0.965))
            pdf.add(figure)


def main() -> None:
    args = parse_args()
    scores, subject_groups, decoder_result = load_decoder_result(args.input)
    n_scouts, n_times, _ = scores.shape
    times = make_time_vector(n_times, decoder_result.get("times_ms"))
    scout_labels = io.load_scout_labels(args.atlas, n_scouts)
    scout_indices = list(range(n_scouts)) if args.scouts is None else args.scouts
    if not scout_indices or len(set(scout_indices)) != len(scout_indices):
        raise ValueError("scouts must contain unique indices and cannot be empty.")
    if min(scout_indices) < 0 or max(scout_indices) >= n_scouts:
        raise IndexError(f"Scout indices must be between 0 and {n_scouts - 1}.")

    grouping = GROUPINGS[args.grouping]
    groups = [g for g in grouping if g[0] in args.groups]
    group_indices = {}
    for key, _, codes in groups:
        indices = np.flatnonzero(np.isin(subject_groups, codes))
        if indices.size < 2:
            raise ValueError(f"Group {key} has fewer than two subjects.")
        group_indices[key] = indices
    adjacency, adjacency_description = stats.load_scout_adjacency(args.scout_adjacency, n_scouts, scout_indices)
    selected_scores = np.ascontiguousarray(scores[scout_indices, :, :])
    selected_labels = [scout_labels[i] for i in scout_indices]
    statistics_directory = args.output_dir / "spatiotemporal_statistics"
    statistics_directory.mkdir(parents=True, exist_ok=True)

    print(f"Input: {args.input}\ndecodeScore: {scores.shape} (scout x time x subject)")
    print("Groups: " + ", ".join(f"{label}={group_indices[key].size}" for key, label, _ in groups))
    print(f"Testing {len(scout_indices)} scouts jointly with {args.n_permutations} permutations per group")
    print(f"Cluster p-values: {GROUPING_TEXT[args.grouping][0]}")
    print(f"Scout adjacency: {adjacency_description}; permutation workers: {args.n_jobs}")

    statistics_by_group, computed = {}, 0
    digest = hashlib.sha256(np.asarray(scout_indices, dtype=np.int64).tobytes()).hexdigest()[:12]
    for group in groups:
        key, label, _ = group
        signature = cache_signature(args, grouping, key, group_indices[key], scout_indices, scores.shape, times,
                                    adjacency)
        print(f"Analyzing {label} (n={group_indices[key].size})...")
        statistics, source = load_or_compute_group(
            statistics_directory / f"group_{key}_scouts_{digest}.pkl", signature,
            selected_scores=selected_scores, times=times, scout_indices=scout_indices,
            selected_scout_labels=selected_labels, grouping_name=args.grouping, group=group,
            subject_indices=group_indices[key], adjacency=adjacency, adjacency_description=adjacency_description,
            args=args)
        statistics_by_group[key] = statistics
        computed += int(source == "computed")
        print(f"  {source}; significant clusters: {len(statistics['significant_clusters'])}")

    summary_path = args.output_dir / "cluster_summary.csv"
    write_cluster_summary(statistics_by_group, summary_path)
    figure_path = args.output_dir / GROUPING_TEXT[args.grouping][3]
    if args.rebuild_figure or computed or not figure_path.exists():
        create_multipage_figure(statistics_by_group, groups, scout_indices, scout_labels, figure_path,
                                args.rows_per_page, args.grouping)
    else:
        print(f"Figure already exists; not rebuilding: {figure_path}")
    io.save_pickle(args.output_dir / "run_metadata.pkl", {
        "input_path": str(args.input.resolve()), "input_axis_order": decoder_result.get("axis_order"),
        "decode_parameters": decoder_result.get("parameters"), "score_shape": scores.shape,
        "time_range_ms": (float(times[0]), float(times[-1])), "epoch_end_exclusive_ms": T_EPOCH_END_EXCLUSIVE_MS,
        "sampling_rate_hz": paths.FS, "grouping": args.grouping, "groups_analyzed": [g[0] for g in groups],
        "group_codes": {label: tuple(codes) for _, label, codes in groups},
        "group_sizes": {label: int(group_indices[key].size) for key, label, _ in groups},
        "scout_indices": scout_indices, "n_scouts_processed": len(scout_indices),
        "test_parameters": next(iter(statistics_by_group.values()))["test"],
    })
    print(f"Group tests computed: {computed}; loaded from cache: {len(groups) - computed}")
    print(f"Significant joint clusters: {sum(len(s['significant_clusters']) for s in statistics_by_group.values())}")
    print(f"Figure: {figure_path}\nCluster summary: {summary_path}")


if __name__ == "__main__":
    main()
