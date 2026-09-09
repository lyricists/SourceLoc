"""Plot scout-wise source-decoding results with temporal cluster inference.

The input decoder output contains one cross-validated balanced-accuracy value
for every scout, time point, and subject. Trial/pseudotrial and fold averaging
have already occurred in ``svm_source_scout.py``; subjects are therefore the
independent observations used here for group-level inference.

For each scout and group, this script runs a one-tailed, one-sample temporal
cluster permutation test on accuracy minus 50% chance. Each panel is cached in
its own pickle file, so interrupted runs resume without recomputing finished
tests. Correction is across time within a panel, not across scouts or groups.
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import re
import sys
import tempfile
from pathlib import Path

import mat73
import matplotlib
import numpy as np
from mne.stats import permutation_cluster_1samp_test
from scipy.stats import t as t_distribution

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from pypdf import PdfReader, PdfWriter
except ModuleNotFoundError:
    # Codex's bundled artifact runtime includes pypdf even when the scientific
    # Python environment used for MNE does not. Append (rather than prepend)
    # the pure-Python package location so NumPy/SciPy continue to come from the
    # active scientific environment.
    bundled_site_packages = (
        Path.home()
        / ".cache"
        / "codex-runtimes"
        / "codex-primary-runtime"
        / "dependencies"
        / "python"
        / "lib"
        / "python3.12"
        / "site-packages"
    )
    if bundled_site_packages.exists():
        sys.path.append(str(bundled_site_packages))
    from pypdf import PdfReader, PdfWriter


BASE_DIRECTORY = Path(
    "/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Code/SourceLoc"
)
DEFAULT_INPUT = BASE_DIRECTORY / "Results" / "svmDecoding_scout_3pc.pkl"
DEFAULT_ATLAS = (
    BASE_DIRECTORY.parent.parent
    / "Data"
    / "Preprocessed data"
    / "DKT_scout.mat"
)
DEFAULT_OUTPUT = BASE_DIRECTORY / "Results" / "CBP_Chance_scout_3pc"

T_START_MS = -200
T_END_MS = 1000
SAMPLING_RATE_HZ = 250
CHANCE_LEVEL = 0.5

GROUP_DEFINITIONS = (
    ("all", "All", None),
    ("control", "Control", 1),
    ("depressed", "Depressed", 2),
    ("suicidal", "Suicidal", 3),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--atlas", type=Path, default=DEFAULT_ATLAS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-permutations", type=int, default=5000)
    parser.add_argument("--threshold-p", type=float, default=0.05)
    parser.add_argument("--cluster-alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    # One process is the portable default. Some managed/macOS environments do
    # not permit the semaphore query used by joblib's process backend.
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--rows-per-page", type=int, default=4)
    parser.add_argument(
        "--rebuild-figure",
        action="store_true",
        help="Rebuild the PDF from cached statistics even if the PDF exists.",
    )
    parser.add_argument(
        "--scouts",
        type=int,
        nargs="*",
        default=None,
        help="Optional zero-based scout indices. The default processes all scouts.",
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


def load_scout_labels(path: Path, n_scouts: int) -> list[str]:
    atlas = mat73.loadmat(str(path))["DKT_scout"]
    raw_labels = atlas.get("Label", atlas.get("Region"))
    if raw_labels is None or len(raw_labels) != n_scouts:
        raise ValueError(f"Atlas must contain {n_scouts} scout labels.")

    labels = []
    for index, label in enumerate(raw_labels):
        clean = re.sub(r"\s+", " ", str(label)).strip()
        labels.append(clean or f"Scout {index + 1}")
    return labels


def make_time_vector(n_times: int) -> np.ndarray:
    step_ms = 1000 / SAMPLING_RATE_HZ
    times = np.arange(T_START_MS, T_END_MS + step_ms, step_ms)
    if times.size != n_times:
        raise ValueError(
            f"Expected {times.size} time points from {T_START_MS} to "
            f"{T_END_MS} ms, but decodeScore contains {n_times}."
        )
    return times


def cluster_indices_to_mask(cluster: object, n_times: int) -> np.ndarray:
    mask = np.zeros(n_times, dtype=bool)
    if isinstance(cluster, tuple):
        mask[cluster] = True
    else:
        values = np.asarray(cluster)
        if values.dtype == bool:
            mask |= values.reshape(-1)
        else:
            mask[values.astype(int).reshape(-1)] = True
    return mask


def contiguous_intervals(mask: np.ndarray, times: np.ndarray) -> list[tuple[float, float]]:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return []
    runs = np.split(indices, np.where(np.diff(indices) > 1)[0] + 1)
    return [(float(times[run[0]]), float(times[run[-1]])) for run in runs]


def compute_panel_statistics(
    group_scores: np.ndarray,
    times: np.ndarray,
    scout_index: int,
    scout_label: str,
    group_key: str,
    group_label: str,
    subject_indices: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    centered_scores = group_scores.T - CHANCE_LEVEL
    degrees_of_freedom = centered_scores.shape[0] - 1
    threshold = float(
        t_distribution.ppf(1.0 - args.threshold_p, degrees_of_freedom)
    )

    t_observed, clusters, cluster_p_values, null_distribution = (
        permutation_cluster_1samp_test(
            centered_scores,
            threshold=threshold,
            n_permutations=args.n_permutations,
            tail=1,
            adjacency=None,
            out_type="indices",
            seed=args.seed,
            n_jobs=args.n_jobs,
            verbose=False,
        )
    )

    cluster_masks = [
        cluster_indices_to_mask(cluster, times.size) for cluster in clusters
    ]
    significant_mask = np.zeros(times.size, dtype=bool)
    significant_clusters = []
    for mask, p_value in zip(cluster_masks, cluster_p_values):
        if p_value < args.cluster_alpha:
            significant_mask |= mask
            cluster_times = times[mask]
            significant_clusters.append(
                {
                    "start_ms": float(cluster_times[0]),
                    "end_ms": float(cluster_times[-1]),
                    "p_value": float(p_value),
                    "n_samples": int(mask.sum()),
                }
            )

    mean_accuracy = group_scores.mean(axis=1)
    sem_accuracy = group_scores.std(axis=1, ddof=1) / np.sqrt(group_scores.shape[1])

    return {
        "scout_index": scout_index,
        "scout_number": scout_index + 1,
        "scout_label": scout_label,
        "group_key": group_key,
        "group_label": group_label,
        "subject_indices": subject_indices,
        "n_subjects": int(subject_indices.size),
        "times_ms": times,
        "chance_level": CHANCE_LEVEL,
        "mean_accuracy": mean_accuracy,
        "sem_accuracy": sem_accuracy,
        "t_observed": np.asarray(t_observed),
        "cluster_masks": cluster_masks,
        "cluster_p_values": np.asarray(cluster_p_values),
        "significant_mask": significant_mask,
        "significant_clusters": significant_clusters,
        "null_distribution": np.asarray(null_distribution),
        "test": {
            "name": "one-sample temporal cluster permutation test",
            "contrast": "cross-validated balanced accuracy minus 0.5",
            "tail": 1,
            "n_permutations": args.n_permutations,
            "threshold_p": args.threshold_p,
            "cluster_forming_t": threshold,
            "cluster_alpha": args.cluster_alpha,
            "seed": args.seed,
            "correction_scope": "time points within this scout/group panel",
        },
    }


def load_or_compute_panel(
    cache_path: Path,
    group_scores: np.ndarray,
    times: np.ndarray,
    scout_index: int,
    scout_label: str,
    group_key: str,
    group_label: str,
    subject_indices: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict, bool]:
    if cache_path.exists():
        with open(cache_path, "rb") as file:
            return pickle.load(file), True

    statistics = compute_panel_statistics(
        group_scores=group_scores,
        times=times,
        scout_index=scout_index,
        scout_label=scout_label,
        group_key=group_key,
        group_label=group_label,
        subject_indices=subject_indices,
        args=args,
    )
    atomic_pickle_dump(statistics, cache_path)
    return statistics, False


def write_cluster_summary(statistics: list[dict], destination: Path) -> None:
    fieldnames = (
        "scout_index",
        "scout_number",
        "scout_label",
        "group",
        "n_subjects",
        "cluster_number",
        "start_ms",
        "end_ms",
        "p_value",
        "n_samples",
    )
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with open(temporary, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for panel in statistics:
            for cluster_number, cluster in enumerate(
                panel["significant_clusters"], start=1
            ):
                writer.writerow(
                    {
                        "scout_index": panel["scout_index"],
                        "scout_number": panel["scout_number"],
                        "scout_label": panel["scout_label"],
                        "group": panel["group_label"],
                        "n_subjects": panel["n_subjects"],
                        "cluster_number": cluster_number,
                        **cluster,
                    }
                )
    os.replace(temporary, destination)


def plot_significant_intervals(
    axis: plt.Axes,
    times: np.ndarray,
    significant_mask: np.ndarray,
    y_position: float,
) -> None:
    indices = np.flatnonzero(significant_mask)
    if indices.size == 0:
        return
    for run in np.split(indices, np.where(np.diff(indices) > 1)[0] + 1):
        axis.plot(
            times[run],
            np.full(run.size, y_position),
            color="red",
            linewidth=4,
            solid_capstyle="butt",
            zorder=5,
        )


def create_multipage_figure(
    statistics_by_panel: dict[tuple[int, str], dict],
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
        [panel["mean_accuracy"] for panel in statistics_by_panel.values()]
    ) * 100
    all_sems = np.concatenate(
        [panel["sem_accuracy"] for panel in statistics_by_panel.values()]
    ) * 100
    lower_limit = min(45.0, float(np.floor(np.min(all_means - all_sems) - 1)))
    upper_limit = max(65.0, float(np.ceil(np.max(all_means + all_sems) + 1)))

    pdf_scratch_directory = BASE_DIRECTORY / "tmp" / "pdfs"
    pdf_scratch_directory.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.pdf")
    page_paths = []

    # Saving pages as independent PDFs avoids a Matplotlib PdfPages resource-
    # reuse bug observed with several data-dense middle pages. Merge the fully
    # rendered single-page files afterward for a stable multi-page document.
    with tempfile.TemporaryDirectory(
        prefix="scout_decoding_pages_", dir=pdf_scratch_directory
    ) as temporary_directory:
        for page_start in range(0, len(scout_indices), rows_per_page):
            page_scouts = scout_indices[page_start : page_start + rows_per_page]
            figure, axes = plt.subplots(
                len(page_scouts),
                len(GROUP_DEFINITIONS),
                figsize=(24, 4.2 * len(page_scouts)),
                sharex=True,
                sharey=True,
                squeeze=False,
            )

            for row, scout_index in enumerate(page_scouts):
                for column, (group_key, group_label, _) in enumerate(
                    GROUP_DEFINITIONS
                ):
                    axis = axes[row, column]
                    panel = statistics_by_panel[(scout_index, group_key)]
                    times = panel["times_ms"]
                    mean = panel["mean_accuracy"] * 100
                    sem = panel["sem_accuracy"] * 100

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
                        panel["significant_mask"],
                        y_position=lower_limit + 0.8,
                    )
                    axis.set_xlim(T_START_MS, T_END_MS)
                    axis.set_ylim(lower_limit, upper_limit)
                    axis.grid(axis="y", color="0.9", linewidth=0.7)
                    axis.tick_params(labelsize=10)

                    if row == 0:
                        axis.set_title(
                            f"{group_label} (n={panel['n_subjects']})",
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
                "Scout-wise sentiment decoding\n"
                "Mean ± SEM; red bars: temporal cluster p < 0.05 vs 50%",
                fontsize=17,
                fontweight="bold",
                y=0.998,
            )
            figure.text(
                0.5,
                0.004,
                "Cluster correction is within each scout/group panel; "
                "there is no additional correction across scouts or groups.",
                ha="center",
                fontsize=9,
            )
            figure.tight_layout(rect=(0.02, 0.025, 1, 0.965))
            # Keep every PDF page on the same fixed canvas. Per-page tight
            # bounding boxes can produce inconsistent clipping when row labels
            # have substantially different lengths.
            page_path = Path(temporary_directory) / f"page_{len(page_paths) + 1:03d}.pdf"
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


def main() -> None:
    args = parse_args()
    if args.n_permutations < 2:
        raise ValueError("n-permutations must be at least 2.")
    if args.rows_per_page < 1:
        raise ValueError("rows-per-page must be at least 1.")

    scores, subject_groups, decoder_result = load_decoder_result(args.input)
    n_scouts, n_times, n_subjects = scores.shape
    times = make_time_vector(n_times)
    scout_labels = load_scout_labels(args.atlas, n_scouts)

    scout_indices = list(range(n_scouts)) if args.scouts is None else args.scouts
    if not scout_indices or len(set(scout_indices)) != len(scout_indices):
        raise ValueError("scouts must contain unique indices and cannot be empty.")
    if min(scout_indices) < 0 or max(scout_indices) >= n_scouts:
        raise IndexError(f"Scout indices must be between 0 and {n_scouts - 1}.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    statistics_directory = args.output_dir / "statistics"
    statistics_directory.mkdir(parents=True, exist_ok=True)

    group_indices = {}
    for group_key, _, group_code in GROUP_DEFINITIONS:
        if group_code is None:
            indices = np.arange(n_subjects)
        else:
            indices = np.flatnonzero(subject_groups == group_code)
        if indices.size < 2:
            raise ValueError(f"Group {group_key} has fewer than two subjects.")
        group_indices[group_key] = indices

    print(f"Input: {args.input}")
    print(f"decodeScore: {scores.shape} (scout x time x subject)")
    print(
        "Groups: "
        + ", ".join(
            f"{label}={group_indices[key].size}"
            for key, label, _ in GROUP_DEFINITIONS
        )
    )
    print(f"Processing {len(scout_indices)} scouts with {args.n_permutations} permutations")

    statistics_by_panel = {}
    panel_statistics = []
    loaded_count = 0
    computed_count = 0

    for scout_position, scout_index in enumerate(scout_indices, start=1):
        for group_key, group_label, _ in GROUP_DEFINITIONS:
            indices = group_indices[group_key]
            cache_path = (
                statistics_directory
                / f"scout_{scout_index:03d}_{group_key}.pkl"
            )
            panel, loaded = load_or_compute_panel(
                cache_path=cache_path,
                # Index the scout first so NumPy keeps time x subject order;
                # combining a scalar and advanced index in one operation would
                # move the subject axis in front of time.
                group_scores=scores[scout_index][:, indices],
                times=times,
                scout_index=scout_index,
                scout_label=scout_labels[scout_index],
                group_key=group_key,
                group_label=group_label,
                subject_indices=indices,
                args=args,
            )
            statistics_by_panel[(scout_index, group_key)] = panel
            panel_statistics.append(panel)
            loaded_count += int(loaded)
            computed_count += int(not loaded)

        print(
            f"Scout {scout_position:02d}/{len(scout_indices):02d}: "
            f"{scout_labels[scout_index]}"
        )

    summary_path = args.output_dir / "cluster_summary.csv"
    write_cluster_summary(panel_statistics, summary_path)

    figure_path = args.output_dir / "scout_decoding_all_groups.pdf"
    create_multipage_figure(
        statistics_by_panel=statistics_by_panel,
        scout_indices=scout_indices,
        scout_labels=scout_labels,
        destination=figure_path,
        rows_per_page=args.rows_per_page,
        rebuild=args.rebuild_figure,
    )

    metadata = {
        "input_path": str(args.input),
        "input_axis_order": decoder_result.get("axis_order"),
        "decode_parameters": decoder_result.get("parameters"),
        "score_shape": scores.shape,
        "time_range_ms": (float(times[0]), float(times[-1])),
        "sampling_rate_hz": SAMPLING_RATE_HZ,
        "group_sizes": {
            label: int(group_indices[key].size)
            for key, label, _ in GROUP_DEFINITIONS
        },
        "n_scouts_processed": len(scout_indices),
        "test_parameters": {
            "n_permutations": args.n_permutations,
            "threshold_p": args.threshold_p,
            "cluster_alpha": args.cluster_alpha,
            "tail": 1,
            "seed": args.seed,
            "correction_scope": "time points within each scout/group panel",
        },
    }
    atomic_pickle_dump(metadata, args.output_dir / "run_metadata.pkl")

    significant_panels = sum(
        bool(panel["significant_clusters"]) for panel in panel_statistics
    )
    significant_clusters = sum(
        len(panel["significant_clusters"]) for panel in panel_statistics
    )
    print(f"Panel caches loaded: {loaded_count}")
    print(f"Panel tests computed: {computed_count}")
    print(f"Significant panels: {significant_panels}/{len(panel_statistics)}")
    print(f"Significant clusters: {significant_clusters}")
    print(f"Figure: {figure_path}")
    print(f"Cluster summary: {summary_path}")


if __name__ == "__main__":
    main()
