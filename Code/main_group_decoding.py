"""Time-resolved group decoding (control vs depressed + suicidal) within each DKT scout.

Sentence decoding classifies trials within a subject; here the examples are subjects.
At every scout and time point a ridge classifier is trained on the congruency contrast
(decoders/group_time_decoder.py) with the outer splits shared by the classifiers, and
AUC is pooled over each repeat's out-of-fold scores and averaged over repeats.

Each scout and time point yields one AUC, so inference uses label permutation (the same
permuted labels for every scout and time point). A point is supra-threshold when its
AUC exceeds the --threshold-p upper quantile of its own null; clusters are contiguous
time runs within a scout with mass = sum of null z-scores, and each cluster p-value
comes from the maximum cluster mass over all selected scouts and times in every
permutation (one-sided, FWE across scouts x time).

Examples (run from Code/):
    python3 main_group_decoding.py --smoke-test
    python3 main_group_decoding.py --jobs 2
    python3 main_group_decoding.py --rebuild-figure
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from decoders.group_time_decoder import decode_scout, subject_contrasts
from utils import cv, io, paths, plotting, source, stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pair", choices=source.PAIRS, default="control-vs-depressed+suicidal")
    p.add_argument("--eeg", type=Path, default=paths.EEG)
    p.add_argument("--tmatrix", type=Path, default=paths.TMATRIX)
    p.add_argument("--atlas", type=Path, default=paths.ATLAS)
    p.add_argument("--subject-index", type=Path, default=paths.SUBJECT_INDEX)
    p.add_argument("--trial-index", type=Path, default=paths.TOI_INDEX)
    p.add_argument("--category", default="Intention")
    p.add_argument("--cache-dir", type=Path, default=paths.SUBJECT_CACHE)
    p.add_argument("--projection-root", type=Path, default=paths.PROJECTIONS,
                   help="Directory whose projections/ holds (or will hold) the per-fold scout matrices")
    p.add_argument("--output-dir", type=Path, default=paths.RESULTS / "group_decoding_scout")
    p.add_argument("--n-components", type=int, default=32)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=42, help="CV seed (shared with the classifiers)")
    p.add_argument("--alpha", type=float, default=1.0, help="Ridge penalty per training subject")
    p.add_argument("--n-permutations", type=int, default=1000)
    p.add_argument("--permutation-seed", type=int, default=0)
    p.add_argument("--threshold-p", type=float, default=0.05, help="Point-wise permutation p for clusters")
    p.add_argument("--cluster-alpha", type=float, default=0.05)
    p.add_argument("--scouts", type=int, nargs="*", default=None,
                   help="Zero-based scouts (default all); correction covers only these")
    p.add_argument("--jobs", type=int, default=2)
    p.add_argument("--blas-threads", type=int, default=1)
    p.add_argument("--scouts-per-page", type=int, default=16)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--rebuild-figure", action="store_true")
    p.add_argument("--smoke-test", action="store_true", help="Short run on synthetic data")
    args = p.parse_args()
    for name in ("n_components", "folds", "repeats", "n_permutations", "jobs", "blas_threads", "scouts_per_page"):
        if getattr(args, name) < 1:
            p.error(f"--{name.replace('_', '-')} must be positive")
    if args.alpha <= 0:
        p.error("--alpha must be positive")
    if not (0 < args.threshold_p < 0.5 and 0 < args.cluster_alpha < 1):
        p.error("--threshold-p must be in (0, 0.5) and --cluster-alpha in (0, 1)")
    return args


def projection_cfg(args: argparse.Namespace) -> dict:
    """The settings that build and locate the scout projections (shared with main_group_classifier.py)."""
    return {"projection_root": str(args.projection_root), "pair": args.pair, "cache_dir": str(args.cache_dir),
            "trial_index": str(args.trial_index), "category": args.category, "n_components": args.n_components,
            "folds": args.folds, "repeats": args.repeats, "seed": args.seed, "window": [0, 800], "max_shift": 3}


def compute_maps(args: argparse.Namespace, cfg: dict, meta: dict, scouts: list[int], selected: np.ndarray,
                 labels: np.ndarray, splits: list) -> tuple[dict, bool]:
    """Observed and permutation AUC maps, and whether they were reused from an earlier run."""
    sig = {"cache": meta["signature"], "trial_index": io.file_signature(cfg["trial_index"]),
           "projections": source.projection_signature(cfg, meta), "scouts": scouts, "alpha": args.alpha,
           "n_permutations": args.n_permutations, "permutation_seed": args.permutation_seed}
    path = args.output_dir / "decoding_maps.pkl"
    if args.resume and path.exists():
        cached = io.load_pickle(path)
        if cached["signature"] == sig:
            print("Reusing decoding maps", flush=True)
            return cached, True
        print("Decoding maps were computed with other settings; recomputing", flush=True)
    print(f"Loading {len(selected)} subjects' congruency contrasts...", flush=True)
    contrasts = subject_contrasts(cfg, meta, selected)
    rng = np.random.default_rng(args.permutation_seed)
    labelings = np.column_stack([labels] + [rng.permutation(labels) for _ in range(args.n_permutations)])
    labelings = labelings.astype(np.float64)
    print(f"Decoding {len(scouts)} scouts x {contrasts.shape[2]} time points with "
          f"{args.n_permutations} permutations on {args.jobs} job(s)...", flush=True)
    from joblib import Parallel, delayed
    results = {}
    for scout, auc in Parallel(n_jobs=args.jobs, return_as="generator_unordered")(
            delayed(decode_scout)(cfg, s, contrasts, labelings, splits, args.alpha, args.blas_threads)
            for s in scouts):
        results[scout] = auc
        print(f"  scout {scout:2d} {meta['labels'][scout]:<28} peak AUC {auc[:, :, 0].mean(0).max():.3f}", flush=True)
    stacked = np.stack([results[s] for s in scouts])  # scout x repeat x time x labeling
    maps = {"signature": sig, "auc_by_repeat": stacked[..., 0].transpose(1, 0, 2),
            "auc": stacked[..., 0].mean(1), "null": stacked[..., 1:].mean(1).astype(np.float32)}
    io.save_pickle(path, maps)
    return maps, False


def plot(path: Path, results: dict, args: argparse.Namespace) -> None:
    times, auc, labels = results["times_ms"], results["auc"], results["scout_labels"]
    scouts, mask = results["scout_indices"], results["significant_mask"].T  # scout x time
    low, high, null_mean = results["null_low"], results["null_high"], results["null_mean"]
    y_low = min(0.38, float(np.floor(min(auc.min(), low.min()) * 50) / 50))
    y_high = max(0.66, float(np.ceil(max(auc.max(), high.max()) * 50) / 50))
    title = "Time-resolved group decoding: Control vs. Patients (D+S)"
    note = (f"Ridge per time point on the congruency contrast (32 scout PCs); AUC averaged over "
            f"{args.repeats} x {args.folds}-fold CV. Gray: 95% label-permutation null. Red: cluster p < "
            f"{args.cluster_alpha:g}, FWE across {len(scouts)} scouts x time ({args.n_permutations} permutations).")
    with plotting.MultipagePdf(path) as pdf:
        fig, axis = plt.subplots(figsize=(13, 0.16 * len(scouts) + 2.2))
        span = max(0.05, float(np.quantile(np.abs(auc - 0.5), 0.995)))
        image = axis.imshow(auc, aspect="auto", cmap="RdBu_r", vmin=0.5 - span, vmax=0.5 + span,
                            extent=(times[0], times[-1], len(scouts) - 0.5, -0.5), interpolation="nearest")
        step = times[1] - times[0]
        for i, row in enumerate(mask):
            for run in stats.contiguous_runs(row):
                axis.add_patch(matplotlib.patches.Rectangle((times[run[0]] - step / 2, i - 0.5), run.size * step,
                                                            1, fill=False, edgecolor=plotting.INK, linewidth=1.2))
        axis.axvline(0, color=plotting.INK_SECONDARY, linestyle=":", linewidth=1)
        axis.set_yticks(range(len(scouts)), [f"{s + 1:02d} {labels[i]}" for i, s in enumerate(scouts)], fontsize=6)
        axis.set_xlabel("Time from stimulus onset (ms)", fontsize=10)
        axis.set_title(f"{title}\nOutlined: significant clusters", fontsize=12, loc="left")
        fig.colorbar(image, ax=axis, fraction=0.025, pad=0.01).set_label("Cross-validated AUC", fontsize=9)
        fig.tight_layout()
        pdf.add(fig)

        n_cols = 4
        for start in range(0, len(scouts), args.scouts_per_page):
            chunk = list(range(start, min(start + args.scouts_per_page, len(scouts))))
            n_rows = int(np.ceil(len(chunk) / n_cols))
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.0 * n_rows + 1.2), sharex=True, sharey=True,
                                     squeeze=False)
            for axis in axes.ravel()[len(chunk):]:
                axis.set_visible(False)
            for axis, i in zip(axes.ravel(), chunk):
                axis.fill_between(times, low[i], high[i], color=plotting.NEUTRAL_LIGHT, linewidth=0)
                axis.plot(times, null_mean[i], color=plotting.NEUTRAL, linewidth=1, linestyle="--")
                axis.plot(times, auc[i], color=plotting.BLUE, linewidth=1.8)
                axis.axhline(0.5, color=plotting.INK, linestyle=":", linewidth=1)
                axis.axvline(0, color=plotting.INK_SECONDARY, linestyle=":", linewidth=1)
                plotting.plot_significant_intervals(axis, times, mask[i], y_low + 0.008)
                axis.set_title(f"{scouts[i] + 1:02d}. {labels[i]}", fontsize=10, fontweight="bold")
                axis.set_xlim(times[0], times[-1])
                axis.set_ylim(y_low, y_high)
                axis.grid(axis="y", color="0.92", linewidth=0.7)
                axis.tick_params(labelsize=8)
            for axis in axes[:, 0]:
                axis.set_ylabel("AUC", fontsize=9)
            for axis in axes[-1]:
                axis.set_xlabel("Time (ms)", fontsize=9)
            fig.suptitle(title, fontsize=14, fontweight="bold")
            fig.text(0.5, 0.005, note, ha="center", fontsize=8, wrap=True)
            fig.tight_layout(rect=(0, 0.025, 1, 0.97))
            pdf.add(fig)


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        args.cache_dir, args.trial_index = source.build_synthetic_cache()
        args.projection_root = args.cache_dir.parent / "output"
        args.output_dir = args.projection_root / "group_decoding"
        args.repeats, args.n_permutations = 1, 200
    meta = source.ensure_cache(args.cache_dir, args.eeg, args.tmatrix, args.atlas, args.subject_index)
    cfg = projection_cfg(args)
    n_scouts = len(meta["vertices"])
    scouts = list(range(n_scouts)) if args.scouts is None else sorted(set(args.scouts))
    if not scouts or min(scouts) < 0 or max(scouts) >= n_scouts:
        raise ValueError(f"--scouts must be between 0 and {n_scouts - 1}")
    selected, labels, originals = source.select_pair(np.asarray(meta["groups"]), args.pair)
    splits = cv.outer_splits(originals, args.folds, args.repeats, args.seed)
    source.ensure_projections(cfg, meta, scouts, args.jobs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"{args.pair}: {int((labels == 0).sum())} vs {int(labels.sum())} subjects | {len(scouts)} scouts | "
          f"{args.repeats} x {args.folds}-fold CV | output {args.output_dir}", flush=True)

    maps, reused = compute_maps(args, cfg, meta, scouts, selected, labels, splits)
    test = stats.permutation_cluster_maps(maps["auc"], maps["null"], args.threshold_p)
    times = np.asarray(meta["times_ms"])
    clusters = []
    for c in np.flatnonzero(test["p_values"] < args.cluster_alpha):
        i, t = np.nonzero(test["run"] == c + 1)
        peak = t[np.argmax(maps["auc"][i[0], t])]
        clusters.append({"scout_index": scouts[i[0]], "scout_number": scouts[i[0]] + 1,
                         "scout_label": meta["labels"][scouts[i[0]]], "start_ms": float(times[t.min()]),
                         "end_ms": float(times[t.max()]), "n_time_samples": int(t.size),
                         "cluster_mass_z": float(test["masses"][c]), "peak_auc": float(maps["auc"][i[0], peak]),
                         "peak_time_ms": float(times[peak]), "p_value": float(test["p_values"][c]),
                         "mask_run": int(c + 1)})
    clusters.sort(key=lambda c: (c["p_value"], c["start_ms"]))
    for number, c in enumerate(clusters, 1):
        c["cluster_number"] = number
    significant = np.isin(test["run"], [c["mask_run"] for c in clusters])

    results = {"pair": args.pair, "times_ms": times, "scout_indices": scouts,
               "scout_labels": [meta["labels"][s] for s in scouts], "subject_indices": selected, "labels": labels,
               "auc": maps["auc"], "auc_by_repeat": maps["auc_by_repeat"], "null_mean": test["null_mean"],
               "null_low": test["null_low"], "null_high": test["null_high"],
               "null_threshold": test["null_threshold"], "z": test["z"],
               "significant_mask": significant.T,  # time x scout, as in the other CBP outputs
               "significant_clusters": clusters, "null_max_cluster_mass": test["null_max_mass"],
               "parameters": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
               "axis_order": {"auc": "scout x time", "auc_by_repeat": "repeat x scout x time",
                              "significant_mask": "time x scout"}}
    io.save_pickle(args.output_dir / "group_decoding.pkl", results)
    fields = ["cluster_number", "scout_index", "scout_number", "scout_label", "start_ms", "end_ms",
              "n_time_samples", "cluster_mass_z", "peak_auc", "peak_time_ms", "p_value"]
    io.save_csv(args.output_dir / "cluster_summary.csv", [{k: c[k] for k in fields} for c in clusters], fields)
    figure = args.output_dir / "group_decoding_scouts.pdf"
    if args.rebuild_figure or not figure.exists() or not reused:
        plot(figure, results, args)
    else:
        print(f"Figure already exists; not rebuilding (use --rebuild-figure): {figure}")

    print(f"\nSignificant clusters (FWE p < {args.cluster_alpha:g}): {len(clusters)}")
    for c in clusters:
        print(f"  {c['scout_number']:02d} {c['scout_label']:<28} {c['start_ms']:6.0f} to {c['end_ms']:6.0f} ms | "
              f"peak AUC {c['peak_auc']:.3f} at {c['peak_time_ms']:.0f} ms | p = {c['p_value']:.4f}")
    print(f"Largest null cluster mass, 95th percentile: {np.quantile(test['null_max_mass'], 0.95):.1f}")
    print(f"Figure: {figure}\nSummary: {args.output_dir / 'cluster_summary.csv'}")


if __name__ == "__main__":
    main()
