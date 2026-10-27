"""Time-resolved group decoding (control vs depressed + suicidal) within each DKT scout.

Sentence decoding classifies trials within a subject; here the examples are subjects.
Each subject's feature at time t is its incongruent - congruent average projected to
the scout PCs of scout_subject_classifier.py (orientation and PCs fitted on the
training subjects of each outer fold, PCs scaled by the subject's baseline SD; the
matrices cached by multi_scout_classifier.py are reused). A ridge classifier
(lambda = --alpha x number of training subjects, standardized features) is fitted at
every time point, and AUC is pooled over each repeat's out-of-fold scores and
averaged over repeats.

Each scout and time point yields one AUC, so inference uses label permutation. Ridge
predictions are linear in the labels, so all permutations come from one matrix
product; the same permuted labels are used for every scout and time point. A point
is supra-threshold when its AUC exceeds the --threshold-p upper quantile of its own
null. Clusters are contiguous time runs within a scout, their mass is the sum of the
points' null z-scores, and each cluster p-value comes from the maximum cluster mass
over all selected scouts and times in every permutation (one-sided, FWE across
scouts x time).

Examples:
    python3 group_decoding_scout.py --smoke-test
    python3 group_decoding_scout.py --jobs 2
    python3 group_decoding_scout.py --rebuild-figure
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import tempfile
from pathlib import Path

import matplotlib
import numpy as np
from pypdf import PdfReader, PdfWriter
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

import multi_scout_classifier as msc
import scout_subject_classifier as ssc

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, RED = "#2a78d6", "red"
INK, INK_SECONDARY, NEUTRAL, NEUTRAL_LIGHT = "#0b0b0b", "#52514e", "#8a8984", "#dcdbd6"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pair", choices=ssc.PAIRS, default="control-vs-depressed+suicidal")
    p.add_argument("--eeg", type=Path, default=ssc.PREPROCESSED / "Data_sen_bl.pkl")
    p.add_argument("--tmatrix", type=Path, default=ssc.PREPROCESSED / "tMatrix.pkl")
    p.add_argument("--atlas", type=Path, default=ssc.PREPROCESSED / "DKT_scout.mat")
    p.add_argument("--subject-index", type=Path, default=ssc.BEHAVIOR / "subject_index.mat")
    p.add_argument("--trial-index", type=Path, default=ssc.BEHAVIOR / "senIdx_TOI.pkl")
    p.add_argument("--category", default="Intention")
    p.add_argument("--cache-dir", type=Path, default=ssc.ROOT / "Results" / "subject_sensor_cache")
    p.add_argument("--projection-root", type=Path, default=ssc.ROOT / "Results" / "multi_scout_classifier",
                   help="Directory whose projections/ holds (or will hold) the per-fold scout matrices")
    p.add_argument("--output-dir", type=Path, default=ssc.ROOT / "Results" / "group_decoding_scout")
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
    """The settings multi_scout_classifier.py uses to build and find its projections."""
    return {"output_dir": str(args.projection_root), "resume": True, "pair": args.pair,
            "cache_dir": str(args.cache_dir), "trial_index": str(args.trial_index), "category": args.category,
            "n_components": args.n_components, "folds": args.folds, "repeats": args.repeats, "seed": args.seed,
            "window": [0, 800], "max_shift": 3}


def subject_contrasts(cfg: dict, meta: dict, selected: np.ndarray) -> np.ndarray:
    """Subject x channel x time: incongruent minus congruent average of baseline-corrected trials."""
    sets = ssc.load_trial_sets(Path(cfg["trial_index"]), cfg["category"], len(meta["groups"]))
    baseline = np.asarray(meta["times_ms"]) < ssc.BASELINE_END_MS
    out = []
    for s in selected:
        trials = ssc.load_subject(Path(cfg["cache_dir"]), s, sets[s], np.array([0]), baseline)["trials"]
        out.append(trials[1].mean(0, dtype=np.float64) - trials[0].mean(0, dtype=np.float64))
    return np.stack(out)


def decode_scout(cfg: dict, scout: int, contrasts: np.ndarray, labels: np.ndarray, splits: list,
                 alpha: float, blas_threads: int) -> tuple[int, np.ndarray]:
    """Repeat x time x labeling AUC; labeling 0 is the observed labels, the rest permutations."""
    with threadpool_limits(blas_threads):
        n, n_times = contrasts.shape[0], contrasts.shape[2]
        n1 = int(labels[:, 0].sum())
        out = np.empty((len(splits), n_times, labels.shape[1]), dtype=np.float32)
        for r, folds in enumerate(splits):
            scores = np.empty((n, n_times, labels.shape[1]), dtype=np.float32)
            for f, (train, test) in enumerate(folds):
                m = np.load(msc.projection_path(cfg, scout, r, f)).astype(np.float64)
                x = np.einsum("npc,nct->tnp", m, contrasts, optimize=True)  # time x subject x PC
                mean, sd = x[:, train].mean(1, keepdims=True), x[:, train].std(1, keepdims=True)
                sd[sd == 0] = 1  # zero-padded PCs stay zero
                x_train, x_test = (x[:, train] - mean) / sd, (x[:, test] - mean) / sd
                gram = x_train.transpose(0, 2, 1) @ x_train + alpha * len(train) * np.eye(x.shape[2])
                hat = x_test @ np.linalg.solve(gram, x_train.transpose(0, 2, 1))  # time x test x train
                y_train = labels[train]
                y_mean = y_train.mean(0)
                scores[test] = (hat @ (y_train - y_mean) + y_mean).transpose(1, 0, 2)
            ranks = rankdata(scores, axis=0)
            out[r] = (np.einsum("itk,ik->tk", ranks, labels) - n1 * (n1 + 1) / 2) / (n1 * (n - n1))
    return scout, out


def runs_and_masses(mask: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Label contiguous time runs of ``mask`` (scout x time) and sum ``z`` within each."""
    n_scouts, n_times = mask.shape
    padded = np.zeros((n_scouts, n_times + 1), dtype=bool)
    padded[:, :n_times] = mask
    flat = padded.ravel()
    starts = flat & ~np.concatenate(([False], flat[:-1]))
    run = (np.cumsum(starts) * flat).reshape(n_scouts, n_times + 1)[:, :n_times]
    masses = np.bincount(run.ravel(), weights=np.where(mask, z, 0).ravel())[1:]
    return run, masses


def cluster_test(observed: np.ndarray, null: np.ndarray, threshold_p: float) -> dict:
    """observed: scout x time; null: scout x time x permutation."""
    mean, sd = null.mean(-1), null.std(-1)
    sd = np.where(sd > 0, sd, np.inf)
    threshold = np.quantile(null, 1 - threshold_p, axis=-1)
    null_max = np.zeros(null.shape[-1])
    for k in range(null.shape[-1]):
        _, masses = runs_and_masses(null[..., k] > threshold, (null[..., k] - mean) / sd)
        null_max[k] = masses.max(initial=0.0)
    z = (observed - mean) / sd
    run, masses = runs_and_masses(observed > threshold, z)
    p_values = (1 + (null_max[None, :] >= masses[:, None] - 1e-12).sum(1)) / (1 + len(null_max))
    return {"null_mean": mean, "null_threshold": threshold, "z": z, "run": run, "masses": masses,
            "p_values": p_values, "null_max_mass": null_max,
            "null_low": np.quantile(null, 0.025, axis=-1), "null_high": np.quantile(null, 0.975, axis=-1)}


def compute_maps(args: argparse.Namespace, cfg: dict, meta: dict, scouts: list[int], selected: np.ndarray,
                 labels: np.ndarray, splits: list) -> tuple[dict, bool]:
    """Observed and permutation AUC maps, and whether they were reused from an earlier run."""
    sig = {"cache": meta["signature"], "trial_index": ssc.file_signature(cfg["trial_index"]),
           "projections": msc.projection_signature(cfg, meta), "scouts": scouts, "alpha": args.alpha,
           "n_permutations": args.n_permutations, "permutation_seed": args.permutation_seed}
    path = args.output_dir / "decoding_maps.pkl"
    if args.resume and path.exists():
        with open(path, "rb") as f:
            cached = pickle.load(f)
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
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(maps, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    return maps, False


def write_summary(path: Path, clusters: list[dict]) -> None:
    fields = ["cluster_number", "scout_index", "scout_number", "scout_label", "start_ms", "end_ms",
              "n_time_samples", "cluster_mass_z", "peak_auc", "peak_time_ms", "p_value"]
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: c[k] for k in fields} for c in clusters)
    os.replace(tmp, path)


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
    scratch = path.parent / ".pdf_scratch"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as tmp_dir:
        pages = []

        fig, axis = plt.subplots(figsize=(13, 0.16 * len(scouts) + 2.2))
        span = max(0.05, float(np.quantile(np.abs(auc - 0.5), 0.995)))
        image = axis.imshow(auc, aspect="auto", cmap="RdBu_r", vmin=0.5 - span, vmax=0.5 + span,
                            extent=(times[0], times[-1], len(scouts) - 0.5, -0.5), interpolation="nearest")
        step = times[1] - times[0]
        for i, row in enumerate(mask):
            idx = np.flatnonzero(row)
            for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1) if idx.size else []:
                axis.add_patch(matplotlib.patches.Rectangle((times[run[0]] - step / 2, i - 0.5), run.size * step,
                                                            1, fill=False, edgecolor=INK, linewidth=1.2))
        axis.axvline(0, color=INK_SECONDARY, linestyle=":", linewidth=1)
        axis.set_yticks(range(len(scouts)), [f"{s + 1:02d} {labels[i]}" for i, s in enumerate(scouts)], fontsize=6)
        axis.set_xlabel("Time from stimulus onset (ms)", fontsize=10)
        axis.set_title(f"{title}\nOutlined: significant clusters", fontsize=12, loc="left")
        fig.colorbar(image, ax=axis, fraction=0.025, pad=0.01).set_label("Cross-validated AUC", fontsize=9)
        fig.tight_layout()
        pages.append(Path(tmp_dir) / "page_000.pdf")
        fig.savefig(pages[-1], format="pdf")
        plt.close(fig)

        per_page = args.scouts_per_page
        n_cols = 4
        for start in range(0, len(scouts), per_page):
            chunk = list(range(start, min(start + per_page, len(scouts))))
            n_rows = int(np.ceil(len(chunk) / n_cols))
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.0 * n_rows + 1.2), sharex=True, sharey=True,
                                     squeeze=False)
            for axis in axes.ravel()[len(chunk):]:
                axis.set_visible(False)
            for axis, i in zip(axes.ravel(), chunk):
                axis.fill_between(times, low[i], high[i], color=NEUTRAL_LIGHT, linewidth=0)
                axis.plot(times, null_mean[i], color=NEUTRAL, linewidth=1, linestyle="--")
                axis.plot(times, auc[i], color=BLUE, linewidth=1.8)
                axis.axhline(0.5, color=INK, linestyle=":", linewidth=1)
                axis.axvline(0, color=INK_SECONDARY, linestyle=":", linewidth=1)
                for run in np.split(np.flatnonzero(mask[i]), np.flatnonzero(np.diff(np.flatnonzero(mask[i])) > 1) + 1):
                    if run.size:
                        axis.plot(times[run], np.full(run.size, y_low + 0.008), color=RED, linewidth=4,
                                  solid_capstyle="butt")
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
            pages.append(Path(tmp_dir) / f"page_{len(pages):03d}.pdf")
            fig.savefig(pages[-1], format="pdf")
            plt.close(fig)

        writer = PdfWriter()
        for page in pages:
            writer.add_page(PdfReader(page).pages[0])
        tmp = path.with_suffix(".tmp.pdf")
        with open(tmp, "wb") as f:
            writer.write(f)
    os.replace(tmp, path)


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        args.selected = None  # make_smoke_cache also fills multi-scout options
        msc.make_smoke_cache(args)
        args.projection_root, args.output_dir = args.output_dir, args.output_dir / "group_decoding"
        args.repeats, args.n_permutations = 1, 200
    meta = ssc.ensure_cache(args)
    cfg = projection_cfg(args)
    n_scouts = len(meta["vertices"])
    scouts = list(range(n_scouts)) if args.scouts is None else sorted(set(args.scouts))
    if not scouts or min(scouts) < 0 or max(scouts) >= n_scouts:
        raise ValueError(f"--scouts must be between 0 and {n_scouts - 1}")
    selected, labels, originals = ssc.select_pair(np.asarray(meta["groups"]), args.pair)
    splits = msc.make_splits(cfg, originals)

    sig_path = args.projection_root / "projections" / "signature.json"
    if sig_path.exists() and json.loads(sig_path.read_text()) != msc.projection_signature(cfg, meta):
        raise ValueError(f"{sig_path.parent} was built with other settings; pass another --projection-root")
    msc.ensure_projections(cfg, meta, scouts, args.jobs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"{args.pair}: {int((labels == 0).sum())} vs {int(labels.sum())} subjects | {len(scouts)} scouts | "
          f"{args.repeats} x {args.folds}-fold CV | output {args.output_dir}", flush=True)

    maps, reused = compute_maps(args, cfg, meta, scouts, selected, labels, splits)
    stats = cluster_test(maps["auc"], maps["null"], args.threshold_p)
    times = np.asarray(meta["times_ms"])
    clusters = []
    for c in np.flatnonzero(stats["p_values"] < args.cluster_alpha):
        i, t = np.nonzero(stats["run"] == c + 1)
        peak = t[np.argmax(maps["auc"][i[0], t])]
        clusters.append({"scout_index": scouts[i[0]], "scout_number": scouts[i[0]] + 1,
                         "scout_label": meta["labels"][scouts[i[0]]], "start_ms": float(times[t.min()]),
                         "end_ms": float(times[t.max()]), "n_time_samples": int(t.size),
                         "cluster_mass_z": float(stats["masses"][c]), "peak_auc": float(maps["auc"][i[0], peak]),
                         "peak_time_ms": float(times[peak]), "p_value": float(stats["p_values"][c]),
                         "mask_run": int(c + 1)})
    clusters.sort(key=lambda c: (c["p_value"], c["start_ms"]))
    for number, c in enumerate(clusters, 1):
        c["cluster_number"] = number
    significant = np.isin(stats["run"], [c["mask_run"] for c in clusters])

    results = {"pair": args.pair, "times_ms": times, "scout_indices": scouts,
               "scout_labels": [meta["labels"][s] for s in scouts], "subject_indices": selected, "labels": labels,
               "auc": maps["auc"], "auc_by_repeat": maps["auc_by_repeat"], "null_mean": stats["null_mean"],
               "null_low": stats["null_low"], "null_high": stats["null_high"],
               "null_threshold": stats["null_threshold"], "z": stats["z"],
               "significant_mask": significant.T,  # time x scout, as in the other CBP outputs
               "significant_clusters": clusters, "null_max_cluster_mass": stats["null_max_mass"],
               "parameters": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
               "axis_order": {"auc": "scout x time", "auc_by_repeat": "repeat x scout x time",
                              "significant_mask": "time x scout"}}
    tmp = args.output_dir / "group_decoding.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, args.output_dir / "group_decoding.pkl")
    write_summary(args.output_dir / "cluster_summary.csv", clusters)
    figure = args.output_dir / "group_decoding_scouts.pdf"
    if args.rebuild_figure or not figure.exists() or not reused:
        plot(figure, results, args)
    else:
        print(f"Figure already exists; not rebuilding (use --rebuild-figure): {figure}")

    print(f"\nSignificant clusters (FWE p < {args.cluster_alpha:g}): {len(clusters)}")
    for c in clusters:
        print(f"  {c['scout_number']:02d} {c['scout_label']:<28} {c['start_ms']:6.0f} to {c['end_ms']:6.0f} ms | "
              f"peak AUC {c['peak_auc']:.3f} at {c['peak_time_ms']:.0f} ms | p = {c['p_value']:.4f}")
    print(f"Largest null cluster mass, 95th percentile: {np.quantile(stats['null_max_mass'], 0.95):.1f}")
    print(f"Figure: {figure}\nSummary: {args.output_dir / 'cluster_summary.csv'}")


if __name__ == "__main__":
    main()
