"""Group classification from several DKT scouts combined in one network, with its ablation.

The PC time courses of a set of scouts are stacked as input rows of the flat EEGNet
(its first 1x1 convolution mixes all scouts), like giving an EEG model more channels.
Each scout is preprocessed as in main_scout_screen.py, and the outer splits and
training recipe match it, so predictions pair subject by subject with the screen.

Input sets (arms):
  selected   scouts whose single-scout AUC in --screen-csv exceeds --threshold (or
             --selected); picked with all subjects' labels, so optimistic
  random     --n-random random sets of the same size, drawn from all scouts
  all        every scout
  sensor     the sensor channels, each scaled by the subject's baseline SD
The best single scout of --screen-csv is compared from its saved predictions.

Comparison (per arm): label permutation against chance; the selected arm's percentile
among the random sets; paired subject bootstraps (stratified by the original groups)
against the best single scout, all scouts, and the sensor arm; Holm correction across
those comparisons.

Examples (run from Code/):
    python3 main_group_classifier.py --smoke-test
    python3 main_group_classifier.py --device cpu --jobs 12
    python3 main_group_classifier.py --compare-only
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from decoders import subject_classifier as sc
from utils import cv, io, paths, plotting, source, stats

ARM_KINDS = ("selected", "random", "all", "sensor")
MODEL = "flat"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pair", choices=source.PAIRS, default="control-vs-depressed+suicidal")
    p.add_argument("--arms", nargs="+", choices=ARM_KINDS, default=list(ARM_KINDS))
    p.add_argument("--screen-csv", type=Path,
                   default=paths.RESULTS / "scout_subject_cds_all" / "control-vs-depressed+suicidal_scouts.csv",
                   help="Single-scout results (main_scout_screen.py) used to pick scouts")
    p.add_argument("--threshold", type=float, default=0.60, help="Single-scout soft-vote AUC for selection")
    p.add_argument("--selected", nargs="+", default=None,
                   help="Zero-based scouts to use instead of the --screen-csv threshold")
    p.add_argument("--set-name", default=None,
                   help="Name of the selected set (arm flat_<name>, comparison_<name>/); default selected<k>")
    p.add_argument("--single-scout-dir", type=Path, default=paths.RESULTS / "scout_subject_cds_all",
                   help="Single-scout predictions for the best-single-scout comparison")
    p.add_argument("--n-random", type=int, default=30)
    p.add_argument("--random-seed", type=int, default=0)
    p.add_argument("--scout-dropout", type=float, default=0.2, help="Probability of zeroing a whole scout")
    p.add_argument("--pc-dropout", type=float, default=0.1, help="Probability of zeroing one input row")
    p.add_argument("--eeg", type=Path, default=paths.EEG)
    p.add_argument("--tmatrix", type=Path, default=paths.TMATRIX)
    p.add_argument("--atlas", type=Path, default=paths.ATLAS)
    p.add_argument("--subject-index", type=Path, default=paths.SUBJECT_INDEX)
    p.add_argument("--trial-index", type=Path, default=paths.TOI_INDEX)
    p.add_argument("--category", default="Intention")
    p.add_argument("--cache-dir", type=Path, default=paths.SUBJECT_CACHE)
    p.add_argument("--output-dir", type=Path, default=paths.RESULTS / "group_classifier")
    p.add_argument("--projection-root", type=Path, default=paths.PROJECTIONS,
                   help="Directory whose projections/ holds the per-fold scout matrices")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--fast-eegnet", action=argparse.BooleanOptionalAction, default=True,
                   help="Matrix-multiply implementation of EEGNet's 1x1 convolutions (same model, 4-16x faster; "
                        "results equal up to rounding, so --no-fast-eegnet reproduces Results/multi_scout_classifier)")
    p.add_argument("--arms-per-task", type=int, default=10, help="Arms trained per parallel task (shares data loading)")
    p.add_argument("--n-components", type=int, default=32)
    p.add_argument("--window", nargs=2, type=int, default=[0, 800], metavar=("START_MS", "END_MS"))
    p.add_argument("--avg-range", nargs=2, type=int, default=[8, 24], metavar=("LO", "HI"))
    p.add_argument("--draws-per-epoch", type=int, default=16)
    p.add_argument("--tta-avg", type=int, default=16)
    p.add_argument("--tta-draws", type=int, default=32)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--avg-epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--max-shift", type=int, default=3)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-bootstrap", type=int, default=10_000)
    p.add_argument("--n-permutations", type=int, default=10_000)
    p.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    p.add_argument("--jobs", type=int, default=1, help="Parallel CPU folds")
    p.add_argument("--compare-only", action="store_true", help="Skip training; recompute the comparison")
    p.add_argument("--smoke-test", action="store_true", help="Short run on synthetic data")
    args = p.parse_args()
    if args.selected is not None:
        try:
            args.selected = sorted({int(x) for tok in args.selected for x in tok.replace(",", " ").split()})
        except ValueError:
            p.error("--selected takes zero-based integers")
    for name in ("n_random", "arms_per_task", "n_components", "draws_per_epoch", "tta_avg", "tta_draws", "epochs", "avg_epochs",
                 "batch_size", "folds", "repeats", "n_bootstrap", "n_permutations", "jobs"):
        if getattr(args, name) < 1:
            p.error(f"--{name.replace('_', '-')} must be positive")
    if args.avg_epochs > args.epochs:
        p.error("--avg-epochs cannot exceed --epochs")
    if not (0 <= args.scout_dropout < 1 and 0 <= args.pc_dropout < 1):
        p.error("dropout probabilities must be in [0, 1)")
    if args.jobs > 1 and args.device not in ("auto", "cpu"):
        p.error("--jobs above 1 runs on CPU; drop --device or use --device cpu")
    return args


def selected_scouts(args: argparse.Namespace) -> tuple[list[int], int | None]:
    """The selected scouts (--selected, or the screen threshold) and the screen's best single scout."""
    auc = {}
    if Path(args.screen_csv).exists():
        with open(args.screen_csv, encoding="utf-8") as f:
            auc = {int(r["scout"]): float(r["auc_soft_vote_mean"]) for r in csv.DictReader(f) if r["pair"] == args.pair}
    best_single = max(auc, key=auc.get) if auc else None
    if args.selected is not None:
        return args.selected, best_single
    if not auc:
        raise ValueError(f"{args.screen_csv} has no rows for {args.pair}")
    chosen = sorted(s for s, a in auc.items() if a > args.threshold)
    if len(chosen) < 2:
        raise ValueError(f"Fewer than two scouts exceed AUC {args.threshold}")
    return chosen, best_single


def build_arms(args: argparse.Namespace, n_scouts: int, n_channels: int, selected: list[int]) -> list[sc.Arm]:
    k = len(selected)
    rng = np.random.default_rng(args.random_seed)
    random_sets, seen = [], {tuple(selected)}
    while len(random_sets) < args.n_random:
        subset = tuple(sorted(rng.choice(n_scouts, k, replace=False).tolist()))
        if subset not in seen:
            seen.add(subset)
            random_sets.append(list(subset))
    arms = []  # main arms first, random sets last
    if "selected" in args.arms:
        arms.append(sc.Arm(f"{MODEL}_{args.set_name}", "selected", MODEL, selected))
    if "all" in args.arms:
        arms.append(sc.Arm(f"{MODEL}_all{n_scouts}", "all", MODEL, list(range(n_scouts))))
    if "sensor" in args.arms:
        arms.append(sc.Arm(f"sensor{n_channels}", "sensor", "eegnet", None))
    if "random" in args.arms:
        arms += [sc.Arm(f"{MODEL}_random{k}_{i + 1:03d}", "random", MODEL, s) for i, s in enumerate(random_sets)]
    return arms


# ---------------------------------------------------------------- comparison

def compare(cfg: dict, meta: dict, arms: list[sc.Arm], best_single: int | None) -> None:
    selected, labels, originals = source.select_pair(np.asarray(meta["groups"]), cfg["pair"])
    position = {int(s): j for j, s in enumerate(selected)}
    fold_of = np.full((cfg["repeats"], len(selected)), -1)
    for r, folds in enumerate(cv.outer_splits(originals, cfg["folds"], cfg["repeats"], cfg["seed"])):
        for f, (_, test) in enumerate(folds):
            fold_of[r, test] = f
    rng = np.random.default_rng(cfg["random_seed"] + 1)
    permuted = np.stack([rng.permutation(labels) for _ in range(cfg["n_permutations"])]).astype(np.float64)
    by_group = [np.flatnonzero(originals == g) for g in np.unique(originals)]
    boot = [np.concatenate([rng.choice(ix, len(ix), replace=True) for ix in by_group])
            for _ in range(cfg["n_bootstrap"])]

    probs, missing = {}, []
    for arm in arms:
        loaded = sc.load_arm_probs(cfg, arm, position, len(selected), sc.arm_signature(cfg, meta, arm))
        if loaded is None:
            missing.append(arm.name)
        else:
            probs[arm.name] = loaded[0]
    if missing:
        print(f"Comparison skipped; arms not finished: {', '.join(missing[:6])}{' ...' if len(missing) > 6 else ''}")
        return
    single_name = None
    if best_single is not None:
        single = sc.load_single_scout_probs(cfg["single_scout_dir"], cfg["pair"], best_single, position, fold_of)
        if single is None:
            print(f"No complete single-scout predictions for scout {best_single}; that comparison is skipped")
        else:
            single_name = f"single_scout{best_single:02d}"
            probs[single_name] = single

    main = {a.kind: a.name for a in arms if a.kind in ("selected", "all", "sensor")}
    randoms = [a.name for a in arms if a.kind == "random"]
    if "selected" not in main:
        print("No selected arm to compare")
        return
    sel = main["selected"]
    arm_metrics = {name: stats.repeat_metrics(labels, probs[name], permuted)
                   for name in [single_name, *main.values()] if name}
    random_aucs = np.array([np.mean([stats.auc(labels, qr) for qr in probs[name]]) for name in randoms])
    target = arm_metrics[sel]["auc_mean"]
    rows = []
    if len(random_aucs):
        rows.append({"comparison": f"{sel} vs random sets ({len(random_aucs)})", "method": "percentile among random sets",
                     "difference": target - float(np.median(random_aucs)), "ci_low": "", "ci_high": "",
                     "percentile": float(100 * np.mean(random_aucs < target)),
                     "p_one_sided": float((1 + np.sum(random_aucs >= target - 1e-12)) / (1 + len(random_aucs)))})
    for other in (single_name, main.get("all"), main.get("sensor")):
        if other is None:
            continue
        d = stats.paired_bootstrap_auc_difference(labels, probs[sel], probs[other], boot)
        rows.append({"comparison": f"{sel} vs {other}", "method": "paired subject bootstrap",
                     "difference": target - arm_metrics[other]["auc_mean"],
                     "ci_low": float(np.percentile(d, 2.5)), "ci_high": float(np.percentile(d, 97.5)),
                     "percentile": "", "p_one_sided": float((1 + np.sum(d <= 0)) / (1 + len(d)))})
    for row, adjusted in zip(rows, stats.holm([row["p_one_sided"] for row in rows])):
        row["p_holm"] = adjusted
    summary_rows = [{"arm": name, **{k: (";".join(map(str, v)) if isinstance(v, list) else v)
                                     for k, v in values.items()}} for name, values in arm_metrics.items()]
    if len(random_aucs):
        summary_rows.append({"arm": f"{MODEL}_random (median of {len(random_aucs)})",
                             "auc_mean": float(np.median(random_aucs)), "auc_sd": float(random_aucs.std()),
                             "auc_by_repeat": "", "bal_acc_mean": "", "p_vs_chance": ""})

    out = Path(cfg["output_dir"]) / f"comparison_{cfg['set_name']}"
    io.save_csv(out / "arms_summary.csv", summary_rows)
    if rows:
        io.save_csv(out / "comparisons.csv", rows)
    io.save_json(out / "comparison_summary.json", {
        "best_single_scout_0based": best_single, "arms": summary_rows, "comparisons": rows,
        "random_set_aucs": random_aucs.tolist(), "n_bootstrap": cfg["n_bootstrap"],
        "n_permutations": cfg["n_permutations"]})
    plot_comparison(out / "comparison.pdf", sel, main, arm_metrics, random_aucs, single_name, meta,
                    best_single)

    print("\nArm                              AUC mean ± SD   by repeat                 bal acc  p vs chance")
    for row in summary_rows:
        if row["auc_by_repeat"]:
            print(f"{row['arm']:<32} {row['auc_mean']:.3f} ± {row['auc_sd']:.3f}   {row['auc_by_repeat']:<25} "
                  f"{row['bal_acc_mean']:.3f}    {row['p_vs_chance']:.4f}")
        else:
            print(f"{row['arm']:<32} {row['auc_mean']:.3f} ± {row['auc_sd']:.3f}")
    if rows:
        print("\nComparison                                   diff     95% CI             p (one-sided)  p (Holm)")
        for row in rows:
            ci = "" if row["ci_low"] == "" else f"[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]"
            extra = "" if row["percentile"] == "" else f"  ({row['percentile']:.0f}th percentile)"
            print(f"{row['comparison']:<44} {row['difference']:+.3f}   {ci:<18} {row['p_one_sided']:.4f}         "
                  f"{row['p_holm']:.4f}{extra}")
    print(f"\nComparison: {out / 'comparisons.csv'}")


def plot_comparison(path: Path, sel: str, main: dict, arm_metrics: dict, random_aucs: np.ndarray,
                    single_name: str | None, meta: dict, best_single: int | None) -> None:
    fig, axis = plt.subplots(figsize=(8, 4.8))
    rows = []
    if single_name:
        rows.append((single_name, f"Best single scout ({best_single + 1:02d} {meta['labels'][best_single]})",
                     plotting.AQUA))
    rows.append((sel, f"Selected scouts ({sel.removeprefix(MODEL + '_')})", plotting.BLUE))
    if len(random_aucs):
        rows.append(("random", f"Random sets ({len(random_aucs)})", plotting.NEUTRAL))
    if "all" in main:
        rows.append((main["all"], f"All scouts ({main['all'].rsplit('all', 1)[1]})", plotting.ORANGE))
    if "sensor" in main:
        rows.append((main["sensor"], f"Sensor channels ({main['sensor'][len('sensor'):]})", plotting.INK_SECONDARY))
    for y_pos, (name, _, color) in enumerate(rows[::-1]):
        if name == "random":
            jitter = np.random.default_rng(0).uniform(-0.12, 0.12, len(random_aucs))
            axis.plot(random_aucs, y_pos + jitter, "o", color=plotting.NEUTRAL_LIGHT, markersize=5, markeredgewidth=0)
            median = float(np.median(random_aucs))
            axis.plot(median, y_pos, "o", color=plotting.NEUTRAL, markersize=10, markeredgecolor="white",
                      markeredgewidth=1.5)
            axis.text(random_aucs.max() + 0.006, y_pos, f"median {median:.3f}", va="center", fontsize=9,
                      color=plotting.INK_SECONDARY)
            continue
        values = arm_metrics[name]
        axis.plot(values["auc_by_repeat"], [y_pos] * len(values["auc_by_repeat"]), "o", color=color, markersize=6,
                  alpha=0.45, markeredgewidth=0)
        axis.plot(values["auc_mean"], y_pos, "o", color=color, markersize=10, markeredgecolor="white",
                  markeredgewidth=1.5)
        axis.text(max(values["auc_by_repeat"]) + 0.006, y_pos, f"{values['auc_mean']:.3f}", va="center", fontsize=10,
                  color=plotting.INK)
    axis.set_yticks(range(len(rows)), [r[1] for r in rows[::-1]], fontsize=9, color=plotting.INK)
    axis.axvline(0.5, color=plotting.INK_SECONDARY, linestyle=":", linewidth=1)
    axis.set_title(f"Flat multi-scout EEGNet ({sel}): Control vs. Patients (D+S)", fontsize=12, color=plotting.INK,
                   loc="left", pad=16)
    axis.text(0, 1.01, "Scouts chosen with these subjects' labels: comparisons with random sets, all scouts, and "
              "sensors are optimistic", transform=axis.transAxes, fontsize=8, color=plotting.INK_SECONDARY)
    axis.set_xlabel("Out-of-fold ROC AUC (small: each repeat or random set; large: mean / median)", fontsize=9,
                    color=plotting.INK)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(colors=plotting.INK_SECONDARY, labelsize=9)
    axis.grid(axis="x", color="#ececea", linewidth=0.7)
    axis.set_axisbelow(True)
    fig.tight_layout()
    plotting.save_figure(fig, path)


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        args.cache_dir, args.trial_index = source.build_synthetic_cache()
        args.output_dir = args.projection_root = args.cache_dir.parent / "output"
        args.category, args.selected = "Intention", args.selected or [0, 1, 2]
        args.screen_csv = args.cache_dir / "no_screen.csv"  # no single-scout reference on synthetic data
        args.n_random, args.repeats, args.epochs, args.avg_epochs = 2, 1, 3, 2
        args.n_bootstrap, args.n_permutations = 200, 200
    meta = source.ensure_cache(args.cache_dir, args.eeg, args.tmatrix, args.atlas, args.subject_index)
    n_scouts = len(meta["vertices"])
    n_channels = int(np.load(args.cache_dir / "sensor_000.npy", mmap_mode="r").shape[1])
    selected, best_single = selected_scouts(args)
    if max(selected) >= n_scouts:
        raise ValueError(f"selected scouts must be between 0 and {n_scouts - 1}")
    args.set_name = args.set_name or f"selected{len(selected)}"
    arms = build_arms(args, n_scouts, n_channels, selected)
    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    device = sc.choose_device("cpu" if args.jobs > 1 else args.device)
    print(f"{args.pair} | selected scouts (0-based): {selected} | {len(arms)} arms | "
          f"{args.repeats} x {args.folds}-fold CV | output {args.output_dir}", flush=True)
    io.save_json(args.output_dir / f"comparison_{args.set_name}" / "arms.json",
                 {"set_name": args.set_name, "selected": selected, "best_single_scout": best_single,
                  "arms": [vars(a) for a in arms]})

    if not args.compare_only:
        scouts = sorted({s for a in arms if a.scouts for s in a.scouts})
        if scouts:
            source.ensure_projections(cfg, meta, scouts, args.jobs)
        chunks = [arms[i:i + args.arms_per_task] for i in range(0, len(arms), args.arms_per_task)]
        tasks = [(chunk, r, f) for chunk in chunks for r in range(args.repeats) for f in range(args.folds)]
        print(f"Training {len(arms) * args.repeats * args.folds} folds in {len(tasks)} tasks on {device}...", flush=True)
        if args.jobs == 1:
            for chunk, r, f in tasks:
                for result in sc.run_arm_chunk(cfg, meta, chunk, r, f, device):
                    print(sc.fold_line(result), flush=True)
        else:
            from joblib import Parallel, delayed
            for results in Parallel(n_jobs=args.jobs, return_as="generator_unordered")(
                    delayed(sc.run_arm_chunk)(cfg, meta, chunk, r, f, device) for chunk, r, f in tasks):
                for result in results:
                    print(sc.fold_line(result), flush=True)
    compare(cfg, meta, arms, best_single)
    if args.smoke_test:
        print(f"Smoke-test output: {args.output_dir}")


if __name__ == "__main__":
    main()
