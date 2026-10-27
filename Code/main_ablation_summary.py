"""One table and figure across the selected-scout sets of main_group_classifier.py.

For every set run by main_group_classifier.py (folders comparison_<set>/ under
--output-dir): its selected arm, its random sets of the same size, all scouts, the
sensors, and the screen's best single scout. All arms share the outer splits and
training seeds, so comparisons are paired by subject. Every arm must have been trained
with the same settings (checked from the fold signatures).

Every set was chosen with the labels of the same subjects (screen AUCs or time-resolved
group-decoding clusters), and the best single scout is the maximum over all scouts, so
these AUCs and the comparisons with random sets, all scouts, and sensors are optimistic.

Comparisons (AUC averaged over repeats):
  within each set   selected vs its random sets (percentile p), vs best single scout,
                    all scouts, and sensors (paired subject bootstrap, one-sided);
                    Holm across the set's comparisons
  between sets      every pair of selected sets (paired bootstrap, two-sided); Holm
                    across pairs
Each arm is also tested against chance with label permutation.

Example (run from Code/):
    python3 main_ablation_summary.py --sets timeres3 timeres6 selected17
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from decoders import subject_classifier as sc
from utils import cv, io, paths, plotting, source, stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", type=Path, default=paths.RESULTS / "group_classifier")
    p.add_argument("--sets", nargs="+", default=None, help="Set names (default: every comparison_<set>/ found)")
    p.add_argument("--pair", choices=source.PAIRS, default="control-vs-depressed+suicidal")
    p.add_argument("--cache-dir", type=Path, default=paths.SUBJECT_CACHE)
    p.add_argument("--single-scout-dir", type=Path, default=paths.RESULTS / "scout_subject_cds_all")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-bootstrap", type=int, default=10_000)
    p.add_argument("--n-permutations", type=int, default=10_000)
    p.add_argument("--random-seed", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sets = args.sets or sorted(d.name.removeprefix("comparison_") for d in args.output_dir.glob("comparison_*")
                               if (d / "arms.json").exists())
    if not sets:
        raise ValueError(f"No comparison_<set>/arms.json under {args.output_dir}")
    info = {name: json.loads((args.output_dir / f"comparison_{name}" / "arms.json").read_text()) for name in sets}
    meta = source.ensure_cache(args.cache_dir)
    selected, labels, originals = source.select_pair(np.asarray(meta["groups"]), args.pair)
    position = {int(s): j for j, s in enumerate(selected)}
    fold_of = np.full((args.repeats, len(selected)), -1)
    for r, folds in enumerate(cv.outer_splits(originals, args.folds, args.repeats, args.seed)):
        for f, (_, test) in enumerate(folds):
            fold_of[r, test] = f
    cfg = {"output_dir": str(args.output_dir), "repeats": args.repeats, "folds": args.folds}
    reference = {}  # training settings of the first arm loaded; every arm must match them

    def load(arm: dict) -> np.ndarray:
        loaded = sc.load_arm_probs(cfg, sc.Arm(**arm), position, len(selected))
        if loaded is None:
            raise ValueError(f"Arm {arm['name']} is not finished")
        probs, settings = loaded
        reference.setdefault("settings", settings)
        if settings != reference["settings"]:
            differ = sorted(k for k in settings.keys() | reference["settings"].keys()
                            if settings.get(k) != reference["settings"].get(k))
            raise ValueError(f"Arm {arm['name']} was trained with other settings ({', '.join(differ)})")
        return probs

    shared, per_set = {}, {}
    for name in sets:
        arms = info[name]["arms"]
        per_set[name] = {"selected": next(a for a in arms if a["kind"] == "selected"),
                         "random": [a for a in arms if a["kind"] == "random"]}
        for a in arms:
            if a["kind"] in ("all", "sensor"):
                shared[a["kind"]] = a
    best_single = next((i["best_single_scout"] for i in info.values() if i.get("best_single_scout") is not None), None)

    rng = np.random.default_rng(args.random_seed)
    permuted = np.stack([rng.permutation(labels) for _ in range(args.n_permutations)]).astype(np.float64)
    by_group = [np.flatnonzero(originals == g) for g in np.unique(originals)]
    boot = [np.concatenate([rng.choice(ix, len(ix), replace=True) for ix in by_group]) for _ in range(args.n_bootstrap)]

    probs, metrics = {}, {}
    if best_single is not None:
        single = sc.load_single_scout_probs(args.single_scout_dir, args.pair, best_single, position, fold_of)
        if single is not None:
            probs["single"] = single
    for kind, arm in shared.items():
        probs[kind] = load(arm)
    random_aucs = {}
    for name in sets:
        probs[name] = load(per_set[name]["selected"])
        random_aucs[name] = np.array([np.mean([stats.auc(labels, qr) for qr in load(a)])
                                      for a in per_set[name]["random"]])
    for key, q in probs.items():
        metrics[key] = stats.repeat_metrics(labels, q, permuted)
    label = {"single": f"best single scout ({best_single + 1:02d} {meta['labels'][best_single]})"
             if best_single is not None else "best single scout",
             "all": f"all {len(meta['vertices'])} scouts", "sensor": "sensors"}

    comparisons = []
    for name in sets:
        target, rows = metrics[name]["auc_mean"], []
        if len(random_aucs[name]):
            r = random_aucs[name]
            rows.append({"family": name, "comparison": f"{name} vs random {len(per_set[name]['selected']['scouts'])}-scout"
                         f" sets ({len(r)})", "method": "percentile among random sets",
                         "difference": target - float(np.median(r)), "ci_low": "", "ci_high": "",
                         "percentile": float(100 * np.mean(r < target)),
                         "p": float((1 + np.sum(r >= target - 1e-12)) / (1 + len(r)))})
        for other in ("single", "all", "sensor"):
            if other in probs:
                d = stats.paired_bootstrap_auc_difference(labels, probs[name], probs[other], boot)
                rows.append({"family": name, "comparison": f"{name} vs {label[other]}",
                             "method": "paired subject bootstrap (one-sided)",
                             "difference": target - metrics[other]["auc_mean"], "ci_low": float(np.percentile(d, 2.5)),
                             "ci_high": float(np.percentile(d, 97.5)), "percentile": "",
                             "p": float((1 + np.sum(d <= 0)) / (1 + len(d)))})
        for row, adjusted in zip(rows, stats.holm([row["p"] for row in rows])):
            row["p_holm"] = adjusted
        comparisons += rows
    rows = []
    for a, b in itertools.combinations(sets, 2):
        d = stats.paired_bootstrap_auc_difference(labels, probs[a], probs[b], boot)
        rows.append({"family": "between sets", "comparison": f"{a} vs {b}", "method": "paired subject bootstrap (two-sided)",
                     "difference": metrics[a]["auc_mean"] - metrics[b]["auc_mean"],
                     "ci_low": float(np.percentile(d, 2.5)), "ci_high": float(np.percentile(d, 97.5)), "percentile": "",
                     "p": float(min(1.0, 2 * min((1 + np.sum(d <= 0)), (1 + np.sum(d >= 0))) / (1 + len(d))))})
    for row, adjusted in zip(rows, stats.holm([row["p"] for row in rows]) if rows else []):
        row["p_holm"] = adjusted
    comparisons += rows

    arm_rows = []
    for key in ["single", *sets, "all", "sensor"]:
        if key in metrics:
            arm_rows.append({"arm": label.get(key, key), **{k: (";".join(map(str, v)) if isinstance(v, list) else v)
                                                            for k, v in metrics[key].items()}})
        if key in random_aucs and len(random_aucs[key]):
            r = random_aucs[key]
            arm_rows.append({"arm": f"random {len(per_set[key]['selected']['scouts'])}-scout sets (median of {len(r)})",
                             "auc_mean": float(np.median(r)), "auc_sd": float(r.std()),
                             "auc_by_repeat": f"95% range {np.percentile(r, 2.5):.3f}-{np.percentile(r, 97.5):.3f}",
                             "bal_acc_mean": "", "p_vs_chance": ""})
    out = args.output_dir / "ablation_summary"
    io.save_csv(out / "arms.csv", arm_rows)
    io.save_csv(out / "comparisons.csv", comparisons)
    io.save_json(out / "summary.json", {"sets": {n: {"scouts_0based": per_set[n]["selected"]["scouts"],
                                                     "random_set_aucs": random_aucs[n].tolist()} for n in sets},
                                        "best_single_scout_0based": best_single, "arms": arm_rows,
                                        "comparisons": comparisons})
    plot(out / "ablation_summary.pdf", sets, per_set, metrics, random_aucs, label)

    print("Arm                                          AUC mean ± SD   bal acc  p vs chance")
    for row in arm_rows:
        extra = (f"   {row['bal_acc_mean']:.3f}    {row['p_vs_chance']:.4f}" if row["p_vs_chance"] != ""
                 else f"   ({row['auc_by_repeat']})")
        print(f"{row['arm']:<44} {row['auc_mean']:.3f} ± {row['auc_sd']:.3f}{extra}")
    print("\nComparison                                          diff     95% CI              p        p (Holm)")
    for row in comparisons:
        ci = "" if row["ci_low"] == "" else f"[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]"
        extra = "" if row["percentile"] == "" else f"  ({row['percentile']:.0f}th percentile)"
        print(f"{row['comparison']:<51} {row['difference']:+.3f}   {ci:<19} {row['p']:.4f}   {row['p_holm']:.4f}{extra}")
    print(f"\nOutput: {out}")


def plot(path: Path, sets: list[str], per_set: dict, metrics: dict, random_aucs: dict, label: dict) -> None:
    rows = []
    if "single" in metrics:
        rows.append(("single", label["single"], plotting.AQUA))
    for name in sets:
        rows.append((name, f"{name} ({len(per_set[name]['selected']['scouts'])} scouts)", plotting.BLUE))
        if len(random_aucs[name]):
            rows.append((("random", name), f"random {len(per_set[name]['selected']['scouts'])}-scout sets "
                                           f"({len(random_aucs[name])})", plotting.NEUTRAL))
    for key, color in (("all", plotting.ORANGE), ("sensor", plotting.INK_SECONDARY)):
        if key in metrics:
            rows.append((key, label[key], color))
    fig, axis = plt.subplots(figsize=(9, 0.55 * len(rows) + 1.4))
    for y_pos, (key, _, color) in enumerate(rows[::-1]):
        if isinstance(key, tuple):
            r = random_aucs[key[1]]
            jitter = np.random.default_rng(0).uniform(-0.15, 0.15, len(r))
            axis.plot(r, y_pos + jitter, "o", color=plotting.NEUTRAL_LIGHT, markersize=3.5, markeredgewidth=0)
            axis.plot(np.median(r), y_pos, "o", color=plotting.NEUTRAL, markersize=9, markeredgecolor="white",
                      markeredgewidth=1.5)
            axis.text(r.max() + 0.006, y_pos, f"median {np.median(r):.3f}", va="center", fontsize=8,
                      color=plotting.INK_SECONDARY)
            continue
        m = metrics[key]
        axis.plot(m["auc_by_repeat"], [y_pos] * len(m["auc_by_repeat"]), "o", color=color, markersize=5, alpha=0.45,
                  markeredgewidth=0)
        axis.plot(m["auc_mean"], y_pos, "o", color=color, markersize=9, markeredgecolor="white", markeredgewidth=1.5)
        axis.text(max(m["auc_by_repeat"]) + 0.006, y_pos, f"{m['auc_mean']:.3f}", va="center", fontsize=9,
                  color=plotting.INK)
    axis.set_yticks(range(len(rows)), [r[1] for r in rows[::-1]], fontsize=9, color=plotting.INK)
    axis.axvline(0.5, color=plotting.INK_SECONDARY, linestyle=":", linewidth=1)
    axis.set_title("Flat multi-scout EEGNet: selected scout sets vs random sets, all scouts, sensors",
                   fontsize=11, color=plotting.INK, loc="left", pad=16)
    axis.text(0, 1.01, "Every set was chosen with these subjects' labels and the best single scout is the maximum of "
              "all scouts: AUCs and comparisons are optimistic", transform=axis.transAxes, fontsize=7.5,
              color=plotting.INK_SECONDARY)
    axis.set_xlabel("Out-of-fold ROC AUC (small: each repeat or random set; large: mean / median)", fontsize=9)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(colors=plotting.INK_SECONDARY, labelsize=9)
    axis.grid(axis="x", color="#ececea", linewidth=0.7)
    axis.set_axisbelow(True)
    fig.tight_layout()
    plotting.save_figure(fig, path)


if __name__ == "__main__":
    main()
