"""Group time courses of sentence decoding with a time-series cluster permutation test.

Plots mean ± SEM decoding accuracy per group from main_sentence_decoding.py, either
for one scout (--scout) or averaged over scouts, and marks time points in clusters
where accuracy exceeds 50% chance. The test is a one-sample, one-tailed cluster
permutation test on accuracy - 50% (sign-flip permutations,
utils.stats.cluster_based_permutation_test_timeseries_1samp). Cluster p-values are
Bonferroni-corrected across the plotted groups unless --no-correct-across-groups.

Examples (run from Code/):
    python3 main_decoding_plot.py --input ../Results/svmDecoding_scout_con_Int.pkl
    python3 main_decoding_plot.py --input ../Results/svmDecoding_scout_con_Int.pkl --scout 27
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from utils import io, paths, plotting, stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=paths.RESULTS / "svmDecoding_scout_con_Int.pkl")
    p.add_argument("--atlas", type=Path, default=paths.ATLAS)
    p.add_argument("--subject-index", type=Path, default=paths.SUBJECT_INDEX,
                   help="Used only when the decoding result has no subIdx")
    p.add_argument("--scout", type=int, default=None, help="Zero-based atlas scout; default: average over the result's scouts")
    p.add_argument("--grouping", choices=io.GROUPINGS, default="three")
    p.add_argument("--n-permutations", type=int, default=5000)
    p.add_argument("--threshold-p", type=float, default=0.05)
    p.add_argument("--cluster-alpha", type=float, default=0.05)
    p.add_argument("--correct-across-groups", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--output-dir", type=Path, default=paths.RESULTS / "decoding_plot")
    return p.parse_args()


def load_time_courses(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, dict]:
    """Subject x time accuracy (%), group codes, times (ms), a description of the selection, and
    the decoding parameters."""
    result = io.load_pickle(args.input)
    scores = np.asarray(result["decodeScore"], dtype=np.float64)
    groups = (np.asarray(result["subIdx"]).squeeze() if "subIdx" in result
              else io.load_subject_groups(args.subject_index))
    if scores.ndim == 3:  # scout x time x subject
        if args.scout is None:
            data, label = scores.mean(0).T, f"mean over {scores.shape[0]} scouts"
        else:
            # --scout is an atlas index; a result decoded on a scout subset stores its atlas indices.
            stored = [int(s) for s in result.get("scout_indices", range(scores.shape[0]))]
            if args.scout not in stored:
                raise IndexError(f"--scout {args.scout} is not in this result (atlas scouts {stored[0]}-{stored[-1]}, "
                                 f"{len(stored)} stored)")
            local = stored.index(args.scout)
            names = result.get("scout_labels")
            name = names[local] if names is not None else io.load_atlas(args.atlas)[1][args.scout]
            data, label = scores[local].T, io.pretty_scout_name(name)
    elif scores.ndim == 2:  # subject x time (whole-source decoding)
        if args.scout is not None:
            raise ValueError("--scout needs a scout-wise decoding result")
        data, label = scores, "whole source"
    else:
        raise ValueError(f"decodeScore must be scout x time x subject or subject x time; got {scores.shape}")
    if data.shape[0] != groups.size:
        raise ValueError(f"{data.shape[0]} subjects in decodeScore but {groups.size} group codes")
    times = np.asarray(result.get("times_ms", paths.EPOCH_START_MS + np.arange(data.shape[1]) * 1000 / paths.FS))
    if times.size != data.shape[1]:
        raise ValueError("times_ms does not match decodeScore")
    if data.max() <= 1:
        data = data * 100
    return data, groups, times, label, result.get("parameters", {})


def main() -> None:
    args = parse_args()
    data, groups, times, label, parameters = load_time_courses(args)
    grouping = io.GROUPINGS[args.grouping]
    factor = len(grouping) if args.correct_across_groups else 1
    results, rows = {}, []
    for key, name, codes in grouping:
        members = np.flatnonzero(np.isin(groups, codes))
        _, t_obs, clusters, p_values = stats.cluster_based_permutation_test_timeseries_1samp(
            data[members], chance=50.0, n_permutations=args.n_permutations, threshold_p=args.threshold_p,
            cluster_alpha=args.cluster_alpha, tail=1, seed=args.seed, n_jobs=args.n_jobs)
        mask = np.zeros(times.size, dtype=bool)
        print(f"\n{name} (n={members.size})")
        for i, (cluster, p) in enumerate(zip(clusters, p_values)):
            adjusted = min(1.0, float(p) * factor)
            if adjusted >= args.cluster_alpha:
                continue
            cluster_mask = stats.cluster_to_mask(cluster, t_obs.shape)
            mask |= cluster_mask
            span = times[cluster_mask]
            print(f"  cluster {i}: p = {p:.4f} (group-adjusted {adjusted:.4f}), {span[0]:.0f} to {span[-1]:.0f} ms")
            rows.append({"group": name, "n_subjects": int(members.size), "cluster": i, "p_value": float(p),
                         "group_adjusted_p_value": adjusted, "start_ms": float(span[0]), "end_ms": float(span[-1]),
                         "n_time_samples": int(cluster_mask.sum())})
        results[key] = (name, data[members], mask)

    correction = f"Bonferroni x{factor} across groups" if factor > 1 else "no correction across groups"
    panels = [{"title": f"Sentence decoding: {name}", "mean": d.mean(0), "spread": d.std(0) / np.sqrt(len(d)),
               "mask": mask} for name, d, mask in results.values()]
    avg_num = parameters.get("avg_num")
    ylabel = f"Decoding accuracy (%, {avg_num}-trial averages)" if avg_num else "Decoding accuracy (%)"
    fig = plotting.decoding_panels(panels, times, ylabel, 50.0, "Baseline = 50%", (45.0, 65.0), 1.0, 2.5,
                                   f"{args.input.stem} | {label} | one-sample cluster test vs 50% ({correction})")
    selection = "mean" if args.scout is None else f"scout{args.scout:02d}"
    stem = f"{args.input.stem}_{selection}_{args.grouping}"
    plotting.save_figure(fig, args.output_dir / f"{stem}.pdf")
    io.save_csv(args.output_dir / f"{stem}_clusters.csv", rows,
                ["group", "n_subjects", "cluster", "p_value", "group_adjusted_p_value", "start_ms", "end_ms",
                 "n_time_samples"])
    print(f"\nSaved figure: {args.output_dir / (stem + '.pdf')}")


if __name__ == "__main__":
    main()
