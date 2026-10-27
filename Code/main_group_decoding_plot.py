"""Time-resolved group decoding of selected scouts in the decodingPlot layout.

Reads main_group_decoding.py's result and draws one large panel per scout: the
cross-validated AUC (mean over CV repeats), a band of ± SD across the repeats, chance
(AUC 0.5), and red bars over the scout's significant clusters (label-permutation cluster
test, FWE across all scouts x time). By default the scouts with significant clusters are
shown, ordered by cluster p.

The band shows only how much the AUC changes with the CV split; unlike the ± SEM band
of the sentence-decoding plots it is not an uncertainty across subjects (that is what
the label-permutation test addresses).

Example (run from Code/):
    python3 main_group_decoding_plot.py
    python3 main_group_decoding_plot.py --scouts 27 57 25 48 21
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from utils import io, paths, plotting


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=paths.RESULTS / "group_decoding_scout" / "group_decoding.pkl")
    p.add_argument("--scouts", type=int, nargs="+", default=None,
                   help="Zero-based scouts to plot (default: those with significant clusters)")
    p.add_argument("--output", type=Path, default=None,
                   help="Default: group_decoding_significant_scouts.pdf next to --input")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result = io.load_pickle(args.input)
    times = np.asarray(result["times_ms"], dtype=float)
    scouts = list(result["scout_indices"])
    if args.scouts is None:
        order = sorted(result["significant_clusters"], key=lambda c: c["p_value"])
        chosen = list(dict.fromkeys(c["scout_index"] for c in order))
    else:
        chosen = args.scouts
    missing = [s for s in chosen if s not in scouts]
    if not chosen or missing:
        raise ValueError(f"No scouts to plot, or scouts not in the result: {missing}")
    by_repeat = np.asarray(result["auc_by_repeat"])  # repeat x scout x time
    mask = np.asarray(result["significant_mask"])  # time x scout
    panels = []
    for s in chosen:
        i = scouts.index(s)
        p_values = [c["p_value"] for c in result["significant_clusters"] if c["scout_index"] == s]
        title = io.pretty_scout_name(result["scout_labels"][i])
        panels.append({"title": title, "mean": by_repeat[:, i].mean(0), "spread": by_repeat[:, i].std(0, ddof=1),
                       "mask": mask[:, i], "line_label": "AUC (± SD over CV repeats)"})
        print(f"{title}: peak AUC {by_repeat[:, i].mean(0).max():.3f}; cluster p = "
              f"{', '.join(f'{p:.3f}' for p in p_values) or 'none'}")
    n_repeats = by_repeat.shape[0]
    fig = plotting.decoding_panels(
        panels, times, "AUC-ROC", 0.5, "Chance = 0.5", (0.40, 0.75), 0.01, 0.025,
        f"Control vs Patients (D+S) | shaded: ± SD across {n_repeats} CV repeats | red: cluster p < 0.05 "
        f"(label permutation, FWE across scouts × time)")
    output = args.output or args.input.with_name("group_decoding_significant_scouts.pdf")
    plotting.save_figure(fig, output)
    print(f"Saved figure: {output}")


if __name__ == "__main__":
    main()
