"""Network timeline of sentence decoding and time-resolved group decoding.

A. Sentence (congruency) decoding of all subjects (main_sentence_decoding.py), averaged
   over the scouts of each anatomical network: onset (first time the 20-ms smoothed mean
   reaches half its peak above chance) and center (time by which half of the above-chance
   decoding in 200-1000 ms has occurred), with subject-bootstrap 95% CIs.
B. The center per group (Control vs Depressed + Suicidal), bootstrapped within group;
   descriptive only (the group difference is not tested).
C. Time-resolved group decoding (main_group_decoding.py): every scout's best cluster with
   p < --trend-p, its window and AUC peak (clusters with 0.05 <= p < --trend-p are trends).

Example (run from Code/):
    python3 main_network_timeline.py
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from utils import io, paths, plotting, stats

NETWORKS = {
    "Visual (occipital)": r"^(cuneus|pericalcarine|lingual|lateraloccipital) ",
    "Ventral temporal": "fusiform|inferiortemporal|parahippocampal|entorhinal",
    "Lateral temporal": "superiortemporal|middletemporal|transversetemporal",
    "Insula": "insula",
    "Inferior frontal": "parsopercularis|parstriangularis|parsorbitalis",
    "Orbitofrontal": "orbitofrontal",
    "ACC / medial PFC": "anteriorcingulate|superiorfrontal",
    "Posterior midline": "posteriorcingulate|isthmuscingulate|precuneus",
    "Parietal": "superiorparietal|inferiorparietal|supramarginal",
    "Dorsolateral PFC": "middlefrontal",
    "Sensorimotor": "precentral|postcentral|paracentral",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sentence-input", type=Path, default=paths.RESULTS / "svmDecoding_scout_con_Int.pkl")
    p.add_argument("--group-dir", type=Path, default=paths.RESULTS / "group_decoding_scout")
    p.add_argument("--output-dir", type=Path, default=paths.RESULTS / "network_timeline")
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--trend-p", type=float, default=0.10, help="Group-decoding clusters shown up to this p")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def latencies(curve: np.ndarray, times: np.ndarray) -> tuple[float, float]:
    """Onset (half of peak above chance) and center (half of above-chance area in 200-1000 ms)."""
    post = times >= 0
    tp = times[post]
    c = np.convolve(curve, np.ones(5) / 5, mode="same")[post] - 0.5
    ip = c.argmax()
    onset = tp[np.flatnonzero(c[:ip + 1] >= 0.5 * c[ip])[0]]
    window = (tp >= 200) & (tp <= 1000)
    area = np.clip(c[window], 0, None)
    return float(onset), float(tp[window][np.searchsorted(np.cumsum(area), 0.5 * area.sum())])


def bootstrap(curves: np.ndarray, members: np.ndarray, times: np.ndarray, rng, n: int) -> np.ndarray:
    """n x 2 (onset, center) from resampling ``members`` (columns of curves: time x subject)."""
    return np.array([latencies(curves[:, rng.choice(members, members.size)].mean(1), times) for _ in range(n)])


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    result = io.load_pickle(args.sentence_input)
    scores = np.asarray(result["decodeScore"], dtype=float)  # scout x time x subject
    times = np.asarray(result["times_ms"], dtype=float)
    labels = result["scout_labels"]
    groups = np.asarray(result["subIdx"]).squeeze().astype(int)
    members = {k: [i for i, l in enumerate(labels) if re.search(v, l)] for k, v in NETWORKS.items()}
    if sorted(i for v in members.values() for i in v) != list(range(len(labels))):
        raise ValueError("Every scout must belong to exactly one network")
    everyone, control, patients = np.arange(groups.size), np.flatnonzero(groups == 1), np.flatnonzero(groups > 1)

    rows = []
    for name, idx in members.items():
        curves = scores[idx].mean(0)  # time x subject
        onset, center = latencies(curves.mean(1), times)
        b_all = bootstrap(curves, everyone, times, rng, args.n_bootstrap)
        c_ctl, c_pat = latencies(curves[:, control].mean(1), times)[1], latencies(curves[:, patients].mean(1), times)[1]
        b_ctl, b_pat = bootstrap(curves, control, times, rng, args.n_bootstrap), bootstrap(curves, patients, times, rng,
                                                                                         args.n_bootstrap)
        smooth = np.convolve(curves.mean(1), np.ones(5) / 5, mode="same")
        rows.append({"network": name, "n_scouts": len(idx), "onset_ms": onset,
                     "onset_ci_low": np.percentile(b_all[:, 0], 2.5), "onset_ci_high": np.percentile(b_all[:, 0], 97.5),
                     "center_ms": center, "center_ci_low": np.percentile(b_all[:, 1], 2.5),
                     "center_ci_high": np.percentile(b_all[:, 1], 97.5), "peak_accuracy": float(smooth[times >= 0].max()),
                     "center_control_ms": c_ctl, "center_control_ci": np.percentile(b_ctl[:, 1], [2.5, 97.5]).tolist(),
                     "center_patients_ms": c_pat, "center_patients_ci": np.percentile(b_pat[:, 1], [2.5, 97.5]).tolist(),
                     "curve": smooth})
    rows.sort(key=lambda r: r["center_ms"])

    group = io.load_pickle(args.group_dir / "group_decoding.pkl")
    maps = io.load_pickle(args.group_dir / "decoding_maps.pkl")
    test = stats.permutation_cluster_maps(maps["auc"], maps["null"], 0.05)
    g_times = np.asarray(group["times_ms"])
    clusters = []
    for i, scout in enumerate(group["scout_indices"]):
        ids = np.unique(test["run"][i][test["run"][i] > 0])
        if not ids.size:
            continue
        best = min(ids, key=lambda c: test["p_values"][c - 1])
        p = float(test["p_values"][best - 1])
        if p < args.trend_p:
            span = np.flatnonzero(test["run"][i] == best)
            peak = span[np.argmax(maps["auc"][i, span])]
            clusters.append({"scout": int(scout), "label": group["scout_labels"][i], "start_ms": float(g_times[span[0]]),
                             "end_ms": float(g_times[span[-1]]), "peak_ms": float(g_times[peak]),
                             "peak_auc": float(maps["auc"][i, peak]), "p_value": p,
                             "curve": maps["auc"][i] - test["null_mean"][i]})
    clusters.sort(key=lambda c: c["start_ms"])

    out = args.output_dir
    io.save_csv(out / "sentence_network_latencies.csv",
                [{k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items() if k != "curve"} for r in rows])
    io.save_csv(out / "group_decoding_clusters.csv",
                [{k: v for k, v in c.items() if k != "curve"} for c in clusters])
    sizes = (control.size, patients.size)
    plot(out / "network_timeline.pdf", rows, clusters, times, g_times, args.trend_p, sizes)
    plot(out / "network_timeline.png", rows, clusters, times, g_times, args.trend_p, sizes)

    print("A/B. Sentence decoding (con_Int), networks ordered by center latency")
    for r in rows:
        print(f"  {r['network']:<20} onset {r['onset_ms']:4.0f} [{r['onset_ci_low']:4.0f},{r['onset_ci_high']:4.0f}]  "
              f"center {r['center_ms']:4.0f} [{r['center_ci_low']:4.0f},{r['center_ci_high']:4.0f}]  "
              f"control {r['center_control_ms']:4.0f} / patients {r['center_patients_ms']:4.0f}")
    print(f"C. Group decoding clusters (p < {args.trend_p})")
    for c in clusters:
        print(f"  {c['label']:<28} {c['start_ms']:4.0f}-{c['end_ms']:4.0f} ms, peak {c['peak_ms']:4.0f} ms "
              f"(AUC {c['peak_auc']:.3f}), p = {c['p_value']:.3f}")
    print(f"Figure: {out / 'network_timeline.pdf'}")


def plot(path: Path, rows: list[dict], clusters: list[dict], times: np.ndarray, g_times: np.ndarray,
         trend_p: float, sizes: tuple[int, int]) -> None:
    n_a, n_c = len(rows), len(clusters)
    fig, axes = plt.subplots(3, 1, figsize=(11, 0.42 * (2 * n_a + n_c) + 3.2), sharex=True,
                             gridspec_kw={"height_ratios": [n_a + 1.2, n_a + 1.2, n_c + 1.2]})
    ax_a, ax_b, ax_c = axes
    cmap = plt.get_cmap("Blues")
    vmax = max(r["curve"].max() - 0.5 for r in rows)
    for ax in axes:
        for x0, x1, label in ((300, 500, "N400 window"), (500, 1000, "late (LPP) window")):
            ax.axvspan(x0, x1, color="#f3f2ef" if label.startswith("N") else "#f8f7f5", zorder=0, linewidth=0)
        ax.axvline(0, color=plotting.INK_SECONDARY, linestyle=":", linewidth=1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=plotting.INK_SECONDARY, labelsize=9)
    for x, label in ((400, "N400 window"), (750, "late (LPP) window")):
        ax_a.text(x, -0.9, label, ha="center", va="bottom", fontsize=8, color=plotting.INK_SECONDARY)

    # A: decoding strength strip + onset and center per network
    for y, r in enumerate(rows):
        strength = np.clip(r["curve"] - 0.5, 0, None) / vmax
        ax_a.imshow(strength[None, :], extent=(times[0], times[-1], y + 0.32, y - 0.32), aspect="auto",
                    cmap=cmap, vmin=0, vmax=1.05, zorder=1, interpolation="nearest")
        ax_a.plot([r["center_ci_low"], r["center_ci_high"]], [y, y], color=plotting.INK, linewidth=1.6, zorder=3)
        ax_a.plot(r["center_ms"], y, "D", color=plotting.ORANGE, markersize=7, markeredgecolor="white", zorder=4)
        ax_a.plot(r["onset_ms"], y, "|", color=plotting.INK, markersize=12, markeredgewidth=2, zorder=4)
    ax_a.set_yticks(range(n_a), [f"{r['network']} ({r['n_scouts']})" for r in rows], fontsize=9)
    ax_a.set_ylim(n_a - 0.4, -1)
    ax_a.set_title("A. Sentence decoding (all subjects)", fontsize=11, loc="left", color=plotting.INK)
    ax_a.legend(handles=[Line2D([], [], marker="|", color=plotting.INK, linestyle="", markersize=10, markeredgewidth=2,
                                label="onset (half peak)"),
                         Line2D([], [], marker="D", color=plotting.ORANGE, linestyle="-", markeredgecolor="white",
                                label="center ± 95% CI"),
                         plt.Rectangle((0, 0), 1, 1, color=cmap(0.7), label="decoding above chance (shade)")],
                fontsize=8, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3, frameon=False)

    # B: center per group
    for y, r in enumerate(rows):
        for value, ci, color, dy in ((r["center_control_ms"], r["center_control_ci"], plotting.BLUE, -0.14),
                                     (r["center_patients_ms"], r["center_patients_ci"], plotting.ORANGE, 0.14)):
            ax_b.plot(ci, [y + dy, y + dy], color=color, linewidth=1.4, alpha=0.6)
            ax_b.plot(value, y + dy, "o", color=color, markersize=6, markeredgecolor="white")
        ax_b.plot([r["center_control_ms"], r["center_patients_ms"]], [y - 0.14, y + 0.14], color=plotting.NEUTRAL,
                  linewidth=0.8)
    ax_b.set_yticks(range(n_a), [r["network"] for r in rows], fontsize=9)
    ax_b.set_ylim(n_a - 0.4, -0.8)
    ax_b.set_title("B. Center by group (descriptive, not tested)", fontsize=11,
                   loc="left", color=plotting.INK)
    ax_b.legend(handles=[Line2D([], [], marker="o", color=plotting.BLUE, label=f"Control (n={sizes[0]})"),
                         Line2D([], [], marker="o", color=plotting.ORANGE, label=f"Patients D+S (n={sizes[1]})")],
                fontsize=8, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=False)

    # C: group decoding clusters
    for y, c in enumerate(clusters):
        strength = np.clip(c["curve"], 0, None) / max(cl["curve"].max() for cl in clusters)
        ax_c.imshow(strength[None, :], extent=(g_times[0], g_times[-1], y + 0.32, y - 0.32), aspect="auto",
                    cmap=plt.get_cmap("Greys"), vmin=-0.2, vmax=2.2, zorder=1, interpolation="nearest")
        significant = c["p_value"] < 0.05
        ax_c.plot([c["start_ms"], c["end_ms"]], [y, y], color="red" if significant else "#f1a08f",
                  linewidth=7, solid_capstyle="butt", zorder=3)
        ax_c.plot(c["peak_ms"], y, "D", color=plotting.INK, markersize=6, markeredgecolor="white", zorder=4)
        ax_c.text(c["start_ms"] - 10, y, f"p = {c['p_value']:.3f}", va="center", ha="right", fontsize=8,
                  color=plotting.INK_SECONDARY)
    ax_c.set_yticks(range(n_c), [c["label"] for c in clusters], fontsize=9)
    ax_c.set_ylim(n_c - 0.4, -0.8)
    ax_c.set_title("C. Group decoding clusters (light: n.s. trends)",
                   fontsize=11, loc="left", color=plotting.INK)
    ax_c.legend(handles=[Line2D([], [], color="red", linewidth=6, label="cluster p < 0.05"),
                         Line2D([], [], color="#f1a08f", linewidth=6, label=f"trend p < {trend_p:g}"),
                         Line2D([], [], marker="D", color=plotting.INK, linestyle="", label="AUC peak")],
                fontsize=8, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3, frameon=False)
    ax_c.set_xlim(-100, 1000)
    ax_c.set_xlabel("Time from stimulus onset (ms)", fontsize=10)
    fig.tight_layout()
    plotting.save_figure(fig, path)


if __name__ == "__main__":
    main()
