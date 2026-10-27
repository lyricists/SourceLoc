"""Shared plotting pieces: colors, significance bars, and multipage PDFs."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pypdf import PdfReader, PdfWriter

from utils.stats import contiguous_runs

# Reference categorical slots 1-3 and neutral inks.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_SECONDARY, NEUTRAL, NEUTRAL_LIGHT = "#0b0b0b", "#52514e", "#8a8984", "#d6d5d0"


def plot_significant_intervals(axis: plt.Axes, times: np.ndarray, mask: np.ndarray, y_position: float,
                               color: str = "red", linewidth: float = 4) -> None:
    """Horizontal bars over every contiguous significant run."""
    for run in contiguous_runs(mask):
        axis.plot(times[run], np.full(run.size, y_position), color=color, linewidth=linewidth,
                  solid_capstyle="butt", zorder=5, label="_nolegend_")


def decoding_panels(panels: list[dict], times: np.ndarray, ylabel: str, chance: float, chance_label: str,
                    default_ylim: tuple[float, float], step: float, bar_offset: float,
                    suptitle: str | None = None) -> plt.Figure:
    """Large-format decoding figure (the decodingPlot layout): one panel per entry of ``panels``.

    Each panel dict has ``title``, ``mean`` and ``spread`` (time vectors; shaded mean ± spread),
    and ``mask`` (significant time points, drawn as red bars). The y-range covers
    ``default_ylim`` and all data, rounded outward to ``step``.
    """
    from matplotlib.lines import Line2D
    lower = min(default_ylim[0], float(np.floor((min(p["mean"].min() - p["spread"].max() for p in panels) - step)
                                                / step) * step))
    upper = max(default_ylim[1], float(np.ceil((max(p["mean"].max() + p["spread"].max() for p in panels) + step)
                                               / step) * step))
    fig, axes = plt.subplots(1, len(panels), figsize=(40 / 3 * len(panels), 10), squeeze=False)
    fig.subplots_adjust(wspace=0.28)
    for axis, p in zip(axes[0], panels):
        handles = [axis.plot(times, p["mean"], linewidth=4, label=p.get("line_label", "Decoding"))[0],
                   axis.plot(times, np.full(times.size, chance), linewidth=4, linestyle="--", color="k",
                             label=chance_label)[0]]
        axis.fill_between(times, p["mean"] - p["spread"], p["mean"] + p["spread"], alpha=0.4, linewidth=4,
                          label="_nolegend_")
        plot_significant_intervals(axis, times, p["mask"], y_position=lower + bar_offset, linewidth=6)
        axis.set_xlabel("Time from stimulus onset (ms)", fontsize=30, labelpad=15)
        axis.set_title(p["title"], fontsize=30, fontweight="bold", pad=25)
        if p["mask"].any():
            handles.append(Line2D([], [], color="red", linewidth=6, label="p < 0.05"))
        axis.legend(handles=handles, fontsize=20)
        axis.tick_params(axis="both", labelsize=30, pad=10)
        axis.set_ylim(lower, upper)
        # Extend to the end of the last sample (996 ms + 4 ms = 1000 ms) so the epoch end is labeled.
        axis.set_xlim(times[0], times[-1] + (times[1] - times[0]))
        axis.set_xticks(np.arange(np.ceil(times[0] / 200) * 200, times[-1] + (times[1] - times[0]) + 1, 200))
    axes[0, 0].set_ylabel(ylabel, fontsize=30, labelpad=15)
    if suptitle:
        fig.suptitle(suptitle, fontsize=22, y=1.02)
    return fig


class MultipagePdf:
    """Collect figures as separate one-page PDFs, then merge them atomically.

    Saving pages independently avoids a Matplotlib PdfPages resource-reuse bug seen
    with data-dense pages.
    """

    def __init__(self, destination: Path) -> None:
        self.destination = Path(destination)
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        scratch = self.destination.parent / ".pdf_scratch"
        scratch.mkdir(exist_ok=True)
        self._dir = tempfile.TemporaryDirectory(prefix="pages_", dir=scratch)
        self.pages: list[Path] = []

    def add(self, figure: plt.Figure) -> None:
        page = Path(self._dir.name) / f"page_{len(self.pages):03d}.pdf"
        figure.savefig(page, format="pdf")
        plt.close(figure)
        self.pages.append(page)

    def close(self) -> None:
        writer = PdfWriter()
        for page in self.pages:
            reader = PdfReader(page)
            if len(reader.pages) != 1:
                raise ValueError(f"Expected one page in {page}.")
            writer.add_page(reader.pages[0])
        tmp = self.destination.with_suffix(".tmp.pdf")
        with open(tmp, "wb") as f:
            writer.write(f)
        os.replace(tmp, self.destination)
        self._dir.cleanup()

    def __enter__(self) -> "MultipagePdf":
        return self

    def __exit__(self, exc_type, *_) -> None:
        if exc_type is None:
            self.close()
        else:
            self._dir.cleanup()


def save_figure(figure: plt.Figure, destination: Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(".tmp" + destination.suffix)
    figure.savefig(tmp, format=destination.suffix.lstrip("."), bbox_inches="tight")
    plt.close(figure)
    os.replace(tmp, destination)
