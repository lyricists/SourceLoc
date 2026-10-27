"""Plot sensor ERPs for congruent vs. incongruent intention trials by group.

Each subject's ERP is the mean of its trials in a condition; each group's ERP
is the equal-weight mean of its subjects' ERPs, shaded by the across-subject
standard error. Rows are channels, columns are Control, Depressed, Suicidal.

By default each subject's EEG is z-scored per channel with one prestimulus
baseline mean and SD pooled over the baseline samples of both conditions'
trials, so both conditions share the same scaling and subjects with very
different raw amplitudes contribute equally. '--normalization none' plots µV.

"""

from __future__ import annotations

import argparse
import io
import os
import pickle
import pickletools
import struct
from pathlib import Path

import mat73
import matplotlib
import numpy as np
from scipy import io as sio

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent  # SourceLoc; this script lives in Exp/
DATA = ROOT.parent.parent / "Data"
PREPROCESSED = DATA / "Preprocessed data"
BEHAVIOR = DATA / "Behavior"

VERSION = 2
CONDITIONS = ("congruent", "incongruent")
CONDITION_COLORS = {"congruent": "#1f5fbf", "incongruent": "#d1342f"}
GROUPS = {"c": (1, "Control"), "d": (2, "Depressed"), "s": (3, "Suicidal")}
VOLTS_TO_MICROVOLTS = 1e6
NORMALIZATIONS = {"baseline-zscore": ("zscore", "z (baseline SD)"), "none": ("uV", "µV")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eeg", type=Path, default=PREPROCESSED / "Data_sen_bl.pkl")
    parser.add_argument(
        "--indices",
        type=Path,
        default=BEHAVIOR / "senIdx_congruency_Intention.pkl",
        help="'positive' trials are congruent, 'negative' trials incongruent.",
    )
    parser.add_argument(
        "--subject-index", type=Path, default=BEHAVIOR / "subject_index.mat"
    )
    parser.add_argument(
        "--channel-names", type=Path, default=PREPROCESSED / "chName.mat"
    )
    parser.add_argument(
        "--good-channels",
        type=Path,
        default=BEHAVIOR / "GoodChannel.mat",
        help="1-based MATLAB channel indices to plot as rows.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "Results" / "ERP_congruency_intention"
    )
    parser.add_argument("--groups", nargs="+", choices=list(GROUPS), default=list(GROUPS))
    parser.add_argument(
        "--normalization", choices=list(NORMALIZATIONS), default="baseline-zscore"
    )
    parser.add_argument("--sampling-rate", type=float, default=250.0)
    parser.add_argument("--epoch-start-ms", type=float, default=-200.0)
    parser.add_argument("--baseline-start-ms", type=float, default=-200.0)
    parser.add_argument("--baseline-end-ms", type=float, default=-4.0)
    parser.add_argument("--row-height", type=float, default=1.15, help="Inches per row.")
    parser.add_argument(
        "--rebuild-figure",
        action="store_true",
        help="Redraw the PDF even if it exists (it is always redrawn after recomputing ERPs).",
    )
    return parser.parse_args()


def file_signature(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def load_pickle(path: Path):
    with path.open("rb") as file:
        return pickle.load(file)


def load_mat(path: Path) -> dict:
    try:
        return sio.loadmat(str(path), squeeze_me=True)
    except NotImplementedError:
        return mat73.loadmat(str(path))


def atomic_pickle_dump(value: object, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as file:
        pickle.dump(value, file, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, destination)


def memmap_pickled_array(path: Path) -> np.ndarray | None:
    """Memory-map a pickled ndarray's raw buffer, or return None if unsupported.

    A protocol-4+ ndarray pickle stores its shape, dtype, and Fortran flag in a
    short header followed by one BINBYTES8 payload; mapping that payload avoids
    holding the whole array in memory.
    """
    with path.open("rb") as file:
        head = file.read(4096)
    ints, strings, booleans = [], [], []
    shape = None
    try:
        for opcode, argument, _ in pickletools.genops(io.BytesIO(head)):
            if opcode.name.startswith("BININT"):
                ints.append(argument)
            elif opcode.name in ("TUPLE", "TUPLE2", "TUPLE3") and shape is None:
                # TUPLE closes a MARK (ints reset there); TUPLE2/3 take the last items.
                count = {"TUPLE2": 2, "TUPLE3": 3}.get(opcode.name, len(ints))
                if count >= 2 and len(ints) >= count:
                    shape = tuple(ints[-count:])
            elif opcode.name == "MARK":
                ints = []
            elif "UNICODE" in opcode.name:
                strings.append(argument)
            elif opcode.name in ("NEWTRUE", "NEWFALSE"):
                booleans.append(opcode.name == "NEWTRUE")
    except ValueError:
        pass  # genops stops at the payload, which lies beyond the header bytes.
    if shape is None or not booleans or not {"numpy", "ndarray"} <= set(strings):
        return None
    dtype_codes = [s for s in strings if len(s) in (2, 3) and s[0] in "fiuc" and s[1:].isdigit()]
    if not dtype_codes or "<" not in strings:
        return None
    dtype = np.dtype("<" + dtype_codes[0])
    n_bytes = int(np.prod(shape)) * dtype.itemsize
    # BINBYTES8, BINBYTES, and SHORT_BINBYTES payload openers.
    markers = [b"\x8e" + struct.pack("<Q", n_bytes)]
    if n_bytes < 2**32:
        markers.append(b"B" + struct.pack("<I", n_bytes))
    if n_bytes < 256:
        markers.append(b"C" + struct.pack("<B", n_bytes))
    found = [(head.find(marker), marker) for marker in markers if head.find(marker) >= 0]
    if not found:
        return None
    position, marker = min(found)
    offset = position + len(marker)
    if path.stat().st_size < offset + n_bytes:
        return None
    order = "F" if booleans[-1] else "C"
    return np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=shape, order=order)


def load_eeg(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path, mmap_mode="r")
    if path.suffix != ".pkl":
        raise ValueError("EEG must be a .pkl or .npy array.")
    mapped = memmap_pickled_array(path)
    if mapped is not None:
        return mapped
    print("Could not memory-map the EEG pickle; loading it fully.", flush=True)
    return np.asarray(load_pickle(path))


def read_trial_indices(path: Path, n_subjects: int, n_trials: int) -> list[tuple]:
    entries = load_pickle(path)["Sentiment"]
    if len(entries) != n_subjects:
        raise ValueError("Trial-index file does not match the EEG subject count.")
    result = []
    for subject, entry in enumerate(entries):
        indices = []
        for name, key in zip(CONDITIONS, ("positive", "negative")):
            trial_indices = np.asarray(entry[key], dtype=np.intp).reshape(-1)
            if trial_indices.size == 0:
                raise ValueError(f"Subject {subject}: {name} has no trials.")
            if np.any((trial_indices < 0) | (trial_indices >= n_trials)):
                raise IndexError(f"Subject {subject}: {name} has an out-of-range trial index.")
            if np.unique(trial_indices).size != trial_indices.size:
                raise ValueError(f"Subject {subject}: {name} contains duplicate trials.")
            indices.append(trial_indices)
        if np.intersect1d(*indices).size:
            raise ValueError(f"Subject {subject}: congruent and incongruent trials overlap.")
        result.append(tuple(indices))
    return result


def compute_subject_erps(
    eeg: np.ndarray,
    trial_indices: list[tuple],
    normalization: str,
    baseline_indices: np.ndarray,
) -> np.ndarray:
    """Return subject x condition x channel x time trial-mean ERPs in plot units."""
    n_channels, n_times, _, n_subjects = eeg.shape
    erps = np.empty((n_subjects, len(CONDITIONS), n_channels, n_times), dtype=np.float64)
    for subject in range(n_subjects):
        data = np.asarray(eeg[:, :, :, subject])
        if normalization == "baseline-zscore":
            trials = np.concatenate(trial_indices[subject])
            baseline = data[:, baseline_indices][:, :, trials]
            center = baseline.mean(axis=(1, 2))[:, None]
            scale = baseline.std(axis=(1, 2))[:, None]
            if np.any(scale <= 0):
                raise ValueError(f"Subject {subject}: a channel has zero baseline variance.")
        else:
            center, scale = 0.0, 1 / VOLTS_TO_MICROVOLTS
        for condition, indices in enumerate(trial_indices[subject]):
            # Averaging before scaling equals scaling each trial with shared parameters.
            erps[subject, condition] = (data[:, :, indices].mean(axis=2) - center) / scale
        if (subject + 1) % 10 == 0 or subject + 1 == n_subjects:
            print(f"  {subject + 1}/{n_subjects} subjects", flush=True)
    return erps


def load_or_compute_erps(
    args: argparse.Namespace, output_dir: Path
) -> tuple[np.ndarray, bool]:
    signature = {
        "version": VERSION,
        "eeg": file_signature(args.eeg),
        "indices": file_signature(args.indices),
        "normalization": args.normalization,
        "sampling_rate": args.sampling_rate,
        "epoch_start_ms": args.epoch_start_ms,
        "baseline_ms": (args.baseline_start_ms, args.baseline_end_ms),
    }
    cache_path = output_dir / f"subject_erps_{NORMALIZATIONS[args.normalization][0]}.pkl"
    if cache_path.exists():
        cached = load_pickle(cache_path)
        if cached.get("signature") == signature:
            print(f"Reusing subject ERPs from {cache_path}", flush=True)
            return cached["erps"], False
    print("Loading EEG...", flush=True)
    eeg = load_eeg(args.eeg)
    if eeg.ndim != 4:
        raise ValueError("EEG must have channel x time x trial x subject axes.")
    print(f"EEG {eeg.shape}, {eeg.dtype}; averaging trials per subject...", flush=True)
    trial_indices = read_trial_indices(args.indices, eeg.shape[3], eeg.shape[2])
    times_ms = make_times_ms(args, eeg.shape[1])
    baseline_indices = np.flatnonzero(
        (times_ms >= args.baseline_start_ms - 1e-9) & (times_ms <= args.baseline_end_ms + 1e-9)
    )
    if args.normalization == "baseline-zscore" and (
        baseline_indices.size < 2
        or args.baseline_start_ms < times_ms[0] - 1e-9
        or args.baseline_end_ms > -1e-9
    ):
        raise ValueError("Baseline window must be prestimulus, inside the epoch, and span 2+ samples.")
    erps = compute_subject_erps(eeg, trial_indices, args.normalization, baseline_indices)
    atomic_pickle_dump(
        {
            "signature": signature,
            "erps": erps,
            "axes": f"subject x condition x channel x time ({NORMALIZATIONS[args.normalization][1]})",
            "conditions": CONDITIONS,
        },
        cache_path,
    )
    return erps, True


def make_times_ms(args: argparse.Namespace, n_times: int) -> np.ndarray:
    return args.epoch_start_ms + np.arange(n_times) * 1000 / args.sampling_rate


def plot_erps(
    erps: np.ndarray,
    groups: np.ndarray,
    group_keys: list[str],
    channel_indices: np.ndarray,
    channel_names: list[str],
    times_ms: np.ndarray,
    unit: str,
    title: str,
    row_height: float,
    destination: Path,
) -> None:
    columns = []
    for key in group_keys:
        code, name = GROUPS[key]
        members = erps[groups == code]
        if members.shape[0] < 2:
            raise ValueError(f"Group {name} needs at least two subjects for an SEM.")
        mean = members.mean(axis=0)
        sem = members.std(axis=0, ddof=1) / np.sqrt(members.shape[0])
        columns.append((name, members.shape[0], mean, sem))

    n_rows, n_columns = channel_indices.size, len(columns)
    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(3.4 * n_columns + 0.6, row_height * n_rows + 1.2),
        sharex=True,
        squeeze=False,
    )
    for row, channel in enumerate(channel_indices):
        low = min((mean[:, channel] - sem[:, channel]).min() for _, _, mean, sem in columns)
        high = max((mean[:, channel] + sem[:, channel]).max() for _, _, mean, sem in columns)
        margin = 0.05 * (high - low) or 1.0
        for column, (name, count, mean, sem) in enumerate(columns):
            axis = axes[row, column]
            axis.axhline(0, color="#9a9a9a", linewidth=0.5)
            axis.axvline(0, color="#9a9a9a", linewidth=0.5)
            for condition, label in enumerate(CONDITIONS):
                trace = mean[condition, channel]
                error = sem[condition, channel]
                color = CONDITION_COLORS[label]
                axis.fill_between(times_ms, trace - error, trace + error, color=color, alpha=0.2, linewidth=0)
                axis.plot(times_ms, trace, color=color, linewidth=1.0, label=label.capitalize())
            axis.set_ylim(low - margin, high + margin)
            axis.set_xlim(times_ms[0], times_ms[-1])
            axis.tick_params(labelsize=6, length=2, colors="#444444")
            for spine in ("top", "right"):
                axis.spines[spine].set_visible(False)
            if row == 0:
                axis.set_title(f"{name} (n={count})", fontsize=10)
            if column == 0:
                axis.set_ylabel(f"{channel_names[channel]}\n{unit}", fontsize=8)
            if row == n_rows - 1:
                axis.set_xlabel("Time (ms)", fontsize=8)
            if row == 0:
                axis.tick_params(labeltop=True)

    # Margins are fixed in inches so the header does not grow with the row count.
    height = figure.get_figheight()
    title_height = 0.2 * (title.count("\n") + 1)
    figure.subplots_adjust(
        left=0.9 / figure.get_figwidth(),
        right=1 - 0.15 / figure.get_figwidth(),
        top=1 - (0.85 + title_height) / height,
        bottom=0.45 / height,
        hspace=0.35,
        wspace=0.22,
    )
    figure.suptitle(
        title,
        fontsize=11,
        y=1 - 0.12 / height,
        va="top",
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1 - (0.18 + title_height) / height),
        ncol=2, frameon=False, fontsize=9,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    figure.savefig(temporary, format="pdf")
    plt.close(figure)
    os.replace(temporary, destination)


def main() -> None:
    args = parse_args()
    if args.sampling_rate <= 0 or args.row_height <= 0:
        raise ValueError("sampling-rate and row-height must be positive.")
    group_keys = [key for key in GROUPS if key in args.groups]
    output_dir = args.output_dir

    erps, computed = load_or_compute_erps(args, output_dir)
    n_subjects, _, n_channels, n_times = erps.shape

    groups = np.asarray(load_mat(args.subject_index)["subject_index"]).reshape(-1).astype(int)
    if groups.size != n_subjects:
        raise ValueError("subject_index does not match the EEG subject count.")
    if not np.isin(groups, [code for code, _ in GROUPS.values()]).all():
        raise ValueError("subject_index must contain only group codes 1, 2, and 3.")

    channel_names = [str(name) for name in load_mat(args.channel_names)["chName"]]
    if len(channel_names) != n_channels:
        raise ValueError("chName does not match the EEG channel count.")
    good = np.asarray(load_mat(args.good_channels)["Channel"]).reshape(-1)
    if np.any(good != np.round(good)):
        raise ValueError("GoodChannel must contain integer channel indices.")
    channel_indices = good.astype(int) - 1
    if np.unique(channel_indices).size != channel_indices.size:
        raise ValueError("GoodChannel contains duplicate channels.")
    if np.any((channel_indices < 0) | (channel_indices >= n_channels)):
        raise IndexError("GoodChannel contains an out-of-range channel index.")

    times_ms = make_times_ms(args, n_times)
    suffix, unit = NORMALIZATIONS[args.normalization]
    destination = output_dir / f"ERP_congruent_vs_incongruent_intention_{suffix}.pdf"
    title = "Intention ERPs: congruent vs. incongruent (mean ± SEM across subjects)"
    if args.normalization == "baseline-zscore":
        title += (
            f"\nz-scored per subject and channel by the pooled "
            f"{args.baseline_start_ms:g} to {args.baseline_end_ms:g} ms baseline"
        )
    if destination.exists() and not (computed or args.rebuild_figure):
        print(f"Figure already exists; not rebuilding: {destination}")
        return
    plot_erps(
        erps,
        groups,
        group_keys,
        channel_indices,
        channel_names,
        times_ms,
        unit,
        title,
        args.row_height,
        destination,
    )
    print(f"Saved {destination}")


if __name__ == "__main__":
    main()
