"""Subject-level group classification from signed vertex time series in DKT scouts.

For each outer fold, training subjects determine one X/Y/Z PCA direction per
atlas vertex. The fixed directions turn each subject's three inverse-kernel
rows into one scalar source row. Bootstrapped condition averages feed a compact
spatial CNN + temporal residual CNN + Transformer classifier.

Examples:
    python3 scout_vertex_group_classifier.py --smoke-test
    python3 scout_vertex_group_classifier.py --scout 35 52
    python3 scout_vertex_group_classifier.py --scout 35,52 --representation fusion
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import pickle
import shutil
import tempfile
import time
from fractions import Fraction
from pathlib import Path

import mat73
import numpy as np
import torch
from scipy.signal import resample_poly
from sklearn.model_selection import StratifiedKFold, train_test_split
from threadpoolctl import threadpool_limits
from torch import nn

from scout_group_classifier import (
    BEHAVIOR, DEFAULT_PAIRS, PAIRS, PREPROCESSED, ROOT, choose_device,
    file_signature, save_json, subject_metrics,
)


VERSION = 3
CACHE_VERSION = 2
FS_INPUT = 250
START_MS = -200
BASELINE_END_MS = 0
MODEL_START_MS = 0
DEFAULT_END_MS = 800
REPRESENTATIONS = ("congruent", "incongruent", "contrast", "fusion", "all")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--scout", dest="scouts", nargs="+", help="Zero-based DKT scouts, separated by spaces or commas")
    parser.add_argument("--pair", dest="pairs", nargs="+", choices=PAIRS, default=list(DEFAULT_PAIRS))
    parser.add_argument("--representation", choices=REPRESENTATIONS, default="contrast",
                        help="Single condition/contrast, shared two-condition fusion, or three independent encoders")
    parser.add_argument("--eeg", type=Path, default=PREPROCESSED / "Data_sen_bl.pkl")
    parser.add_argument("--tmatrix", type=Path, default=PREPROCESSED / "tMatrix.pkl")
    parser.add_argument("--atlas", type=Path, default=PREPROCESSED / "DKT_scout.mat")
    parser.add_argument("--subject-index", type=Path, default=BEHAVIOR / "subject_index.mat")
    parser.add_argument("--condition-index", type=Path, default=BEHAVIOR / "senIdx_congruency_Intention.pkl")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "Results" / "scout_vertex_cache_v2")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "Results" / "scout_vertex_classifier_v3")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--end-ms", type=int, default=DEFAULT_END_MS,
                        help="Exclusive right endpoint of the model window; -200 to 0 ms is retained for baseline Z-scoring")
    parser.add_argument("--resample-rate", type=int, default=FS_INPUT,
                        help="Model sampling rate; 250 Hz keeps the original samples")
    parser.add_argument("--avg-num", type=int, default=16,
                        help="Trials per training bootstrap average, sampled with replacement")
    parser.add_argument("--draws", type=int, default=200,
                        help="Training bootstrap averages per subject")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--scratch-dtype", choices=("float16", "float32"), default="float16",
                        help="Temporary normalized pseudotrial storage; float16 halves disk use")
    parser.add_argument("--smoke-test", action="store_true", help="Train tiny synthetic folds without study data")
    args = parser.parse_args()
    if not args.smoke_test and not args.scouts:
        parser.error("--scout is required for a full experiment")
    try:
        args.scouts = [int(part) for token in (args.scouts or ["0"])
                       for part in token.split(",") if part.strip()]
    except ValueError:
        parser.error("--scout needs zero-based integer indices")
    if not args.scouts or min(args.scouts) < 0 or len(args.scouts) != len(set(args.scouts)):
        parser.error("--scout needs distinct nonnegative indices")
    if len(args.pairs) != len(set(args.pairs)):
        parser.error("--pair names must be distinct")
    for name in ("resample_rate", "avg_num", "draws", "epochs", "patience", "batch_size", "threads"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.resample_rate > FS_INPUT or args.end_ms <= MODEL_START_MS:
        parser.error("resample rate must be at most 250 Hz and end-ms must exceed zero")
    if args.lr <= 0 or args.weight_decay < 0:
        parser.error("learning rate must be positive and weight decay nonnegative")
    return args


def cache_signature(args: argparse.Namespace, scout: int) -> dict:
    return {
        "version": CACHE_VERSION, "scout": scout, "end_ms": args.end_ms,
        "resample_rate": args.resample_rate,
        "inputs": {name: file_signature(getattr(args, name)) for name in
                   ("eeg", "tmatrix", "atlas", "subject_index", "condition_index")},
    }


def source_orientation_cov(kernel: np.ndarray, sensor_cov: np.ndarray) -> np.ndarray:
    """Per-vertex 3x3 covariance without materializing XYZ source trials."""
    intermediate = np.einsum("vic,cd->vid", kernel, sensor_cov, optimize=True)
    return np.einsum("vid,vjd->vij", intermediate, kernel, optimize=True)


def prepare_cache(args: argparse.Namespace, targets: dict[int, Path]) -> None:
    atlas = mat73.loadmat(str(args.atlas))["DKT_scout"]
    labels = [str(label).strip() for label in atlas.get("Label", atlas.get("Region"))]
    infos = {}
    for scout, directory in targets.items():
        if scout >= len(labels):
            raise ValueError(f"scout must be between 0 and {len(labels) - 1}")
        vertices = np.asarray(atlas["Vertices"][scout], dtype=np.int64).ravel() - 1
        if not len(vertices) or np.any(vertices < 0) or len(np.unique(vertices)) != len(vertices):
            raise ValueError(f"scout {scout} contains invalid vertex indices")
        rows = (3 * vertices[:, None] + np.arange(3)[None, :]).ravel()
        directory.mkdir(parents=True, exist_ok=True)
        infos[scout] = {"rows": rows, "vertices": len(vertices), "label": labels[scout],
                        "directory": directory}
    print(f"Building vertex caches for {len(infos)} scout(s); loading EEG and inverse kernels...", flush=True)
    with open(args.eeg, "rb") as file:
        eeg = np.asarray(pickle.load(file))
    with open(args.tmatrix, "rb") as file:
        tmatrix = pickle.load(file)
    with open(args.condition_index, "rb") as file:
        conditions = pickle.load(file)["Sentiment"]
    groups = np.asarray(mat73.loadmat(str(args.subject_index))["subject_index"]).ravel().astype(int)
    if eeg.ndim != 4:
        raise ValueError("EEG must have channel x time x trial x subject axes")
    n_channels, n_original_times, n_trials, n_subjects = eeg.shape
    if not (n_subjects == len(groups) == len(conditions) == len(tmatrix["kernel"]) == len(tmatrix["GoodChannel"])):
        raise ValueError("Subject counts differ between EEG, labels, conditions, and kernels")
    old_times = START_MS + np.arange(n_original_times) * 1000 / FS_INPUT
    selected_times = np.flatnonzero(old_times < args.end_ms)
    if len(selected_times) < 25 or selected_times[0] != 0 or old_times[selected_times[-1]] < 0:
        raise ValueError("EEG does not span the requested baseline and response interval")
    ratio = Fraction(args.resample_rate, FS_INPUT)
    n_resampled = (len(selected_times) * ratio.numerator + ratio.denominator - 1) // ratio.denominator
    times = START_MS + np.arange(n_resampled) * 1000 / args.resample_rate
    if np.sum((times >= START_MS) & (times < BASELINE_END_MS)) < 4:
        raise ValueError("Fewer than four baseline time points")
    started = time.time()
    with threadpool_limits(limits=args.threads):
        for subject in range(n_subjects):
            condition_indices = []
            for key in ("positive", "negative"):
                indices = np.asarray(conditions[subject][key], dtype=int).ravel()
                if len(indices) != 40 or len(np.unique(indices)) != 40 or np.any(indices < 0) or np.any(indices >= n_trials):
                    raise ValueError(f"subject {subject}: {key} must have 40 distinct valid trials")
                condition_indices.append(indices)
            if np.intersect1d(*condition_indices).size:
                raise ValueError(f"subject {subject}: condition trial sets overlap")
            trial_ids = np.concatenate(condition_indices)
            good = np.asarray(tmatrix["GoodChannel"][subject], dtype=int).ravel() - 1
            if len(good) == 0 or np.any(good < 0) or np.any(good >= n_channels):
                raise ValueError(f"subject {subject}: bad GoodChannel indices")
            kernel = np.asarray(tmatrix["kernel"][subject], dtype=np.float64)
            subject_eeg = eeg[:, :, :, subject]
            if kernel.shape[1] == len(good):
                sensor = subject_eeg[good][:, selected_times, :][:, :, trial_ids]
            elif kernel.shape[1] == n_channels:
                sensor = subject_eeg[:, selected_times, :][:, :, trial_ids]
            else:
                raise ValueError(f"subject {subject}: inverse kernel columns do not match EEG channels")
            valid = np.isfinite(sensor).all(axis=(0, 1)) & np.any(sensor != 0, axis=(0, 1))
            if valid[:40].sum() < args.avg_num or valid[40:].sum() < args.avg_num:
                raise ValueError(f"subject {subject}: fewer than {args.avg_num} valid trials in one condition")
            sensor = np.asarray(sensor[:, :, valid], dtype=np.float64)
            if args.resample_rate != FS_INPUT:
                sensor = resample_poly(sensor, ratio.numerator, ratio.denominator, axis=1)
            if sensor.shape[1] != len(times):
                raise ValueError("Unexpected time dimension after resampling")
            flat = sensor.reshape(sensor.shape[0], -1)
            flat -= flat.mean(axis=1, keepdims=True)
            sensor_cov = flat @ flat.T / max(flat.shape[1] - 1, 1)
            a = sensor[:, :, :valid[:40].sum()].transpose(2, 0, 1).astype(np.float32)
            b = sensor[:, :, valid[:40].sum():].transpose(2, 0, 1).astype(np.float32)
            for scout, info in infos.items():
                if info["rows"].max() >= kernel.shape[0]:
                    raise ValueError(f"subject {subject}, scout {scout}: vertex index exceeds kernel")
                k = kernel[info["rows"]].reshape(info["vertices"], 3, -1)
                cov = source_orientation_cov(k, sensor_cov)
                cov /= np.maximum(np.trace(cov, axis1=1, axis2=2), np.finfo(float).tiny)[:, None, None]
                target = info["directory"] / f"subject_{subject:03d}.npz"
                temporary = target.with_suffix(".tmp")
                with open(temporary, "wb") as file:
                    np.savez(file, congruent=a, incongruent=b, kernel=k.astype(np.float32),
                             orientation_cov=cov.astype(np.float32))
                os.replace(temporary, target)
            if (subject + 1) % 10 == 0 or subject + 1 == n_subjects:
                print(f"  cached {subject + 1}/{n_subjects} subjects in {time.time() - started:.0f}s", flush=True)
    del eeg, tmatrix
    for scout, info in infos.items():
        save_json(info["directory"] / "metadata.json", {
            "signature": cache_signature(args, scout), "scout_label": info["label"],
            "n_vertices": info["vertices"], "groups": groups.tolist(),
            "times_ms": times.tolist(), "n_subjects": n_subjects,
            "conditions": "congruent,incongruent; positive,negative keys in the intention index",
            "axes": "subject files contain trial x sensor x time and vertex x XYZ x sensor",
        })


def ensure_caches(args: argparse.Namespace) -> dict[int, Path]:
    paths = {scout: args.cache_dir / f"scout_{scout:02d}" for scout in args.scouts}
    missing = {}
    for scout, directory in paths.items():
        expected = cache_signature(args, scout)
        metadata = directory / "metadata.json"
        if not args.rebuild_cache and metadata.exists():
            with open(metadata, encoding="utf-8") as file:
                existing = json.load(file)
            if existing.get("signature") == expected and all(
                (directory / f"subject_{subject:03d}.npz").exists()
                for subject in range(existing.get("n_subjects", 0))
            ):
                print(f"Using vertex cache: {directory}", flush=True)
                continue
        missing[scout] = directory
    if missing:
        prepare_cache(args, missing)
    return paths


def fit_vertex_pca(cache_dir: Path, train_subjects: np.ndarray, n_vertices: int) -> tuple[np.ndarray, np.ndarray]:
    """One common XYZ direction per vertex, learned solely from train subjects."""
    total = np.zeros((n_vertices, 3, 3), dtype=np.float64)
    for subject in train_subjects:
        with np.load(cache_dir / f"subject_{subject:03d}.npz") as data:
            covariance = np.asarray(data["orientation_cov"], dtype=np.float64)
            # Equalize subjects at each vertex so a large inverse-kernel scale
            # does not determine the common orientation by itself.
            scale = np.maximum(np.trace(covariance, axis1=1, axis2=2), np.finfo(float).tiny)
            total += covariance / scale[:, None, None]
    total /= len(train_subjects)
    eigenvalues, eigenvectors = np.linalg.eigh(total)
    weights = eigenvectors[:, :, -1]
    largest = np.argmax(np.abs(weights), axis=1)
    signs = np.sign(weights[np.arange(n_vertices), largest])
    weights *= np.where(signs == 0, 1, signs)[:, None]
    denominator = np.maximum(eigenvalues.sum(axis=1), np.finfo(float).tiny)
    explained = np.clip(eigenvalues[:, -1] / denominator, 0, 1)
    return weights.astype(np.float32), explained.astype(np.float32)


def scalar_source_trials(sensor: np.ndarray, kernel: np.ndarray,
                         weights: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """Project through collapsed kernel and baseline-z-score each original trial."""
    effective = np.einsum("vkc,vk->vc", kernel, weights, optimize=True)
    n_trials, n_channels, n_times = sensor.shape
    source = (effective @ sensor.transpose(1, 0, 2).reshape(n_channels, -1)).reshape(
        len(weights), n_trials, n_times).transpose(1, 0, 2)
    mean = source[:, :, baseline].mean(axis=-1, keepdims=True)
    std = source[:, :, baseline].std(axis=-1, keepdims=True)
    floor = np.maximum(np.median(std, axis=0, keepdims=True) * 1e-4, 1e-30)
    source = (source - mean) / np.maximum(std, floor)
    if not np.isfinite(source).all():
        raise ValueError("Nonfinite source values after baseline normalization")
    return source.astype(np.float32, copy=False)


def bootstrap_chunk(trials: np.ndarray, counts: np.ndarray, avg_num: int) -> np.ndarray:
    return ((counts.astype(np.float32) @ trials.reshape(len(trials), -1)) / avg_num).reshape(
        len(counts), *trials.shape[1:])


def branch_count(representation: str) -> int:
    return 3 if representation == "all" else 2 if representation == "fusion" else 1


def make_examples(cache_dir: Path, output: Path, subjects: np.ndarray, weights: np.ndarray,
                  times: np.ndarray, args: argparse.Namespace,
                  rng: np.random.Generator | None = None, *, full_average: bool = False) -> np.ndarray:
    """Stream bootstrapped or full-condition subject averages to a temporary memmap."""
    model_window = times >= MODEL_START_MS
    n_vertices, n_times = len(weights), int(model_window.sum())
    if n_times < 1:
        raise ValueError("No samples in the model window")
    n_branches = branch_count(args.representation)
    examples_per_subject = 1 if full_average else args.draws
    if not full_average and rng is None:
        raise ValueError("Bootstrap examples require a random generator")
    dtype = np.dtype(args.scratch_dtype)
    required = len(subjects) * examples_per_subject * n_branches * n_vertices * n_times * dtype.itemsize
    free = shutil.disk_usage(output.parent).free
    if required > free * 0.85:
        raise OSError(f"Not enough free disk for {required / 2**30:.1f} GiB of temporary pseudotrials")
    data_out = np.lib.format.open_memmap(
        output, mode="w+", dtype=dtype,
        shape=(len(subjects) * examples_per_subject, n_branches, n_vertices, n_times),
    )
    def store_examples(where: slice, a: np.ndarray, b: np.ndarray) -> None:
        if args.representation == "congruent":
            data_out[where, 0] = a
        elif args.representation == "incongruent":
            data_out[where, 0] = b
        elif args.representation == "contrast":
            data_out[where, 0] = b - a
        else:
            data_out[where, 0] = a
            data_out[where, 1] = b
            if args.representation == "all":
                data_out[where, 2] = b - a

    baseline = (times >= START_MS) & (times < BASELINE_END_MS)
    if baseline.sum() < 4:
        raise ValueError("Fewer than four baseline samples")
    for position, subject in enumerate(subjects):
        with np.load(cache_dir / f"subject_{subject:03d}.npz") as data:
            kernel = data["kernel"]
            congruent = scalar_source_trials(data["congruent"], kernel, weights, baseline)[:, :, model_window]
            incongruent = scalar_source_trials(data["incongruent"], kernel, weights, baseline)[:, :, model_window]
        if min(len(congruent), len(incongruent)) < (1 if full_average else args.avg_num):
            raise ValueError(f"subject {subject}: insufficient valid condition trials")
        if full_average:
            a = congruent.mean(axis=0, keepdims=True, dtype=np.float32)
            b = incongruent.mean(axis=0, keepdims=True, dtype=np.float32)
            store_examples(slice(position, position + 1), a, b)
        else:
            for start in range(0, args.draws, 16):
                stop = min(start + 16, args.draws)
                n = stop - start
                a = bootstrap_chunk(congruent, rng.multinomial(
                    args.avg_num, np.full(len(congruent), 1 / len(congruent)), size=n), args.avg_num)
                b = bootstrap_chunk(incongruent, rng.multinomial(
                    args.avg_num, np.full(len(incongruent), 1 / len(incongruent)), size=n), args.avg_num)
                sl = slice(position * args.draws + start, position * args.draws + stop)
                store_examples(sl, a, b)
        if (position + 1) % 10 == 0 or position + 1 == len(subjects):
            print(f"  {output.stem}: {position + 1}/{len(subjects)} subjects", flush=True)
    data_out.flush()
    del data_out
    return np.load(output, mmap_mode="r")


class ResidualTemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=2,
                      padding=kernel_size // 2, bias=False),
            nn.GroupNorm(8, out_channels), nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2,
                      bias=False), nn.GroupNorm(8, out_channels),
        )
        self.skip = nn.Conv1d(in_channels, out_channels, 1, stride=2, bias=False)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.main(x) + self.skip(x))


class VertexEncoder(nn.Module):
    def __init__(self, n_vertices: int) -> None:
        super().__init__()
        self.spatial = nn.Sequential(nn.Conv1d(n_vertices, 64, 1, bias=False),
                                     nn.GroupNorm(8, 64), nn.GELU(), nn.Dropout(0.15))
        self.temporal = nn.Sequential(ResidualTemporalBlock(64, 96, 7),
                                      ResidualTemporalBlock(96, 128, 5))
        layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=256,
                                           dropout=0.1, batch_first=True, norm_first=False)
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal(self.spatial(x)).transpose(1, 2)
        positions = torch.arange(x.shape[1], device=x.device, dtype=x.dtype)[:, None]
        scales = torch.exp(torch.arange(0, x.shape[2], 2, device=x.device, dtype=x.dtype)
                           * (-np.log(10000.0) / x.shape[2]))
        encoding = torch.zeros((x.shape[1], x.shape[2]), device=x.device, dtype=x.dtype)
        encoding[:, 0::2] = torch.sin(positions * scales)
        encoding[:, 1::2] = torch.cos(positions * scales)
        x = self.transformer(x + encoding[None])
        return x.mean(dim=1)


class ScoutVertexNet(nn.Module):
    def __init__(self, n_vertices: int, representation: str) -> None:
        super().__init__()
        if representation not in REPRESENTATIONS:
            raise ValueError(f"Unknown representation: {representation}")
        self.representation = representation
        self.encoders = nn.ModuleList(
            [VertexEncoder(n_vertices) for _ in range(3 if representation == "all" else 1)]
        )
        head_width = 384 if representation in ("fusion", "all") else 128
        self.head = nn.Sequential(nn.LayerNorm(head_width), nn.Linear(head_width, 64),
                                  nn.GELU(), nn.Dropout(0.25), nn.Linear(64, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.representation == "all":
            features = torch.cat([encoder(x[:, branch]) for branch, encoder in enumerate(self.encoders)], dim=1)
        elif self.representation == "fusion":
            a = self.encoders[0](x[:, 0])
            b = self.encoders[0](x[:, 1])
            features = torch.cat((a, b, b - a), dim=1)
        else:
            features = self.encoders[0](x[:, 0])
        return self.head(features)


@torch.no_grad()
def subject_probabilities(model: nn.Module, examples: np.ndarray, n_subjects: int,
                          examples_per_subject: int, device: torch.device, batch_size: int) -> np.ndarray:
    """Return one probability vector per subject (a direct prediction for one example)."""
    model.eval()
    sums = np.zeros((n_subjects, 2), dtype=np.float64)
    for start in range(0, len(examples), batch_size):
        stop = min(start + batch_size, len(examples))
        batch = torch.as_tensor(np.asarray(examples[start:stop], dtype=np.float32), device=device)
        p = torch.softmax(model(batch), dim=1).cpu().numpy()
        for offset, row in enumerate(p):
            sums[(start + offset) // examples_per_subject] += row
    return sums / examples_per_subject


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def synthetic_cache(directory: Path, n_vertices: int = 12,
                    end_ms: int = DEFAULT_END_MS, sample_rate: int = FS_INPUT) -> dict:
    """Create synthetic sensor data and inverse kernels for an end-to-end smoke test."""
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(991)
    groups = np.repeat([1, 2, 3], 10)
    times = np.arange(START_MS, end_ms, 1000 / sample_rate)
    pulse = np.exp(-0.5 * ((times - 30) / 30) ** 2)
    common_kernel = rng.normal(size=(n_vertices, 3, 8)).astype(np.float32) / 8
    for subject, group in enumerate(groups):
        a = rng.normal(size=(20, 8, len(times))).astype(np.float32)
        b = rng.normal(size=(20, 8, len(times))).astype(np.float32)
        b[:, 0, :] += (group - 1) * pulse[None] * 0.4
        kernel = common_kernel + rng.normal(scale=0.02, size=common_kernel.shape)
        all_sensor = np.concatenate((a, b), axis=0).transpose(1, 0, 2).reshape(8, -1)
        covariance = np.cov(all_sensor)
        np.savez(directory / f"subject_{subject:03d}.npz", congruent=a, incongruent=b,
                 kernel=kernel.astype(np.float32),
                 orientation_cov=source_orientation_cov(kernel, covariance).astype(np.float32))
    meta = {"signature": {"synthetic": True, "version": VERSION}, "scout_label": "synthetic scout",
            "n_vertices": n_vertices, "groups": groups.tolist(), "times_ms": times.tolist(),
            "n_subjects": len(groups)}
    save_json(directory / "metadata.json", meta)
    return meta


def select_pair(groups: np.ndarray, pair: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first, second = PAIRS[pair]
    selected = np.flatnonzero(np.isin(groups, first + second))
    originals = groups[selected]
    labels = np.isin(originals, second).astype(np.int64)
    if any(np.sum(originals == code) < 5 for code in first + second):
        raise ValueError(f"each original group needs at least five subjects for 5-fold CV ({pair})")
    return selected, labels, originals


def train_fold(args: argparse.Namespace, cache_dir: Path, meta: dict, selected: np.ndarray,
               labels: np.ndarray, train: np.ndarray, val: np.ndarray, test: np.ndarray,
               fold: int, output: Path, device: torch.device) -> tuple[np.ndarray, dict]:
    train_subjects = selected[train]
    weights, explained = fit_vertex_pca(cache_dir, train_subjects, meta["n_vertices"])
    np.save(output / f"fold_{fold:02d}_xyz_weights.npy", weights)
    np.save(output / f"fold_{fold:02d}_xyz_explained.npy", explained)
    print(f"fold {fold:02d}: XYZ PCA median retained variance={np.median(explained):.3f}; "
          f"10th percentile={np.percentile(explained, 10):.3f}", flush=True)
    times = np.asarray(meta["times_ms"], dtype=np.float32)
    torch.manual_seed(args.seed + fold)
    model = ScoutVertexNet(meta["n_vertices"], args.representation).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    counts = np.bincount(labels[train], minlength=2)
    class_weights = len(train) / (2 * counts)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=device, dtype=torch.float32))
    best_loss, best_epoch, history = np.inf, 0, []
    checkpoint = output / f"fold_{fold:02d}_best.pt"
    with tempfile.TemporaryDirectory(prefix=f"fold_{fold:02d}_scratch_", dir=output) as scratch:
        scratch = Path(scratch)
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, fold, 0]))
        x_train = make_examples(cache_dir, scratch / "train.npy", selected[train],
                                weights, times, args, rng)
        x_val = make_examples(cache_dir, scratch / "val.npy", selected[val],
                              weights, times, args, full_average=True)
        x_test = make_examples(cache_dir, scratch / "test.npy", selected[test],
                               weights, times, args, full_average=True)
        y_train = np.repeat(labels[train], args.draws)
        for epoch in range(1, args.epochs + 1):
            started = time.time()
            model.train()
            order = np.random.default_rng(np.random.SeedSequence([args.seed, fold, epoch])).permutation(len(x_train))
            running = 0.0
            train_total_by_class = torch.zeros(2, device=device)
            train_correct_by_class = torch.zeros(2, device=device)
            for start in range(0, len(order), args.batch_size):
                indices = order[start:start + args.batch_size]
                batch = torch.as_tensor(np.asarray(x_train[indices], dtype=np.float32), device=device)
                target = torch.as_tensor(y_train[indices], dtype=torch.long, device=device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch)
                loss = criterion(logits, target)
                loss.backward()
                optimizer.step()
                running += float(loss.detach().cpu()) * len(indices)
                predicted = logits.detach().argmax(dim=1)
                train_total_by_class += torch.stack(((target == 0).sum(), (target == 1).sum()))
                train_correct_by_class += torch.stack((
                    ((target == 0) & (predicted == 0)).sum(),
                    ((target == 1) & (predicted == 1)).sum(),
                ))
            totals = train_total_by_class.cpu().numpy()
            correct = train_correct_by_class.cpu().numpy()
            train_accuracy = float(correct.sum() / totals.sum())
            train_balanced_accuracy = float(np.mean(correct / totals))
            p_val = subject_probabilities(model, x_val, len(val), 1, device, args.batch_size)
            metrics = subject_metrics(labels[val], p_val)
            row = {"fold": fold, "epoch": epoch, "train_loss": running / len(x_train),
                   "train_pseudotrial_accuracy": train_accuracy,
                   "train_pseudotrial_balanced_accuracy": train_balanced_accuracy,
                   "val_nll": metrics["nll"], "val_auc": metrics["auc"],
                   "val_balanced_accuracy": metrics["balanced_accuracy"],
                   "seconds": time.time() - started}
            history.append(row)
            print(f"fold {fold:02d}/05 epoch {epoch:03d} train_loss={row['train_loss']:.4f} "
                  f"train_pseudo_acc={train_accuracy:.3f} train_pseudo_bal_acc={train_balanced_accuracy:.3f} "
                  f"val_nll={row['val_nll']:.4f} val_auc={row['val_auc']:.3f} "
                  f"val_bal_acc={row['val_balanced_accuracy']:.3f} "
                  f"time={row['seconds']:.1f}s", flush=True)
            if metrics["nll"] < best_loss - 1e-4:
                best_loss, best_epoch = metrics["nll"], epoch
                torch.save(model.state_dict(), checkpoint)
            if epoch - best_epoch >= args.patience:
                print(f"fold {fold:02d}: early stop at epoch {epoch}; best={best_epoch}", flush=True)
                break
        save_csv(output / f"fold_{fold:02d}_history.csv", history)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        p_test = subject_probabilities(model, x_test, len(test), 1, device, args.batch_size)
        del x_train, x_val, x_test
    return p_test, {"best_epoch": best_epoch, "best_val_nll": best_loss,
                    "median_xyz_variance": float(np.median(explained))}


def run_one(args: argparse.Namespace, scout: int, pair: str, cache_dir: Path,
            meta: dict, device: torch.device, smoke: bool) -> dict:
    groups = np.asarray(meta["groups"], dtype=np.int64)
    selected, labels, originals = select_pair(groups, pair)
    output = args.output_dir / f"scout_{scout:02d}" / pair / args.representation
    output.mkdir(parents=True, exist_ok=True)
    print(f"Scout {scout}: {meta['scout_label']} ({meta['n_vertices']} vertices) | "
          f"pair {pair} | subjects {np.bincount(labels).tolist()} | device {device}", flush=True)
    print(f"Parameters: {sum(p.numel() for p in ScoutVertexNet(meta['n_vertices'], args.representation).parameters()):,}", flush=True)
    config = {"version": VERSION, "cache_signature": meta["signature"], "scout": scout,
              "pair": pair, "representation": args.representation, "seed": args.seed,
              "baseline_start_ms": START_MS, "baseline_end_ms": BASELINE_END_MS,
              "model_start_ms": MODEL_START_MS, "model_end_ms": args.end_ms,
              "validation_test_averaging": "all_valid_trials_once_per_condition",
              "avg_num": args.avg_num, "draws": args.draws, "epochs": args.epochs,
              "patience": args.patience, "batch_size": args.batch_size, "lr": args.lr,
              "weight_decay": args.weight_decay, "scratch_dtype": args.scratch_dtype}
    config_path = output / "run_config.json"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as file:
            if json.load(file) != config:
                raise ValueError(f"Output {output} belongs to a different configuration; choose another --output-dir")
    else:
        save_json(config_path, config)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    predictions, fold_rows = [], []
    for fold, (trainval, test) in enumerate(cv.split(selected, originals), start=1):
        if smoke and fold > 1:
            break
        train, val = train_test_split(trainval, test_size=0.125,
                                      stratify=originals[trainval], random_state=args.seed + fold)
        result_path = output / f"fold_{fold:02d}_result.json"
        if args.resume and result_path.exists():
            with open(result_path, encoding="utf-8") as file:
                saved = json.load(file)
            if saved["test_subjects"] != selected[test].tolist():
                raise ValueError(f"fold {fold}: saved test subjects do not match")
            p_test = np.asarray(saved["probabilities"], dtype=np.float64)
            details = saved["details"]
            print(f"fold {fold:02d}: reused completed fold", flush=True)
        else:
            print(f"fold {fold:02d}: train={len(train)} val={len(val)} test={len(test)}; "
                  "fitting training-subject vertex orientations", flush=True)
            p_test, details = train_fold(args, cache_dir, meta, selected, labels,
                                         train, val, test, fold, output, device)
            save_json(result_path, {"test_subjects": selected[test].tolist(),
                                    "probabilities": p_test.tolist(), "details": details})
        metrics = subject_metrics(labels[test], p_test)
        print(f"fold {fold:02d} TEST: AUC={metrics['auc']:.3f} "
              f"bal_acc={metrics['balanced_accuracy']:.3f}", flush=True)
        fold_rows.append({"fold": fold, "best_epoch": details["best_epoch"],
                          "best_val_nll": details["best_val_nll"], **metrics})
        for position, local_subject in enumerate(test):
            predictions.append({"fold": fold, "subject_index_0based": int(selected[local_subject]),
                                "true_group_code": int(originals[local_subject]),
                                "true_binary_label": int(labels[local_subject]),
                                "probability_second_group": float(p_test[position, 1])})
        save_csv(output / "fold_metrics.csv", fold_rows)
        save_csv(output / "subject_predictions.csv", predictions)
    if len(predictions) == len(selected):
        truth = np.asarray([row["true_binary_label"] for row in predictions])
        p = np.asarray([row["probability_second_group"] for row in predictions])
        summary = subject_metrics(truth, np.column_stack((1 - p, p)))
        print(f"OUT-OF-FOLD SUBJECTS: AUC={summary['auc']:.3f} "
              f"balanced_accuracy={summary['balanced_accuracy']:.3f}", flush=True)
    else:
        summary = {"smoke_test": True, "tested_subjects": len(predictions)}
    save_json(output / "summary.json", {"config": config, "summary": summary})
    return {"scout": scout, "scout_label": meta["scout_label"], "n_vertices": meta["n_vertices"],
            "pair": pair, "representation": args.representation,
            "n_subjects": len(selected), "auc": summary.get("auc"),
            "balanced_accuracy": summary.get("balanced_accuracy")}


def run(args: argparse.Namespace) -> None:
    device = choose_device(args.device)
    torch.set_num_threads(args.threads)
    print(f"Sampling {args.resample_rate} Hz | baseline {START_MS} to <{BASELINE_END_MS} ms "
          f"| model input {MODEL_START_MS} to <{args.end_ms} ms", flush=True)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    if args.smoke_test:
        args.epochs = min(args.epochs, 1)
        args.draws = min(args.draws, 4)
        args.avg_num = min(args.avg_num, 3)
        args.batch_size = min(args.batch_size, 8)
        args.output_dir = Path(tempfile.mkdtemp(prefix="scout_vertex_smoke_"))
        cache_paths = {}
        for scout in args.scouts:
            cache_paths[scout] = args.output_dir / "cache" / f"scout_{scout:02d}"
            synthetic_cache(cache_paths[scout], end_ms=args.end_ms, sample_rate=args.resample_rate)
        print(f"Synthetic smoke output: {args.output_dir}", flush=True)
    else:
        cache_paths = ensure_caches(args)
    print(f"Training: {args.draws} bootstrap averages of {args.avg_num} trials per subject; "
          "validation/test: one average of all valid trials per condition per subject", flush=True)
    all_rows = []
    with threadpool_limits(limits=args.threads):
        for position, pair in enumerate(args.pairs, start=1):
            print(f"\n=== Comparison {position}/{len(args.pairs)}: {pair} ===", flush=True)
            rows = []
            for scout in args.scouts:
                with open(cache_paths[scout] / "metadata.json", encoding="utf-8") as file:
                    meta = json.load(file)
                rows.append(run_one(args, scout, pair, cache_paths[scout], meta, device, args.smoke_test))
                gc.collect()
            path = args.output_dir / f"{pair}_{args.representation}_scouts.csv"
            save_csv(path, rows)
            print(f"Comparison summary: {path}", flush=True)
            all_rows.extend(rows)
    if len(args.pairs) > 1:
        path = args.output_dir / f"{args.representation}_comparisons.csv"
        save_csv(path, all_rows)
        print(f"Combined summary: {path}", flush=True)


if __name__ == "__main__":
    run(parse_args())
