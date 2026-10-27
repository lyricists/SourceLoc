"""Binary group classification from one scout's intention responses.

The expensive source projection/PCA is cached once per scout. Each subject has
its own PCA basis; every outer fold builds the shared alignment reference from
training subjects only. Three compact temporal encoders process congruent,
incongruent, and their contrast before late fusion. All splits and metrics are
at the subject level.

Run ``python3 scout_group_classifier.py --smoke-test`` for a synthetic check.
Run ``python3 scout_group_classifier.py --scout 0 1 2`` for the default
control-vs-depressed+suicidal and depressed-vs-suicidal comparisons (zero-based
scout indices). The full experiment reads the
large sensor EEG and inverse-kernel pickles once to build the requested,
uncached scout caches.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import pickle
import random
import tempfile
import time
from pathlib import Path

import mat73
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from torch import nn


ROOT = Path(__file__).resolve().parent.parent  # SourceLoc; this script lives in Exp/
DATA = ROOT.parent.parent / "Data"
PREPROCESSED = DATA / "Preprocessed data"
BEHAVIOR = DATA / "Behavior"
GROUPS = {"control": 1, "depressed": 2, "suicidal": 3}
PAIRS = {
    "control-vs-depressed+suicidal": ((1,), (2, 3)),
    "depressed-suicidal": ((2,), (3,)),
    "control-depressed": ((1,), (2,)),
    "control-suicidal": ((1,), (3,)),
}
DEFAULT_PAIRS = ("control-vs-depressed+suicidal", "depressed-suicidal")
FS = 250
START_MS = -200
END_MS = 1000
BASELINE_START_MS = -200
BASELINE_END_MS = 0
CACHE_VERSION = 2
MODEL_VERSION = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scout", dest="scouts", nargs="+",
        help="One or more zero-based DKT scout indices, e.g. --scout 0 1 2",
    )
    parser.add_argument(
        "--pair", dest="pairs", nargs="+", choices=PAIRS,
        default=list(DEFAULT_PAIRS),
        help="Comparisons to run in order; default: control vs pooled patients, then depressed vs suicidal",
    )
    parser.add_argument(
        "--branches", choices=("all", "conditions"), default="all",
        help="Use all three encoders, or only congruent and incongruent for an ablation",
    )
    parser.add_argument("--eeg", type=Path, default=PREPROCESSED / "Data_sen_bl.pkl")
    parser.add_argument("--tmatrix", type=Path, default=PREPROCESSED / "tMatrix.pkl")
    parser.add_argument("--atlas", type=Path, default=PREPROCESSED / "DKT_scout.mat")
    parser.add_argument("--subject-index", type=Path, default=BEHAVIOR / "subject_index.mat")
    parser.add_argument(
        "--condition-index", type=Path,
        default=BEHAVIOR / "senIdx_congruency_Intention.pkl",
    )
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "Results" / "scout_group_cache")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True,
        help="Reuse completed folds from a matching run",
    )
    parser.add_argument("--avg-num", type=int, default=12)
    parser.add_argument("--train-draws", type=int, default=100)
    parser.add_argument("--eval-draws", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if not args.smoke_test and args.scouts is None:
        parser.error("--scout is required for a full experiment")
    try:
        args.scouts = [
            int(item) for token in (args.scouts or ["0"])
            for item in token.split(",") if item.strip()
        ]
    except ValueError:
        parser.error("--scout accepts integer indices, separated by spaces or commas")
    if not args.scouts or len(set(args.scouts)) != len(args.scouts) or min(args.scouts) < 0:
        parser.error("--scout needs unique, nonnegative indices")
    if len(set(args.pairs)) != len(args.pairs):
        parser.error("--pair needs unique comparison names")
    for name in ("avg_num", "train_draws", "eval_draws", "epochs", "patience", "batch_size", "threads"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.lr <= 0 or args.weight_decay < 0:
        parser.error("learning rate must be positive and weight decay nonnegative")
    return args


def file_signature(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def cache_signature(args: argparse.Namespace, scout: int) -> dict:
    return {
        "version": CACHE_VERSION,
        "scout": scout,
        "fs": FS,
        "start_ms": START_MS,
        "end_ms": END_MS,
        "inputs": {
            name: file_signature(getattr(args, name))
            for name in ("eeg", "tmatrix", "atlas", "subject_index", "condition_index")
        },
    }


def load_cache_files(directory: Path) -> dict:
    with open(directory / "metadata.json", encoding="utf-8") as file:
        meta = json.load(file)
    return {
        "scores": np.load(directory / "scores.npy", mmap_mode="r"),
        "components": np.load(directory / "components.npy", mmap_mode="r"),
        "explained_variance": np.load(directory / "explained_variance.npy", mmap_mode="r"),
        "valid": np.load(directory / "valid.npy", mmap_mode="r"),
        "groups": np.asarray(meta["groups"], dtype=np.int64),
        "times": np.asarray(meta["times_ms"], dtype=np.float32),
        "label": meta["scout_label"],
        "meta": meta,
    }


def make_scout_caches(
    args: argparse.Namespace, targets: dict[int, tuple[Path, dict]]
) -> None:
    """Build several scout caches in one EEG read and one subject pass."""
    atlas = mat73.loadmat(str(args.atlas))["DKT_scout"]
    labels = [str(x).strip() for x in atlas.get("Label", atlas.get("Region"))]
    scout_infos = []
    for scout, (directory, signature) in targets.items():
        if scout < 0 or scout >= len(labels):
            raise ValueError(f"scout must be between 0 and {len(labels) - 1}")
        vertices = np.asarray(atlas["Vertices"][scout], dtype=np.int64).ravel() - 1
        if vertices.size == 0 or np.any(vertices < 0):
            raise ValueError(f"scout {scout} has invalid vertices")
        rows = (3 * vertices[:, None] + np.arange(3)[None, :]).ravel()
        scout_infos.append({
            "scout": scout, "directory": directory, "signature": signature,
            "label": labels[scout], "rows": rows,
        })
    print(
        f"Building {len(scout_infos)} scout cache(s); loading EEG and inverse kernels...",
        flush=True,
    )
    with open(args.eeg, "rb") as file:
        eeg = np.asarray(pickle.load(file))
    with open(args.tmatrix, "rb") as file:
        tmatrix = pickle.load(file)
    groups = np.asarray(mat73.loadmat(str(args.subject_index))["subject_index"]).ravel().astype(int)
    with open(args.condition_index, "rb") as file:
        conditions = pickle.load(file)["Sentiment"]
    if eeg.ndim != 4:
        raise ValueError("EEG must be channel x time x trial x subject")
    n_channels, n_original_times, n_trials, n_subjects = eeg.shape
    if not (len(tmatrix["kernel"]) == len(tmatrix["GoodChannel"]) == len(groups) == len(conditions) == n_subjects):
        raise ValueError("subject counts differ across EEG, kernels, groups, or conditions")
    times = START_MS + np.arange(n_original_times) * (1000 / FS)
    time_indices = np.flatnonzero(times <= END_MS)
    times = times[time_indices]
    if times.size < 25 or times[0] != START_MS or times[-1] <= 0:
        raise ValueError("EEG does not cover the requested epoch and baseline")
    for info in scout_infos:
        directory = info["directory"]
        directory.mkdir(parents=True, exist_ok=True)
        info["files"] = {
            name: directory / f"{name}.tmp.npy"
            for name in ("scores", "components", "explained_variance", "valid")
        }
        info["scores"] = np.lib.format.open_memmap(
            info["files"]["scores"], mode="w+", dtype="float32",
            shape=(n_subjects, 2, 40, 3, times.size),
        )
        info["components"] = np.lib.format.open_memmap(
            info["files"]["components"], mode="w+", dtype="float32",
            shape=(n_subjects, 3, info["rows"].size),
        )
        info["explained_variance"] = np.lib.format.open_memmap(
            info["files"]["explained_variance"], mode="w+", dtype="float32",
            shape=(n_subjects, 3),
        )
        info["valid"] = np.lib.format.open_memmap(
            info["files"]["valid"], mode="w+", dtype="bool",
            shape=(n_subjects, 2, 40),
        )
        info["scores"][:] = np.nan
        info["valid"][:] = False
    started = time.time()
    with threadpool_limits(limits=args.threads):
        for subject in range(n_subjects):
            selected = []
            for key in ("positive", "negative"):
                indices = np.asarray(conditions[subject][key], dtype=int).ravel()
                if indices.size != 40 or np.unique(indices).size != 40:
                    raise ValueError(f"subject {subject}: {key} must contain 40 unique trials")
                if np.any(indices < 0) or np.any(indices >= n_trials):
                    raise ValueError(f"subject {subject}: {key} has out-of-range trials")
                selected.append(indices)
            if np.intersect1d(selected[0], selected[1]).size:
                raise ValueError(f"subject {subject}: conditions overlap")
            trial_indices = np.concatenate(selected)
            kernel = np.asarray(tmatrix["kernel"][subject], dtype=np.float64)
            for info in scout_infos:
                if info["rows"].max() >= kernel.shape[0]:
                    raise ValueError(
                        f"subject {subject}: scout {info['scout']} vertices exceed kernel rows"
                    )
            good = np.asarray(tmatrix["GoodChannel"][subject], dtype=int).ravel() - 1
            if np.any(good < 0) or np.any(good >= n_channels):
                raise ValueError(f"subject {subject}: invalid good-channel indices")
            subject_eeg = eeg[:, :, :, subject][:, time_indices, :][:, :, trial_indices]
            if kernel.shape[1] == good.size:
                subject_eeg = subject_eeg[good]
            elif kernel.shape[1] != n_channels:
                raise ValueError(f"subject {subject}: kernel/EEG channels do not match")
            keep = np.isfinite(subject_eeg).all(axis=(0, 1)) & np.any(subject_eeg != 0, axis=(0, 1))
            if keep[:40].sum() < args.avg_num or keep[40:].sum() < args.avg_num:
                raise ValueError(f"subject {subject}: too few valid trials in a condition")
            selected_eeg = np.asarray(subject_eeg[:, :, keep], dtype=np.float64)
            flat = selected_eeg.reshape(selected_eeg.shape[0], -1)
            sensor_mean = flat.mean(axis=1, keepdims=True)
            centered = flat - sensor_mean
            sensor_cov = centered @ centered.T / (centered.shape[1] - 1)
            eigenvalues, eigenvectors = np.linalg.eigh(sensor_cov)
            sqrt_cov = eigenvectors * np.sqrt(np.clip(eigenvalues, 0, None))
            valid_positions = np.flatnonzero(keep)
            for info in scout_infos:
                info["valid"][subject] = keep.reshape(2, 40)
                kernel_scout = kernel[info["rows"]]
                left, singular, _ = np.linalg.svd(kernel_scout @ sqrt_cov, full_matrices=False)
                if singular.size < 3 or singular[2] <= 0:
                    raise ValueError(
                        f"subject {subject}: scout {info['scout']} source rank below 3"
                    )
                basis = left[:, :3].T
                info["components"][subject] = basis.astype(np.float32)
                info["explained_variance"][subject] = (
                    singular[:3] ** 2 / np.sum(singular ** 2)
                ).astype(np.float32)
                pc_scores = (basis @ kernel_scout @ centered).reshape(3, times.size, keep.sum())
                for k, position in enumerate(valid_positions):
                    info["scores"][subject, position // 40, position % 40] = pc_scores[:, :, k]
            if (subject + 1) % 10 == 0 or subject + 1 == n_subjects:
                print(
                    f"  PCA {subject + 1}/{n_subjects} subjects across "
                    f"{len(scout_infos)} scout(s); {time.time() - started:.0f}s",
                    flush=True,
                )
    del eeg, tmatrix
    for info in scout_infos:
        directory = info["directory"]
        for name, temporary in info["files"].items():
            info[name].flush()
            del info[name]
            os.replace(temporary, directory / f"{name}.npy")
        meta = {
            "signature": info["signature"],
            "scout_label": info["label"],
            "groups": groups.tolist(),
            "times_ms": times.tolist(),
            "axes": "subject x condition(congruent,incongruent) x trial x PC x time",
            "pca_scope": "one PCA per subject/scout using both 40-trial intention conditions",
        }
        save_json(directory / "metadata.json", meta)


def ensure_scout_caches(args: argparse.Namespace, scouts: list[int]) -> dict[int, Path]:
    atlas = mat73.loadmat(str(args.atlas))["DKT_scout"]
    n_scouts = len(atlas["Vertices"])
    if any(scout >= n_scouts for scout in scouts):
        raise ValueError(f"scout indices must be between 0 and {n_scouts - 1}")
    paths = {scout: args.cache_dir / f"scout_{scout:02d}" for scout in scouts}
    missing = {}
    for scout, directory in paths.items():
        signature = cache_signature(args, scout)
        if not args.rebuild_cache and (directory / "metadata.json").exists():
            with open(directory / "metadata.json", encoding="utf-8") as file:
                existing_meta = json.load(file)
            if existing_meta.get("signature") == signature:
                print(f"Using scout cache: {directory}", flush=True)
                continue
        missing[scout] = (directory, signature)
    if missing:
        make_scout_caches(args, missing)
    return paths


def orient_largest_positive(components: np.ndarray) -> np.ndarray:
    largest = np.argmax(np.abs(components), axis=1)
    signs = np.sign(components[np.arange(components.shape[0]), largest])
    signs[signs == 0] = 1
    return components * signs[:, None]


def align_fold(cache: dict, selected: np.ndarray, train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Align selected subjects to a reference learned only from train subjects."""
    bases = np.asarray(cache["components"][selected], dtype=np.float64)
    stacked = bases[train].reshape(-1, bases.shape[-1])
    _, _, right = np.linalg.svd(stacked, full_matrices=False)
    reference = orient_largest_positive(right[:3])
    baseline = (cache["times"] >= BASELINE_START_MS) & (cache["times"] <= BASELINE_END_MS)
    if baseline.sum() < 2:
        raise ValueError("fewer than two baseline time points")
    n, _, n_trials, _, n_times = cache["scores"][selected].shape
    aligned = np.full((n, 2, n_trials, 3, n_times), np.nan, dtype=np.float32)
    for local, global_subject in enumerate(selected):
        u, _, vt = np.linalg.svd(reference @ bases[local].T)
        rotation = (u @ vt).astype(np.float32)
        for condition in range(2):
            valid = np.asarray(cache["valid"][global_subject, condition], dtype=bool)
            raw = np.asarray(cache["scores"][global_subject, condition, valid], dtype=np.float32)
            projected = np.einsum("ab,nbt->nat", rotation, raw, optimize=True)
            mean = projected[:, :, baseline].mean(axis=-1, keepdims=True)
            std = projected[:, :, baseline].std(axis=-1, keepdims=True)
            std = np.maximum(std, 1e-8)
            aligned[local, condition, valid] = (projected - mean) / std
    return aligned, reference.astype(np.float32)


def bootstrap_examples(
    aligned: np.ndarray, subjects: np.ndarray, labels: np.ndarray,
    draws: int, avg_num: int, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate congruent, incongruent, and exact difference triplets."""
    n_times = aligned.shape[-1]
    x = np.empty((len(subjects) * draws, 3, 3, n_times), dtype=np.float32)
    y = np.repeat(labels[subjects], draws).astype(np.int64)
    for offset, subject in enumerate(subjects):
        conditions = []
        for condition in range(2):
            trials = aligned[subject, condition]
            trials = trials[np.isfinite(trials).all(axis=(1, 2))]
            if not len(trials):
                raise ValueError(f"subject {subject}: no valid {condition} trials")
            counts = rng.multinomial(avg_num, np.full(len(trials), 1 / len(trials)), size=draws)
            average = counts.astype(np.float32) @ trials.reshape(len(trials), -1)
            conditions.append((average / avg_num).reshape(draws, 3, n_times))
        sl = slice(offset * draws, (offset + 1) * draws)
        x[sl, 0] = conditions[0]
        x[sl, 1] = conditions[1]
        x[sl, 2] = conditions[1] - conditions[0]
    return x, y


class TemporalBinPool(nn.Module):
    """Average into fixed temporal bins without MPS adaptive pooling."""

    def __init__(self, n_bins: int = 8) -> None:
        super().__init__()
        self.n_bins = n_bins

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        width = x.shape[-1]
        if width < self.n_bins:
            raise ValueError(f"temporal width {width} is smaller than {self.n_bins} bins")
        padding = (-width) % self.n_bins
        if padding:
            x = torch.cat(
                (x, x[..., -1:].expand(*x.shape[:-1], padding)), dim=-1
            )
        return x.reshape(*x.shape[:-1], self.n_bins, x.shape[-1] // self.n_bins).mean(dim=-1)


class TemporalEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, (1, 25), padding=(0, 12), bias=False),
            nn.BatchNorm2d(8), nn.ELU(),
            nn.Conv2d(8, 16, (3, 1), groups=8, bias=False),
            nn.BatchNorm2d(16), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(0.3),
            nn.Conv2d(16, 16, (1, 15), padding=(0, 7), groups=16, bias=False),
            nn.Conv2d(16, 16, (1, 1), bias=False),
            nn.BatchNorm2d(16), nn.ELU(),
            TemporalBinPool(8), nn.Flatten(),
            nn.Linear(16 * 8, 16), nn.ELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x.unsqueeze(1))


class ScoutGroupNet(nn.Module):
    def __init__(self, n_branches: int = 3) -> None:
        super().__init__()
        if n_branches not in (2, 3):
            raise ValueError("n_branches must be 2 or 3")
        self.encoders = nn.ModuleList([TemporalEncoder() for _ in range(n_branches)])
        self.head = nn.Sequential(nn.Linear(16 * n_branches, 16), nn.ELU(), nn.Dropout(0.3), nn.Linear(16, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = torch.cat([encoder(x[:, branch]) for branch, encoder in enumerate(self.encoders)], dim=1)
        return self.head(latent)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS was requested but is unavailable")
    return torch.device(requested)


@torch.no_grad()
def probabilities(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    tensor = torch.from_numpy(x)
    parts = []
    for start in range(0, len(x), batch_size):
        batch = tensor[start:start + batch_size].to(device)
        parts.append(torch.softmax(model(batch), dim=1).cpu().numpy())
    return np.concatenate(parts, axis=0)


def subject_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 1e-7, 1 - 1e-7)
    p = p / p.sum(axis=1, keepdims=True)
    pred = (p[:, 1] >= 0.5).astype(int)
    return {
        "n": int(len(y)),
        "auc": float(roc_auc_score(y, p[:, 1])) if len(np.unique(y)) == 2 else None,
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "sensitivity": float(np.mean(pred[y == 1] == 1)) if np.any(y == 1) else None,
        "specificity": float(np.mean(pred[y == 0] == 0)) if np.any(y == 0) else None,
        "nll": float(log_loss(y, p, labels=[0, 1])),
    }


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(value, file, indent=2)
    os.replace(temporary, path)


def linear_baseline(
    aligned: np.ndarray, train: np.ndarray, val: np.ndarray, test: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Fixed, train-scaled regularized linear comparator on condition ERPs."""
    erps = np.nanmean(aligned[:, :2], axis=2).reshape(len(aligned), -1)
    best_nll = np.inf
    best_model = None
    for c_value in (0.01, 0.1, 1.0):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c_value, class_weight="balanced", max_iter=2000, solver="liblinear"),
        )
        model.fit(erps[train], y[train])
        p_val = model.predict_proba(erps[val])
        score = log_loss(y[val], p_val, labels=[0, 1])
        if score < best_nll:
            best_nll, best_model = score, model
    return best_model.predict_proba(erps[test])


def train_fold(
    args: argparse.Namespace, aligned: np.ndarray, y: np.ndarray,
    train: np.ndarray, val: np.ndarray, test: np.ndarray,
    fold: int, directory: Path, device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict]:
    torch.manual_seed(args.seed + fold)
    np.random.seed(args.seed + fold)
    random.seed(args.seed + fold)
    model = ScoutGroupNet(3 if args.branches == "all" else 2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    class_counts = np.bincount(y[train], minlength=2)
    class_weight = len(train) / (2 * class_counts)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weight, dtype=torch.float32, device=device))
    val_x, _ = bootstrap_examples(
        aligned, val, y, args.eval_draws, args.avg_num,
        np.random.default_rng(np.random.SeedSequence([args.seed, fold, 1])),
    )
    test_x, _ = bootstrap_examples(
        aligned, test, y, args.eval_draws, args.avg_num,
        np.random.default_rng(np.random.SeedSequence([args.seed, fold, 2])),
    )
    rng = np.random.default_rng(np.random.SeedSequence([args.seed, fold, 0]))
    best_loss = np.inf
    best_epoch = 0
    history = []
    checkpoint = directory / f"fold_{fold:02d}_best.pt"
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train_x, train_y = bootstrap_examples(aligned, train, y, args.train_draws, args.avg_num, rng)
        train_tensor = torch.from_numpy(train_x).to(device)
        label_tensor = torch.from_numpy(train_y).to(device)
        order = torch.randperm(len(train_x), device=device)
        model.train()
        loss_sum = 0.0
        for batch_indices in order.split(args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            logits = model(train_tensor[batch_indices])
            loss = criterion(logits, label_tensor[batch_indices])
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(batch_indices)
        p_val = probabilities(model, val_x, device, args.batch_size).reshape(len(val), args.eval_draws, 2).mean(axis=1)
        metrics = subject_metrics(y[val], p_val)
        row = {
            "fold": fold, "epoch": epoch, "train_loss": loss_sum / len(train_x),
            "val_nll": metrics["nll"], "val_auc": metrics["auc"],
            "val_balanced_accuracy": metrics["balanced_accuracy"],
            "val_accuracy": metrics["accuracy"], "seconds": time.time() - started,
        }
        history.append(row)
        print(
            f"fold {fold:02d}/10 epoch {epoch:03d} "
            f"train_loss={row['train_loss']:.4f} val_nll={row['val_nll']:.4f} "
            f"val_auc={row['val_auc']:.3f} val_bal_acc={row['val_balanced_accuracy']:.3f} "
            f"time={row['seconds']:.1f}s", flush=True,
        )
        if metrics["nll"] < best_loss - 1e-4:
            best_loss = metrics["nll"]
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint)
        if epoch - best_epoch >= args.patience:
            print(f"fold {fold:02d}: early stop at epoch {epoch}; best={best_epoch}", flush=True)
            break
    save_csv(directory / f"fold_{fold:02d}_history.csv", history)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    p_test = probabilities(model, test_x, device, args.batch_size).reshape(len(test), args.eval_draws, 2).mean(axis=1)
    p_linear = linear_baseline(aligned, train, val, test, y)
    details = {"best_epoch": best_epoch, "best_val_nll": best_loss}
    return p_test, p_linear, details


def synthetic_cache() -> dict:
    """Small aligned-subspace test data; no user EEG is opened."""
    rng = np.random.default_rng(991)
    n, trials, n_times, n_rows = 60, 40, 101, 12
    times = START_MS + np.arange(n_times) * (1000 / FS)
    base, _ = np.linalg.qr(rng.normal(size=(n_rows, 3)))
    base = base.T
    components = np.empty((n, 3, n_rows), dtype=np.float32)
    scores = np.empty((n, 2, trials, 3, n_times), dtype=np.float32)
    groups = np.repeat([1, 2, 3], n // 3)
    pulse = np.exp(-0.5 * ((times - 90) / 35) ** 2)
    for subject in range(n):
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        components[subject] = (rotation @ base).astype(np.float32)
        common = rng.normal(scale=0.1, size=(3, n_times))
        for condition in range(2):
            for trial in range(trials):
                canonical = common + rng.normal(scale=0.7, size=(3, n_times))
                canonical[0] += (groups[subject] - 1) * condition * 0.4 * pulse
                scores[subject, condition, trial] = rotation @ canonical
    return {
        "scores": scores,
        "components": components,
        "explained_variance": np.ones((n, 3), dtype=np.float32) / 3,
        "valid": np.ones((n, 2, trials), dtype=bool),
        "groups": groups,
        "times": times.astype(np.float32),
        "label": "synthetic scout",
        "meta": {"signature": {"synthetic": True}},
    }


def select_pair_subjects(groups: np.ndarray, pair: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return subject indices, binary labels, and original labels for stratification."""
    negative_codes, positive_codes = PAIRS[pair]
    selected = np.flatnonzero(np.isin(groups, negative_codes + positive_codes))
    original_labels = np.asarray(groups[selected], dtype=np.int64)
    y = np.isin(original_labels, positive_codes).astype(np.int64)
    counts = {int(code): int((original_labels == code).sum()) for code in negative_codes + positive_codes}
    if min(counts.values()) < 10:
        raise ValueError(f"each original group needs at least ten subjects for 10-fold CV; got {counts}")
    return selected, y, original_labels


def run_one_scout(
    args: argparse.Namespace, cache: dict, device: torch.device, max_folds: int
) -> dict:
    selected, y, original_labels = select_pair_subjects(cache["groups"], args.pair)
    output = args.output_dir / f"scout_{args.scout:02d}" / args.pair / args.branches
    output.mkdir(parents=True, exist_ok=True)
    print(
        f"Scout {args.scout}: {cache['label']} | pair {args.pair} | "
        f"subjects {np.bincount(y).tolist()} | device {device} | output {output}",
        flush=True,
    )
    n_branches = 3 if args.branches == "all" else 2
    print(f"Parameters: {sum(p.numel() for p in ScoutGroupNet(n_branches).parameters()):,}", flush=True)
    retained = np.asarray(cache["explained_variance"][selected]).sum(axis=1)
    print(
        f"Three-PC variance retained: median={np.median(retained):.3f}, "
        f"minimum={np.min(retained):.3f}", flush=True,
    )
    run_config = {
        "model_version": MODEL_VERSION,
        "cache_signature": cache["meta"]["signature"],
        "scout": args.scout, "pair": args.pair, "branches": args.branches,
        "seed": args.seed, "avg_num": args.avg_num,
        "train_draws": args.train_draws, "eval_draws": args.eval_draws,
        "epochs": args.epochs, "patience": args.patience,
        "batch_size": args.batch_size, "lr": args.lr,
        "weight_decay": args.weight_decay, "device": str(device),
    }
    config_path = output / "run_config.json"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as file:
            previous_config = json.load(file)
        if previous_config != run_config:
            raise ValueError(
                f"Output directory {output} contains a run with different settings; "
                "choose another --output-dir"
            )
    else:
        save_json(config_path, run_config)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=args.seed)
    prediction_rows = []
    fold_rows = []
    for fold, (trainval, test) in enumerate(cv.split(selected, original_labels), start=1):
        if fold > max_folds:
            break
        train, val = train_test_split(
            trainval, test_size=1 / 9, stratify=original_labels[trainval],
            random_state=args.seed + fold,
        )
        result_path = output / f"fold_{fold:02d}_result.json"
        if args.resume and result_path.exists():
            with open(result_path, encoding="utf-8") as file:
                saved = json.load(file)
            if saved["test_subjects"] != selected[test].tolist():
                raise ValueError(f"fold {fold}: saved test subjects do not match this split")
            p_net = np.asarray(saved["cnn_probabilities"], dtype=np.float64)
            p_linear = np.asarray(saved["linear_probabilities"], dtype=np.float64)
            details = saved["details"]
            print(f"fold {fold:02d}: reused completed fold", flush=True)
        else:
            print(
                f"fold {fold:02d}: train={len(train)} val={len(val)} test={len(test)}; "
                "aligning PCs from training reference", flush=True,
            )
            aligned, reference = align_fold(cache, selected, train)
            np.save(output / f"fold_{fold:02d}_reference.npy", reference)
            p_net, p_linear, details = train_fold(args, aligned, y, train, val, test, fold, output, device)
            save_json(result_path, {
                "test_subjects": selected[test].tolist(),
                "cnn_probabilities": p_net.tolist(),
                "linear_probabilities": p_linear.tolist(),
                "details": details,
            })
        net_metrics = subject_metrics(y[test], p_net)
        linear_metrics = subject_metrics(y[test], p_linear)
        print(
            f"fold {fold:02d} TEST: CNN auc={net_metrics['auc']:.3f} "
            f"bal_acc={net_metrics['balanced_accuracy']:.3f} | "
            f"linear auc={linear_metrics['auc']:.3f} "
            f"bal_acc={linear_metrics['balanced_accuracy']:.3f}", flush=True,
        )
        fold_rows.append({
            "fold": fold, "best_epoch": details["best_epoch"],
            "best_val_nll": details["best_val_nll"],
            **{f"cnn_{k}": v for k, v in net_metrics.items()},
            **{f"linear_{k}": v for k, v in linear_metrics.items()},
        })
        for position, local_subject in enumerate(test):
            prediction_rows.append({
                "fold": fold,
                "subject_index_0based": int(selected[local_subject]),
                "true_group_code": int(cache["groups"][selected[local_subject]]),
                "true_binary_label": int(y[local_subject]),
                "cnn_probability_second_group": float(p_net[position, 1]),
                "linear_probability_second_group": float(p_linear[position, 1]),
            })
        save_csv(output / "fold_metrics.csv", fold_rows)
        save_csv(output / "subject_predictions.csv", prediction_rows)
    if len(prediction_rows) == len(selected):
        true = np.array([row["true_binary_label"] for row in prediction_rows])
        summary = {}
        for model_name in ("cnn", "linear"):
            positive = np.array([row[f"{model_name}_probability_second_group"] for row in prediction_rows])
            summary[model_name] = subject_metrics(true, np.column_stack((1 - positive, positive)))
        print(
            f"OUT-OF-FOLD SUBJECTS: CNN AUC={summary['cnn']['auc']:.3f} "
            f"balanced_accuracy={summary['cnn']['balanced_accuracy']:.3f}; "
            f"linear AUC={summary['linear']['auc']:.3f} "
            f"balanced_accuracy={summary['linear']['balanced_accuracy']:.3f}", flush=True,
        )
    else:
        summary = {"smoke_test": True, "tested_subjects": len(prediction_rows)}
    manifest = {
        "config": {
            "scout": args.scout, "scout_label": cache["label"], "pair": args.pair,
            "branches": args.branches,
            "seed": args.seed, "folds": max_folds, "avg_num": args.avg_num,
            "train_draws": args.train_draws, "eval_draws": args.eval_draws,
            "epochs": args.epochs, "patience": args.patience,
            "batch_size": args.batch_size, "lr": args.lr,
            "weight_decay": args.weight_decay, "device": str(device),
        },
        "cache_signature": cache["meta"]["signature"],
        "summary": summary,
    }
    save_json(output / "summary.json", manifest)
    print(f"Saved predictions and metrics to {output}", flush=True)
    return {
        "scout": args.scout,
        "scout_label": cache["label"],
        "pair": args.pair,
        "branches": args.branches,
        "n_subjects": len(selected),
        "cnn_auc": summary.get("cnn", {}).get("auc"),
        "cnn_balanced_accuracy": summary.get("cnn", {}).get("balanced_accuracy"),
        "linear_auc": summary.get("linear", {}).get("auc"),
        "linear_balanced_accuracy": summary.get("linear", {}).get("balanced_accuracy"),
    }


def run(args: argparse.Namespace) -> None:
    device = choose_device(args.device)
    torch.set_num_threads(args.threads)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    if args.smoke_test:
        args.epochs = min(args.epochs, 2)
        args.train_draws = min(args.train_draws, 8)
        args.eval_draws = min(args.eval_draws, 8)
        args.patience = min(args.patience, 2)
        if args.output_dir is None:
            args.output_dir = Path(tempfile.mkdtemp(prefix="scout_group_smoke_"))
        max_folds = 2
        cache_paths = None
    else:
        if args.output_dir is None:
            args.output_dir = ROOT / "Results" / "scout_group_classifier_v2"
        max_folds = 10
        cache_paths = ensure_scout_caches(args, args.scouts)
    all_summaries = []
    for pair_position, pair in enumerate(args.pairs, start=1):
        args.pair = pair
        pair_summaries = []
        print(f"\n=== Comparison {pair_position}/{len(args.pairs)}: {pair} ===", flush=True)
        for scout_position, scout in enumerate(args.scouts, start=1):
            args.scout = scout
            if args.smoke_test:
                cache = synthetic_cache()
                cache["label"] = f"synthetic scout {scout}"
            else:
                cache = load_cache_files(cache_paths[scout])
            print(
                f"\n=== Scout {scout_position}/{len(args.scouts)}: index {scout} ===",
                flush=True,
            )
            pair_summaries.append(run_one_scout(args, cache, device, max_folds))
            del cache
            gc.collect()
        summary_path = args.output_dir / f"{pair}_{args.branches}_scouts.csv"
        save_csv(summary_path, pair_summaries)
        print(f"Comparison summary: {summary_path}", flush=True)
        all_summaries.extend(pair_summaries)
    if len(args.pairs) > 1:
        summary_path = args.output_dir / f"{args.branches}_comparisons.csv"
        save_csv(summary_path, all_summaries)
        print(f"Combined summary: {summary_path}", flush=True)


if __name__ == "__main__":
    run(parse_args())
