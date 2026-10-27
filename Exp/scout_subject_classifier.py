"""Subject-level group classification from one DKT scout with an EEGNet-style network.

Each subject's intention responses are projected to the scout, reduced to shared
scout PCs, and turned into incongruent - congruent bootstrap averages. The
orientation and PC basis are fitted on the training subjects of every outer fold.
The network trains for a fixed number of epochs with a one-cycle learning rate,
and the weights of the final epochs are averaged. Test subjects are scored by a
soft vote over bootstrap averages.

Examples:
    python3 scout_subject_classifier.py --smoke-test
    python3 scout_subject_classifier.py --scout 13
    python3 scout_subject_classifier.py --scout 13 48 --pair control-vs-depressed+suicidal
    python3 scout_subject_classifier.py --scout all --device cpu --jobs 12
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.optim.swa_utils import AveragedModel

ROOT = Path(__file__).resolve().parent.parent  # SourceLoc; this script lives in Exp/
DATA = ROOT.parent.parent / "Data"
PREPROCESSED = DATA / "Preprocessed data"
BEHAVIOR = DATA / "Behavior"
PAIRS = {
    "control-vs-depressed+suicidal": ((1,), (2, 3)),
    "depressed-suicidal": ((2,), (3,)),
    "control-depressed": ((1,), (2,)),
    "control-suicidal": ((1,), (3,)),
}
DEFAULT_PAIRS = ("control-vs-depressed+suicidal", "depressed-suicidal")
FS = 250
START_MS = -200
BASELINE_END_MS = 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scout", dest="scouts", nargs="+",
                   help="Zero-based DKT scouts separated by spaces or commas, or 'all'")
    p.add_argument("--pair", dest="pairs", nargs="+", choices=PAIRS, default=list(DEFAULT_PAIRS))
    p.add_argument("--eeg", type=Path, default=PREPROCESSED / "Data_sen_bl.pkl")
    p.add_argument("--tmatrix", type=Path, default=PREPROCESSED / "tMatrix.pkl")
    p.add_argument("--atlas", type=Path, default=PREPROCESSED / "DKT_scout.mat")
    p.add_argument("--subject-index", type=Path, default=BEHAVIOR / "subject_index.mat")
    p.add_argument("--trial-index", type=Path, default=BEHAVIOR / "senIdx_TOI.pkl")
    p.add_argument("--category", default="Intention",
                   help="Key under each subject in --trial-index; 'none' if the file has no categories")
    p.add_argument("--cache-dir", type=Path, default=ROOT / "Results" / "subject_sensor_cache")
    p.add_argument("--output-dir", type=Path, default=ROOT / "Results" / "scout_subject_classifier")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--n-components", type=int, default=32)
    p.add_argument("--representation", choices=("contrast", "fusion"), default="contrast",
                   help="contrast: incongruent - congruent input; fusion: shared encoder on both conditions, "
                        "latent features joined as [congruent, incongruent, difference]")
    p.add_argument("--window", nargs=2, type=int, default=[0, 800], metavar=("START_MS", "END_MS"),
                   help="Model input window; the end is exclusive")
    p.add_argument("--avg-range", nargs=2, type=int, default=[8, 24], metavar=("LO", "HI"),
                   help="Trials per training bootstrap average, drawn per example")
    p.add_argument("--draws-per-epoch", type=int, default=16, help="New bootstrap averages per subject per epoch")
    p.add_argument("--tta-avg", type=int, default=16, help="Trials per test bootstrap average")
    p.add_argument("--tta-draws", type=int, default=32, help="Test bootstrap averages per subject")
    p.add_argument("--n-models", type=int, default=1,
                   help="Networks trained per fold with different seeds; their predictions are averaged")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--avg-epochs", type=int, default=10, help="Average the weights of the final N epochs")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3, help="Peak of the one-cycle learning rate")
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--max-shift", type=int, default=3, help="Time-shift augmentation in samples")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--permute-seed", type=int, default=None, help="Shuffle group labels to build a null distribution")
    p.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    p.add_argument("--jobs", type=int, default=1,
                   help="Parallel CPU folds; with more than one, per-epoch metrics go to the history files only")
    p.add_argument("--smoke-test", action="store_true", help="Short run on synthetic data")
    args = p.parse_args()
    if args.smoke_test:
        args.scouts = args.scouts or ["0"]
    elif not args.scouts:
        p.error("--scout is required")
    if [s.lower() for s in args.scouts] == ["all"]:
        args.scouts = "all"
    else:
        try:
            args.scouts = [int(x) for tok in args.scouts for x in tok.split(",") if x.strip()]
        except ValueError:
            p.error("--scout takes zero-based integers or 'all'")
        if not args.scouts or min(args.scouts) < 0 or len(set(args.scouts)) != len(args.scouts):
            p.error("--scout needs distinct nonnegative indices")
    if len(set(args.pairs)) != len(args.pairs):
        p.error("--pair names must be distinct")
    lo, hi = args.avg_range
    if lo < 1 or hi < lo:
        p.error("--avg-range needs 1 <= LO <= HI")
    if args.window[0] < START_MS or args.window[1] <= args.window[0]:
        p.error("--window must start at or after -200 ms and end after it starts")
    for name in ("n_components", "n_models", "draws_per_epoch", "tta_avg", "tta_draws", "epochs", "avg_epochs",
                 "batch_size", "folds", "repeats", "jobs"):
        if getattr(args, name) < 1:
            p.error(f"--{name.replace('_', '-')} must be positive")
    if args.avg_epochs > args.epochs:
        p.error("--avg-epochs cannot exceed --epochs")
    if args.jobs > 1 and args.device not in ("auto", "cpu"):
        p.error("--jobs above 1 runs on CPU; drop --device or use --device cpu")
    return args


def file_signature(path: Path | str) -> dict:
    stat = Path(path).stat()
    return {"path": str(Path(path).resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def save_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS was requested but is unavailable")
    return torch.device(requested)


def cache_signature(args: argparse.Namespace) -> dict:
    return {name: file_signature(getattr(args, name)) for name in ("eeg", "tmatrix", "atlas", "subject_index")}


def build_cache(args: argparse.Namespace) -> None:
    """Per subject: sensor trials (trial x channel x time) and the inverse kernel, both float32."""
    import mat73
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    atlas = mat73.loadmat(str(args.atlas))["DKT_scout"]
    labels = [str(x).strip() for x in atlas.get("Label", atlas.get("Region"))]
    vertices = [(np.asarray(v, dtype=np.int64).ravel() - 1).tolist() for v in atlas["Vertices"]]
    groups = np.asarray(mat73.loadmat(str(args.subject_index))["subject_index"]).ravel().astype(int)
    print("Building the subject cache from the EEG and inverse-kernel pickles...", flush=True)
    with open(args.eeg, "rb") as f:
        eeg = pickle.load(f)
    with open(args.tmatrix, "rb") as f:
        tmatrix = pickle.load(f)
    if eeg.ndim != 4 or eeg.shape[3] != len(groups) or len(tmatrix["kernel"]) != len(groups):
        raise ValueError(f"EEG shape {eeg.shape} does not match {len(groups)} subjects and kernels")
    n_channels, n_times = eeg.shape[:2]
    for s in range(len(groups)):
        kernel = np.asarray(tmatrix["kernel"][s], dtype=np.float32)
        good = np.asarray(tmatrix["GoodChannel"][s]).ravel().astype(int) - 1
        sensor = np.ascontiguousarray(eeg[:, :, :, s].transpose(2, 0, 1), dtype=np.float32)
        if kernel.shape[1] == len(good) and len(good) != n_channels:
            sensor = sensor[:, good]
        elif kernel.shape[1] != n_channels:
            raise ValueError(f"subject {s}: kernel columns do not match channels")
        if 3 * max(max(v) for v in vertices) + 2 >= kernel.shape[0]:
            raise ValueError(f"subject {s}: atlas vertices exceed kernel rows")
        for name, arr in (("sensor", sensor), ("kernel", kernel)):
            tmp = args.cache_dir / f"{name}_{s:03d}.tmp.npy"
            np.save(tmp, arr)
            os.replace(tmp, args.cache_dir / f"{name}_{s:03d}.npy")
    save_json(args.cache_dir / "metadata.json", {
        "signature": cache_signature(args), "groups": groups.tolist(), "labels": labels, "vertices": vertices,
        "times_ms": (START_MS + np.arange(n_times) * 1000 / FS).tolist(),
    })


def build_synthetic_cache(directory: Path) -> None:
    """Groups 2 and 3 carry an extra 10-Hz burst in incongruent trials."""
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    groups = np.repeat([1, 2, 3], 12)
    n_ch, n_vert, n_times, n_trials = 16, 30, 300, 80
    times = START_MS + np.arange(n_times) * 1000 / FS
    burst = np.sin(2 * np.pi * 10 * times / 1000) * np.exp(-0.5 * ((times - 300) / 80) ** 2)
    erp = np.exp(-0.5 * ((times - 250) / 50) ** 2)
    base_kernel = rng.normal(size=(3 * n_vert, n_ch)).astype(np.float32)
    pattern = rng.normal(size=n_ch)
    for s, g in enumerate(groups):
        x = rng.normal(size=(n_trials, n_ch, n_times)) + 0.5 * erp * pattern[:, None]
        if g > 1:
            x[1::2] += 0.8 * burst * pattern[:, None]
        np.save(directory / f"sensor_{s:03d}.npy", (x * rng.uniform(0.5, 50)).astype(np.float32))
        np.save(directory / f"kernel_{s:03d}.npy",
                (base_kernel + 0.05 * rng.normal(size=base_kernel.shape)).astype(np.float32))
    index = {"Sentiment": [{"Intention": {"positive": np.arange(0, n_trials, 2),
                                          "negative": np.arange(1, n_trials, 2)}} for _ in groups]}
    with open(directory / "trial_index.pkl", "wb") as f:
        pickle.dump(index, f)
    save_json(directory / "metadata.json", {
        "signature": {"synthetic": True}, "groups": groups.tolist(), "labels": ["synthetic scout"],
        "vertices": [list(range(n_vert))], "times_ms": times.tolist(),
    })


def ensure_cache(args: argparse.Namespace) -> dict:
    path = args.cache_dir / "metadata.json"
    if args.smoke_test:
        if not path.exists():
            build_synthetic_cache(args.cache_dir)
    elif not path.exists() or json.loads(path.read_text())["signature"] != cache_signature(args):
        build_cache(args)
    return json.loads(path.read_text())


def load_trial_sets(path: Path, category: str, n_subjects: int) -> list[tuple[np.ndarray, np.ndarray]]:
    with open(path, "rb") as f:
        entries = pickle.load(f)["Sentiment"]
    if len(entries) != n_subjects:
        raise ValueError("The trial index and the cache have different subject counts")
    sets = []
    for s, entry in enumerate(entries):
        if category.lower() != "none":
            entry = entry[category]
        congruent, incongruent = (np.asarray(entry[k], dtype=int).ravel() for k in ("positive", "negative"))
        if np.intersect1d(congruent, incongruent).size:
            raise ValueError(f"subject {s}: congruent and incongruent trials overlap")
        sets.append((congruent, incongruent))
    return sets


def select_pair(groups: np.ndarray, pair: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first, second = PAIRS[pair]
    selected = np.flatnonzero(np.isin(groups, first + second))
    originals = groups[selected]
    return selected, np.isin(originals, second).astype(np.int64), originals


def load_subject(cache: Path, s: int, sets: tuple, vertices: np.ndarray, baseline: np.ndarray) -> dict:
    sensor = np.load(cache / f"sensor_{s:03d}.npy", mmap_mode="r")
    conditions = []
    for idx in sets:
        if idx.min() < 0 or idx.max() >= sensor.shape[0]:
            raise ValueError(f"subject {s}: trial index outside 0-{sensor.shape[0] - 1}")
        x = np.array(sensor[idx], dtype=np.float32)
        x = x[np.isfinite(x).all(axis=(1, 2)) & np.any(x != 0, axis=(1, 2))]
        if len(x) < 8:
            raise ValueError(f"subject {s}: fewer than 8 valid trials in a condition")
        x -= x[:, :, baseline].mean(axis=-1, keepdims=True)
        conditions.append(x)
    rows = (3 * vertices[:, None] + np.arange(3)).ravel()
    kernel = np.asarray(np.load(cache / f"kernel_{s:03d}.npy", mmap_mode="r")[rows], dtype=np.float64)
    trials = np.concatenate(conditions).astype(np.float64)
    whole = trials.transpose(1, 0, 2).reshape(trials.shape[1], -1)
    base = trials[:, :, baseline].transpose(1, 0, 2).reshape(trials.shape[1], -1)
    return {"trials": conditions, "kernel": kernel.reshape(len(vertices), 3, -1),
            "cov": whole @ whole.T / whole.shape[1], "base_cov": base @ base.T / base.shape[1]}


def fit_spatial(subjects: list[dict], train: np.ndarray, n_components: int) -> dict:
    """Vertex orientations and scout PCs from training subjects only."""
    n_vertices = subjects[0]["kernel"].shape[0]
    orient = np.zeros((n_vertices, 3, 3))
    for i in train:
        k = subjects[i]["kernel"]
        c = np.einsum("vic,cd,vjd->vij", k, subjects[i]["cov"], k, optimize=True)
        orient += c / np.maximum(np.trace(c, axis1=1, axis2=2), 1e-30)[:, None, None]
    _, vecs = np.linalg.eigh(orient / len(train))
    w = vecs[:, :, -1]
    w *= np.where(w[np.arange(n_vertices), np.abs(w).argmax(1)] < 0, -1, 1)[:, None]
    vcov = np.zeros((n_vertices, n_vertices))
    for i in train:
        eff = np.einsum("vkc,vk->vc", subjects[i]["kernel"], w)
        c = eff @ subjects[i]["cov"] @ eff.T
        vcov += c / max(np.trace(c), 1e-30)
    vals, vecs = np.linalg.eigh(vcov / len(train))
    vals, vecs = vals[::-1], vecs[:, ::-1]
    k = int(min(n_components, np.sum(vals > vals[0] * 1e-9)))
    basis = vecs[:, :k] * np.where(vecs[np.abs(vecs[:, :k]).argmax(0), np.arange(k)] < 0, -1, 1)
    return {"orientation": w, "basis": basis, "explained": float(vals[:k].sum() / vals.sum())}


def projection(subject: dict, spatial: dict) -> np.ndarray:
    """PC x channel map from sensor data to scout PCs, scaled by the subject's pooled baseline SD."""
    eff = np.einsum("vkc,vk->vc", subject["kernel"], spatial["orientation"])
    m = spatial["basis"].T @ eff
    m /= np.sqrt(np.maximum(np.einsum("rc,cd,rd->r", m, subject["base_cov"], m), 1e-30))[:, None]
    return m


def project(subject: dict, spatial: dict) -> list[np.ndarray]:
    """Scout PC trials per condition, scaled by the subject's pooled baseline SD."""
    m = projection(subject, spatial)
    return [np.einsum("rc,nct->nrt", m.astype(np.float32), t, optimize=True) for t in subject["trials"]]


class EEGNet(nn.Module):
    def __init__(self, n_in: int, n_times: int, fusion: bool = False) -> None:
        super().__init__()
        self.fusion = fusion
        self.features = nn.Sequential(
            nn.Conv1d(n_in, 16, 1, bias=False),
            nn.Conv1d(16, 16, 31, padding=15, groups=16, bias=False), nn.GroupNorm(4, 16), nn.ELU(),
            nn.AvgPool1d(4), nn.Dropout(0.25),
            nn.Conv1d(16, 16, 15, padding=7, groups=16, bias=False), nn.Conv1d(16, 16, 1, bias=False),
            nn.GroupNorm(4, 16), nn.ELU(), nn.AvgPool1d(8), nn.Dropout(0.25), nn.Flatten())
        width = 16 * (n_times // 32) * (3 if fusion else 1)
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Dropout(0.3), nn.Linear(width, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fusion:
            a, b = self.features(x[:, 0]), self.features(x[:, 1])
            return self.head(torch.cat((a, b, b - a), 1))
        return self.head(self.features(x))


def bootstrap_weights(rng: np.random.Generator, n: int, sizes: np.ndarray) -> np.ndarray:
    counts = np.stack([rng.multinomial(m, np.full(n, 1 / n)) for m in sizes]).astype(np.float32)
    return counts / sizes[:, None].astype(np.float32)


def model_input(trials: list[np.ndarray], weights: list[np.ndarray], representation: str) -> np.ndarray:
    """Weighted condition averages, one example per row of the weights.

    contrast gives example x PC x time (incongruent - congruent); fusion gives
    example x condition x PC x time.
    """
    a, b = ((w @ t.reshape(len(t), -1)).reshape(len(w), *t.shape[1:]) for w, t in zip(weights, trials))
    return b - a if representation == "contrast" else np.stack((a, b), 1)


def draw_inputs(trials: list[np.ndarray], rng: np.random.Generator, sizes: np.ndarray,
                representation: str) -> np.ndarray:
    return model_input(trials, [bootstrap_weights(rng, len(t), sizes) for t in trials], representation)


def predict(model: nn.Module, x: np.ndarray, owner: np.ndarray, n_subjects: int,
            device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        p = np.concatenate([torch.softmax(model(torch.from_numpy(x[k:k + 256]).to(device)), 1)[:, 1].cpu().numpy()
                            for k in range(0, len(x), 256)])
    return np.array([p[owner == j].mean() for j in range(n_subjects)])


def fold_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return {"loss": float(log_loss(y, p, labels=[0, 1])), "auc": float(roc_auc_score(y, p)),
            "bal_acc": float(balanced_accuracy_score(y, (p >= 0.5).astype(int)))}


def train_fold(cfg: dict, trials: list[list[np.ndarray]], labels: np.ndarray, train: np.ndarray,
               test: np.ndarray, window: tuple[int, int], seed: int, device: torch.device,
               log) -> tuple[dict, list[dict]]:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    i0, i1 = window
    n_times = i1 - i0
    shift_max = min(cfg["max_shift"], i0, trials[0][0].shape[-1] - i1)
    lo, hi = cfg["avg_range"]
    draws = cfg["draws_per_epoch"]
    rep = cfg["representation"]

    def subject_draws(subjects, n, low, high):
        return np.concatenate([draw_inputs(trials[i], rng, rng.integers(low, high + 1, size=n), rep)
                               for i in subjects])

    def training_set():
        # Both conditions of a fusion example share one shift, scale, and PC mask.
        x = subject_draws(train, draws, lo, hi)
        n, extra = len(x), (1,) * (x.ndim - 3)
        shifts = rng.integers(-shift_max, shift_max + 1, size=n)
        x = np.take_along_axis(x, (i0 + shifts).reshape(n, *extra, 1, 1) + np.arange(n_times), axis=-1)
        x *= np.exp(rng.normal(0, 0.1, size=(n, *extra, 1, 1))).astype(np.float32)
        x *= (rng.random((n, *extra, x.shape[-2], 1)) > 0.1).astype(np.float32)
        return x

    # One unused set of draws keeps the random stream identical to the screening runs.
    subject_draws(train, draws, lo, hi)
    y_train = torch.from_numpy(np.repeat(labels[train], draws)).to(device)
    y_test = labels[test]
    x_full = np.stack([model_input(trials[i], [np.full((1, len(t)), 1 / len(t), dtype=np.float32)
                                               for t in trials[i]], rep)[0] for i in test])[..., i0:i1]
    monitor_rng = np.random.default_rng([seed, 1])
    x_monitor = np.concatenate([draw_inputs(trials[i], monitor_rng, np.full(cfg["tta_draws"], cfg["tta_avg"]), rep)
                                for i in test])[..., i0:i1]
    owner_tta = np.repeat(np.arange(len(test)), cfg["tta_draws"])

    model = EEGNet(trials[0][0].shape[1], n_times, fusion=rep == "fusion").to(device)
    averaged = AveragedModel(model)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps = int(np.ceil(len(train) * draws / cfg["batch_size"]))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=cfg["epochs"] * steps,
                                                pct_start=0.15)
    counts = np.bincount(labels[train], minlength=2)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(len(train) / (2 * counts), dtype=torch.float32, device=device),
                                  label_smoothing=cfg["label_smoothing"])
    history = []
    for epoch in range(1, cfg["epochs"] + 1):
        started = time.time()
        x = torch.from_numpy(training_set()).to(device)
        order = rng.permutation(len(x))
        model.train()
        loss_sum = torch.zeros((), device=device)
        correct = torch.zeros(2, device=device)
        for k in range(steps):
            idx = torch.from_numpy(order[k * cfg["batch_size"]:(k + 1) * cfg["batch_size"]]).to(device)
            logits = model(x[idx])
            target = y_train[idx]
            loss = loss_fn(logits, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.detach() * len(idx)
            hit = logits.detach().argmax(1) == target
            correct += torch.stack(((hit & (target == 0)).sum(), (hit & (target == 1)).sum()))
        lr = opt.param_groups[0]["lr"]
        if epoch > cfg["epochs"] - cfg["avg_epochs"]:
            averaged.update_parameters(model)
        totals = np.bincount(labels[train], minlength=2) * draws
        correct = correct.cpu().numpy()
        monitor = fold_metrics(y_test, predict(model, x_monitor, owner_tta, len(test), device))
        row = {"epoch": epoch, "lr": lr, "train_loss": float(loss_sum.item() / len(x)),
               "train_acc": float(correct.sum() / totals.sum()), "train_bal_acc": float(np.mean(correct / totals)),
               "test_loss": monitor["loss"], "test_auc": monitor["auc"], "test_bal_acc": monitor["bal_acc"],
               "averaged": epoch > cfg["epochs"] - cfg["avg_epochs"], "seconds": time.time() - started}
        history.append(row)
        log(f"  epoch {epoch:03d}/{cfg['epochs']} lr {lr:.1e}{' avg' if row['averaged'] else '    '} | "
            f"train loss {row['train_loss']:.4f} acc {row['train_acc']:.3f} bal_acc {row['train_bal_acc']:.3f} | "
            f"test [monitor] loss {monitor['loss']:.4f} auc {monitor['auc']:.3f} bal_acc {monitor['bal_acc']:.3f} | "
            f"{row['seconds']:.1f}s")
    net = averaged.module
    x_tta = subject_draws(test, cfg["tta_draws"], cfg["tta_avg"], cfg["tta_avg"])[..., i0:i1]
    result = {"p_full": predict(net, x_full, np.arange(len(test)), len(test), device).tolist(),
              "p_tta": predict(net, x_tta, owner_tta, len(test), device).tolist()}
    return result, history


def signature(cfg: dict, meta: dict) -> dict:
    keys = ("n_components", "representation", "n_models", "window", "avg_range", "draws_per_epoch", "tta_avg", "tta_draws", "epochs", "avg_epochs",
            "batch_size", "lr", "weight_decay", "label_smoothing", "max_shift", "folds", "seed", "permute_seed",
            "category")
    return {"cache": meta["signature"], "trial_index": file_signature(cfg["trial_index"]),
            **{k: cfg[k] for k in keys}}


def output_root(cfg: dict) -> Path:
    root = Path(cfg["output_dir"])
    return root if cfg["permute_seed"] is None else root / f"permutation_{cfg['permute_seed']:04d}"


def fold_path(cfg: dict, scout: int, pair: str, repeat: int, fold: int) -> Path:
    return output_root(cfg) / f"scout_{scout:02d}" / pair / f"repeat{repeat + 1}_fold{fold + 1}.json"


class ScoutData:
    """One scout's subjects for a pair, loaded on first use."""

    def __init__(self, cfg: dict, meta: dict, scout: int, pair: str) -> None:
        groups = np.asarray(meta["groups"])
        self.selected, self.labels, originals = select_pair(groups, pair)
        if any(np.sum(originals == g) < cfg["folds"] for g in np.unique(originals)):
            raise ValueError(f"{pair}: every group needs at least {cfg['folds']} subjects")
        if cfg["permute_seed"] is not None:
            order = np.random.default_rng(cfg["permute_seed"]).permutation(len(self.selected))
            self.labels, originals = self.labels[order], originals[order]
        self.originals = originals
        times = np.asarray(meta["times_ms"])
        hits = np.flatnonzero((times >= cfg["window"][0]) & (times < cfg["window"][1]))
        self.window = (int(hits[0]), int(hits[-1]) + 1)
        self._args = (cfg, meta, scout, times < BASELINE_END_MS)
        self.subjects = None

    def splits(self, cfg: dict, repeat: int):
        cv = StratifiedKFold(cfg["folds"], shuffle=True, random_state=cfg["seed"] + 1009 * repeat)
        return list(cv.split(self.selected, self.originals))

    def load(self) -> None:
        if self.subjects is None:
            cfg, meta, scout, baseline = self._args
            sets = load_trial_sets(Path(cfg["trial_index"]), cfg["category"], len(meta["groups"]))
            vertices = np.asarray(meta["vertices"][scout], dtype=np.int64)
            self.subjects = [load_subject(Path(cfg["cache_dir"]), s, sets[s], vertices, baseline)
                             for s in self.selected]


def run_fold(cfg: dict, meta: dict, scout: int, pair: str, repeat: int, fold: int, device: torch.device,
             verbose: bool, data: ScoutData | None = None) -> dict:
    path = fold_path(cfg, scout, pair, repeat, fold)
    sig = signature(cfg, meta)
    if cfg["resume"] and path.exists():
        saved = json.loads(path.read_text())
        if saved["signature"] == sig:
            return {**saved, "reused": True}
    if device.type == "cpu":
        torch.set_num_threads(1)
    data = data or ScoutData(cfg, meta, scout, pair)
    data.load()
    train, test = data.splits(cfg, repeat)[fold]
    started = time.time()
    spatial = fit_spatial(data.subjects, train, cfg["n_components"])
    trials = [project(s, spatial) for s in data.subjects]
    log = print if verbose else (lambda *_: None)
    log(f"scout {scout} {meta['labels'][scout]} | {pair} | repeat {repeat + 1}/{cfg['repeats']} "
        f"fold {fold + 1}/{cfg['folds']} | train {len(train)} test {len(test)} | "
        f"{spatial['basis'].shape[1]} PCs, {100 * spatial['explained']:.2f}% of scout variance")
    seed = cfg["seed"] + 7919 * repeat + 101 * fold
    predictions, history = [], []
    for k in range(cfg["n_models"]):
        if cfg["n_models"] > 1:
            log(f" model {k + 1}/{cfg['n_models']}")
        p, h = train_fold(cfg, trials, data.labels, train, test, data.window, seed + 100003 * k, device, log)
        predictions.append(p)
        history += [{"model": k + 1, **row} for row in h]
    result = {key: np.mean([p[key] for p in predictions], axis=0).tolist() for key in ("p_full", "p_tta")}
    path.parent.mkdir(parents=True, exist_ok=True)
    save_csv(path.with_name(path.stem + "_history.csv"), history)
    out = {"signature": sig, "scout": scout, "scout_label": meta["labels"][scout], "pair": pair,
           "repeat": repeat, "fold": fold, "test_subjects": data.selected[test].tolist(),
           "test_groups": data.originals[test].tolist(), "labels": data.labels[test].tolist(),
           "n_components": int(spatial["basis"].shape[1]), "explained": spatial["explained"],
           "seconds": time.time() - started, **result}
    save_json(path, out)
    return out


def fold_line(r: dict, cfg: dict, with_scout: bool) -> str:
    y = np.array(r["labels"])
    tta, full = fold_metrics(y, np.array(r["p_tta"])), fold_metrics(y, np.array(r["p_full"]))
    head = f"scout {r['scout']} {r['pair']} repeat {r['repeat'] + 1} " if with_scout else ""
    tail = "reused" if r.get("reused") else f"{r['seconds']:.0f}s"
    return (f"{head}fold {r['fold'] + 1}/{cfg['folds']} | test AUC soft vote {tta['auc']:.3f}, "
            f"all-trial {full['auc']:.3f} | bal_acc {tta['bal_acc']:.3f}, {full['bal_acc']:.3f} | {tail}")


def summarize_scout(cfg: dict, scout: int, pair: str, results: list[dict]) -> dict:
    rows, per_repeat = [], {"p_tta": [], "p_full": []}
    for repeat in range(cfg["repeats"]):
        folds = sorted((r for r in results if r["repeat"] == repeat), key=lambda r: r["fold"])
        y = np.concatenate([r["labels"] for r in folds])
        for key in per_repeat:
            per_repeat[key].append(fold_metrics(y, np.concatenate([r[key] for r in folds])))
        for r in folds:
            for j, s in enumerate(r["test_subjects"]):
                rows.append({"repeat": repeat + 1, "fold": r["fold"] + 1, "subject_index_0based": s,
                             "group_code": r["test_groups"][j], "label": r["labels"][j],
                             "p_soft_vote": r["p_tta"][j], "p_all_trial": r["p_full"][j]})
    directory = output_root(cfg) / f"scout_{scout:02d}" / pair
    save_csv(directory / "subject_predictions.csv", rows)
    summary = {"scout": scout, "scout_label": results[0]["scout_label"], "pair": pair,
               "n_subjects": len(rows) // cfg["repeats"]}
    for key, name in (("p_tta", "soft_vote"), ("p_full", "all_trial")):
        aucs = np.array([m["auc"] for m in per_repeat[key]])
        bals = np.array([m["bal_acc"] for m in per_repeat[key]])
        summary.update({f"auc_{name}_mean": float(aucs.mean()), f"auc_{name}_sd": float(aucs.std()),
                        f"bal_acc_{name}_mean": float(bals.mean()),
                        f"auc_{name}_by_repeat": [round(float(a), 4) for a in aucs]})
    save_json(directory / "summary.json", summary)
    print(f"==> scout {scout} {summary['scout_label']} | {pair} | out-of-fold AUC soft vote "
          f"{summary['auc_soft_vote_mean']:.3f} ± {summary['auc_soft_vote_sd']:.3f} "
          f"{summary['auc_soft_vote_by_repeat']} | all-trial {summary['auc_all_trial_mean']:.3f} ± "
          f"{summary['auc_all_trial_sd']:.3f}", flush=True)
    return summary


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        scratch = Path(tempfile.mkdtemp(prefix="scout_subject_smoke_"))
        args.cache_dir, args.output_dir = scratch / "cache", scratch / "output"
        args.repeats, args.epochs, args.avg_epochs = 1, 3, 2
        args.trial_index, args.category = args.cache_dir / "trial_index.pkl", "Intention"
    meta = ensure_cache(args)
    scouts = list(range(len(meta["vertices"]))) if args.scouts == "all" else args.scouts
    if max(scouts) >= len(meta["vertices"]):
        raise ValueError(f"scouts must be between 0 and {len(meta['vertices']) - 1}")
    device = choose_device("cpu" if args.jobs > 1 else args.device)
    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    print(f"Device {device} | {len(scouts)} scout(s) | pairs {', '.join(args.pairs)} | "
          f"{args.repeats} x {args.folds}-fold CV | output {output_root(cfg)}", flush=True)
    output_root(cfg).mkdir(parents=True, exist_ok=True)
    summaries = {pair: [] for pair in args.pairs}
    if args.jobs == 1:
        for pair in args.pairs:
            print(f"\n=== {pair} ===", flush=True)
            for scout in scouts:
                data = ScoutData(cfg, meta, scout, pair)
                results = []
                for repeat in range(args.repeats):
                    for fold in range(args.folds):
                        r = run_fold(cfg, meta, scout, pair, repeat, fold, device, True, data)
                        print(fold_line(r, cfg, False), flush=True)
                        results.append(r)
                summaries[pair].append(summarize_scout(cfg, scout, pair, results))
    else:
        from joblib import Parallel, delayed
        tasks = [(s, p, r, f) for p in args.pairs for s in scouts for r in range(args.repeats)
                 for f in range(args.folds)]
        collected = {}
        for r in Parallel(n_jobs=args.jobs, return_as="generator_unordered")(
                delayed(run_fold)(cfg, meta, s, p, rep, f, device, False) for s, p, rep, f in tasks):
            print(fold_line(r, cfg, True), flush=True)
            collected.setdefault((r["pair"], r["scout"]), []).append(r)
        for pair in args.pairs:
            print(f"\n=== {pair} ===", flush=True)
            for scout in scouts:
                summaries[pair].append(summarize_scout(cfg, scout, pair, collected[(pair, scout)]))
    for pair, rows in summaries.items():
        save_csv(output_root(cfg) / f"{pair}_scouts.csv",
                 [{k: (";".join(map(str, v)) if isinstance(v, list) else v) for k, v in row.items()} for row in rows])
    if args.smoke_test:
        print(f"Smoke-test output: {args.output_dir}")


if __name__ == "__main__":
    main()
