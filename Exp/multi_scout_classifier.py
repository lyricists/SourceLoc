"""Subject-level group classification from several DKT scouts combined in one network.

The PC time courses of a set of scouts are stacked as input channels (early fusion),
like giving an EEG model more channels. Every scout is preprocessed exactly as in
scout_subject_classifier.py: one orientation per vertex and --n-components shared PCs
fitted on the training subjects of each outer fold, each PC scaled by the subject's
pooled baseline SD. The outer splits and training recipe also match that script, so
predictions pair subject by subject with its single-scout runs.

Input sets (arms):
  selected   scouts whose single-scout AUC in --screen-csv exceeds --threshold
             (or --selected); picked with all subjects' labels, so optimistic
  random     --n-random random sets of the same size, drawn from all scouts
  all        every scout
  sensor     the sensor channels, each scaled by the subject's baseline SD

Models for scout inputs (--models):
  grouped    a 1x1 convolution per scout (PCs -> --per-scout virtual sources, weights
             not shared), temporal convolutions per virtual source, then one 1x1
             convolution that mixes all scouts
  flat       the single-scout EEGNet, whose first 1x1 convolution sees all stacked PCs
The sensor arm uses the single-scout EEGNet on the sensor channels.

Bootstrap trial averages are formed on the sensors and projected into the scouts one
batch at a time (projection is linear, so this equals averaging projected trials);
memory therefore does not grow with the number of scouts. Each fold's per-subject
projection matrices are computed once per scout and cached under projections/.

After training, every arm is compared with the best single scout from --screen-csv,
the random sets, all scouts, and the sensor arm (see compare()).

Examples:
    python3 multi_scout_classifier.py --smoke-test
    python3 multi_scout_classifier.py --jobs 12
    python3 multi_scout_classifier.py --arms selected all sensor --models grouped
    python3 multi_scout_classifier.py --compare-only
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
from scipy.stats import rankdata
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.optim.swa_utils import AveragedModel

import scout_subject_classifier as ssc

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARM_KINDS = ("selected", "random", "all", "sensor")
MODELS = ("grouped", "flat")
# Reference categorical slots 1-3 and neutral inks.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_SECONDARY, NEUTRAL, NEUTRAL_LIGHT = "#0b0b0b", "#52514e", "#8a8984", "#d6d5d0"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pair", choices=ssc.PAIRS, default="control-vs-depressed+suicidal")
    p.add_argument("--arms", nargs="+", choices=ARM_KINDS, default=list(ARM_KINDS))
    p.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    p.add_argument("--screen-csv", type=Path,
                   default=ssc.ROOT / "Results" / "scout_subject_cds_all" / "control-vs-depressed+suicidal_scouts.csv",
                   help="Single-scout results (scout_subject_classifier.py) used to pick scouts")
    p.add_argument("--threshold", type=float, default=0.60, help="Single-scout soft-vote AUC for selection")
    p.add_argument("--selected", nargs="+", default=None,
                   help="Zero-based scouts to use instead of the --screen-csv threshold")
    p.add_argument("--single-scout-dir", type=Path, default=ssc.ROOT / "Results" / "scout_subject_cds_all",
                   help="Single-scout predictions for the best-single-scout comparison")
    p.add_argument("--n-random", type=int, default=30)
    p.add_argument("--random-seed", type=int, default=0)
    p.add_argument("--per-scout", type=int, default=4, help="Virtual sources per scout in the grouped model")
    p.add_argument("--scout-dropout", type=float, default=0.2, help="Probability of zeroing a whole scout")
    p.add_argument("--pc-dropout", type=float, default=0.1, help="Probability of zeroing one input row")
    p.add_argument("--eeg", type=Path, default=ssc.PREPROCESSED / "Data_sen_bl.pkl")
    p.add_argument("--tmatrix", type=Path, default=ssc.PREPROCESSED / "tMatrix.pkl")
    p.add_argument("--atlas", type=Path, default=ssc.PREPROCESSED / "DKT_scout.mat")
    p.add_argument("--subject-index", type=Path, default=ssc.BEHAVIOR / "subject_index.mat")
    p.add_argument("--trial-index", type=Path, default=ssc.BEHAVIOR / "senIdx_TOI.pkl")
    p.add_argument("--category", default="Intention")
    p.add_argument("--cache-dir", type=Path, default=ssc.ROOT / "Results" / "subject_sensor_cache")
    p.add_argument("--output-dir", type=Path, default=ssc.ROOT / "Results" / "multi_scout_classifier")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
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
            args.selected = sorted({int(x) for tok in args.selected for x in tok.split(",") if x.strip()})
        except ValueError:
            p.error("--selected takes zero-based integers")
    for name in ("n_random", "per_scout", "n_components", "draws_per_epoch", "tta_avg", "tta_draws", "epochs",
                 "avg_epochs", "batch_size", "folds", "repeats", "n_bootstrap", "n_permutations", "jobs"):
        if getattr(args, name) < 1:
            p.error(f"--{name.replace('_', '-')} must be positive")
    if args.avg_epochs > args.epochs:
        p.error("--avg-epochs cannot exceed --epochs")
    if not (0 <= args.scout_dropout < 1 and 0 <= args.pc_dropout < 1):
        p.error("dropout probabilities must be in [0, 1)")
    if args.jobs > 1 and args.device not in ("auto", "cpu"):
        p.error("--jobs above 1 runs on CPU; drop --device or use --device cpu")
    return args


class Arm:
    def __init__(self, name: str, kind: str, model: str, scouts: list[int] | None) -> None:
        self.name, self.kind, self.model, self.scouts = name, kind, model, scouts


def selected_scouts(args: argparse.Namespace) -> tuple[list[int], int | None]:
    """The selected scouts and the best single scout (None when no screen CSV is used)."""
    if args.selected is not None:
        return args.selected, None
    with open(args.screen_csv, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["pair"] == args.pair]
    if not rows:
        raise ValueError(f"{args.screen_csv} has no rows for {args.pair}")
    auc = {int(r["scout"]): float(r["auc_soft_vote_mean"]) for r in rows}
    chosen = sorted(s for s, a in auc.items() if a > args.threshold)
    if len(chosen) < 2:
        raise ValueError(f"Fewer than two scouts exceed AUC {args.threshold}")
    return chosen, max(auc, key=auc.get)


def build_arms(args: argparse.Namespace, n_scouts: int, n_channels: int, selected: list[int]) -> list[Arm]:
    k = len(selected)
    rng = np.random.default_rng(args.random_seed)
    random_sets, seen = [], {tuple(selected)}
    while len(random_sets) < args.n_random:
        subset = tuple(sorted(rng.choice(n_scouts, k, replace=False).tolist()))
        if subset not in seen:
            seen.add(subset)
            random_sets.append(list(subset))
    arms = []
    for model in args.models:  # main arms first, random sets last
        if "selected" in args.arms:
            arms.append(Arm(f"{model}_selected{k}", "selected", model, selected))
        if "all" in args.arms:
            arms.append(Arm(f"{model}_all{n_scouts}", "all", model, list(range(n_scouts))))
    if "sensor" in args.arms:
        arms.append(Arm(f"sensor{n_channels}", "sensor", "eegnet", None))
    if "random" in args.arms:
        for model in args.models:
            arms += [Arm(f"{model}_random{k}_{i + 1:02d}", "random", model, s) for i, s in enumerate(random_sets)]
    return arms


def window_indices(cfg: dict, meta: dict) -> tuple[int, int, int]:
    """Window start/end sample and the time-shift margin kept on both sides."""
    times = np.asarray(meta["times_ms"])
    hits = np.flatnonzero((times >= cfg["window"][0]) & (times < cfg["window"][1]))
    i0, i1 = int(hits[0]), int(hits[-1]) + 1
    return i0, i1, min(cfg["max_shift"], i0, len(times) - i1)


def make_splits(cfg: dict, originals: np.ndarray) -> list[list]:
    """The outer splits of scout_subject_classifier.py (same seeds, same stratification)."""
    return [list(StratifiedKFold(cfg["folds"], shuffle=True, random_state=cfg["seed"] + 1009 * r)
                 .split(originals, originals)) for r in range(cfg["repeats"])]


def load_subjects(cfg: dict, meta: dict, selected: np.ndarray, trials: bool = True) -> list[dict]:
    """Baseline-corrected sensor trials (cropped to the window plus shift margins) and covariances."""
    sets = ssc.load_trial_sets(Path(cfg["trial_index"]), cfg["category"], len(meta["groups"]))
    baseline = np.asarray(meta["times_ms"]) < ssc.BASELINE_END_MS
    i0, i1, shift = window_indices(cfg, meta)
    subjects = []
    for s in selected:
        subject = ssc.load_subject(Path(cfg["cache_dir"]), s, sets[s], np.array([0]), baseline)
        entry = {"cov": subject["cov"], "base_cov": subject["base_cov"]}
        if trials:
            entry["trials"] = [np.ascontiguousarray(t[..., i0 - shift:i1 + shift]) for t in subject["trials"]]
        subjects.append(entry)
    return subjects


def projection_signature(cfg: dict, meta: dict) -> dict:
    return {"cache": meta["signature"], "trial_index": ssc.file_signature(cfg["trial_index"]), "pair": cfg["pair"],
            **{k: cfg[k] for k in ("category", "n_components", "folds", "repeats", "seed")}}


def projection_path(cfg: dict, scout: int, repeat: int, fold: int) -> Path:
    return Path(cfg["output_dir"]) / "projections" / f"scout_{scout:02d}" / f"repeat{repeat + 1}_fold{fold + 1}.npy"


def compute_projections(cfg: dict, meta: dict, scout: int, covs: list[dict], splits: list) -> int:
    """Per fold: orientation and PCs from training subjects, then every subject's PC x channel matrix."""
    selected = ssc.select_pair(np.asarray(meta["groups"]), cfg["pair"])[0]
    vertices = np.asarray(meta["vertices"][scout], dtype=np.int64)
    rows = (3 * vertices[:, None] + np.arange(3)).ravel()
    subjects = []
    for s, cov in zip(selected, covs):
        kernel = np.load(Path(cfg["cache_dir"]) / f"kernel_{s:03d}.npy", mmap_mode="r")[rows]
        subjects.append({"kernel": np.asarray(kernel, dtype=np.float64).reshape(len(vertices), 3, -1), **cov})
    for repeat in range(cfg["repeats"]):
        for fold in range(cfg["folds"]):
            spatial = ssc.fit_spatial(subjects, splits[repeat][fold][0], cfg["n_components"])
            m = np.zeros((len(subjects), cfg["n_components"], subjects[0]["kernel"].shape[-1]), dtype=np.float32)
            for i, subject in enumerate(subjects):
                p = ssc.projection(subject, spatial)
                m[i, :len(p)] = p  # scouts with fewer vertices than PCs keep zero rows
            path = projection_path(cfg, scout, repeat, fold)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp.npy")
            np.save(tmp, m)
            os.replace(tmp, path)
    return scout


def ensure_projections(cfg: dict, meta: dict, scouts: list[int], jobs: int) -> None:
    sig_path = Path(cfg["output_dir"]) / "projections" / "signature.json"
    sig = projection_signature(cfg, meta)
    valid = cfg["resume"] and sig_path.exists() and json.loads(sig_path.read_text()) == sig
    todo = [s for s in scouts if not valid or not all(projection_path(cfg, s, r, f).exists()
                                                       for r in range(cfg["repeats"]) for f in range(cfg["folds"]))]
    if not todo:
        return
    print(f"Computing fold projections for {len(todo)} scout(s)...", flush=True)
    groups = np.asarray(meta["groups"])
    selected, _, originals = ssc.select_pair(groups, cfg["pair"])
    covs = load_subjects(cfg, meta, selected, trials=False)
    splits = make_splits(cfg, originals)
    sig_path.parent.mkdir(parents=True, exist_ok=True)
    if sig_path.exists() and not valid:
        sig_path.unlink()  # mark the cache stale until every requested scout is rebuilt
    from joblib import Parallel, delayed
    for done in Parallel(n_jobs=jobs, return_as="generator_unordered")(
            delayed(compute_projections)(cfg, meta, s, covs, splits) for s in todo):
        print(f"  scout {done} {meta['labels'][done]}", flush=True)
    ssc.save_json(sig_path, sig)


class GroupedScoutEEGNet(nn.Module):
    """Per-scout spatial filters, temporal filters per virtual source, then cross-scout mixing."""

    def __init__(self, n_scouts: int, n_pcs: int, n_times: int, per_scout: int) -> None:
        super().__init__()
        c = n_scouts * per_scout
        self.features = nn.Sequential(
            nn.Conv1d(n_scouts * n_pcs, c, 1, groups=n_scouts, bias=False),
            nn.Conv1d(c, c, 31, padding=15, groups=c, bias=False), nn.GroupNorm(n_scouts, c), nn.ELU(),
            nn.AvgPool1d(4), nn.Dropout(0.25),
            nn.Conv1d(c, c, 15, padding=7, groups=c, bias=False), nn.Conv1d(c, 16, 1, bias=False),
            nn.GroupNorm(4, 16), nn.ELU(), nn.AvgPool1d(8), nn.Dropout(0.25), nn.Flatten())
        width = 16 * (n_times // 32)
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Dropout(0.3), nn.Linear(width, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def build_model(arm: Arm, cfg: dict, n_rows: int, n_times: int) -> nn.Module:
    if arm.model == "grouped":
        return GroupedScoutEEGNet(len(arm.scouts), cfg["n_components"], n_times, cfg["per_scout"])
    return ssc.EEGNet(n_rows, n_times)


def predict(model: nn.Module, x: np.ndarray, owner: np.ndarray, position: np.ndarray, n_test: int,
            m: torch.Tensor, device: torch.device) -> np.ndarray:
    """Mean probability per test subject; ``owner`` indexes ``m``, ``position`` the test subject."""
    model.eval()
    with torch.no_grad():
        p = np.concatenate([
            torch.softmax(model(torch.bmm(m[owner[k:k + 128]], torch.from_numpy(x[k:k + 128]).to(device))), 1)[:, 1]
            .cpu().numpy() for k in range(0, len(x), 128)])
    return np.array([p[position == j].mean() for j in range(n_test)])


def train_fold(cfg: dict, arm: Arm, subjects: list[dict], m_np: np.ndarray, labels: np.ndarray,
               train: np.ndarray, test: np.ndarray, n_times: int, shift: int, seed: int,
               device: torch.device) -> tuple[dict, list[dict], int]:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    lo, hi = cfg["avg_range"]
    draws = cfg["draws_per_epoch"]
    m = torch.from_numpy(m_np).to(device)
    n_rows = m_np.shape[1]
    n_scouts = 0 if arm.scouts is None else len(arm.scouts)

    def contrasts(members: np.ndarray, n: int, low: int, high: int) -> np.ndarray:
        """Incongruent - congruent bootstrap averages of sensor trials, n per subject."""
        out = []
        for i in members:
            sizes = rng.integers(low, high + 1, size=n)
            a, b = (ssc.bootstrap_weights(rng, len(t), sizes) @ t.reshape(len(t), -1) for t in subjects[i]["trials"])
            out.append((b - a).reshape(n, *subjects[i]["trials"][0].shape[1:]))
        return np.concatenate(out)

    owner_train = np.repeat(train, draws)
    y_train = torch.from_numpy(np.repeat(labels[train], draws)).to(device)
    x_full = np.stack([s["trials"][1].mean(0) - s["trials"][0].mean(0) for s in (subjects[i] for i in test)])
    x_full = np.ascontiguousarray(x_full[..., shift:shift + n_times])
    x_tta = np.ascontiguousarray(contrasts(test, cfg["tta_draws"], cfg["tta_avg"], cfg["tta_avg"])
                                 [..., shift:shift + n_times])
    position_tta = np.repeat(np.arange(len(test)), cfg["tta_draws"])

    model = build_model(arm, cfg, n_rows, n_times).to(device)
    n_parameters = sum(p.numel() for p in model.parameters())
    averaged = AveragedModel(model)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps = int(np.ceil(len(owner_train) / cfg["batch_size"]))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=cfg["epochs"] * steps,
                                                pct_start=0.15)
    counts = np.bincount(labels[train], minlength=2)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(len(train) / (2 * counts), dtype=torch.float32, device=device),
                                  label_smoothing=cfg["label_smoothing"])
    history = []
    for epoch in range(1, cfg["epochs"] + 1):
        started = time.time()
        x = contrasts(train, draws, lo, hi)
        n = len(x)
        shifts = rng.integers(-shift, shift + 1, size=n)
        x = np.take_along_axis(x, (shift + shifts).reshape(n, 1, 1) + np.arange(n_times), axis=-1)
        x *= np.exp(rng.normal(0, 0.1, size=(n, 1, 1))).astype(np.float32)
        keep = rng.random((n, n_rows)) > cfg["pc_dropout"]
        if n_scouts:
            keep &= np.repeat(rng.random((n, n_scouts)) > cfg["scout_dropout"], cfg["n_components"], axis=1)
        keep = torch.from_numpy(keep.astype(np.float32)).to(device)
        x = torch.from_numpy(np.ascontiguousarray(x)).to(device)
        order = rng.permutation(n)
        model.train()
        loss_sum = torch.zeros((), device=device)
        correct = torch.zeros(2, device=device)
        for k in range(steps):
            idx = torch.from_numpy(order[k * cfg["batch_size"]:(k + 1) * cfg["batch_size"]]).to(device)
            owners = torch.from_numpy(owner_train[idx.cpu().numpy()]).to(device)
            xb = torch.bmm(m[owners], x[idx]) * keep[idx][:, :, None]
            logits = model(xb)
            target = y_train[idx]
            loss = loss_fn(logits, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.detach() * len(idx)
            hit = logits.detach().argmax(1) == target
            correct += torch.stack(((hit & (target == 0)).sum(), (hit & (target == 1)).sum()))
        if epoch > cfg["epochs"] - cfg["avg_epochs"]:
            averaged.update_parameters(model)
        totals = counts * draws
        correct = correct.cpu().numpy()
        history.append({"epoch": epoch, "lr": opt.param_groups[0]["lr"], "train_loss": float(loss_sum.item() / n),
                        "train_acc": float(correct.sum() / totals.sum()),
                        "train_bal_acc": float(np.mean(correct / totals)),
                        "averaged": epoch > cfg["epochs"] - cfg["avg_epochs"], "seconds": time.time() - started})
    net = averaged.module
    result = {"p_full": predict(net, x_full, test, np.arange(len(test)), len(test), m, device).tolist(),
              "p_tta": predict(net, x_tta, np.repeat(test, cfg["tta_draws"]), position_tta, len(test), m,
                               device).tolist()}
    return result, history, n_parameters


def signature(cfg: dict, meta: dict, arm: Arm) -> dict:
    keys = ("category", "n_components", "window", "avg_range", "draws_per_epoch", "tta_avg", "tta_draws", "epochs",
            "avg_epochs", "batch_size", "lr", "weight_decay", "label_smoothing", "max_shift", "folds", "seed",
            "per_scout", "scout_dropout", "pc_dropout")
    return {"cache": meta["signature"], "trial_index": ssc.file_signature(cfg["trial_index"]), "pair": cfg["pair"],
            "arm": arm.name, "model": arm.model, "scouts": arm.scouts, **{k: cfg[k] for k in keys}}


def fold_path(cfg: dict, arm: Arm, repeat: int, fold: int) -> Path:
    return Path(cfg["output_dir"]) / "arms" / arm.name / f"repeat{repeat + 1}_fold{fold + 1}.json"


def run_fold(cfg: dict, meta: dict, arm: Arm, repeat: int, fold: int, device: torch.device) -> dict:
    path = fold_path(cfg, arm, repeat, fold)
    sig = signature(cfg, meta, arm)
    if cfg["resume"] and path.exists():
        saved = json.loads(path.read_text())
        if saved["signature"] == sig:
            return {**saved, "reused": True}
    if device.type == "cpu":
        torch.set_num_threads(1)
    started = time.time()
    selected, labels, originals = ssc.select_pair(np.asarray(meta["groups"]), cfg["pair"])
    subjects = load_subjects(cfg, meta, selected)
    train, test = make_splits(cfg, originals)[repeat][fold]
    i0, i1, shift = window_indices(cfg, meta)
    if arm.scouts is None:
        m = np.stack([np.diag(1 / np.sqrt(np.maximum(np.diag(s["base_cov"]), 1e-30))) for s in subjects])
    else:
        m = np.concatenate([np.load(projection_path(cfg, s, repeat, fold)) for s in arm.scouts], axis=1)
    seed = cfg["seed"] + 7919 * repeat + 101 * fold  # the same seed for every arm (paired comparisons)
    result, history, n_parameters = train_fold(cfg, arm, subjects, m.astype(np.float32), labels, train, test,
                                               i1 - i0, shift, seed, device)
    path.parent.mkdir(parents=True, exist_ok=True)
    ssc.save_csv(path.with_name(path.stem + "_history.csv"), history)
    out = {"signature": sig, "arm": arm.name, "kind": arm.kind, "model": arm.model, "scouts": arm.scouts,
           "repeat": repeat, "fold": fold, "test_subjects": selected[test].tolist(),
           "test_groups": originals[test].tolist(), "labels": labels[test].tolist(), "n_input_rows": int(m.shape[1]),
           "n_parameters": int(n_parameters), "seconds": time.time() - started, **result}
    ssc.save_json(path, out)
    return out


# ---------------------------------------------------------------- comparison

def auc(y: np.ndarray, p: np.ndarray) -> float:
    """Mann-Whitney ROC AUC (ties count one half)."""
    n1 = int(y.sum())
    return float((rankdata(p)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def load_arm(cfg: dict, arm: Arm, position: dict, n: int) -> tuple[np.ndarray, list] | None:
    """Repeat x subject soft-vote probabilities, or None when a fold is missing."""
    probs = np.full((cfg["repeats"], n), np.nan)
    for repeat in range(cfg["repeats"]):
        for fold in range(cfg["folds"]):
            path = fold_path(cfg, arm, repeat, fold)
            if not path.exists():
                return None
            r = json.loads(path.read_text())
            probs[repeat, [position[s] for s in r["test_subjects"]]] = r["p_tta"]
    return probs


def load_single_scout(cfg: dict, scout: int, position: dict, fold_of: np.ndarray) -> np.ndarray | None:
    path = Path(cfg["single_scout_dir"]) / f"scout_{scout:02d}" / cfg["pair"] / "subject_predictions.csv"
    if not path.exists():
        return None
    probs = np.full(fold_of.shape, np.nan)
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            r, j = int(row["repeat"]) - 1, position[int(row["subject_index_0based"])]
            if r < len(probs):
                if int(row["fold"]) - 1 != fold_of[r, j]:
                    raise ValueError(f"{path}: outer splits differ from this script's splits")
                probs[r, j] = float(row["p_soft_vote"])
    return None if np.isnan(probs).any() else probs


def metrics(y: np.ndarray, q: np.ndarray, permuted: np.ndarray) -> dict:
    aucs = np.array([auc(y, qr) for qr in q])
    n1 = int(y.sum())
    ranks = np.stack([rankdata(qr) for qr in q])
    null = ((permuted @ ranks.T - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))).mean(1)
    return {"auc_mean": float(aucs.mean()), "auc_sd": float(aucs.std()), "auc_by_repeat": aucs.round(4).tolist(),
            "bal_acc_mean": float(np.mean([balanced_accuracy_score(y, (qr >= 0.5).astype(int)) for qr in q])),
            "p_vs_chance": float((1 + np.sum(null >= aucs.mean() - 1e-12)) / (1 + len(null)))}


def holm(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted, running = np.empty(len(p_values)), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[i])
        adjusted[i] = min(1.0, running)
    return adjusted.tolist()


def compare(cfg: dict, meta: dict, arms: list[Arm], best_single: int | None) -> None:
    groups = np.asarray(meta["groups"])
    selected, labels, originals = ssc.select_pair(groups, cfg["pair"])
    position = {int(s): j for j, s in enumerate(selected)}
    fold_of = np.full((cfg["repeats"], len(selected)), -1)
    for r, folds in enumerate(make_splits(cfg, originals)):
        for f, (_, test) in enumerate(folds):
            fold_of[r, test] = f
    rng = np.random.default_rng(cfg["random_seed"] + 1)
    permuted = np.stack([rng.permutation(labels) for _ in range(cfg["n_permutations"])]).astype(np.float64)
    by_group = [np.flatnonzero(originals == g) for g in np.unique(originals)]
    boot = [np.concatenate([rng.choice(ix, len(ix), replace=True) for ix in by_group])
            for _ in range(cfg["n_bootstrap"])]

    probs, missing = {}, []
    for arm in arms:
        loaded = load_arm(cfg, arm, position, len(selected))
        if loaded is None:
            missing.append(arm.name)
        else:
            probs[arm.name] = loaded
    if missing:
        print(f"Comparison skipped; arms not finished: {', '.join(missing[:6])}{' ...' if len(missing) > 6 else ''}")
        return
    single_name = None
    if best_single is not None:
        single = load_single_scout(cfg, best_single, position, fold_of)
        if single is None:
            print(f"No complete single-scout predictions for scout {best_single}; that comparison is skipped")
        else:
            single_name = f"single_scout{best_single:02d}"
            probs[single_name] = single

    def boot_mean_auc(q: np.ndarray, idx: np.ndarray) -> float:
        return float(np.mean([auc(labels[idx], qr[idx]) for qr in q]))

    summary_rows, comparison_rows, panels = [], [], []
    sensor = next((a.name for a in arms if a.kind == "sensor"), None)
    shared = {name: metrics(labels, probs[name], permuted) for name in (single_name, sensor) if name}
    for model in cfg["models"]:
        main = {a.kind: a.name for a in arms if a.model == model and a.kind in ("selected", "all")}
        randoms = [a.name for a in arms if a.model == model and a.kind == "random"]
        if "selected" not in main:
            continue
        sel = main["selected"]
        arm_metrics = {name: metrics(labels, probs[name], permuted) for name in main.values()}
        arm_metrics.update(shared)
        random_aucs = np.array([np.mean([auc(labels, qr) for qr in probs[name]]) for name in randoms])
        target = arm_metrics[sel]["auc_mean"]
        rows = []
        if len(random_aucs):
            rows.append({"model": model, "comparison": f"{sel} vs random sets ({len(random_aucs)})",
                         "method": "percentile among random sets",
                         "difference": target - float(np.median(random_aucs)), "ci_low": "", "ci_high": "",
                         "percentile": float(100 * np.mean(random_aucs < target)),
                         "p_one_sided": float((1 + np.sum(random_aucs >= target - 1e-12)) / (1 + len(random_aucs)))})
        for other in (single_name, main.get("all"), sensor):
            if other is None:
                continue
            d = np.array([boot_mean_auc(probs[sel], idx) - boot_mean_auc(probs[other], idx) for idx in boot])
            rows.append({"model": model, "comparison": f"{sel} vs {other}", "method": "paired subject bootstrap",
                         "difference": target - arm_metrics[other]["auc_mean"],
                         "ci_low": float(np.percentile(d, 2.5)), "ci_high": float(np.percentile(d, 97.5)),
                         "percentile": "", "p_one_sided": float((1 + np.sum(d <= 0)) / (1 + len(d)))})
        for row, adjusted in zip(rows, holm([row["p_one_sided"] for row in rows])):
            row["p_holm"] = adjusted
        comparison_rows += rows
        for name, values in arm_metrics.items():
            summary_rows.append({"model": model, "arm": name, **{k: (";".join(map(str, v)) if isinstance(v, list) else v)
                                                                 for k, v in values.items()}})
        if len(random_aucs):
            summary_rows.append({"model": model, "arm": f"{model}_random (median of {len(random_aucs)})",
                                 "auc_mean": float(np.median(random_aucs)), "auc_sd": float(random_aucs.std()),
                                 "auc_by_repeat": "", "bal_acc_mean": "", "p_vs_chance": ""})
        panels.append((model, sel, main.get("all"), arm_metrics, random_aucs))

    out = Path(cfg["output_dir"])
    if not summary_rows:
        print("No selected arm to compare")
        return
    ssc.save_csv(out / "arms_summary.csv", summary_rows)
    if comparison_rows:
        ssc.save_csv(out / "comparisons.csv", comparison_rows)
    ssc.save_json(out / "comparison_summary.json", {
        "best_single_scout_0based": best_single, "arms": summary_rows, "comparisons": comparison_rows,
        "n_bootstrap": cfg["n_bootstrap"], "n_permutations": cfg["n_permutations"]})
    plot_comparison(out / "multi_scout_comparison.pdf", panels, single_name, sensor, meta, best_single)

    print("\nArm                              AUC mean ± SD   by repeat                 bal acc  p vs chance")
    for row in summary_rows:
        if row["auc_by_repeat"]:
            print(f"{row['arm']:<32} {row['auc_mean']:.3f} ± {row['auc_sd']:.3f}   {row['auc_by_repeat']:<25} "
                  f"{row['bal_acc_mean']:.3f}    {row['p_vs_chance']:.4f}")
        else:
            print(f"{row['arm']:<32} {row['auc_mean']:.3f} ± {row['auc_sd']:.3f}")
    if comparison_rows:
        print("\nComparison                                   diff     95% CI             p (one-sided)  p (Holm)")
        for row in comparison_rows:
            ci = "" if row["ci_low"] == "" else f"[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]"
            extra = "" if row["percentile"] == "" else f"  ({row['percentile']:.0f}th percentile)"
            print(f"{row['comparison']:<44} {row['difference']:+.3f}   {ci:<18} {row['p_one_sided']:.4f}         "
                  f"{row['p_holm']:.4f}{extra}")
    print(f"\nComparison: {out / 'comparisons.csv'}")


def plot_comparison(path: Path, panels: list, single_name: str | None, sensor: str | None, meta: dict,
                    best_single: int | None) -> None:
    fig, axes = plt.subplots(1, len(panels), figsize=(7.5 * len(panels), 4.8), sharex=True, squeeze=False)
    for axis, (model, sel, all_name, arm_metrics, random_aucs) in zip(axes[0], panels):
        rows = []
        if single_name:
            rows.append((single_name, f"Best single scout ({best_single + 1:02d} {meta['labels'][best_single]})", AQUA))
        rows.append((sel, f"Selected scouts ({sel.rsplit('selected', 1)[1]})", BLUE))
        if len(random_aucs):
            rows.append(("random", f"Random sets ({len(random_aucs)})", NEUTRAL))
        if all_name:
            rows.append((all_name, f"All scouts ({all_name.rsplit('all', 1)[1]})", ORANGE))
        if sensor:
            rows.append((sensor, f"Sensor channels ({sensor[len('sensor'):]})", INK_SECONDARY))
        for y_pos, (name, _, color) in enumerate(rows[::-1]):
            if name == "random":
                jitter = np.random.default_rng(0).uniform(-0.12, 0.12, len(random_aucs))
                axis.plot(random_aucs, y_pos + jitter, "o", color=NEUTRAL_LIGHT, markersize=5, markeredgewidth=0)
                median = float(np.median(random_aucs))
                axis.plot(median, y_pos, "o", color=NEUTRAL, markersize=10, markeredgecolor="white",
                          markeredgewidth=1.5)
                axis.text(random_aucs.max() + 0.006, y_pos, f"median {median:.3f}", va="center", fontsize=9,
                          color=INK_SECONDARY)
                continue
            values = arm_metrics[name]
            axis.plot(values["auc_by_repeat"], [y_pos] * len(values["auc_by_repeat"]), "o", color=color,
                      markersize=6, alpha=0.45, markeredgewidth=0)
            axis.plot(values["auc_mean"], y_pos, "o", color=color, markersize=10, markeredgecolor="white",
                      markeredgewidth=1.5)
            axis.text(max(values["auc_by_repeat"]) + 0.006, y_pos, f"{values['auc_mean']:.3f}", va="center",
                      fontsize=10, color=INK)
        axis.set_yticks(range(len(rows)), [r[1] for r in rows[::-1]], fontsize=9, color=INK)
        axis.axvline(0.5, color=INK_SECONDARY, linestyle=":", linewidth=1)
        axis.set_title(f"{model} multi-scout model", fontsize=12, color=INK, loc="left")
        axis.set_xlabel("Out-of-fold ROC AUC (small: each repeat or random set; large: mean / median)",
                        fontsize=9, color=INK)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(colors=INK_SECONDARY, labelsize=9)
        axis.grid(axis="x", color="#ececea", linewidth=0.7)
        axis.set_axisbelow(True)
    fig.tight_layout()
    tmp = path.with_suffix(".tmp.pdf")
    fig.savefig(tmp, format="pdf")
    plt.close(fig)
    os.replace(tmp, path)


def make_smoke_cache(args: argparse.Namespace) -> None:
    """Synthetic subjects from scout_subject_classifier.py, with the 30 vertices split into 6 scouts."""
    scratch = Path(tempfile.mkdtemp(prefix="multi_scout_smoke_"))
    args.cache_dir, args.output_dir = scratch / "cache", scratch / "output"
    ssc.build_synthetic_cache(args.cache_dir)
    meta = json.loads((args.cache_dir / "metadata.json").read_text())
    meta["vertices"] = [list(range(i, i + 5)) for i in range(0, 30, 5)]
    meta["labels"] = [f"synthetic {i}" for i in range(6)]
    ssc.save_json(args.cache_dir / "metadata.json", meta)
    args.trial_index, args.category = args.cache_dir / "trial_index.pkl", "Intention"
    args.selected = args.selected or [0, 1, 2]
    args.n_random, args.repeats, args.epochs, args.avg_epochs = 2, 1, 3, 2
    args.n_bootstrap, args.n_permutations = 200, 200


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        make_smoke_cache(args)
    meta = ssc.ensure_cache(args)
    n_scouts = len(meta["vertices"])
    n_channels = int(np.load(args.cache_dir / "sensor_000.npy", mmap_mode="r").shape[1])
    selected, best_single = selected_scouts(args)
    if max(selected) >= n_scouts:
        raise ValueError(f"selected scouts must be between 0 and {n_scouts - 1}")
    arms = build_arms(args, n_scouts, n_channels, selected)
    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    device = ssc.choose_device("cpu" if args.jobs > 1 else args.device)
    print(f"{args.pair} | selected scouts (0-based): {selected} | {len(arms)} arms | "
          f"{args.repeats} x {args.folds}-fold CV | output {args.output_dir}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ssc.save_json(args.output_dir / "arms.json", [vars(a) for a in arms])

    if not args.compare_only:
        scouts = sorted({s for a in arms if a.scouts for s in a.scouts})
        if scouts:
            ensure_projections(cfg, meta, scouts, args.jobs)
        tasks = [(a, r, f) for a in arms for r in range(args.repeats) for f in range(args.folds)]
        print(f"Training {len(tasks)} folds on {device}...", flush=True)

        def line(r: dict) -> str:
            y = np.array(r["labels"])
            tail = "reused" if r.get("reused") else f"{r['seconds']:.0f}s"
            return (f"{r['arm']} repeat {r['repeat'] + 1} fold {r['fold'] + 1} | {r['n_input_rows']} rows, "
                    f"{r['n_parameters']} params | test AUC soft vote {auc(y, np.array(r['p_tta'])):.3f}, "
                    f"all-trial {auc(y, np.array(r['p_full'])):.3f} | {tail}")

        if args.jobs == 1:
            for a, r, f in tasks:
                print(line(run_fold(cfg, meta, a, r, f, device)), flush=True)
        else:
            from joblib import Parallel, delayed
            for result in Parallel(n_jobs=args.jobs, return_as="generator_unordered")(
                    delayed(run_fold)(cfg, meta, a, r, f, device) for a, r, f in tasks):
                print(line(result), flush=True)
    compare(cfg, meta, arms, best_single)
    if args.smoke_test:
        print(f"Smoke-test output: {args.output_dir}")


if __name__ == "__main__":
    main()
