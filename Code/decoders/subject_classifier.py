"""Subject-level group classification with EEGNet: one scout, or several scouts stacked.

Both runners share the outer splits of ``utils.cv.outer_splits``, fit every
subject-pooling step on the training subjects of a fold, train for a fixed number of
epochs with a one-cycle learning rate, average the weights of the final epochs, and
score test subjects by a soft vote over bootstrap averages. Each fold is saved as JSON
with a signature of its settings and inputs and is reused when the signature matches.

Single scout: trials are projected to the scout's PCs once per fold.
Multi scout (and sensors): bootstrap averages are formed on the sensors and projected
into the stacked scout PCs one batch at a time (projection is linear, so this equals
averaging projected trials), keeping memory independent of the number of scouts.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.metrics import balanced_accuracy_score, log_loss, roc_auc_score
from torch import nn
from torch.optim.swa_utils import AveragedModel

from models.eegnet import EEGNet
from utils import cv, io, source

SIGNATURE_KEYS = ("n_components", "window", "avg_range", "draws_per_epoch", "tta_avg", "tta_draws", "epochs",
                  "avg_epochs", "batch_size", "lr", "weight_decay", "label_smoothing", "max_shift", "folds", "seed",
                  "category")
# Settings older runs stored that the current code fixes (value they must have had).
SINGLE_LEGACY = {"representation": "contrast", "n_models": 1}
MULTI_LEGACY = {"per_scout": 4}


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS was requested but is unavailable")
    return torch.device(requested)


def fold_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return {"loss": float(log_loss(y, p, labels=[0, 1])), "auc": float(roc_auc_score(y, p)),
            "bal_acc": float(balanced_accuracy_score(y, (p >= 0.5).astype(int)))}


def fold_line(r: dict, with_name: bool = True) -> str:
    y = np.array(r["labels"])
    tta, full = fold_metrics(y, np.array(r["p_tta"])), fold_metrics(y, np.array(r["p_full"]))
    name = r.get("arm") or f"scout {r['scout']} {r['pair']}"
    head = f"{name} repeat {r['repeat'] + 1} fold {r['fold'] + 1} | " if with_name else f"fold {r['fold'] + 1} | "
    tail = "reused" if r.get("reused") else f"{r['seconds']:.0f}s"
    return (f"{head}test AUC soft vote {tta['auc']:.3f}, all-trial {full['auc']:.3f} | "
            f"bal_acc {tta['bal_acc']:.3f}, {full['bal_acc']:.3f} | {tail}")


def load_saved(path: Path, sig: dict, legacy: dict, resume: bool) -> dict | None:
    if resume and path.exists():
        saved = json.loads(path.read_text())
        if io.signatures_match(saved["signature"], sig, legacy):
            return {**saved, "reused": True}
    return None


# ---------------------------------------------------------------- single scout

def contrast_input(trials: list[np.ndarray], weights: list[np.ndarray]) -> np.ndarray:
    """Incongruent - congruent weighted averages, one example per weight row (example x PC x time)."""
    a, b = ((w @ t.reshape(len(t), -1)).reshape(len(w), *t.shape[1:]) for w, t in zip(weights, trials))
    return b - a


def draw_contrasts(trials: list[np.ndarray], rng: np.random.Generator, sizes: np.ndarray) -> np.ndarray:
    return contrast_input(trials, [cv.bootstrap_weights(rng, len(t), sizes) for t in trials])


def predict(model: nn.Module, x: np.ndarray, owner: np.ndarray, n_subjects: int, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        p = np.concatenate([torch.softmax(model(torch.from_numpy(x[k:k + 256]).to(device)), 1)[:, 1].cpu().numpy()
                            for k in range(0, len(x), 256)])
    return np.array([p[owner == j].mean() for j in range(n_subjects)])


def train_single_scout(cfg: dict, trials: list[list[np.ndarray]], labels: np.ndarray, train: np.ndarray,
                       test: np.ndarray, window: tuple[int, int], seed: int, device: torch.device,
                       log) -> tuple[dict, list[dict]]:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    i0, i1 = window
    n_times = i1 - i0
    shift_max = min(cfg["max_shift"], i0, trials[0][0].shape[-1] - i1)
    lo, hi = cfg["avg_range"]
    draws = cfg["draws_per_epoch"]

    def subject_draws(subjects, n, low, high):
        return np.concatenate([draw_contrasts(trials[i], rng, rng.integers(low, high + 1, size=n)) for i in subjects])

    def training_set():
        x = subject_draws(train, draws, lo, hi)
        n = len(x)
        shifts = rng.integers(-shift_max, shift_max + 1, size=n)
        x = np.take_along_axis(x, (i0 + shifts).reshape(n, 1, 1) + np.arange(n_times), axis=-1)
        x *= np.exp(rng.normal(0, 0.1, size=(n, 1, 1))).astype(np.float32)
        x *= (rng.random((n, x.shape[-2], 1)) > 0.1).astype(np.float32)
        return x

    # One unused set of draws keeps the random stream identical to the screening runs.
    subject_draws(train, draws, lo, hi)
    y_train = torch.from_numpy(np.repeat(labels[train], draws)).to(device)
    y_test = labels[test]
    x_full = np.stack([contrast_input(trials[i], [np.full((1, len(t)), 1 / len(t), dtype=np.float32)
                                                  for t in trials[i]])[0] for i in test])[..., i0:i1]
    monitor_rng = np.random.default_rng([seed, 1])
    x_monitor = np.concatenate([draw_contrasts(trials[i], monitor_rng, np.full(cfg["tta_draws"], cfg["tta_avg"]))
                                for i in test])[..., i0:i1]
    owner_tta = np.repeat(np.arange(len(test)), cfg["tta_draws"])

    model = EEGNet(trials[0][0].shape[1], n_times).to(device)
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
        totals = counts * draws
        correct = correct.cpu().numpy()
        # Test metrics per epoch are only a monitor (separate draws); nothing is selected from them.
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


class ScoutData:
    """One scout's subjects for a pair, loaded on first use."""

    def __init__(self, cfg: dict, meta: dict, scout: int, pair: str) -> None:
        groups = np.asarray(meta["groups"])
        self.selected, self.labels, originals = source.select_pair(groups, pair)
        if any(np.sum(originals == g) < cfg["folds"] for g in np.unique(originals)):
            raise ValueError(f"{pair}: every group needs at least {cfg['folds']} subjects")
        if cfg["permute_seed"] is not None:
            order = np.random.default_rng(cfg["permute_seed"]).permutation(len(self.selected))
            self.labels, originals = self.labels[order], originals[order]
        self.originals = originals
        times = np.asarray(meta["times_ms"])
        hits = np.flatnonzero((times >= cfg["window"][0]) & (times < cfg["window"][1]))
        self.window = (int(hits[0]), int(hits[-1]) + 1)
        self._args = (cfg, meta, scout, times < source.BASELINE_END_MS)
        self.subjects = None

    def splits(self, cfg: dict, repeat: int):
        return cv.outer_splits(self.originals, cfg["folds"], repeat + 1, cfg["seed"])[repeat]

    def load(self) -> None:
        if self.subjects is None:
            cfg, meta, scout, baseline = self._args
            sets = source.load_trial_sets(Path(cfg["trial_index"]), cfg["category"], len(meta["groups"]))
            vertices = np.asarray(meta["vertices"][scout], dtype=np.int64)
            self.subjects = [source.load_subject(Path(cfg["cache_dir"]), s, sets[s], vertices, baseline)
                             for s in self.selected]


def scout_signature(cfg: dict, meta: dict) -> dict:
    return {"cache": meta["signature"], "trial_index": io.file_signature(cfg["trial_index"]),
            **{k: cfg[k] for k in SIGNATURE_KEYS + ("permute_seed",)}}


def scout_output_root(cfg: dict) -> Path:
    root = Path(cfg["output_dir"])
    return root if cfg["permute_seed"] is None else root / f"permutation_{cfg['permute_seed']:04d}"


def scout_fold_path(cfg: dict, scout: int, pair: str, repeat: int, fold: int) -> Path:
    return scout_output_root(cfg) / f"scout_{scout:02d}" / pair / f"repeat{repeat + 1}_fold{fold + 1}.json"


def run_scout_fold(cfg: dict, meta: dict, scout: int, pair: str, repeat: int, fold: int, device: torch.device,
                   verbose: bool, data: ScoutData | None = None) -> dict:
    path = scout_fold_path(cfg, scout, pair, repeat, fold)
    sig = scout_signature(cfg, meta)
    saved = load_saved(path, sig, SINGLE_LEGACY, cfg["resume"])
    if saved is not None:
        return saved
    if device.type == "cpu":
        torch.set_num_threads(1)
    data = data or ScoutData(cfg, meta, scout, pair)
    data.load()
    train, test = data.splits(cfg, repeat)[fold]
    started = time.time()
    spatial = source.fit_spatial(data.subjects, train, cfg["n_components"])
    trials = [source.project(s, spatial) for s in data.subjects]
    log = print if verbose else (lambda *_: None)
    log(f"scout {scout} {meta['labels'][scout]} | {pair} | repeat {repeat + 1}/{cfg['repeats']} "
        f"fold {fold + 1}/{cfg['folds']} | train {len(train)} test {len(test)} | "
        f"{spatial['basis'].shape[1]} PCs, {100 * spatial['explained']:.2f}% of scout variance")
    seed = cfg["seed"] + 7919 * repeat + 101 * fold
    result, history = train_single_scout(cfg, trials, data.labels, train, test, data.window, seed, device, log)
    path.parent.mkdir(parents=True, exist_ok=True)
    io.save_csv(path.with_name(path.stem + "_history.csv"), history)
    out = {"signature": sig, "scout": scout, "scout_label": meta["labels"][scout], "pair": pair,
           "repeat": repeat, "fold": fold, "test_subjects": data.selected[test].tolist(),
           "test_groups": data.originals[test].tolist(), "labels": data.labels[test].tolist(),
           "n_components": int(spatial["basis"].shape[1]), "explained": spatial["explained"],
           "seconds": time.time() - started, **result}
    io.save_json(path, out)
    return out


def summarize_scout(cfg: dict, scout: int, pair: str, results: list[dict]) -> dict:
    """Out-of-fold AUC per repeat; writes subject_predictions.csv and summary.json."""
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
    directory = scout_output_root(cfg) / f"scout_{scout:02d}" / pair
    io.save_csv(directory / "subject_predictions.csv", rows)
    summary = {"scout": scout, "scout_label": results[0]["scout_label"], "pair": pair,
               "n_subjects": len(rows) // cfg["repeats"]}
    for key, name in (("p_tta", "soft_vote"), ("p_full", "all_trial")):
        aucs = np.array([m["auc"] for m in per_repeat[key]])
        bals = np.array([m["bal_acc"] for m in per_repeat[key]])
        summary.update({f"auc_{name}_mean": float(aucs.mean()), f"auc_{name}_sd": float(aucs.std()),
                        f"bal_acc_{name}_mean": float(bals.mean()),
                        f"auc_{name}_by_repeat": [round(float(a), 4) for a in aucs]})
    io.save_json(directory / "summary.json", summary)
    print(f"==> scout {scout} {summary['scout_label']} | {pair} | out-of-fold AUC soft vote "
          f"{summary['auc_soft_vote_mean']:.3f} ± {summary['auc_soft_vote_sd']:.3f} "
          f"{summary['auc_soft_vote_by_repeat']} | all-trial {summary['auc_all_trial_mean']:.3f} ± "
          f"{summary['auc_all_trial_sd']:.3f}", flush=True)
    return summary


# ---------------------------------------------------------------- several scouts (or sensors)

class Arm:
    """One input set: scouts stacked as rows (``scouts``), or the sensors (``scouts`` None)."""

    def __init__(self, name: str, kind: str, model: str, scouts: list[int] | None) -> None:
        self.name, self.kind, self.model, self.scouts = name, kind, model, scouts


def predict_projected(model: nn.Module, x: np.ndarray, owner: np.ndarray, position: np.ndarray, n_test: int,
                      m: torch.Tensor, device: torch.device) -> np.ndarray:
    """Mean probability per test subject; ``owner`` indexes ``m``, ``position`` the test subject."""
    model.eval()
    with torch.no_grad():
        p = np.concatenate([
            torch.softmax(model(torch.bmm(m[owner[k:k + 128]], torch.from_numpy(x[k:k + 128]).to(device))), 1)[:, 1]
            .cpu().numpy() for k in range(0, len(x), 128)])
    return np.array([p[position == j].mean() for j in range(n_test)])


def train_projected(cfg: dict, arm: Arm, subjects: list[dict], m_np: np.ndarray, labels: np.ndarray,
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
            a, b = (cv.bootstrap_weights(rng, len(t), sizes) @ t.reshape(len(t), -1) for t in subjects[i]["trials"])
            out.append((b - a).reshape(n, *subjects[i]["trials"][0].shape[1:]))
        return np.concatenate(out)

    owner_train = np.repeat(train, draws)
    y_train = torch.from_numpy(np.repeat(labels[train], draws)).to(device)
    x_full = np.stack([s["trials"][1].mean(0) - s["trials"][0].mean(0) for s in (subjects[i] for i in test)])
    x_full = np.ascontiguousarray(x_full[..., shift:shift + n_times])
    x_tta = np.ascontiguousarray(contrasts(test, cfg["tta_draws"], cfg["tta_avg"], cfg["tta_avg"])
                                 [..., shift:shift + n_times])
    position_tta = np.repeat(np.arange(len(test)), cfg["tta_draws"])

    model = EEGNet(n_rows, n_times, fast=cfg.get("fast_eegnet", False)).to(device)
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
        # Window of each example starting at shift + its time shift (same values as take_along_axis).
        x = sliding_window_view(x, n_times, axis=-1)[np.arange(n), :, shift + shifts]
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
    result = {"p_full": predict_projected(net, x_full, test, np.arange(len(test)), len(test), m, device).tolist(),
              "p_tta": predict_projected(net, x_tta, np.repeat(test, cfg["tta_draws"]), position_tta, len(test), m,
                                         device).tolist()}
    return result, history, n_parameters


def arm_signature(cfg: dict, meta: dict, arm: Arm) -> dict:
    sig = {"cache": meta["signature"], "trial_index": io.file_signature(cfg["trial_index"]), "pair": cfg["pair"],
           "arm": arm.name, "model": arm.model, "scouts": arm.scouts,
           **{k: cfg[k] for k in SIGNATURE_KEYS + ("scout_dropout", "pc_dropout")}}
    if cfg.get("fast_eegnet"):
        sig["eegnet_impl"] = "fast"  # results differ from the convolution implementation at rounding level
    return sig


def arm_fold_path(cfg: dict, arm: Arm, repeat: int, fold: int) -> Path:
    return Path(cfg["output_dir"]) / "arms" / arm.name / f"repeat{repeat + 1}_fold{fold + 1}.json"


class FoldData:
    """Subjects, labels, and the split of one outer fold, loaded on first use and shared by arms."""

    def __init__(self, cfg: dict, meta: dict, repeat: int, fold: int) -> None:
        self.args = (cfg, meta, repeat, fold)
        self.loaded = False

    def load(self) -> "FoldData":
        if not self.loaded:
            cfg, meta, repeat, fold = self.args
            self.selected, self.labels, self.originals = source.select_pair(np.asarray(meta["groups"]), cfg["pair"])
            self.subjects = source.load_subjects(cfg, meta, self.selected)
            self.train, self.test = cv.outer_splits(self.originals, cfg["folds"], cfg["repeats"], cfg["seed"])[repeat][fold]
            i0, i1, self.shift = source.window_indices(cfg, meta)
            self.n_times = i1 - i0
            self.loaded = True
        return self


def run_arm_fold(cfg: dict, meta: dict, arm: Arm, repeat: int, fold: int, device: torch.device,
                 data: FoldData | None = None) -> dict:
    path = arm_fold_path(cfg, arm, repeat, fold)
    sig = arm_signature(cfg, meta, arm)
    saved = load_saved(path, sig, MULTI_LEGACY, cfg["resume"])
    if saved is not None:
        return saved
    if device.type == "cpu":
        torch.set_num_threads(1)
    started = time.time()
    d = (data or FoldData(cfg, meta, repeat, fold)).load()
    if arm.scouts is None:
        m = np.stack([np.diag(1 / np.sqrt(np.maximum(np.diag(s["base_cov"]), 1e-30))) for s in d.subjects])
    else:
        m = np.concatenate([np.load(source.projection_path(cfg, s, repeat, fold)) for s in arm.scouts], axis=1)
    seed = cfg["seed"] + 7919 * repeat + 101 * fold  # the same seed for every arm (paired comparisons)
    result, history, n_parameters = train_projected(cfg, arm, d.subjects, m.astype(np.float32), d.labels, d.train,
                                                    d.test, d.n_times, d.shift, seed, device)
    path.parent.mkdir(parents=True, exist_ok=True)
    io.save_csv(path.with_name(path.stem + "_history.csv"), history)
    out = {"signature": sig, "arm": arm.name, "kind": arm.kind, "model": arm.model, "scouts": arm.scouts,
           "repeat": repeat, "fold": fold, "test_subjects": d.selected[d.test].tolist(),
           "test_groups": d.originals[d.test].tolist(), "labels": d.labels[d.test].tolist(),
           "n_input_rows": int(m.shape[1]), "n_parameters": int(n_parameters), "seconds": time.time() - started,
           **result}
    io.save_json(path, out)
    return out


def run_arm_chunk(cfg: dict, meta: dict, arms: list[Arm], repeat: int, fold: int, device: torch.device) -> list[dict]:
    """Several arms of one outer fold; the fold's subjects are loaded once (and only if needed)."""
    data = FoldData(cfg, meta, repeat, fold)
    return [run_arm_fold(cfg, meta, arm, repeat, fold, device, data) for arm in arms]


def arm_settings(sig: dict) -> dict:
    """The training settings of an arm signature: without the arm's identity and legacy fields."""
    return {k: v for k, v in sig.items()
            if k not in ("arm", "model", "scouts") and not (k in MULTI_LEGACY and v == MULTI_LEGACY[k])}


def load_arm_probs(cfg: dict, arm: Arm, position: dict, n: int,
                   sig: dict | None = None) -> tuple[np.ndarray, dict] | None:
    """Repeat x subject soft-vote probabilities of an arm and its training settings, or None
    when a fold is missing.

    Every fold must belong to ``arm`` and share one set of settings; with ``sig`` (the
    current arm signature) the folds must also match the current settings, so stale
    folds from other settings are never mixed into a comparison.
    """
    probs, settings = np.full((cfg["repeats"], n), np.nan), None
    for repeat in range(cfg["repeats"]):
        for fold in range(cfg["folds"]):
            path = arm_fold_path(cfg, arm, repeat, fold)
            if not path.exists():
                return None
            r = json.loads(path.read_text())
            saved = r["signature"]
            if (saved.get("arm") != arm.name or saved.get("scouts") != arm.scouts
                    or settings not in (None, arm_settings(saved))
                    or (sig is not None and not io.signatures_match(saved, sig, MULTI_LEGACY))):
                raise ValueError(f"{path} was trained with other settings than this arm's other folds or the "
                                 "current arguments; rerun main_group_classifier.py with the intended settings")
            settings = arm_settings(saved)
            probs[repeat, [position[s] for s in r["test_subjects"]]] = r["p_tta"]
    return probs, settings


def load_single_scout_probs(single_scout_dir: Path, pair: str, scout: int, position: dict,
                            fold_of: np.ndarray) -> np.ndarray | None:
    """Repeat x subject soft-vote probabilities of one screened scout (checks the outer splits)."""
    path = Path(single_scout_dir) / f"scout_{scout:02d}" / pair / "subject_predictions.csv"
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
