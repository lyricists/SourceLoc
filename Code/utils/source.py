"""Subject sensor cache, scout orientation + PCA, and the per-fold scout projection cache.

Everything that pools subjects (orientation, PCs) is fitted on the training subjects
of an outer fold. A scout's projection for one subject is a PC x channel matrix, so a
scout time course is that matrix times baseline-corrected sensor data.
"""

from __future__ import annotations

import json
import os
import pickle
import tempfile
from pathlib import Path

import numpy as np

from utils import cv, io, paths

PAIRS = {
    "control-vs-depressed+suicidal": ((1,), (2, 3)),
    "depressed-suicidal": ((2,), (3,)),
    "control-depressed": ((1,), (2,)),
    "control-suicidal": ((1,), (3,)),
}
BASELINE_END_MS = 0


# ---------------------------------------------------------------- subject cache

def cache_signature(eeg: Path, tmatrix: Path, atlas: Path, subject_index: Path) -> dict:
    return {name: io.file_signature(path) for name, path in
            (("eeg", eeg), ("tmatrix", tmatrix), ("atlas", atlas), ("subject_index", subject_index))}


def build_cache(cache_dir: Path, eeg: Path, tmatrix: Path, atlas: Path, subject_index: Path) -> None:
    """Per subject: sensor trials (trial x channel x time) and the inverse kernel, both float32."""
    import mat73
    cache_dir.mkdir(parents=True, exist_ok=True)
    atlas_data = mat73.loadmat(str(atlas))["DKT_scout"]
    labels = [str(x).strip() for x in atlas_data.get("Label", atlas_data.get("Region"))]
    vertices = [(np.asarray(v, dtype=np.int64).ravel() - 1).tolist() for v in atlas_data["Vertices"]]
    groups = io.load_subject_groups(subject_index)
    print("Building the subject cache from the EEG and inverse-kernel pickles...", flush=True)
    data = io.load_pickle(eeg)
    t_matrix = io.load_pickle(tmatrix)
    if data.ndim != 4 or data.shape[3] != len(groups) or len(t_matrix["kernel"]) != len(groups):
        raise ValueError(f"EEG shape {data.shape} does not match {len(groups)} subjects and kernels")
    n_channels, n_times = data.shape[:2]
    for s in range(len(groups)):
        kernel = np.asarray(t_matrix["kernel"][s], dtype=np.float32)
        good = np.asarray(t_matrix["GoodChannel"][s]).ravel().astype(int) - 1
        sensor = np.ascontiguousarray(data[:, :, :, s].transpose(2, 0, 1), dtype=np.float32)
        if kernel.shape[1] == len(good) and len(good) != n_channels:
            sensor = sensor[:, good]
        elif kernel.shape[1] != n_channels:
            raise ValueError(f"subject {s}: kernel columns do not match channels")
        if 3 * max(max(v) for v in vertices) + 2 >= kernel.shape[0]:
            raise ValueError(f"subject {s}: atlas vertices exceed kernel rows")
        for name, arr in (("sensor", sensor), ("kernel", kernel)):
            tmp = cache_dir / f"{name}_{s:03d}.tmp.npy"
            np.save(tmp, arr)
            os.replace(tmp, cache_dir / f"{name}_{s:03d}.npy")
    io.save_json(cache_dir / "metadata.json", {
        "signature": cache_signature(eeg, tmatrix, atlas, subject_index), "groups": groups.tolist(),
        "labels": labels, "vertices": vertices,
        "times_ms": (paths.EPOCH_START_MS + np.arange(n_times) * 1000 / paths.FS).tolist(),
    })


def ensure_cache(cache_dir: Path, eeg: Path = paths.EEG, tmatrix: Path = paths.TMATRIX,
                 atlas: Path = paths.ATLAS, subject_index: Path = paths.SUBJECT_INDEX) -> dict:
    """The cache metadata, rebuilding the cache when an input file changed."""
    path = Path(cache_dir) / "metadata.json"
    meta = json.loads(path.read_text()) if path.exists() else None
    if meta is None or (not meta["signature"].get("synthetic")
                        and meta["signature"] != cache_signature(eeg, tmatrix, atlas, subject_index)):
        build_cache(Path(cache_dir), eeg, tmatrix, atlas, subject_index)
        meta = json.loads(path.read_text())
    return meta


def build_synthetic_cache(n_scouts: int = 6) -> tuple[Path, Path]:
    """Synthetic subjects for smoke tests: groups 2 and 3 carry an extra 10-Hz burst in
    incongruent trials. Returns (cache directory, trial-index file)."""
    directory = Path(tempfile.mkdtemp(prefix="sourceloc_smoke_")) / "cache"
    directory.mkdir(parents=True)
    rng = np.random.default_rng(7)
    groups = np.repeat([1, 2, 3], 12)
    n_ch, n_vert, n_times, n_trials = 16, 30, 300, 80
    times = paths.EPOCH_START_MS + np.arange(n_times) * 1000 / paths.FS
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
    size = n_vert // n_scouts
    io.save_json(directory / "metadata.json", {
        "signature": {"synthetic": True}, "groups": groups.tolist(),
        "labels": [f"synthetic {i}" for i in range(n_scouts)],
        "vertices": [list(range(i * size, (i + 1) * size)) for i in range(n_scouts)], "times_ms": times.tolist(),
    })
    return directory, directory / "trial_index.pkl"


def load_trial_sets(path: Path, category: str, n_subjects: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """(congruent, incongruent) trial indices per subject."""
    entries = io.load_trial_index(path)
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
    """Subjects in the pair, their binary labels (second group = 1), and original group codes."""
    first, second = PAIRS[pair]
    selected = np.flatnonzero(np.isin(groups, first + second))
    originals = groups[selected]
    return selected, np.isin(originals, second).astype(np.int64), originals


def load_subject(cache: Path, s: int, sets: tuple, vertices: np.ndarray, baseline: np.ndarray) -> dict:
    """Baseline-corrected condition trials, the scout's kernel rows, and sensor covariances."""
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


# ---------------------------------------------------------------- projection cache

def window_indices(cfg: dict, meta: dict) -> tuple[int, int, int]:
    """Window start/end sample and the time-shift margin kept on both sides."""
    times = np.asarray(meta["times_ms"])
    hits = np.flatnonzero((times >= cfg["window"][0]) & (times < cfg["window"][1]))
    i0, i1 = int(hits[0]), int(hits[-1]) + 1
    return i0, i1, min(cfg["max_shift"], i0, len(times) - i1)


def load_subjects(cfg: dict, meta: dict, selected: np.ndarray, trials: bool = True, crop: bool = True) -> list[dict]:
    """Baseline-corrected sensor trials (cropped to the window plus shift margins unless
    ``crop`` is False) and sensor covariances."""
    sets = load_trial_sets(Path(cfg["trial_index"]), cfg["category"], len(meta["groups"]))
    baseline = np.asarray(meta["times_ms"]) < BASELINE_END_MS
    i0, i1, shift = window_indices(cfg, meta) if crop else (0, len(meta["times_ms"]), 0)
    subjects = []
    for s in selected:
        subject = load_subject(Path(cfg["cache_dir"]), s, sets[s], np.array([0]), baseline)
        entry = {"cov": subject["cov"], "base_cov": subject["base_cov"]}
        if trials:
            entry["trials"] = [np.ascontiguousarray(t[..., i0 - shift:i1 + shift]) for t in subject["trials"]]
        subjects.append(entry)
    return subjects


def projection_signature(cfg: dict, meta: dict) -> dict:
    return {"cache": meta["signature"], "trial_index": io.file_signature(cfg["trial_index"]), "pair": cfg["pair"],
            **{k: cfg[k] for k in ("category", "n_components", "folds", "repeats", "seed")}}


def projection_path(cfg: dict, scout: int, repeat: int, fold: int) -> Path:
    return (Path(cfg["projection_root"]) / "projections" / f"scout_{scout:02d}"
            / f"repeat{repeat + 1}_fold{fold + 1}.npy")


def compute_projections(cfg: dict, meta: dict, scout: int, covs: list[dict], splits: list) -> int:
    """Per fold: orientation and PCs from training subjects, then every subject's PC x channel matrix."""
    selected = select_pair(np.asarray(meta["groups"]), cfg["pair"])[0]
    vertices = np.asarray(meta["vertices"][scout], dtype=np.int64)
    rows = (3 * vertices[:, None] + np.arange(3)).ravel()
    subjects = []
    for s, cov in zip(selected, covs):
        kernel = np.load(Path(cfg["cache_dir"]) / f"kernel_{s:03d}.npy", mmap_mode="r")[rows]
        subjects.append({"kernel": np.asarray(kernel, dtype=np.float64).reshape(len(vertices), 3, -1), **cov})
    for repeat in range(cfg["repeats"]):
        for fold in range(cfg["folds"]):
            spatial = fit_spatial(subjects, splits[repeat][fold][0], cfg["n_components"])
            m = np.zeros((len(subjects), cfg["n_components"], subjects[0]["kernel"].shape[-1]), dtype=np.float32)
            for i, subject in enumerate(subjects):
                p = projection(subject, spatial)
                m[i, :len(p)] = p  # scouts with fewer vertices than PCs keep zero rows
            path = projection_path(cfg, scout, repeat, fold)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp.npy")
            np.save(tmp, m)
            os.replace(tmp, path)
    return scout


def ensure_projections(cfg: dict, meta: dict, scouts: list[int], jobs: int) -> None:
    """Build missing scout projections; refuse to reuse or extend a cache built with other
    settings or of unknown origin (projection files without a signature)."""
    sig_path = Path(cfg["projection_root"]) / "projections" / "signature.json"
    sig = projection_signature(cfg, meta)
    if sig_path.exists():
        if json.loads(sig_path.read_text()) != sig:
            raise ValueError(f"{sig_path.parent} was built with other settings; pass another --projection-root")
    elif any(sig_path.parent.glob("scout_*/*.npy")):
        raise ValueError(f"{sig_path.parent} holds projections without a signature.json; delete them or pass "
                         "another --projection-root")
    todo = [s for s in scouts if not all(projection_path(cfg, s, r, f).exists()
                                         for r in range(cfg["repeats"]) for f in range(cfg["folds"]))]
    if not todo:
        return
    io.save_json(sig_path, sig)  # before computing, so an interrupted run's files stay attributed
    print(f"Computing fold projections for {len(todo)} scout(s)...", flush=True)
    selected, _, originals = select_pair(np.asarray(meta["groups"]), cfg["pair"])
    covs = load_subjects(cfg, meta, selected, trials=False)
    splits = cv.outer_splits(originals, cfg["folds"], cfg["repeats"], cfg["seed"])
    from joblib import Parallel, delayed
    for done in Parallel(n_jobs=jobs, return_as="generator_unordered")(
            delayed(compute_projections)(cfg, meta, s, covs, splits) for s in todo):
        print(f"  scout {done} {meta['labels'][done]}", flush=True)
