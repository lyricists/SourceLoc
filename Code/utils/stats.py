"""Cluster-based permutation tests, AUC, and comparison statistics."""

from __future__ import annotations

import hashlib
import os
import pickle
import tempfile
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.stats import rankdata
from scipy.stats import t as t_distribution
from sklearn.metrics import balanced_accuracy_score

# Keep MNE's optional JIT cache in a writable location during permutation tests.
_NUMBA_CACHE = Path(tempfile.gettempdir()) / "mne_numba_cache"
_NUMBA_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", str(_NUMBA_CACHE))


# ---------------------------------------------------------------- clusters: helpers

def cluster_to_mask(cluster: object, shape: tuple[int, ...]) -> np.ndarray:
    """Boolean mask of one MNE cluster (index tuple, index array, or boolean array)."""
    if isinstance(cluster, tuple):
        mask = np.zeros(shape, dtype=bool)
        mask[cluster] = True
        return mask
    values = np.asarray(cluster)
    if values.dtype == bool:
        if values.shape != shape:
            raise ValueError(f"Unexpected cluster shape {values.shape}; expected {shape}.")
        return values
    mask = np.zeros(shape, dtype=bool)
    mask[values.astype(int)] = True
    return mask


def contiguous_runs(mask: np.ndarray) -> list[np.ndarray]:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return []
    return list(np.split(indices, np.where(np.diff(indices) > 1)[0] + 1))


# ---------------------------------------------------------------- time-series CBP vs chance

def _t_threshold(threshold_p: float, df: int, tail: int) -> float:
    if tail == 0:
        return float(t_distribution.ppf(1 - threshold_p / 2, df))
    if tail == 1:
        return float(t_distribution.ppf(1 - threshold_p, df))
    if tail == -1:
        return float(-t_distribution.ppf(1 - threshold_p, df))
    raise ValueError("tail must be 0, 1, or -1.")


def _significant(t_obs: np.ndarray, clusters: list, p_values: np.ndarray, cluster_alpha: float) -> np.ndarray:
    mask = np.zeros(t_obs.shape, dtype=bool)
    for cluster, p in zip(clusters, p_values):
        if p < cluster_alpha:
            mask |= cluster_to_mask(cluster, t_obs.shape)
    return mask


def cluster_based_permutation_test_timeseries_1samp(data, chance=0.0, n_permutations=5000, threshold_p=0.05,
                                                    cluster_alpha=0.05, tail=1, seed=42, n_jobs=-1):
    """One-sample cluster permutation test of subjects x time data against ``chance``.

    Sign-flip permutations of (data - chance), one-sample t per time point. Returns
    (significant mask, t values, clusters, cluster p-values).
    """
    from mne.stats import permutation_cluster_1samp_test
    x = np.asarray(data, dtype=np.float64) - chance
    if x.ndim != 2:
        raise ValueError(f"data must be subjects x time; got {x.shape}.")
    threshold = None if threshold_p is None else _t_threshold(threshold_p, x.shape[0] - 1, tail)
    t_obs, clusters, p_values, _ = permutation_cluster_1samp_test(
        x, threshold=threshold, n_permutations=n_permutations, tail=tail, out_type="indices",
        rng=seed, n_jobs=n_jobs, verbose=False)
    return _significant(t_obs, clusters, p_values, cluster_alpha), t_obs, clusters, p_values


# ---------------------------------------------------------------- scout x time CBP

def _read_adjacency(path: Path) -> sparse.csr_matrix:
    if path.suffix.lower() == ".npz":
        try:
            return sparse.csr_matrix(sparse.load_npz(path))
        except (KeyError, ValueError):
            with np.load(path) as archive:
                if "adjacency" not in archive:
                    raise ValueError(f"{path} is neither a SciPy sparse matrix nor an NPZ with an 'adjacency' array.")
                return sparse.csr_matrix(archive["adjacency"])
    if path.suffix.lower() == ".npy":
        return sparse.csr_matrix(np.load(path))
    if path.suffix.lower() in {".pkl", ".pickle"}:
        with open(path, "rb") as f:
            value = pickle.load(f)
        return sparse.csr_matrix(value["adjacency"] if isinstance(value, dict) and "adjacency" in value else value)
    raise ValueError("Scout adjacency must be a .npz, .npy, .pkl, or .pickle file.")


def load_scout_adjacency(path: Path | None, n_scouts: int, scout_indices: list[int]) -> tuple[sparse.csr_matrix, str]:
    """Undirected scout adjacency for the selected scouts; no file means no spatial edges."""
    n_selected = len(scout_indices)
    if path is None:
        return sparse.csr_matrix((n_selected, n_selected), dtype=np.int8), "disconnected scouts (temporal clusters only)"
    adjacency = _read_adjacency(path)
    if adjacency.shape == (n_scouts, n_scouts):
        adjacency = adjacency[scout_indices][:, scout_indices]
    elif adjacency.shape != (n_selected, n_selected):
        raise ValueError(f"Scout adjacency must be {n_scouts} or {n_selected} square; received {adjacency.shape}.")
    if adjacency.data.size and not np.isfinite(adjacency.data).all():
        raise ValueError("Scout adjacency contains NaN or infinite values.")
    adjacency = sparse.csr_matrix((adjacency + adjacency.T) != 0, dtype=np.int8)
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    return adjacency, f"anatomical adjacency from {path.resolve()}"


def adjacency_digest(adjacency: sparse.csr_matrix) -> str:
    adjacency = adjacency.tocsr()
    digest = hashlib.sha256()
    digest.update(np.asarray(adjacency.shape, dtype=np.int64).tobytes())
    digest.update(adjacency.indptr.astype(np.int64, copy=False).tobytes())
    digest.update(adjacency.indices.astype(np.int64, copy=False).tobytes())
    digest.update(adjacency.data.astype(np.int8, copy=False).tobytes())
    return digest.hexdigest()


def scout_time_cluster_1samp(centered: np.ndarray, threshold_p: float, n_permutations: int,
                             adjacency: sparse.csr_matrix, seed: int, n_jobs: int, buffer_size: int):
    """One-tailed one-sample spatiotemporal CBP; ``centered`` is subject x time x scout.

    Returns (t values, clusters as index tuples, cluster p-values, max-cluster null, t threshold).
    """
    from mne.stats import spatio_temporal_cluster_1samp_test
    threshold = float(t_distribution.ppf(1.0 - threshold_p, centered.shape[0] - 1))
    t_obs, clusters, p_values, null = spatio_temporal_cluster_1samp_test(
        centered, threshold=threshold, n_permutations=n_permutations, tail=1, adjacency=adjacency, max_step=1,
        t_power=1, out_type="indices", check_disjoint=True, buffer_size=buffer_size, rng=seed, n_jobs=n_jobs,
        verbose=False)
    return np.asarray(t_obs), clusters, np.asarray(p_values), np.asarray(null), threshold


# ---------------------------------------------------------------- permutation-null clusters (group decoding)

def runs_and_masses(mask: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Label contiguous time runs of ``mask`` (scout x time) and sum ``z`` within each."""
    n_scouts, n_times = mask.shape
    padded = np.zeros((n_scouts, n_times + 1), dtype=bool)
    padded[:, :n_times] = mask
    flat = padded.ravel()
    starts = flat & ~np.concatenate(([False], flat[:-1]))
    run = (np.cumsum(starts) * flat).reshape(n_scouts, n_times + 1)[:, :n_times]
    masses = np.bincount(run.ravel(), weights=np.where(mask, z, 0).ravel())[1:]
    return run, masses


def permutation_cluster_maps(observed: np.ndarray, null: np.ndarray, threshold_p: float) -> dict:
    """Cluster test of an observed scout x time map against label-permutation maps.

    ``null`` is scout x time x permutation. A point is supra-threshold above the
    (1 - threshold_p) quantile of its own null; cluster mass is the sum of null
    z-scores; p-values come from the max cluster mass of every permutation.
    """
    mean, sd = null.mean(-1), null.std(-1)
    sd = np.where(sd > 0, sd, np.inf)
    threshold = np.quantile(null, 1 - threshold_p, axis=-1)
    null_max = np.zeros(null.shape[-1])
    for k in range(null.shape[-1]):
        _, masses = runs_and_masses(null[..., k] > threshold, (null[..., k] - mean) / sd)
        null_max[k] = masses.max(initial=0.0)
    z = (observed - mean) / sd
    run, masses = runs_and_masses(observed > threshold, z)
    p_values = (1 + (null_max[None, :] >= masses[:, None] - 1e-12).sum(1)) / (1 + len(null_max))
    return {"null_mean": mean, "null_threshold": threshold, "z": z, "run": run, "masses": masses,
            "p_values": p_values, "null_max_mass": null_max,
            "null_low": np.quantile(null, 0.025, axis=-1), "null_high": np.quantile(null, 0.975, axis=-1)}


# ---------------------------------------------------------------- AUC and comparisons

def auc(y: np.ndarray, p: np.ndarray) -> float:
    """Mann-Whitney ROC AUC (ties count one half)."""
    n1 = int(y.sum())
    return float((rankdata(p)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def repeat_metrics(y: np.ndarray, q: np.ndarray, permuted: np.ndarray) -> dict:
    """AUC per repeat (rows of ``q``), balanced accuracy at 0.5, and a one-sided label-permutation
    p for the repeat-averaged AUC (``permuted`` is permutation x subject)."""
    aucs = np.array([auc(y, qr) for qr in q])
    n1 = int(y.sum())
    ranks = np.stack([rankdata(qr) for qr in q])
    null = ((permuted @ ranks.T - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))).mean(1)
    return {"auc_mean": float(aucs.mean()), "auc_sd": float(aucs.std()), "auc_by_repeat": aucs.round(4).tolist(),
            "bal_acc_mean": float(np.mean([balanced_accuracy_score(y, (qr >= 0.5).astype(int)) for qr in q])),
            "p_vs_chance": float((1 + np.sum(null >= aucs.mean() - 1e-12)) / (1 + len(null)))}


def paired_bootstrap_auc_difference(y: np.ndarray, q1: np.ndarray, q2: np.ndarray, boot: list[np.ndarray]) -> np.ndarray:
    """Repeat-averaged AUC of q1 minus q2 on each bootstrap resample of subjects."""
    def mean_auc(q, idx):
        return np.mean([auc(y[idx], qr[idx]) for qr in q])
    return np.array([mean_auc(q1, idx) - mean_auc(q2, idx) for idx in boot])


def holm(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted, running = np.empty(len(p_values)), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[i])
        adjusted[i] = min(1.0, running)
    return adjusted.tolist()
