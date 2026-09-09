import numpy as np
from scipy.stats import ttest_ind, t
from mne.stats import permutation_cluster_test


def stat_fun_timeseries(x, y):
    """
    x, y shape:
        subjects x time
    """
    tvals, _ = ttest_ind(
        x,
        y,
        axis=0,
        equal_var=False,
        nan_policy="omit",
    )
    return tvals


def cluster_based_permutation_test_timeseries(
    data1,
    data2,
    n_permutations=5000,
    threshold_p=0.05,
    cluster_alpha=0.05,
    tail=0,
    seed=42,
    n_jobs=-1,
):
    """
    Cluster-based permutation test for time-series data.

    data1, data2:
        subjects x time
    """

    data1 = np.asarray(data1, dtype=np.float64)
    data2 = np.asarray(data2, dtype=np.float64)

    if data1.ndim != 2 or data2.ndim != 2:
        raise ValueError(
            f"data1 and data2 must be subjects x time. "
            f"Got {data1.shape} and {data2.shape}."
        )

    if data1.shape[1] != data2.shape[1]:
        raise ValueError(f"Time mismatch: {data1.shape[1]} vs {data2.shape[1]}")

    if threshold_p is not None:
        df = data1.shape[0] + data2.shape[0] - 2

        if tail == 0:
            threshold = t.ppf(1 - threshold_p / 2, df)
        elif tail == 1:
            threshold = t.ppf(1 - threshold_p, df)
        elif tail == -1:
            threshold = -t.ppf(1 - threshold_p, df)
        else:
            raise ValueError("tail must be 0, 1, or -1.")
    else:
        threshold = None

    T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
        [data1, data2],
        stat_fun=stat_fun_timeseries,
        n_permutations=n_permutations,
        threshold=threshold,
        tail=tail,
        out_type="indices",
        seed=seed,
        n_jobs=n_jobs,
    )

    sig_mask = np.zeros(T_obs.shape, dtype=bool)

    for clu, pval in zip(clusters, cluster_p_values):
        if pval < cluster_alpha:

            clu_mask = np.zeros(T_obs.shape, dtype=bool)

            if isinstance(clu, tuple):
                clu_mask[clu] = True
            else:
                clu_mask[np.asarray(clu)] = True

            sig_mask |= clu_mask

    return sig_mask, T_obs, clusters, cluster_p_values
