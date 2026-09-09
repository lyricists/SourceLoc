import os
import gc
import pickle
import mat73
import numpy as np

from sklearn.decomposition import PCA
from joblib import Parallel, delayed
from tqdm import tqdm
from threadpoolctl import threadpool_limits

# ============================================================
# Settings
# ============================================================
fpath = "/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Data/Preprocessed data/"

eegName = "Data_sen_lepoch.pkl"
scoutName = "DKT_scout.mat"
tMatName = "tMatrix.pkl"

save_name = "Data_sen_lepoch_DKT_scout_flattenedPCA_zscore.pkl"
save_path = os.path.join(fpath, save_name)

# Subject-level parallelism should stay OFF
USE_SCOUT_PARALLEL = True

# Start conservatively. Try 2 first. If stable, try 4.
N_SCOUT_JOBS = 4

# Prevent BLAS/PCA from using many threads inside each job
BLAS_THREADS = 1

FS = 250
T_START = -200
T_END = 1000
BASELINE_WIN = [-200, 0]

EPS = 1e-8


# ============================================================
# Load data
# ============================================================
with open(os.path.join(fpath, eegName), "rb") as f:
    eeg_data = pickle.load(f)

scout_vertices = mat73.loadmat(os.path.join(fpath, scoutName))["DKT_scout"]["Vertices"]

with open(os.path.join(fpath, tMatName), "rb") as f:
    t_matrix = pickle.load(f)

GoodChannel = t_matrix["GoodChannel"]
Kernel = t_matrix["kernel"]


# ============================================================
# Helper functions
# ============================================================
def get_subject_item(x, subj_idx):
    if isinstance(x, (list, tuple)):
        return x[subj_idx]
    elif isinstance(x, np.ndarray) and x.dtype == object:
        return x[subj_idx]
    else:
        return x[subj_idx]


def matlab_to_python_index(idx):
    idx = np.asarray(idx).squeeze().astype(int)
    return idx - 1


def vertex_to_xyz_rows(vertices_0based):
    vertices_0based = np.asarray(vertices_0based).astype(int).ravel()

    rows = np.empty(vertices_0based.size * 3, dtype=int)
    rows[0::3] = 3 * vertices_0based
    rows[1::3] = 3 * vertices_0based + 1
    rows[2::3] = 3 * vertices_0based + 2

    return rows


def compute_flattened_pc1(source_scout_2d, n_times, n_trials):
    """
    source_scout_2d:
        n_sources_in_scout x (time * trial)

    PCA is run once using all time points and trials
    for this subject and scout.

    Returns:
        time x trial
    """

    X = source_scout_2d.T  # (time * trial) x source

    X = X - X.mean(axis=0, keepdims=True)

    pca = PCA(
        n_components=1,
        svd_solver="full",
    )

    pc1 = pca.fit_transform(X)[:, 0]

    # Deterministic sign convention
    if np.sum(pca.components_[0]) < 0:
        pc1 *= -1

    return pc1.reshape(n_times, n_trials).astype(np.float32)


def baseline_zscore_subject(subj_out, baseline_idx):
    """
    subj_out:
        scout x time x trial

    Baseline z-score is applied separately for each scout and trial.
    """

    baseline_data = subj_out[:, baseline_idx, :]

    baseline_mean = np.mean(
        baseline_data,
        axis=1,
        keepdims=True,
    )

    baseline_std = np.std(
        baseline_data,
        axis=1,
        keepdims=True,
        ddof=0,
    )

    baseline_std[baseline_std < EPS] = EPS

    subj_out_z = (subj_out - baseline_mean) / baseline_std

    return subj_out_z.astype(np.float32)


# ============================================================
# Dimensions and time vector
# ============================================================
n_channels, n_times, n_trials, n_subjects = eeg_data.shape
n_scouts = len(scout_vertices)

times = np.arange(T_START, T_END, 1000 / FS)

if len(times) != n_times:
    raise ValueError(
        f"Time vector length mismatch: len(times)={len(times)}, "
        f"but EEG has n_times={n_times}."
    )

baseline_idx = np.where((times >= BASELINE_WIN[0]) & (times <= BASELINE_WIN[1]))[0]

if len(baseline_idx) == 0:
    raise ValueError("No baseline indices found. Check BASELINE_WIN and time vector.")

print("EEG shape:", eeg_data.shape)
print("Number of scouts:", n_scouts)
print("Number of subjects:", n_subjects)
print("Time vector shape:", times.shape)
print("Baseline window:", BASELINE_WIN)
print("Baseline index range:", baseline_idx[0], "to", baseline_idx[-1])
print(
    "Baseline time range:",
    times[baseline_idx[0]],
    "to",
    times[baseline_idx[-1]],
    "ms",
)


# ============================================================
# Precompute scout source-row indices
# ============================================================
scout_source_rows = []

for scout_idx in range(n_scouts):
    vertices_0based = matlab_to_python_index(scout_vertices[scout_idx])
    rows = vertex_to_xyz_rows(vertices_0based)
    scout_source_rows.append(rows)


# ============================================================
# Subject-level processing function
# ============================================================
def process_subject(subj):

    good_ch = matlab_to_python_index(get_subject_item(GoodChannel, subj))

    K = np.asarray(
        get_subject_item(Kernel, subj),
        dtype=np.float32,
    )

    eeg_subj = eeg_data[:, :, :, subj].astype(
        np.float32,
        copy=False,
    )

    if K.shape[1] == len(good_ch):
        eeg_subj = eeg_subj[good_ch, :, :]

    elif K.shape[1] == n_channels:
        pass

    else:
        raise ValueError(
            f"Subject {subj + 1}: kernel shape mismatch. "
            f"K.shape={K.shape}, len(good_ch)={len(good_ch)}, "
            f"n_channels={n_channels}"
        )

    n_kernel_rows = K.shape[0]

    # --------------------------------------------------------
    # Flatten EEG:
    # channel x time x trial -> channel x (time * trial)
    # --------------------------------------------------------
    eeg_2d = eeg_subj.reshape(
        eeg_subj.shape[0],
        n_times * n_trials,
    )

    subj_out = np.zeros(
        (n_scouts, n_times, n_trials),
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Process one scout
    # Important:
    # We do NOT compute full source_2d = K @ eeg_2d.
    # We only compute K_scout @ eeg_2d.
    # --------------------------------------------------------
    def process_one_scout(scout_idx, rows):

        valid_rows = rows[rows < n_kernel_rows]

        if len(valid_rows) == 0:
            scout_tc = np.full(
                (n_times, n_trials),
                np.nan,
                dtype=np.float32,
            )
            return scout_idx, scout_tc

        K_scout = K[valid_rows, :]

        source_scout_2d = K_scout @ eeg_2d

        scout_tc = compute_flattened_pc1(
            source_scout_2d=source_scout_2d,
            n_times=n_times,
            n_trials=n_trials,
        )

        return scout_idx, scout_tc

    # --------------------------------------------------------
    # Scout-level parallelism using threads
    # Threads share K and eeg_2d memory.
    # --------------------------------------------------------
    if USE_SCOUT_PARALLEL:

        with threadpool_limits(limits=BLAS_THREADS):
            scout_results = Parallel(
                n_jobs=N_SCOUT_JOBS,
                backend="threading",
                prefer="threads",
            )(
                delayed(process_one_scout)(scout_idx, rows)
                for scout_idx, rows in enumerate(scout_source_rows)
            )

        for scout_idx, scout_tc in scout_results:
            subj_out[scout_idx, :, :] = scout_tc

    else:

        with threadpool_limits(limits=BLAS_THREADS):
            for scout_idx, rows in enumerate(scout_source_rows):
                scout_idx, scout_tc = process_one_scout(scout_idx, rows)
                subj_out[scout_idx, :, :] = scout_tc

    # --------------------------------------------------------
    # Baseline z-score normalization
    # scout x time x trial
    # Applied separately for each scout and trial
    # --------------------------------------------------------
    subj_out = baseline_zscore_subject(
        subj_out=subj_out,
        baseline_idx=baseline_idx,
    )

    return subj_out


# ============================================================
# Run processing with memory-mapped output
# ============================================================
print("\nStarting source localization processing...")

memmap_path = save_path.replace(".pkl", "_memmap.dat")

scout_data_out = np.memmap(
    memmap_path,
    dtype=np.float32,
    mode="w+",
    shape=(n_scouts, n_times, n_trials, n_subjects),
)

for subj in tqdm(range(n_subjects)):

    subj_out = process_subject(subj)

    scout_data_out[:, :, :, subj] = subj_out
    scout_data_out.flush()

    del subj_out
    gc.collect()

print("\nFinished subject-wise processing.")
print("Memmap output shape:", scout_data_out.shape)
print("Scout x Time x Trial x Subject")


# ============================================================
# Save final pickle
# ============================================================
print("\nSaving final pickle...")

scout_data_final = np.asarray(scout_data_out, dtype=np.float32)

with open(save_path, "wb") as f:
    pickle.dump(
        scout_data_final,
        f,
        protocol=pickle.HIGHEST_PROTOCOL,
    )

print("\nDone.")
print(f"Saved pickle to:\n{save_path}")
print(f"Temporary memmap file:\n{memmap_path}")
