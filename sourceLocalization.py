import os
import gc
import pickle

import mat73
import numpy as np
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
from tqdm import tqdm

# ============================================================
# Settings
# ============================================================
fpath = "/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Data/Preprocessed data/"

eegName = "Data_sen_lepoch.pkl"
scoutName = "DKT_scout.mat"
tMatName = "tMatrix.pkl"

# Output shape:
# PC x scout x time x trial x subject
save_name = "Data_sen_lepoch_DKT_scout_3PC.pkl"
save_path = os.path.join(fpath, save_name)

# Subject-level parallelism stays off.
USE_SCOUT_PARALLEL = True
N_SCOUT_JOBS = 4

# Prevent BLAS/PCA from starting additional threads inside each job.
BLAS_THREADS = 1

FS = 250
T_START = -200
T_END_ORIGINAL = 1500
ANALYSIS_END = 1000
N_PCS = 3
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

    if isinstance(x, np.ndarray) and x.dtype == object:
        return x[subj_idx]

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


def compute_flattened_pcs(
    source_scout_2d,
    n_times,
    n_trials,
    n_pcs=N_PCS,
):
    """
    Run one PCA for one subject and one scout using all selected
    time samples and all trials together.

    Parameters
    ----------
    source_scout_2d : ndarray
        Shape:
            sources_in_scout x (time * trial)

    n_times : int
        Number of retained time samples.

    n_trials : int
        Number of trials.

    n_pcs : int
        Number of PCA components to retain.

    Returns
    -------
    pc_data : ndarray
        Shape:
            PC x time x trial
    """

    # PCA observations:
    # every time-point/trial combination
    #
    # PCA features:
    # source rows within the scout
    X = source_scout_2d.T

    if X.shape[1] < n_pcs:
        raise ValueError(
            f"Cannot compute {n_pcs} PCs from only " f"{X.shape[1]} source rows."
        )

    # Center each source feature.
    X = X - X.mean(axis=0, keepdims=True)

    pca = PCA(
        n_components=n_pcs,
        svd_solver="full",
    )

    # Shape:
    # (time * trial) x PC
    pc_scores = pca.fit_transform(X)

    # PCA signs are arbitrary. Orient every PC so that its
    # largest absolute loading is positive.
    for pc_idx in range(n_pcs):
        largest_loading_idx = np.argmax(np.abs(pca.components_[pc_idx]))

        if pca.components_[pc_idx, largest_loading_idx] < 0:
            pc_scores[:, pc_idx] *= -1
            pca.components_[pc_idx] *= -1

    # Shape:
    # PC x time x trial
    pc_data = pc_scores.T.reshape(
        n_pcs,
        n_times,
        n_trials,
    )

    return pc_data.astype(
        np.float32,
        copy=False,
    )


def baseline_zscore_subject(subj_out, baseline_idx):
    """
    Baseline-z-score each PC, scout, and trial independently.

    Parameters
    ----------
    subj_out : ndarray
        Shape:
            PC x scout x time x trial

    baseline_idx : ndarray
        Indices of baseline time samples.

    Returns
    -------
    subj_out_z : ndarray
        Shape:
            PC x scout x time x trial
    """

    # Shape:
    # PC x scout x baseline_time x trial
    baseline_data = subj_out[:, :, baseline_idx, :]

    # Mean across baseline time independently for every
    # PC, scout, and trial.
    baseline_mean = np.nanmean(
        baseline_data,
        axis=2,
        keepdims=True,
    )

    # Standard deviation across baseline time independently
    # for every PC, scout, and trial.
    baseline_std = np.nanstd(
        baseline_data,
        axis=2,
        keepdims=True,
        ddof=0,
    )

    baseline_std[baseline_std < EPS] = EPS

    subj_out_z = (subj_out - baseline_mean) / baseline_std

    return subj_out_z.astype(
        np.float32,
        copy=False,
    )


# ============================================================
# Dimensions and selected time vector
# ============================================================
n_channels, n_times_original, n_trials, n_subjects = eeg_data.shape

n_scouts = len(scout_vertices)

# Full input EEG time vector:
# -200 through 1496 ms = 425 samples
times_original = np.arange(
    T_START,
    T_END_ORIGINAL,
    1000 / FS,
)

if len(times_original) != n_times_original:
    raise ValueError(
        "Original time-vector mismatch: "
        f"len(times_original)={len(times_original)}, "
        f"but EEG has {n_times_original} time samples."
    )

# Select -200 through 1000 ms inclusive.
# This produces 301 samples at 250 Hz.
time_idx = np.where((times_original >= T_START) & (times_original <= ANALYSIS_END))[0]

times = times_original[time_idx]
n_times = len(times)

if n_times != 301:
    raise ValueError(
        f"Expected 301 samples from {T_START} to "
        f"{ANALYSIS_END} ms, but found {n_times}."
    )

baseline_idx = np.where((times >= BASELINE_WIN[0]) & (times <= BASELINE_WIN[1]))[0]

if len(baseline_idx) == 0:
    raise ValueError(
        "No baseline indices found. " "Check BASELINE_WIN and the time vector."
    )

print("Original EEG shape:", eeg_data.shape)
print("Number of PCs:", N_PCS)
print("Number of scouts:", n_scouts)
print("Number of trials:", n_trials)
print("Number of subjects:", n_subjects)
print("Selected time samples:", n_times)
print(
    "Selected time range:",
    times[0],
    "to",
    times[-1],
    "ms",
)
print("Baseline window:", BASELINE_WIN)
print(
    "Baseline index range:",
    baseline_idx[0],
    "to",
    baseline_idx[-1],
)
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
# Subject-level processing
# ============================================================
def process_subject(subj):
    good_ch = matlab_to_python_index(get_subject_item(GoodChannel, subj))

    K = np.asarray(
        get_subject_item(Kernel, subj),
        dtype=np.float32,
    )

    # IMPORTANT:
    # Select the subject first. This preserves the original:
    #
    # channel x time x trial
    #
    # axis order.
    eeg_subj = eeg_data[:, :, :, subj]

    # Then select -200 through 1000 ms.
    #
    # Selecting time_idx and the subject simultaneously would
    # trigger NumPy advanced indexing and move the time axis.
    eeg_subj = eeg_subj[:, time_idx, :].astype(
        np.float32,
        copy=False,
    )

    expected_shape = (
        n_channels,
        n_times,
        n_trials,
    )

    if eeg_subj.shape != expected_shape:
        raise ValueError(
            f"Subject {subj + 1}: unexpected EEG shape. "
            f"Expected {expected_shape}, "
            f"got {eeg_subj.shape}."
        )

    # Select good channels if the inverse kernel was built
    # using only the good-channel subset.
    if K.shape[1] == len(good_ch):
        eeg_subj = eeg_subj[good_ch, :, :]

    # Otherwise, keep every channel if the kernel expects
    # the full sensor array.
    elif K.shape[1] == n_channels:
        pass

    else:
        raise ValueError(
            f"Subject {subj + 1}: kernel shape mismatch. "
            f"K.shape={K.shape}, "
            f"len(good_ch)={len(good_ch)}, "
            f"n_channels={n_channels}"
        )

    n_kernel_rows = K.shape[0]

    # Flatten:
    #
    # channel x time x trial
    #
    # into:
    #
    # channel x (time * trial)
    eeg_2d = eeg_subj.reshape(
        eeg_subj.shape[0],
        n_times * n_trials,
    )

    # Subject output:
    #
    # PC x scout x time x trial
    subj_out = np.zeros(
        (
            N_PCS,
            n_scouts,
            n_times,
            n_trials,
        ),
        dtype=np.float32,
    )

    def process_one_scout(scout_idx, rows):
        valid_rows = rows[rows < n_kernel_rows]

        if len(valid_rows) == 0:
            scout_tc = np.full(
                (
                    N_PCS,
                    n_times,
                    n_trials,
                ),
                np.nan,
                dtype=np.float32,
            )

            return scout_idx, scout_tc

        if len(valid_rows) < N_PCS:
            raise ValueError(
                f"Subject {subj + 1}, "
                f"scout {scout_idx + 1}: "
                f"only {len(valid_rows)} valid source rows; "
                f"at least {N_PCS} are required."
            )

        # Select only the inverse-kernel rows belonging
        # to this scout.
        K_scout = K[valid_rows, :]

        # Project EEG into source space for this scout.
        #
        # Shape:
        # source rows x (time * trial)
        source_scout_2d = K_scout @ eeg_2d

        # Retain PC1, PC2, and PC3.
        #
        # Shape:
        # PC x time x trial
        scout_tc = compute_flattened_pcs(
            source_scout_2d=source_scout_2d,
            n_times=n_times,
            n_trials=n_trials,
            n_pcs=N_PCS,
        )

        return scout_idx, scout_tc

    # Process scouts in parallel using shared-memory threads.
    if USE_SCOUT_PARALLEL:
        with threadpool_limits(limits=BLAS_THREADS):
            scout_results = Parallel(
                n_jobs=N_SCOUT_JOBS,
                backend="threading",
                prefer="threads",
            )(
                delayed(process_one_scout)(
                    scout_idx,
                    rows,
                )
                for scout_idx, rows in enumerate(scout_source_rows)
            )

        for scout_idx, scout_tc in scout_results:
            subj_out[:, scout_idx, :, :] = scout_tc

    else:
        with threadpool_limits(limits=BLAS_THREADS):
            for scout_idx, rows in enumerate(scout_source_rows):
                scout_idx, scout_tc = process_one_scout(
                    scout_idx,
                    rows,
                )

                subj_out[:, scout_idx, :, :] = scout_tc

    # Baseline normalization:
    # performed separately for each PC, scout, and trial.
    subj_out = baseline_zscore_subject(
        subj_out=subj_out,
        baseline_idx=baseline_idx,
    )

    return subj_out


# ============================================================
# Run processing with memory-mapped output
# ============================================================
print("\nStarting source localization processing...")

memmap_path = save_path.replace(
    ".pkl",
    "_memmap.dat",
)

output_shape = (
    N_PCS,
    n_scouts,
    n_times,
    n_trials,
    n_subjects,
)

scout_data_out = np.memmap(
    memmap_path,
    dtype=np.float32,
    mode="w+",
    shape=output_shape,
)

for subj in tqdm(range(n_subjects)):
    subj_out = process_subject(subj)

    scout_data_out[:, :, :, :, subj] = subj_out
    scout_data_out.flush()

    del subj_out
    gc.collect()

print("\nFinished subject-wise processing.")
print("Memmap output shape:", scout_data_out.shape)
print("PC x Scout x Time x Trial x Subject")


# ============================================================
# Save final pickle
# ============================================================
print("\nSaving final pickle...")

scout_data_final = np.asarray(
    scout_data_out,
    dtype=np.float32,
)

with open(save_path, "wb") as f:
    pickle.dump(
        scout_data_final,
        f,
        protocol=pickle.HIGHEST_PROTOCOL,
    )

print("\nDone.")
print("Final output shape:", scout_data_final.shape)
print("PC x Scout x Time x Trial x Subject")
print(f"Saved pickle to:\n{save_path}")
print(f"Temporary memmap file:\n{memmap_path}")
