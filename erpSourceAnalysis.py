import os
import pickle
import mat73
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from scipy.stats import ttest_ind
from scipy.stats import t as tdist
from mne.stats import permutation_cluster_test

# ============================================================
# Settings
# ============================================================
data_path = "/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Data/Preprocessed data/"
behav_path = "/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Data/Behavior/"
save_dir = "/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Code/SourceLoc/Results/GroupResponsePlots/"

data_name = "Data_sen_lepoch_DKT.pkl"
idx_name = "subject_index.mat"
log_name = "senIdx_congruent.pkl"
scout_name = "DKT_scout.mat"

FS = 250
T_START = -200
T_END = 1500

N_PERMUTATIONS = 5000
THRESHOLD_P = 0.05
CLUSTER_ALPHA = 0.05
TAIL = 0  # 0 = two-sided group difference test

os.makedirs(save_dir, exist_ok=True)


# ============================================================
# CBP functions
# ============================================================
def stat_fun_timeseries(x, y):
    tvals, _ = ttest_ind(
        x,
        y,
        axis=0,
        equal_var=False,
        nan_policy="omit",
    )
    return tvals


def cluster_to_mask(cluster, shape):
    mask = np.zeros(shape, dtype=bool)

    if isinstance(cluster, tuple):
        mask[cluster] = True
    else:
        arr = np.asarray(cluster)

        if arr.dtype == bool:
            mask |= arr
        else:
            mask[arr.astype(int)] = True

    return mask


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
    Two-sample CBP for subjects x time data.
    Tests whether group1 - group2 differs from 0 over time.
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

    df = data1.shape[0] + data2.shape[0] - 2

    if threshold_p is not None:
        if tail == 0:
            threshold = tdist.ppf(1 - threshold_p / 2, df)
        elif tail == 1:
            threshold = tdist.ppf(1 - threshold_p, df)
        elif tail == -1:
            threshold = -tdist.ppf(1 - threshold_p, df)
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
            sig_mask |= cluster_to_mask(clu, T_obs.shape)

    return sig_mask, T_obs, clusters, cluster_p_values


def plot_sig_line(ax, times, sig_mask, y_pos, color="red", linewidth=2.5):
    sig_mask = np.asarray(sig_mask, dtype=bool)

    if not np.any(sig_mask):
        return

    idx = np.where(sig_mask)[0]
    clusters = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)

    for clu in clusters:
        ax.plot(
            times[clu],
            np.ones(len(clu)) * y_pos,
            color=color,
            linewidth=linewidth,
            solid_capstyle="butt",
        )


# ============================================================
# Load data
# ============================================================
print("Loading source data...")
with open(os.path.join(data_path, data_name), "rb") as f:
    data = pickle.load(f)

data = np.asarray(data, dtype=np.float32)

# Expected shape: scout x time x trial x subject
n_scouts, n_times, n_trials, n_subjects = data.shape
print("Data shape:", data.shape)

print("Loading subject index...")
sub_idx = mat73.loadmat(os.path.join(behav_path, idx_name))["subject_index"].squeeze()

print("Loading sentence indices...")
with open(os.path.join(behav_path, log_name), "rb") as f:
    sen_idx = pickle.load(f)["Sentiment"]

print("Loading scout information...")
scout_mat = mat73.loadmat(os.path.join(data_path, scout_name))["DKT_scout"]

if "Label" in scout_mat:
    scout_labels = scout_mat["Label"]
elif "Labels" in scout_mat:
    scout_labels = scout_mat["Labels"]
else:
    scout_labels = [f"Scout {i + 1}" for i in range(n_scouts)]

clean_labels = []
for lab in scout_labels:
    if isinstance(lab, np.ndarray):
        lab = lab.squeeze()
    if isinstance(lab, bytes):
        lab = lab.decode()
    if isinstance(lab, (list, tuple, np.ndarray)):
        lab = str(lab[0])
    clean_labels.append(str(lab))

scout_labels = clean_labels


# ============================================================
# Time vector
# ============================================================
times = np.arange(T_START, T_END, 1000 / FS)

if len(times) != n_times:
    raise ValueError(
        f"Time vector mismatch: len(times)={len(times)}, n_times={n_times}"
    )


# ============================================================
# Group definitions
# ============================================================
groups = {
    "Control": np.where(sub_idx == 1)[0],
    "Depressed": np.where(sub_idx == 2)[0],
    "Suicidal": np.where(sub_idx == 3)[0],
    "DepressedSuicidal": np.where((sub_idx == 2) | (sub_idx == 3))[0],
}

comparisons = [
    ("Control", "Depressed"),
    ("Control", "Suicidal"),
    ("Depressed", "Suicidal"),
    ("Control", "DepressedSuicidal"),
]


# ============================================================
# Helper functions
# ============================================================
def sem(x, axis=0):
    return np.nanstd(x, axis=axis, ddof=1) / np.sqrt(x.shape[axis])


def get_subject_condition_mean(data, subject, condition):
    """
    Returns:
        scout x time
    """

    trial_idx = sen_idx[subject][condition]
    trial_idx = np.asarray(trial_idx).astype(int)

    return np.nanmean(data[:, :, trial_idx, subject], axis=2)


def compute_group_response(data, subject_list):
    """
    Returns subject-level and group-level responses.

    subject-level arrays:
        subject x scout x time
    """

    pos_all = []
    neg_all = []
    diff_all = []

    for subject in subject_list:
        pos = get_subject_condition_mean(data, subject, "positive")
        neg = get_subject_condition_mean(data, subject, "negative")
        diff = neg - pos

        pos_all.append(pos)
        neg_all.append(neg)
        diff_all.append(diff)

    pos_all = np.stack(pos_all, axis=0)
    neg_all = np.stack(neg_all, axis=0)
    diff_all = np.stack(diff_all, axis=0)

    return {
        "positive_subjects": pos_all,
        "negative_subjects": neg_all,
        "neg_minus_pos_subjects": diff_all,
        "positive_mean": np.nanmean(pos_all, axis=0),
        "negative_mean": np.nanmean(neg_all, axis=0),
        "neg_minus_pos_mean": np.nanmean(diff_all, axis=0),
        "positive_sem": sem(pos_all, axis=0),
        "negative_sem": sem(neg_all, axis=0),
        "neg_minus_pos_sem": sem(diff_all, axis=0),
    }


def compute_difference_response(group1_response, group2_response):
    """
    Difference direction:
        group1 - group2
    """

    diff_response = {}

    for key in ["positive", "negative", "neg_minus_pos"]:

        mean1 = group1_response[f"{key}_mean"]
        mean2 = group2_response[f"{key}_mean"]

        sem1 = group1_response[f"{key}_sem"]
        sem2 = group2_response[f"{key}_sem"]

        diff_response[f"{key}_mean"] = mean1 - mean2
        diff_response[f"{key}_sem"] = np.sqrt(sem1**2 + sem2**2)

    return diff_response


def compute_cbp_masks(group1_response, group2_response):
    """
    Runs CBP for every scout and condition.

    Returns:
        dict of condition -> scout x time boolean mask
    """

    cbp_masks = {}
    cbp_pvals = {}

    for key in ["positive", "negative", "neg_minus_pos"]:

        data1 = group1_response[f"{key}_subjects"]  # subject x scout x time
        data2 = group2_response[f"{key}_subjects"]

        key_masks = np.zeros((n_scouts, n_times), dtype=bool)
        key_pvals = []

        for scout in tqdm(range(n_scouts), desc=f"CBP {key}", leave=False):

            x1 = data1[:, scout, :]  # subject x time
            x2 = data2[:, scout, :]  # subject x time

            sig_mask, T_obs, clusters, cluster_p_values = (
                cluster_based_permutation_test_timeseries(
                    data1=x1,
                    data2=x2,
                    n_permutations=N_PERMUTATIONS,
                    threshold_p=THRESHOLD_P,
                    cluster_alpha=CLUSTER_ALPHA,
                    tail=TAIL,
                    seed=42,
                    n_jobs=-1,
                )
            )

            key_masks[scout, :] = sig_mask
            key_pvals.append(cluster_p_values)

        cbp_masks[key] = key_masks
        cbp_pvals[key] = key_pvals

    return cbp_masks, cbp_pvals


def plot_comparison(comp_name, diff_response, cbp_masks, save_path):
    """
    Plots group difference with shaded SEM and red CBP significance lines.
    """

    conditions = [
        ("positive", "Positive"),
        ("negative", "Negative"),
        ("neg_minus_pos", "Negative - Positive"),
    ]

    fig_height = n_scouts * 1.25

    fig, axes = plt.subplots(
        n_scouts,
        3,
        figsize=(18, fig_height),
        sharex=True,
        sharey=False,
    )

    if n_scouts == 1:
        axes = axes[np.newaxis, :]

    for scout in range(n_scouts):

        for col, (key, _) in enumerate(conditions):

            ax = axes[scout, col]

            y = diff_response[f"{key}_mean"][scout, :]
            y_sem = diff_response[f"{key}_sem"][scout, :]
            sig_mask = cbp_masks[key][scout, :]

            line = ax.plot(
                times,
                y,
                linewidth=1,
            )[0]

            ax.fill_between(
                times,
                y - y_sem,
                y + y_sem,
                alpha=0.25,
                color=line.get_color(),
                linewidth=0,
            )

            ax.axvline(
                0,
                linestyle="--",
                linewidth=0.7,
                color="black",
            )

            ax.axhline(
                0,
                linewidth=0.5,
                color="black",
            )

            # Red significance line near the 0 baseline
            ymin, ymax = ax.get_ylim()
            y_sig = ymin + 0.08 * (ymax - ymin)

            plot_sig_line(
                ax=ax,
                times=times,
                sig_mask=sig_mask,
                y_pos=y_sig,
                color="red",
                linewidth=2.5,
            )

            if scout == n_scouts - 1:
                ax.set_xlabel("Time (ms)")

        axes[scout, 1].text(
            0.5,
            1.08,
            scout_labels[scout],
            transform=axes[scout, 1].transAxes,
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    fig.suptitle(
        comp_name,
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )

    column_x = [0.19, 0.50, 0.81]
    for x, (_, title) in zip(column_x, conditions):
        fig.text(
            x,
            0.982,
            title,
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
        )

    plt.subplots_adjust(
        hspace=1.15,
        wspace=0.30,
        top=0.965,
        bottom=0.03,
        left=0.05,
        right=0.98,
    )

    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# Compute group responses
# ============================================================
print("Precomputing group responses...")

group_response = {}

for group_name, subject_list in groups.items():
    print(f"{group_name}: {len(subject_list)} subjects")
    group_response[group_name] = compute_group_response(data, subject_list)


# ============================================================
# Compute CBP and plot comparisons
# ============================================================
print("Running CBP and plotting comparisons...")

all_cbp_results = {}

for g1, g2 in tqdm(comparisons):

    comp_name = f"{g1} - {g2}"
    print(f"\nProcessing comparison: {comp_name}")

    diff_response = compute_difference_response(
        group_response[g1],
        group_response[g2],
    )

    cbp_masks, cbp_pvals = compute_cbp_masks(
        group_response[g1],
        group_response[g2],
    )

    all_cbp_results[comp_name] = {
        "cbp_masks": cbp_masks,
        "cbp_pvals": cbp_pvals,
    }

    save_base = f"group_response_{g1}_minus_{g2}_CBP"
    pdf_path = os.path.join(save_dir, save_base + ".pdf")

    plot_comparison(
        comp_name=comp_name,
        diff_response=diff_response,
        cbp_masks=cbp_masks,
        save_path=pdf_path,
    )


# ============================================================
# Save CBP results
# ============================================================
cbp_save_path = os.path.join(save_dir, "group_response_CBP_results.pkl")

with open(cbp_save_path, "wb") as f:
    pickle.dump(all_cbp_results, f)

print("\nDone.")
print("Saved figures to:")
print(save_dir)
print("Saved CBP results to:")
print(cbp_save_path)
