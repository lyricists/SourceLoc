"""Per-subject scout PCA, aligned across subjects, for group classification.

For every subject and DKT scout, the scout's source time courses (inverse-kernel
rows of its X/Y/Z dipoles applied to sensor EEG) are reduced to principal
components fitted on all of that subject's trials and retained time points.
Group labels are constant within a subject, so this fit cannot leak them.

Per-subject PCA bases are arbitrary in sign, order, and mixing among PCs of
similar variance, so PC k would not mean the same thing in different subjects.
All subjects share the template source space, so each scout gets a reference
basis (the top right singular vectors of every subject's stacked PC loadings)
and each subject's PC subspace is rotated onto it with an orthogonal Procrustes
rotation. The aligned scores are then baseline z-scored per PC, scout, and trial.

PCA is computed exactly from the sensor covariance: scout source data are
K_scout @ EEG, so their covariance is K_scout @ C_sensor @ K_scout.T, and every
aligned PC score is a fixed spatial filter applied to the sensor EEG. Source
time courses are never materialized.

Outputs:
    <output>.npy            PC x scout x time x trial x subject, float32,
                            Fortran order (each subject block is contiguous);
                            load with np.load(path, mmap_mode="r").
    <output stem>_metadata.pkl
                            times, scout labels, subject groups, reference
                            bases, rotations, explained variance, parameters.

Trials that are non-finite or all zero in the sensor EEG are excluded from the
PCA fit and are NaN in the output.
"""

from __future__ import annotations

import argparse
import gc
import os
import pickle
import shutil
from pathlib import Path

import mat73
import numpy as np
from threadpoolctl import threadpool_limits
from tqdm import tqdm

BASE_DIRECTORY = Path("/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Code/SourceLoc")
DATA_DIRECTORY = BASE_DIRECTORY.parent.parent / "Data"
PREPROCESSED_DIRECTORY = DATA_DIRECTORY / "Preprocessed data"
BEHAVIOR_DIRECTORY = DATA_DIRECTORY / "Behavior"

DEFAULT_EEG = PREPROCESSED_DIRECTORY / "Data_sen_bl.pkl"
DEFAULT_ATLAS = PREPROCESSED_DIRECTORY / "DKT_scout.mat"
DEFAULT_TMATRIX = PREPROCESSED_DIRECTORY / "tMatrix.pkl"
DEFAULT_SUBJECT_INDEX = BEHAVIOR_DIRECTORY / "subject_index.mat"
DEFAULT_OUTPUT = PREPROCESSED_DIRECTORY / "Data_sen_bl_DKT_scout_3PC_aligned.npy"

# Bump when the computation changes; stored in the metadata.
VERSION = 1

FS = 250

T_START_MS = -200
# Analysis keeps input samples up to this time (all of them for a shorter epoch).
ANALYSIS_END_MS = 1000
BASELINE_START_MS = -200
BASELINE_END_MS = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--eeg", type=Path, default=DEFAULT_EEG)
    parser.add_argument("--atlas", type=Path, default=DEFAULT_ATLAS)
    parser.add_argument("--tmatrix", type=Path, default=DEFAULT_TMATRIX)
    parser.add_argument("--subject-index", type=Path, default=DEFAULT_SUBJECT_INDEX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-pcs", type=int, default=3)
    parser.add_argument(
        "--baseline-zscore",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Z-score each PC/scout/trial by its own baseline "
            f"({BASELINE_START_MS} to {BASELINE_END_MS} ms). Removes absolute "
            "amplitude differences between subjects."
        ),
    )
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument(
        "--subjects",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Optional zero-based subject indices; default: every subject. The "
            "alignment reference is built from the selected subjects only."
        ),
    )
    parser.add_argument(
        "--scouts",
        type=int,
        nargs="*",
        default=None,
        help="Optional zero-based scout indices; default: every scout.",
    )
    return parser.parse_args()


def atomic_pickle_dump(value: object, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with open(temporary, "wb") as file:
        pickle.dump(value, file, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, destination)


def file_signature(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def matlab_to_python_index(indices: object) -> np.ndarray:
    return np.asarray(indices).squeeze().astype(int) - 1


def vertex_to_xyz_rows(vertices_0based: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices_0based, dtype=int).reshape(-1)
    rows = np.empty(vertices.size * 3, dtype=int)
    rows[0::3] = 3 * vertices
    rows[1::3] = 3 * vertices + 1
    rows[2::3] = 3 * vertices + 2
    return rows


def orient_largest_loading_positive(components: np.ndarray) -> np.ndarray:
    largest = np.argmax(np.abs(components), axis=1)
    signs = np.sign(components[np.arange(components.shape[0]), largest])
    signs[signs == 0] = 1
    return components * signs[:, None]


class ScoutPCAExtractor:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self._validate_settings()
        self._load_inputs()
        self._prepare_dimensions()
        self._prepare_selection()

    def _validate_settings(self) -> None:
        if self.args.n_pcs < 1:
            raise ValueError("n-pcs must be positive.")
        if self.args.blas_threads < 1:
            raise ValueError("blas-threads must be positive.")
        if self.args.output.suffix != ".npy":
            raise ValueError("output must be a .npy path.")

    def _load_inputs(self) -> None:
        print("Loading sensor EEG...")
        with open(self.args.eeg, "rb") as file:
            self.eeg_data = np.asarray(pickle.load(file))

        print("Loading inverse kernels and good-channel indices...")
        with open(self.args.tmatrix, "rb") as file:
            t_matrix = pickle.load(file)
        self.good_channels = t_matrix["GoodChannel"]
        self.kernels = t_matrix["kernel"]

        print("Loading atlas and subject groups...")
        atlas = mat73.loadmat(str(self.args.atlas))["DKT_scout"]
        self.scout_vertices = atlas["Vertices"]
        raw_labels = atlas.get("Label", atlas.get("Region"))
        self.scout_labels = [str(label).strip() for label in raw_labels]

        self.subject_groups = np.asarray(
            mat73.loadmat(str(self.args.subject_index))["subject_index"]
        ).squeeze()

    def _prepare_dimensions(self) -> None:
        if self.eeg_data.ndim != 4:
            raise ValueError(
                "Sensor EEG must be channel x time x trial x subject; received "
                f"{self.eeg_data.shape}."
            )
        (
            self.n_channels,
            self.n_times_original,
            self.n_trials,
            self.n_subjects,
        ) = self.eeg_data.shape
        self.n_scouts = len(self.scout_vertices)

        if len(self.kernels) != self.n_subjects:
            raise ValueError("Kernel count does not match the EEG subject count.")
        if len(self.good_channels) != self.n_subjects:
            raise ValueError("GoodChannel count does not match the subject count.")
        if self.subject_groups.size != self.n_subjects:
            raise ValueError("subject_index count does not match the EEG data.")
        if len(self.scout_labels) != self.n_scouts:
            raise ValueError("Scout label count does not match the atlas.")

        # Alignment compares loadings across subjects, so every kernel must
        # live in the same source space.
        kernel_rows = {np.shape(kernel)[0] for kernel in self.kernels}
        if len(kernel_rows) != 1:
            raise ValueError(
                "Inverse kernels have different source-row counts across "
                f"subjects ({sorted(kernel_rows)}); loadings cannot be aligned."
            )
        self.n_source_rows = kernel_rows.pop()

        original_times = (
            T_START_MS + np.arange(self.n_times_original, dtype=float) * 1000 / FS
        )
        self.time_indices = np.flatnonzero(original_times <= ANALYSIS_END_MS)
        self.times = original_times[self.time_indices]
        self.n_times = self.times.size
        self.baseline_indices = np.flatnonzero(
            (self.times >= BASELINE_START_MS) & (self.times <= BASELINE_END_MS)
        )
        if self.baseline_indices.size < 2 or self.times[-1] <= BASELINE_END_MS:
            raise ValueError(
                f"EEG epoch {original_times[0]:g} to {original_times[-1]:g} ms does "
                f"not cover the baseline {BASELINE_START_MS} to {BASELINE_END_MS} ms "
                "plus post-stimulus samples."
            )

        self.scout_source_rows = []
        for scout_index, vertices in enumerate(self.scout_vertices):
            vertices_0based = matlab_to_python_index(vertices)
            if np.any(vertices_0based < 0):
                raise IndexError("Atlas contains a non-positive MATLAB vertex index.")
            rows = vertex_to_xyz_rows(vertices_0based)
            if np.any(rows >= self.n_source_rows):
                raise IndexError(
                    f"Scout {scout_index}: atlas source rows exceed the "
                    f"{self.n_source_rows} kernel rows."
                )
            if rows.size < self.args.n_pcs:
                raise ValueError(
                    f"Scout {scout_index} has fewer source rows than requested PCs."
                )
            self.scout_source_rows.append(rows)

        print("Sensor EEG shape:", self.eeg_data.shape)
        print(
            f"EEG epoch: {original_times[0]:g} to {original_times[-1]:g} ms; "
            f"analysis window: {self.times[0]:g} to {self.times[-1]:g} ms "
            f"({self.n_times} samples)"
        )

    @staticmethod
    def _validated_selection(
        requested: list[int] | None, total: int, name: str
    ) -> list[int]:
        selected = list(range(total)) if requested is None else list(requested)
        if not selected:
            raise ValueError(f"{name} selection cannot be empty.")
        if len(set(selected)) != len(selected):
            raise ValueError(f"{name} selection contains duplicates.")
        if min(selected) < 0 or max(selected) >= total:
            raise IndexError(f"{name} indices must be between 0 and {total - 1}.")
        return selected

    def _prepare_selection(self) -> None:
        self.subject_selection = self._validated_selection(
            self.args.subjects, self.n_subjects, "Subject"
        )
        self.scout_selection = self._validated_selection(
            self.args.scouts, self.n_scouts, "Scout"
        )
        self.output_shape = (
            self.args.n_pcs,
            len(self.scout_selection),
            self.n_times,
            self.n_trials,
            len(self.subject_selection),
        )
        print(
            "Output shape (PC x scout x time x trial x subject):", self.output_shape
        )

    def _subject_sensor_data(
        self, subject_index: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the kernel, valid-trial EEG (channel x time x trial), and the
        valid trial indices for one subject."""
        good_channels = matlab_to_python_index(self.good_channels[subject_index])
        if np.any(good_channels < 0) or np.any(good_channels >= self.n_channels):
            raise IndexError(f"Subject {subject_index}: invalid good-channel index.")

        kernel = np.asarray(self.kernels[subject_index])
        # Select the subject first: combining it with the time indices would
        # trigger advanced indexing and move the time axis.
        eeg_subject = self.eeg_data[:, :, :, subject_index]
        eeg_subject = eeg_subject[:, self.time_indices, :]

        if kernel.shape[1] == good_channels.size:
            eeg_subject = eeg_subject[good_channels, :, :]
        elif kernel.shape[1] != self.n_channels:
            raise ValueError(
                f"Subject {subject_index}: kernel columns {kernel.shape[1]} match "
                f"neither {good_channels.size} good channels nor "
                f"{self.n_channels} full channels."
            )

        valid = np.isfinite(eeg_subject).all(axis=(0, 1)) & np.any(
            eeg_subject != 0, axis=(0, 1)
        )
        valid_trials = np.flatnonzero(valid)
        if valid_trials.size == 0:
            raise ValueError(f"Subject {subject_index}: no valid trials.")
        eeg_valid = np.asarray(eeg_subject[:, :, valid_trials], dtype=np.float64)
        return kernel, eeg_valid, valid_trials

    def _fit_subject(self, subject_index: int) -> dict:
        kernel, eeg_valid, valid_trials = self._subject_sensor_data(subject_index)
        eeg_2d = eeg_valid.reshape(eeg_valid.shape[0], -1)
        n_samples = eeg_2d.shape[1]

        sensor_mean = eeg_2d.mean(axis=1)
        centered = eeg_2d - sensor_mean[:, None]
        sensor_covariance = centered @ centered.T / (n_samples - 1)
        del centered, eeg_2d, eeg_valid

        # sensor_covariance = sqrt_covariance @ sqrt_covariance.T, so the scout
        # source covariance is (K @ sqrt_covariance) @ (K @ sqrt_covariance).T
        # and its eigenvectors are the left singular vectors of K @ sqrt_cov.
        eigenvalues, eigenvectors = np.linalg.eigh(sensor_covariance)
        sqrt_covariance = eigenvectors * np.sqrt(np.clip(eigenvalues, 0, None))

        components = []
        variance = np.empty((len(self.scout_selection), self.args.n_pcs))
        explained_variance_ratio = np.empty_like(variance)
        for output_scout, scout_index in enumerate(self.scout_selection):
            kernel_scout = np.asarray(
                kernel[self.scout_source_rows[scout_index]], dtype=np.float64
            )
            left, singular, _ = np.linalg.svd(
                kernel_scout @ sqrt_covariance, full_matrices=False
            )
            if singular.size < self.args.n_pcs:
                raise ValueError(
                    f"Subject {subject_index}, scout {scout_index}: rank "
                    f"{singular.size} is below {self.args.n_pcs} PCs."
                )
            source_variance = singular**2
            total_variance = source_variance.sum()
            if not total_variance > 0:
                raise ValueError(
                    f"Subject {subject_index}, scout {scout_index}: zero variance."
                )
            components.append(left[:, : self.args.n_pcs].T)
            variance[output_scout] = source_variance[: self.args.n_pcs]
            explained_variance_ratio[output_scout] = (
                source_variance[: self.args.n_pcs] / total_variance
            )

        return {
            "components": components,
            "variance": variance,
            "explained_variance_ratio": explained_variance_ratio,
            "valid_trials": valid_trials,
        }

    def _align(self, fits: list[dict]) -> dict:
        n_subjects = len(fits)
        n_scouts = len(self.scout_selection)
        n_pcs = self.args.n_pcs
        references = []
        rotations = np.empty((n_pcs, n_pcs, n_scouts, n_subjects))
        similarity = np.empty((n_scouts, n_pcs, n_subjects))
        aligned_evr = np.empty((n_scouts, n_pcs, n_subjects))

        for output_scout in range(n_scouts):
            # Uncentered SVD: the reference subspace is invariant to the
            # arbitrary sign of each subject's components.
            stacked = np.concatenate(
                [fit["components"][output_scout] for fit in fits], axis=0
            )
            _, _, right = np.linalg.svd(stacked, full_matrices=False)
            reference = orient_largest_loading_positive(right[:n_pcs])
            references.append(reference.astype(np.float32))

            for output_subject, fit in enumerate(fits):
                subject_components = fit["components"][output_scout]
                # Orthogonal Procrustes: argmin ||Q @ C - R|| over orthogonal Q.
                u, _, vt = np.linalg.svd(reference @ subject_components.T)
                rotation = u @ vt
                aligned = rotation @ subject_components
                fit["components"][output_scout] = aligned
                rotations[:, :, output_scout, output_subject] = rotation
                similarity[output_scout, :, output_subject] = np.sum(
                    aligned * reference, axis=1
                )
                # The original PCs are uncorrelated, so an aligned component's
                # variance is the rotation-weighted sum of their variances.
                original_variance = fit["variance"][output_scout]
                total = (
                    original_variance.sum()
                    / fit["explained_variance_ratio"][output_scout].sum()
                )
                aligned_evr[output_scout, :, output_subject] = (
                    rotation**2 @ original_variance
                ) / total

        return {
            "reference_components": references,
            "rotations": rotations.astype(np.float32),
            "reference_similarity": similarity.astype(np.float32),
            "aligned_explained_variance_ratio": aligned_evr.astype(np.float32),
        }

    def _baseline_zscore(
        self, scores: np.ndarray, subject_index: int, scout_index: int
    ) -> None:
        baseline = scores[:, self.baseline_indices, :]
        baseline_mean = baseline.mean(axis=1, keepdims=True)
        baseline_std = baseline.std(axis=1, keepdims=True)
        if not np.isfinite(baseline_std).all() or np.any(baseline_std <= 0):
            raise ValueError(
                f"Subject {subject_index}, scout {scout_index}: zero or "
                "non-finite baseline SD in PC scores."
            )
        scores -= baseline_mean
        scores /= baseline_std

    def _subject_scores(self, subject_index: int, fit: dict) -> np.ndarray:
        kernel, eeg_valid, valid_trials = self._subject_sensor_data(subject_index)
        if not np.array_equal(valid_trials, fit["valid_trials"]):
            raise RuntimeError(f"Subject {subject_index}: valid trials changed.")
        n_valid = valid_trials.size
        eeg_2d = eeg_valid.reshape(eeg_valid.shape[0], -1)
        eeg_2d -= eeg_2d.mean(axis=1, keepdims=True)
        del eeg_valid

        block = np.full(self.output_shape[:4], np.nan, dtype=np.float32)
        for output_scout, scout_index in enumerate(self.scout_selection):
            kernel_scout = np.asarray(
                kernel[self.scout_source_rows[scout_index]], dtype=np.float64
            )
            # Aligned PC loadings mapped back to a PC x channel spatial filter.
            spatial_filter = fit["components"][output_scout] @ kernel_scout
            scores = (spatial_filter @ eeg_2d).reshape(
                self.args.n_pcs, self.n_times, n_valid
            )
            if self.args.baseline_zscore:
                self._baseline_zscore(scores, subject_index, scout_index)
            block[:, output_scout, :, valid_trials] = scores.transpose(2, 0, 1)
        return block

    def _check_disk_space(self, temporary_output: Path) -> None:
        required = int(np.prod(self.output_shape, dtype=np.int64)) * 4
        free = shutil.disk_usage(temporary_output.parent).free
        print(f"Output size: {required / 1e9:.2f} GB (free: {free / 1e9:.2f} GB)")
        if required > free:
            raise OSError("Not enough free disk space for the output array.")

    def run(self) -> None:
        print("\nPass 1/3: per-subject scout PCA...")
        fits = []
        with threadpool_limits(limits=self.args.blas_threads):
            for subject_index in tqdm(self.subject_selection, desc="Subjects"):
                fits.append(self._fit_subject(subject_index))
                gc.collect()

            print("Pass 2/3: Procrustes alignment to per-scout reference bases...")
            alignment = self._align(fits)

            similarity = alignment["reference_similarity"]
            print(
                "Aligned-to-reference cosine per PC (median over scouts/subjects):",
                np.round(np.median(similarity, axis=(0, 2)), 3),
            )

            print("Pass 3/3: projecting aligned PCs and writing output...")
            self.args.output.parent.mkdir(parents=True, exist_ok=True)
            temporary_output = self.args.output.with_name(
                self.args.output.name + ".tmp"
            )
            self._check_disk_space(temporary_output)
            output = np.lib.format.open_memmap(
                temporary_output,
                mode="w+",
                dtype=np.float32,
                shape=self.output_shape,
                fortran_order=True,
            )
            for output_subject, subject_index in enumerate(
                tqdm(self.subject_selection, desc="Subjects")
            ):
                output[..., output_subject] = self._subject_scores(
                    subject_index, fits[output_subject]
                )
                output.flush()
                gc.collect()
            del output
        os.replace(temporary_output, self.args.output)

        n_valid_trials = np.array([fit["valid_trials"].size for fit in fits])
        metadata = {
            "version": VERSION,
            "data_path": str(self.args.output),
            "axis_order": "PC x scout x time x trial x subject",
            "shape": self.output_shape,
            "subIdx": self.subject_groups[self.subject_selection],
            "subject_indices": np.asarray(self.subject_selection, dtype=int),
            "scout_indices": np.asarray(self.scout_selection, dtype=int),
            "scout_labels": [
                self.scout_labels[index] for index in self.scout_selection
            ],
            "times_ms": self.times,
            "valid_trials": [fit["valid_trials"] for fit in fits],
            "n_valid_trials": n_valid_trials,
            "pcaExplainedVarianceRatio": np.stack(
                [fit["explained_variance_ratio"] for fit in fits], axis=-1
            ).astype(np.float32),
            "reference_components": alignment["reference_components"],
            "rotations": alignment["rotations"],
            "reference_similarity": alignment["reference_similarity"],
            "aligned_explained_variance_ratio": alignment[
                "aligned_explained_variance_ratio"
            ],
            "field_shapes": (
                "pcaExplainedVarianceRatio: scout x PC x subject (unaligned PCs); "
                "aligned_explained_variance_ratio, reference_similarity: "
                "scout x PC x subject; rotations: PC x PC x scout x subject; "
                "reference_components: per scout, PC x source row"
            ),
            "parameters": {
                "n_pcs": self.args.n_pcs,
                "pca_fit_scope": (
                    "one exact PCA per subject/scout on all valid trials and "
                    "retained time points; features are the scout's X/Y/Z "
                    "source rows"
                ),
                "alignment": (
                    "orthogonal Procrustes rotation onto the top right singular "
                    "vectors of the selected subjects' stacked loadings; "
                    "reference oriented with its largest loading positive"
                ),
                "baseline_zscore": self.args.baseline_zscore,
                "baseline_window_ms": (BASELINE_START_MS, BASELINE_END_MS),
                "input_signatures": {
                    "eeg": file_signature(self.args.eeg),
                    "atlas": file_signature(self.args.atlas),
                    "tmatrix": file_signature(self.args.tmatrix),
                    "subject_index": file_signature(self.args.subject_index),
                },
            },
        }
        metadata_path = self.args.output.with_name(
            self.args.output.stem + "_metadata.pkl"
        )
        atomic_pickle_dump(metadata, metadata_path)

        print(f"\nSaved PC scores to {self.args.output}")
        print(f"Saved metadata to {metadata_path}")
        print("Shape (PC x scout x time x trial x subject):", self.output_shape)
        print(
            "Valid trials per subject: "
            f"min {n_valid_trials.min()}, max {n_valid_trials.max()} "
            f"of {self.n_trials}"
        )


def main() -> None:
    ScoutPCAExtractor(parse_args()).run()


if __name__ == "__main__":
    main()
