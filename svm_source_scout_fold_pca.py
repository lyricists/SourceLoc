"""Leakage-controlled scout-wise source SVM decoding.

This module combines the source-projection logic from ``sourceLocalization.py``
with the decoding logic from ``svm_source_scout.py`` so that PCA is fitted
inside each subject/scout cross-validation fold using training trials only.

The inverse kernel and anatomical scout definition are fixed, label-independent
transforms and are therefore applied before the fold-specific PCA. For every
subject and scout, source projection is computed once and reused across folds.
Each fold then follows this sequence:

    training-trial PCA fit -> train/test PCA transform -> per-trial baseline
    z-score -> separate train/test pseudotrial construction -> train-derived
    feature scaling -> linear SVM at each time point

Each completed subject/scout is atomically checkpointed. The default output is
separate from the original result so both analyses remain available.
"""

from __future__ import annotations

import argparse
import gc
import os
import pickle
from pathlib import Path

import mat73
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from tqdm import tqdm


BASE_DIRECTORY = Path(
    "/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Code/SourceLoc"
)
DATA_DIRECTORY = BASE_DIRECTORY.parent.parent / "Data"
PREPROCESSED_DIRECTORY = DATA_DIRECTORY / "Preprocessed data"
BEHAVIOR_DIRECTORY = DATA_DIRECTORY / "Behavior"

DEFAULT_EEG = PREPROCESSED_DIRECTORY / "Data_sen_lepoch.pkl"
DEFAULT_ATLAS = PREPROCESSED_DIRECTORY / "DKT_scout.mat"
DEFAULT_TMATRIX = PREPROCESSED_DIRECTORY / "tMatrix.pkl"
DEFAULT_SUBJECT_INDEX = BEHAVIOR_DIRECTORY / "subject_index.mat"
DEFAULT_SENTIMENT_INDEX = BEHAVIOR_DIRECTORY / "senIdx_congruent.pkl"
DEFAULT_OUTPUT = (
    BASE_DIRECTORY / "Results" / "svmDecoding_scout_3pc_fold_pca.pkl"
)

FS = 250
T_START_MS = -200
T_END_ORIGINAL_MS = 1500
ANALYSIS_END_MS = 1000
BASELINE_START_MS = -200
BASELINE_END_MS = 0
EPSILON = 1e-8
ALGORITHM_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eeg", type=Path, default=DEFAULT_EEG)
    parser.add_argument("--atlas", type=Path, default=DEFAULT_ATLAS)
    parser.add_argument("--tmatrix", type=Path, default=DEFAULT_TMATRIX)
    parser.add_argument("--subject-index", type=Path, default=DEFAULT_SUBJECT_INDEX)
    parser.add_argument(
        "--sentiment-index", type=Path, default=DEFAULT_SENTIMENT_INDEX
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--k-fold", type=int, default=5)
    parser.add_argument("--trial-num", type=int, default=125)
    parser.add_argument("--avg-num", type=int, default=12)
    parser.add_argument("--n-pcs", type=int, default=3)
    parser.add_argument("--state", type=int, default=42)
    parser.add_argument("--bootstrap-chunk-size", type=int, default=10)
    parser.add_argument("--pca-transform-trial-chunk", type=int, default=16)
    parser.add_argument(
        "--pca-solver",
        choices=("randomized", "full"),
        default="randomized",
        help=(
            "Randomized SVD is the practical default for retaining 3 components "
            "from large scouts; use full for exact but substantially slower PCA."
        ),
    )
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument(
        "--subjects",
        type=int,
        nargs="*",
        default=None,
        help="Optional zero-based subject indices; default: every subject.",
    )
    parser.add_argument(
        "--scouts",
        type=int,
        nargs="*",
        default=None,
        help="Optional zero-based scout indices; default: every scout.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load compatible subject/scout checkpoints when available.",
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


def get_subject_item(values: object, subject_index: int) -> object:
    return values[subject_index]


class FoldPCAScoutDecoder:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self._validate_settings()
        self._load_inputs()
        self._prepare_dimensions()
        self._prepare_selection()
        self._prepare_output()

    def _validate_settings(self) -> None:
        if self.args.k_fold < 2:
            raise ValueError("k-fold must be at least 2.")
        if self.args.trial_num < 2:
            raise ValueError("trial-num must be at least 2.")
        if self.args.avg_num < 1 or self.args.n_pcs < 1:
            raise ValueError("avg-num and n-pcs must be positive.")
        if self.args.bootstrap_chunk_size < 1:
            raise ValueError("bootstrap-chunk-size must be positive.")
        if self.args.pca_transform_trial_chunk < 1:
            raise ValueError("pca-transform-trial-chunk must be positive.")
        if self.args.blas_threads < 1:
            raise ValueError("blas-threads must be positive.")

        self.n_train_per_class = int(self.args.trial_num * 0.8)
        self.n_test_per_class = int(self.args.trial_num * 0.2)
        if self.n_train_per_class < 1 or self.n_test_per_class < 1:
            raise ValueError("trial-num must yield train and test pseudotrials.")

    def _load_inputs(self) -> None:
        print("Loading sensor EEG...")
        with open(self.args.eeg, "rb") as file:
            self.eeg_data = np.asarray(pickle.load(file))

        print("Loading inverse kernels and good-channel indices...")
        with open(self.args.tmatrix, "rb") as file:
            t_matrix = pickle.load(file)
        self.good_channels = t_matrix["GoodChannel"]
        self.kernels = t_matrix["kernel"]

        print("Loading atlas and behavioral indices...")
        atlas = mat73.loadmat(str(self.args.atlas))["DKT_scout"]
        self.scout_vertices = atlas["Vertices"]
        raw_labels = atlas.get("Label", atlas.get("Region"))
        self.scout_labels = [str(label).strip() for label in raw_labels]

        self.subject_groups = np.asarray(
            mat73.loadmat(str(self.args.subject_index))["subject_index"]
        ).squeeze()
        with open(self.args.sentiment_index, "rb") as file:
            self.sentiment_indices = pickle.load(file)["Sentiment"]

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
        if len(self.sentiment_indices) != self.n_subjects:
            raise ValueError("Sentiment index count does not match the EEG data.")
        if len(self.scout_labels) != self.n_scouts:
            raise ValueError("Scout label count does not match the atlas.")

        original_times = np.arange(
            T_START_MS, T_END_ORIGINAL_MS, 1000 / FS, dtype=float
        )
        if original_times.size != self.n_times_original:
            raise ValueError(
                f"Expected {original_times.size} original time points but EEG has "
                f"{self.n_times_original}."
            )
        self.time_indices = np.flatnonzero(
            (original_times >= T_START_MS) & (original_times <= ANALYSIS_END_MS)
        )
        self.times = original_times[self.time_indices]
        self.n_times = self.times.size
        self.baseline_indices = np.flatnonzero(
            (self.times >= BASELINE_START_MS) & (self.times <= BASELINE_END_MS)
        )
        if self.n_times != 301 or self.baseline_indices.size == 0:
            raise ValueError("Unexpected analysis or baseline time-vector length.")

        self.scout_source_rows = []
        for vertices in self.scout_vertices:
            vertices_0based = matlab_to_python_index(vertices)
            if np.any(vertices_0based < 0):
                raise IndexError("Atlas contains a non-positive MATLAB vertex index.")
            self.scout_source_rows.append(vertex_to_xyz_rows(vertices_0based))

        print("Sensor EEG shape:", self.eeg_data.shape)
        print(
            "Analysis shape components:",
            f"PC={self.args.n_pcs}, scout={self.n_scouts}, time={self.n_times},",
            f"trial={self.n_trials}, subject={self.n_subjects}",
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

    def _prepare_output(self) -> None:
        self.args.output.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint_directory = (
            self.args.output.parent / f"{self.args.output.stem}_subject_scouts"
        )
        self.checkpoint_directory.mkdir(parents=True, exist_ok=True)

        self.parameters = {
            "algorithm_version": ALGORITHM_VERSION,
            "k_fold": self.args.k_fold,
            "Trial_num": self.args.trial_num,
            "avg_num": self.args.avg_num,
            "n_pcs": self.args.n_pcs,
            "state": self.args.state,
            "bootstrap_chunk_size": self.args.bootstrap_chunk_size,
            "pca_solver": self.args.pca_solver,
            "pca_fit_scope": (
                "one PCA per subject/scout/fold; fit on positive and negative "
                "training trials across all retained time points"
            ),
            "baseline_window_ms": (BASELINE_START_MS, BASELINE_END_MS),
            "positive_label": 0,
            "negative_label": 1,
            "input_signatures": {
                "eeg": file_signature(self.args.eeg),
                "atlas": file_signature(self.args.atlas),
                "tmatrix": file_signature(self.args.tmatrix),
                "subject_index": file_signature(self.args.subject_index),
                "sentiment_index": file_signature(self.args.sentiment_index),
            },
        }

    def _make_subject_splits(self, subject_index: int) -> dict:
        sentiment = self.sentiment_indices[subject_index]
        positive = np.asarray(sentiment["positive"], dtype=int).reshape(-1)
        negative = np.asarray(sentiment["negative"], dtype=int).reshape(-1)

        for label, indices in (("positive", positive), ("negative", negative)):
            if indices.size < self.args.k_fold:
                raise ValueError(
                    f"Subject {subject_index}: {label} has fewer trials than folds."
                )
            if np.unique(indices).size != indices.size:
                raise ValueError(f"Subject {subject_index}: duplicate {label} trials.")
            if np.any(indices < 0) or np.any(indices >= self.n_trials):
                raise IndexError(
                    f"Subject {subject_index}: out-of-range {label} trial index."
                )
        if np.intersect1d(positive, negative).size:
            raise ValueError(
                f"Subject {subject_index}: positive and negative trials overlap."
            )

        splitter = KFold(
            n_splits=self.args.k_fold,
            shuffle=True,
            random_state=self.args.state,
        )
        positive_folds = [
            {"train": positive[train], "test": positive[test]}
            for train, test in splitter.split(positive)
        ]
        negative_folds = [
            {"train": negative[train], "test": negative[test]}
            for train, test in splitter.split(negative)
        ]
        return {"positive": positive_folds, "negative": negative_folds}

    def _prepare_subject_sensor_data(
        self, subject_index: int
    ) -> tuple[np.ndarray, np.ndarray]:
        good_channels = matlab_to_python_index(
            get_subject_item(self.good_channels, subject_index)
        )
        if np.any(good_channels < 0) or np.any(good_channels >= self.n_channels):
            raise IndexError(f"Subject {subject_index}: invalid good-channel index.")

        kernel = np.asarray(
            get_subject_item(self.kernels, subject_index), dtype=np.float32
        )
        eeg_subject = self.eeg_data[:, :, :, subject_index]
        eeg_subject = eeg_subject[:, self.time_indices, :].astype(
            np.float32, copy=False
        )

        if kernel.shape[1] == good_channels.size:
            eeg_subject = eeg_subject[good_channels, :, :]
        elif kernel.shape[1] != self.n_channels:
            raise ValueError(
                f"Subject {subject_index}: kernel columns {kernel.shape[1]} match "
                f"neither {good_channels.size} good channels nor "
                f"{self.n_channels} full channels."
            )

        eeg_2d = np.ascontiguousarray(
            eeg_subject.reshape(eeg_subject.shape[0], self.n_times * self.n_trials),
            dtype=np.float32,
        )
        return kernel, eeg_2d

    def _pca_seed(self, subject_index: int, scout_index: int, fold: int) -> int:
        sequence = np.random.SeedSequence(
            [self.args.state, subject_index, scout_index, fold, 9173]
        )
        return int(sequence.generate_state(1, dtype=np.uint32)[0])

    def _fit_training_pca(
        self,
        source_data: np.ndarray,
        training_trials: np.ndarray,
        subject_index: int,
        scout_index: int,
        fold: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Observations are every training-trial/time combination; features are
        # all X/Y/Z source rows belonging to this scout.
        training_source = np.take(source_data, training_trials, axis=2)
        x_train = np.ascontiguousarray(
            training_source.transpose(1, 2, 0).reshape(
                self.n_times * training_trials.size, source_data.shape[0]
            ),
            dtype=np.float32,
        )
        del training_source

        if min(x_train.shape) < self.args.n_pcs:
            raise ValueError(
                f"Cannot fit {self.args.n_pcs} PCs to training shape "
                f"{x_train.shape}."
            )

        pca = PCA(
            n_components=self.args.n_pcs,
            svd_solver=self.args.pca_solver,
            copy=False,
            random_state=(
                self._pca_seed(subject_index, scout_index, fold)
                if self.args.pca_solver == "randomized"
                else None
            ),
        )
        pca.fit(x_train)
        del x_train

        components = pca.components_.astype(np.float32, copy=True)
        for pc_index in range(self.args.n_pcs):
            largest_loading = np.argmax(np.abs(components[pc_index]))
            if components[pc_index, largest_loading] < 0:
                components[pc_index] *= -1

        mean = pca.mean_.astype(np.float32, copy=True)
        explained_variance_ratio = pca.explained_variance_ratio_.astype(
            np.float32, copy=True
        )
        return mean, components, explained_variance_ratio

    def _transform_all_trials(
        self,
        source_data: np.ndarray,
        pca_mean: np.ndarray,
        pca_components: np.ndarray,
    ) -> np.ndarray:
        pc_data = np.empty(
            (self.args.n_pcs, self.n_times, self.n_trials), dtype=np.float32
        )
        chunk_size = self.args.pca_transform_trial_chunk

        for start in range(0, self.n_trials, chunk_size):
            stop = min(start + chunk_size, self.n_trials)
            source_chunk = source_data[:, :, start:stop]
            x_chunk = np.ascontiguousarray(
                source_chunk.transpose(1, 2, 0).reshape(
                    self.n_times * (stop - start), source_data.shape[0]
                ),
                dtype=np.float32,
            )
            x_chunk -= pca_mean
            transformed = x_chunk @ pca_components.T
            pc_data[:, :, start:stop] = transformed.reshape(
                self.n_times, stop - start, self.args.n_pcs
            ).transpose(2, 0, 1)

        return pc_data

    def _baseline_zscore(self, pc_data: np.ndarray) -> np.ndarray:
        baseline = pc_data[:, self.baseline_indices, :]
        baseline_mean = baseline.mean(axis=1, keepdims=True, dtype=np.float32)
        baseline_std = baseline.std(axis=1, keepdims=True, dtype=np.float32)
        baseline_std[baseline_std < EPSILON] = EPSILON
        pc_data -= baseline_mean
        pc_data /= baseline_std
        return pc_data

    def _bootstrap_average(
        self,
        pc_data: np.ndarray,
        trial_pool: np.ndarray,
        sample_count: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        output = np.empty(
            (self.args.n_pcs, self.n_times, sample_count), dtype=np.float32
        )
        for start in range(0, sample_count, self.args.bootstrap_chunk_size):
            stop = min(start + self.args.bootstrap_chunk_size, sample_count)
            selected_trials = rng.choice(
                trial_pool,
                size=(stop - start, self.args.avg_num),
                replace=True,
            )
            selected_data = np.take(pc_data, selected_trials, axis=2)
            output[:, :, start:stop] = selected_data.mean(
                axis=-1, dtype=np.float32
            )
        return output

    def _augment_fold(
        self, pc_data: np.ndarray, split: dict, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        train_positive = self._bootstrap_average(
            pc_data,
            split["positive"]["train"],
            self.n_train_per_class,
            rng,
        )
        train_negative = self._bootstrap_average(
            pc_data,
            split["negative"]["train"],
            self.n_train_per_class,
            rng,
        )
        test_positive = self._bootstrap_average(
            pc_data,
            split["positive"]["test"],
            self.n_test_per_class,
            rng,
        )
        test_negative = self._bootstrap_average(
            pc_data,
            split["negative"]["test"],
            self.n_test_per_class,
            rng,
        )
        return (
            np.concatenate((train_positive, train_negative), axis=-1),
            np.concatenate((test_positive, test_negative), axis=-1),
        )

    @staticmethod
    def _standardize_fold(
        train_data: np.ndarray, test_data: np.ndarray
    ) -> None:
        mean = train_data.mean(axis=-1, keepdims=True, dtype=np.float32)
        scale = train_data.std(axis=-1, keepdims=True, dtype=np.float32)
        scale[scale == 0] = 1.0
        train_data -= mean
        train_data /= scale
        test_data -= mean
        test_data /= scale

    def _decode_fold(
        self, train_data: np.ndarray, test_data: np.ndarray
    ) -> np.ndarray:
        self._standardize_fold(train_data, test_data)
        scores = np.empty(self.n_times, dtype=np.float32)
        y_train = np.concatenate(
            (
                np.zeros(self.n_train_per_class, dtype=np.int8),
                np.ones(self.n_train_per_class, dtype=np.int8),
            )
        )

        for time_index in range(self.n_times):
            classifier = SVC(
                kernel="linear",
                C=1,
                tol=1e-3,
                class_weight=None,
                cache_size=50,
            )
            classifier.fit(train_data[:, time_index, :].T, y_train)
            prediction = classifier.predict(test_data[:, time_index, :].T)
            positive_accuracy = np.mean(
                prediction[: self.n_test_per_class] == 0
            )
            negative_accuracy = np.mean(
                prediction[self.n_test_per_class :] == 1
            )
            scores[time_index] = 0.5 * (
                positive_accuracy + negative_accuracy
            )
        return scores

    def _decode_subject_scout(
        self,
        subject_index: int,
        scout_index: int,
        kernel: np.ndarray,
        eeg_2d: np.ndarray,
        subject_splits: dict,
    ) -> dict:
        source_rows = self.scout_source_rows[scout_index]
        if np.any(source_rows < 0) or np.any(source_rows >= kernel.shape[0]):
            raise IndexError(
                f"Subject {subject_index}, scout {scout_index}: atlas source rows "
                f"do not match kernel shape {kernel.shape}."
            )
        if source_rows.size < self.args.n_pcs:
            raise ValueError(
                f"Scout {scout_index} has fewer source rows than requested PCs."
            )

        source_2d = kernel[source_rows, :] @ eeg_2d
        source_data = source_2d.reshape(
            source_rows.size, self.n_times, self.n_trials
        )
        del source_2d

        fold_scores = np.empty(
            (self.args.k_fold, self.n_times), dtype=np.float32
        )
        explained_variance_ratio = np.empty(
            (self.args.k_fold, self.args.n_pcs), dtype=np.float32
        )

        for fold in range(self.args.k_fold):
            split = {
                "positive": subject_splits["positive"][fold],
                "negative": subject_splits["negative"][fold],
            }
            training_trials = np.concatenate(
                (split["positive"]["train"], split["negative"]["train"])
            )
            pca_mean, pca_components, explained_variance_ratio[fold] = (
                self._fit_training_pca(
                    source_data,
                    training_trials,
                    subject_index,
                    scout_index,
                    fold,
                )
            )
            pc_data = self._transform_all_trials(
                source_data, pca_mean, pca_components
            )
            self._baseline_zscore(pc_data)

            # Reinitializing with the same subject/fold seed for every scout
            # reproduces identical pseudotrial membership across scouts.
            rng = np.random.default_rng(
                np.random.SeedSequence([self.args.state, subject_index, fold])
            )
            train_data, test_data = self._augment_fold(pc_data, split, rng)
            fold_scores[fold] = self._decode_fold(train_data, test_data)
            del (
                pca_mean,
                pca_components,
                pc_data,
                train_data,
                test_data,
            )

        return {
            "subject_index": subject_index,
            "subject_group": self.subject_groups[subject_index],
            "scout_index": scout_index,
            "scout_label": self.scout_labels[scout_index],
            "decodeScore": fold_scores.mean(axis=0, dtype=np.float64).astype(
                np.float32
            ),
            "foldDecodeScore": fold_scores,
            "pcaExplainedVarianceRatio": explained_variance_ratio,
            "parameters": self.parameters,
        }

    def _checkpoint_path(self, subject_index: int, scout_index: int) -> Path:
        return (
            self.checkpoint_directory
            / f"subject_{subject_index:03d}"
            / f"scout_{scout_index:03d}.pkl"
        )

    def _load_checkpoint(
        self, subject_index: int, scout_index: int
    ) -> dict | None:
        if not self.args.resume:
            return None
        path = self._checkpoint_path(subject_index, scout_index)
        if not path.exists():
            return None
        with open(path, "rb") as file:
            checkpoint = pickle.load(file)
        valid = (
            checkpoint.get("parameters") == self.parameters
            and checkpoint.get("subject_index") == subject_index
            and checkpoint.get("scout_index") == scout_index
            and np.asarray(checkpoint.get("decodeScore")).shape == (self.n_times,)
            and np.asarray(checkpoint.get("foldDecodeScore")).shape
            == (self.args.k_fold, self.n_times)
            and np.asarray(checkpoint.get("pcaExplainedVarianceRatio")).shape
            == (self.args.k_fold, self.args.n_pcs)
        )
        if not valid:
            print(f"Ignoring incompatible checkpoint: {path}")
            return None
        return checkpoint

    def _process_subject(self, subject_index: int) -> tuple[dict, dict]:
        subject_splits = self._make_subject_splits(subject_index)
        kernel = None
        eeg_2d = None
        subject_results = {}

        progress = tqdm(
            self.scout_selection,
            desc=f"Subject {subject_index + 1}: scouts",
            leave=False,
        )
        for scout_index in progress:
            checkpoint = self._load_checkpoint(subject_index, scout_index)
            if checkpoint is not None:
                progress.set_postfix_str("checkpoint")
                subject_results[scout_index] = checkpoint
                continue

            progress.set_postfix_str("decoding")
            if kernel is None or eeg_2d is None:
                kernel, eeg_2d = self._prepare_subject_sensor_data(subject_index)
            with threadpool_limits(limits=self.args.blas_threads):
                result = self._decode_subject_scout(
                    subject_index,
                    scout_index,
                    kernel,
                    eeg_2d,
                    subject_splits,
                )
            atomic_pickle_dump(
                result, self._checkpoint_path(subject_index, scout_index)
            )
            subject_results[scout_index] = result
            gc.collect()

        del kernel, eeg_2d
        gc.collect()
        return subject_results, subject_splits

    def run(self) -> dict:
        n_selected_subjects = len(self.subject_selection)
        n_selected_scouts = len(self.scout_selection)
        decode_scores = np.empty(
            (n_selected_scouts, self.n_times, n_selected_subjects),
            dtype=np.float32,
        )
        fold_scores = np.empty(
            (
                self.args.k_fold,
                n_selected_scouts,
                self.n_times,
                n_selected_subjects,
            ),
            dtype=np.float32,
        )
        pca_explained_variance = np.empty(
            (
                self.args.k_fold,
                n_selected_scouts,
                self.args.n_pcs,
                n_selected_subjects,
            ),
            dtype=np.float32,
        )
        split_data = []

        subjects = tqdm(self.subject_selection, desc="Subjects")
        for output_subject_index, subject_index in enumerate(subjects):
            subject_results, subject_splits = self._process_subject(subject_index)
            split_data.append(subject_splits)
            for output_scout_index, scout_index in enumerate(self.scout_selection):
                result = subject_results[scout_index]
                decode_scores[output_scout_index, :, output_subject_index] = result[
                    "decodeScore"
                ]
                fold_scores[
                    :, output_scout_index, :, output_subject_index
                ] = result["foldDecodeScore"]
                pca_explained_variance[
                    :, output_scout_index, :, output_subject_index
                ] = result["pcaExplainedVarianceRatio"]

        output = {
            "split_data": split_data,
            "decodeScore": decode_scores,
            "foldDecodeScore": fold_scores,
            "pcaExplainedVarianceRatio": pca_explained_variance,
            "subIdx": self.subject_groups[self.subject_selection],
            "subject_indices": np.asarray(self.subject_selection, dtype=int),
            "scout_indices": np.asarray(self.scout_selection, dtype=int),
            "scout_labels": [
                self.scout_labels[index] for index in self.scout_selection
            ],
            "times_ms": self.times,
            "axis_order": (
                "decodeScore: scout x time x subject; "
                "foldDecodeScore: fold x scout x time x subject; "
                "pcaExplainedVarianceRatio: fold x scout x PC x subject"
            ),
            "parameters": self.parameters,
        }
        atomic_pickle_dump(output, self.args.output)
        print(f"Saved leakage-controlled result to {self.args.output}")
        print("decodeScore shape:", decode_scores.shape)
        print("Checkpoint directory:", self.checkpoint_directory)
        return output


def main() -> None:
    decoder = FoldPCAScoutDecoder(parse_args())
    decoder.run()


if __name__ == "__main__":
    main()
