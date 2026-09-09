"""Memory-efficient, time-resolved within-subject SVM decoding.

Expected input shape:
    PCs x scouts x time points x trials x subjects

The decoder processes one subject and one fold at a time. Augmented arrays are
discarded immediately after their fold is decoded, and each completed subject
is checkpointed so an interrupted analysis can resume safely.
"""

import os
import pickle
from pathlib import Path

import mat73
import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from tqdm import tqdm


class SVMDecoder:
    def __init__(
        self,
        fpath: str,
        bPath: str,
        fileName: str,
        IdxName: str,
        logName: str,
        k_fold: int = 5,
        Trial_num: int = 125,
        avg_num: int = 12,
        state: int = 42,
        outputPath: str | None = None,
        saveName: str = "svmDecoding_scout_3pc.pkl",
        n_jobs: int = -1,
        bootstrap_chunk_size: int = 10,
        resume: bool = True,
    ):
        self.fpath = Path(fpath)
        self.bPath = Path(bPath)
        self.fileName = fileName
        self.IdxName = IdxName
        self.logName = logName
        self.kfold = k_fold
        self.Trial_num = Trial_num
        self.avg_num = avg_num
        self.state = state
        self.output_directory = Path(outputPath) if outputPath else Path.cwd()
        self.saveName = saveName
        self.n_jobs = n_jobs
        self.bootstrap_chunk_size = bootstrap_chunk_size
        self.resume = resume

        self.n_train_per_class = int(self.Trial_num * 0.8)
        self.n_test_per_class = int(self.Trial_num * 0.2)

        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.checkpoint_directory = (
            self.output_directory / f"{Path(self.saveName).stem}_subjects"
        )
        self.checkpoint_directory.mkdir(parents=True, exist_ok=True)

        self.run()

    def load_inputs(self):
        """Load the PC data, group index, and per-subject trial indices."""
        print("Loading dataset")

        with open(self.fpath / self.fileName, "rb") as file:
            self.dataset = np.asarray(pickle.load(file))

        self.subject_groups = np.asarray(
            mat73.loadmat(str(self.bPath / self.IdxName))["subject_index"]
        ).squeeze()

        with open(self.bPath / self.logName, "rb") as file:
            self.sentiment_indices = pickle.load(file)["Sentiment"]

        self._validate_inputs()

        # The source file is already float32, but this protects against a
        # float64 input silently doubling all intermediate memory usage.
        if self.dataset.dtype != np.float32:
            print(f"Converting input from {self.dataset.dtype} to float32")
            self.dataset = self.dataset.astype(np.float32)

    def _validate_inputs(self):
        if self.dataset.ndim != 5:
            raise ValueError(
                "Dataset must have shape "
                "(PCs, scouts, time, trials, subjects); received "
                f"{self.dataset.shape}."
            )

        if self.dataset.shape[0] != 3:
            raise ValueError(f"Expected 3 PCs; received {self.dataset.shape[0]}.")

        n_subjects = self.dataset.shape[4]
        if self.subject_groups.size != n_subjects:
            raise ValueError(
                "Dataset and subject_index have different subject counts: "
                f"{n_subjects} and {self.subject_groups.size}."
            )
        if len(self.sentiment_indices) != n_subjects:
            raise ValueError(
                "Dataset and Sentiment have different subject counts: "
                f"{n_subjects} and {len(self.sentiment_indices)}."
            )
        if self.kfold < 2:
            raise ValueError("k_fold must be at least 2.")
        if self.n_train_per_class < 1 or self.n_test_per_class < 1:
            raise ValueError(
                "Trial_num must yield at least one train and test sample per class."
            )
        if self.avg_num < 1:
            raise ValueError("avg_num must be at least 1.")
        if self.bootstrap_chunk_size < 1:
            raise ValueError("bootstrap_chunk_size must be at least 1.")

    def _make_subject_splits(self, subject_index):
        """Create positive and negative K-fold splits for one subject."""
        positive = np.asarray(
            self.sentiment_indices[subject_index]["positive"], dtype=int
        ).reshape(-1)
        negative = np.asarray(
            self.sentiment_indices[subject_index]["negative"], dtype=int
        ).reshape(-1)

        if positive.size < self.kfold or negative.size < self.kfold:
            raise ValueError(
                f"Subject {subject_index} needs at least {self.kfold} trials "
                f"per class; found {positive.size} positive and "
                f"{negative.size} negative."
            )

        n_trials = self.dataset.shape[3]
        for label, indices in (("positive", positive), ("negative", negative)):
            if np.any(indices < 0) or np.any(indices >= n_trials):
                raise IndexError(
                    f"Subject {subject_index} has an out-of-range {label} index."
                )

        splitter = KFold(
            n_splits=self.kfold,
            shuffle=True,
            random_state=self.state,
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

    def _bootstrap_average(self, subject_data, trial_pool, sample_count, rng):
        """Generate bootstrap averages in chunks to bound temporary memory."""
        trial_pool = np.asarray(trial_pool, dtype=int)
        output_shape = subject_data.shape[:3] + (sample_count,)
        output = np.empty(output_shape, dtype=np.float32)

        for start in range(0, sample_count, self.bootstrap_chunk_size):
            stop = min(start + self.bootstrap_chunk_size, sample_count)
            selected_trials = rng.choice(
                trial_pool,
                size=(stop - start, self.avg_num),
                replace=True,
            )

            # Shape before averaging:
            # PCs x scouts x time x chunk samples x averaged trials
            selected_data = np.take(subject_data, selected_trials, axis=3)
            output[..., start:stop] = selected_data.mean(
                axis=-1,
                dtype=np.float32,
            )

        return output

    def _augment_fold(self, subject_data, split, rng):
        """Create train and test arrays for one subject/fold."""
        train_positive = self._bootstrap_average(
            subject_data,
            split["positive"]["train"],
            self.n_train_per_class,
            rng,
        )
        train_negative = self._bootstrap_average(
            subject_data,
            split["negative"]["train"],
            self.n_train_per_class,
            rng,
        )
        test_positive = self._bootstrap_average(
            subject_data,
            split["positive"]["test"],
            self.n_test_per_class,
            rng,
        )
        test_negative = self._bootstrap_average(
            subject_data,
            split["negative"]["test"],
            self.n_test_per_class,
            rng,
        )

        train_data = np.concatenate((train_positive, train_negative), axis=-1)
        test_data = np.concatenate((test_positive, test_negative), axis=-1)
        return train_data, test_data

    @staticmethod
    def _standardize_fold(train_data, test_data):
        """Standardize each PC separately for every scout and time point."""
        # StandardScaler also accumulates these statistics in float64. The
        # small float64 statistics arrays retain its numerical behavior while
        # the much larger train/test arrays remain float32.
        mean = train_data.mean(axis=-1, keepdims=True, dtype=np.float64)
        scale = train_data.std(axis=-1, keepdims=True, dtype=np.float64)
        scale[scale == 0] = 1.0

        # Modify these disposable fold arrays in place to reduce allocations.
        train_data -= mean
        train_data /= scale
        test_data -= mean
        test_data /= scale

    def _decode_scout(self, scout, train_data, test_data):
        """Decode all time points for one scout in one fold."""
        n_times = train_data.shape[2]
        scores = np.empty(n_times, dtype=np.float32)

        y_train = np.concatenate(
            (
                np.zeros(self.n_train_per_class, dtype=np.int8),
                np.ones(self.n_train_per_class, dtype=np.int8),
            )
        )

        for time_point in range(n_times):
            classifier = SVC(
                kernel="linear",
                C=1,
                tol=1e-3,
                # Classes are exactly balanced after augmentation, so
                # class_weight="balanced" would assign both classes weight 1.
                class_weight=None,
                cache_size=50,
            )
            classifier.fit(
                train_data[:, scout, time_point, :].T,
                y_train,
            )
            prediction = classifier.predict(test_data[:, scout, time_point, :].T)

            # Test samples are ordered positive then negative. Computing the
            # two recalls directly avoids balanced_accuracy_score overhead in
            # millions of tiny classifier calls.
            positive_accuracy = np.mean(prediction[: self.n_test_per_class] == 0)
            negative_accuracy = np.mean(prediction[self.n_test_per_class :] == 1)
            scores[time_point] = 0.5 * (positive_accuracy + negative_accuracy)

        return scout, scores

    def _decode_fold(self, train_data, test_data):
        """Decode the scouts concurrently for one fold."""
        self._standardize_fold(train_data, test_data)
        n_scouts = train_data.shape[1]
        n_times = train_data.shape[2]
        fold_scores = np.empty((n_scouts, n_times), dtype=np.float32)

        results = Parallel(
            n_jobs=self.n_jobs,
            prefer="threads",
            batch_size=1,
        )(
            delayed(self._decode_scout)(scout, train_data, test_data)
            for scout in range(n_scouts)
        )

        for scout, scores in results:
            fold_scores[scout] = scores

        return fold_scores

    def _process_subject(self, subject_index):
        """Split, augment, and decode one subject, then release its arrays."""
        # A contiguous subject copy makes repeated trial-axis access much
        # faster than slicing the last axis of the five-dimensional source.
        subject_data = np.ascontiguousarray(
            self.dataset[..., subject_index], dtype=np.float32
        )
        subject_splits = self._make_subject_splits(subject_index)

        n_scouts = subject_data.shape[1]
        n_times = subject_data.shape[2]
        fold_scores = np.empty((self.kfold, n_scouts, n_times), dtype=np.float32)

        fold_progress = tqdm(
            range(self.kfold),
            desc=f"Subject {subject_index + 1}: folds",
            leave=False,
        )
        for fold in fold_progress:
            # Independent seeds make every subject/fold reproducible even if
            # earlier subjects are loaded from checkpoints.
            rng = np.random.default_rng(
                np.random.SeedSequence([self.state, subject_index, fold])
            )
            split = {
                "positive": subject_splits["positive"][fold],
                "negative": subject_splits["negative"][fold],
            }

            train_data, test_data = self._augment_fold(subject_data, split, rng)
            fold_scores[fold] = self._decode_fold(train_data, test_data)

            # No augmented data survives this fold.
            del train_data, test_data

        subject_score = fold_scores.mean(axis=0, dtype=np.float64).astype(np.float32)
        return subject_score, subject_splits

    def _parameters(self):
        return {
            "input_file": self.fileName,
            "k_fold": self.kfold,
            "Trial_num": self.Trial_num,
            "avg_num": self.avg_num,
            "state": self.state,
            "positive_label": 0,
            "negative_label": 1,
        }

    def _checkpoint_path(self, subject_index):
        return self.checkpoint_directory / f"subject_{subject_index:03d}.pkl"

    @staticmethod
    def _atomic_pickle_dump(value, destination):
        """Write a pickle atomically so interruption cannot corrupt it."""
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        with open(temporary, "wb") as file:
            pickle.dump(value, file, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, destination)

    def _load_checkpoint(self, subject_index, expected_shape):
        if not self.resume:
            return None

        path = self._checkpoint_path(subject_index)
        if not path.exists():
            return None

        with open(path, "rb") as file:
            checkpoint = pickle.load(file)

        if checkpoint.get("parameters") != self._parameters():
            print(f"Ignoring stale checkpoint: {path.name}")
            return None

        score = np.asarray(checkpoint.get("decodeScore"))
        if score.shape != expected_shape:
            print(f"Ignoring invalid checkpoint: {path.name}")
            return None

        return checkpoint

    def _save_subject_checkpoint(self, subject_index, score, split_data):
        checkpoint = {
            "subject_index": subject_index,
            "subject_group": self.subject_groups[subject_index],
            "decodeScore": score,
            "split_data": split_data,
            "parameters": self._parameters(),
        }
        self._atomic_pickle_dump(checkpoint, self._checkpoint_path(subject_index))

    def decode_all_subjects(self):
        """Stream subjects through the complete analysis and checkpoint each."""
        n_subjects = self.dataset.shape[4]
        score_shape = (self.dataset.shape[1], self.dataset.shape[2])
        decode_scores = np.empty((n_subjects,) + score_shape, dtype=np.float32)
        split_data = [None] * n_subjects

        subjects = tqdm(range(n_subjects), desc="Subjects")
        for subject_index in subjects:
            checkpoint = self._load_checkpoint(subject_index, score_shape)
            if checkpoint is not None:
                subjects.set_postfix_str("checkpoint")
                decode_scores[subject_index] = checkpoint["decodeScore"]
                split_data[subject_index] = checkpoint["split_data"]
                continue

            subjects.set_postfix_str("decoding")
            score, subject_split = self._process_subject(subject_index)
            decode_scores[subject_index] = score
            split_data[subject_index] = subject_split
            self._save_subject_checkpoint(subject_index, score, subject_split)

        self.results = {
            "split_data": split_data,
            "decodeScore": decode_scores,
            "subIdx": self.subject_groups,
            "axis_order": (
                "input: PC x scout x time x trial x subject; "
                "decodeScore: subject x scout x time"
            ),
            "parameters": self._parameters(),
        }

    def save_final_results(self):
        destination = self.output_directory / self.saveName
        self._atomic_pickle_dump(self.results, destination)
        print(f"Saved combined results to {destination}")
        print(f"Subject checkpoints are in {self.checkpoint_directory}")

    def run(self):
        self.load_inputs()
        self.decode_all_subjects()
        self.save_final_results()


if __name__ == "__main__":
    decoder = SVMDecoder(
        fpath=(
            "/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/" "Data/Preprocessed data/"
        ),
        bPath=("/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/" "Data/Behavior/"),
        fileName="Data_sen_lepoch_DKT_scout_3PC.pkl",
        IdxName="subject_index.mat",
        logName="senIdx_congruent.pkl",
        k_fold=5,
        Trial_num=125,
        avg_num=12,
        state=42,
        outputPath=(
            "/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/" "Code/SourceLoc/Results/"
        ),
        saveName="svmDecoding_scout_3pc.pkl",
        n_jobs=-1,
        bootstrap_chunk_size=10,
        resume=True,
    )
