"""Default data and result locations, in one place."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # SourceLoc
RESULTS = ROOT / "Results"
DATA = ROOT.parent.parent / "Data"
PREPROCESSED = DATA / "Preprocessed data"
BEHAVIOR = DATA / "Behavior"

EEG = PREPROCESSED / "Data_sen_bl.pkl"  # channel x time x trial x subject, 250 Hz, -200 to <1000 ms
TMATRIX = PREPROCESSED / "tMatrix.pkl"  # per-subject inverse kernel and GoodChannel
ATLAS = PREPROCESSED / "DKT_scout.mat"
SUBJECT_INDEX = BEHAVIOR / "subject_index.mat"  # 1 control, 2 depressed, 3 suicidal

# Trial indices; always read as pickle.load(f)["Sentiment"][subject]["positive" | "negative"].
INTENTION_INDEX = BEHAVIOR / "senIdx_congruency_Intention.pkl"  # positive = congruent
SENTIMENT_INDEX = BEHAVIOR / "senIdx_sentiment_noBio.pkl"
TOI_INDEX = BEHAVIOR / "senIdx_TOI.pkl"  # per category ("Intention", ...) under each subject

SUBJECT_CACHE = RESULTS / "subject_sensor_cache"  # per-subject sensor trials and kernels (classifiers)
PROJECTIONS = RESULTS / "multi_scout_classifier"  # holds projections/: per-fold scout PC matrices

FS = 250
EPOCH_START_MS = -200
