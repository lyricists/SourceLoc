"""Atomic writes, input signatures, and the atlas / subject / trial-index loaders."""

from __future__ import annotations

import csv
import json
import os
import pickle
import re
from pathlib import Path

import numpy as np


def _atomic(path: Path, mode: str, write) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, mode, **({"encoding": "utf-8", "newline": ""} if "b" not in mode else {})) as f:
        write(f)
    os.replace(tmp, path)


def save_pickle(path: Path, value: object) -> None:
    _atomic(path, "wb", lambda f: pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL))


def save_json(path: Path, value: object) -> None:
    def write(f):
        json.dump(value, f, indent=1)
        f.write("\n")
    _atomic(path, "w", write)


def save_array(path: Path, value: np.ndarray) -> None:
    _atomic(path, "wb", lambda f: np.save(f, value))


def save_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if not rows and fieldnames is None:
        return

    def write(f):
        writer = csv.DictWriter(f, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _atomic(path, "w", write)


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def file_signature(path: Path | str) -> dict:
    stat = Path(path).stat()
    return {"path": str(Path(path).resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def signatures_match(saved: dict | None, current: dict, legacy: dict | None = None) -> bool:
    """Compare a stored signature with the current one.

    ``legacy`` maps keys that older runs stored but the current code no longer records
    to the value they must have held (None accepts any value), so results from before
    the restructure stay reusable.
    """
    if saved is None:
        return False
    saved = dict(saved)
    for key, value in (legacy or {}).items():
        if key in saved and key not in current and (value is None or saved[key] == value):
            del saved[key]
    return saved == current


# Subject groupings: name -> ((key, label, subject_index codes pooled into the group), ...)
GROUPINGS = {
    "three": (("control", "Control", (1,)), ("depressed", "Depressed", (2,)), ("suicidal", "Suicidal", (3,))),
    "control-patients": (("control", "Control", (1,)), ("patients", "Patients (D+S)", (2, 3))),
}


def matlab_to_python_index(indices: object) -> np.ndarray:
    return np.asarray(indices).squeeze().astype(int) - 1


def vertex_to_xyz_rows(vertices_0based: np.ndarray) -> np.ndarray:
    """Kernel rows of each vertex: vertex v -> rows 3v, 3v + 1, 3v + 2."""
    vertices = np.asarray(vertices_0based, dtype=int).reshape(-1)
    return (3 * vertices[:, None] + np.arange(3)).ravel()


def clean_label(label: object, index: int) -> str:
    clean = re.sub(r"\s+", " ", str(label)).strip()
    return clean or f"Scout {index + 1}"


DKT_NAMES = {
    "caudalanteriorcingulate": "Caudal Anterior Cingulate", "caudalmiddlefrontal": "Caudal Middle Frontal",
    "cuneus": "Cuneus", "entorhinal": "Entorhinal", "fusiform": "Fusiform", "inferiorparietal": "Inferior Parietal",
    "inferiortemporal": "Inferior Temporal", "insula": "Insula", "isthmuscingulate": "Isthmus Cingulate",
    "lateraloccipital": "Lateral Occipital", "lateralorbitofrontal": "Lateral Orbitofrontal", "lingual": "Lingual",
    "medialorbitofrontal": "Medial Orbitofrontal", "middletemporal": "Middle Temporal",
    "parahippocampal": "Parahippocampal", "paracentral": "Paracentral", "parsopercularis": "Pars Opercularis",
    "parsorbitalis": "Pars Orbitalis", "parstriangularis": "Pars Triangularis", "pericalcarine": "Pericalcarine",
    "postcentral": "Postcentral", "posteriorcingulate": "Posterior Cingulate", "precentral": "Precentral",
    "precuneus": "Precuneus", "rostralanteriorcingulate": "Rostral Anterior Cingulate",
    "rostralmiddlefrontal": "Rostral Middle Frontal", "superiorfrontal": "Superior Frontal",
    "superiorparietal": "Superior Parietal", "superiortemporal": "Superior Temporal",
    "supramarginal": "Supramarginal", "transversetemporal": "Transverse Temporal",
}


def pretty_scout_name(label: str) -> str:
    """Readable DKT region name, e.g. 'middletemporal R' -> 'Middle Temporal R'."""
    region, _, hemisphere = label.strip().partition(" ")
    return f"{DKT_NAMES.get(region, region)} {hemisphere}".strip()


def load_atlas(path: Path) -> tuple[list[np.ndarray], list[str]]:
    """Zero-based vertex indices and cleaned labels per DKT scout."""
    import mat73
    atlas = mat73.loadmat(str(path))["DKT_scout"]
    raw_labels = atlas.get("Label", atlas.get("Region"))
    if raw_labels is None or len(raw_labels) != len(atlas["Vertices"]):
        raise ValueError("Atlas must contain one label per scout.")
    vertices = [np.asarray(v, dtype=np.int64).reshape(-1) - 1 for v in atlas["Vertices"]]
    return vertices, [clean_label(label, i) for i, label in enumerate(raw_labels)]


def load_scout_labels(path: Path, n_scouts: int | None = None) -> list[str]:
    labels = load_atlas(path)[1]
    if n_scouts is not None and len(labels) != n_scouts:
        raise ValueError(f"Atlas must contain {n_scouts} scout labels.")
    return labels


def load_subject_groups(path: Path) -> np.ndarray:
    """Group code per subject (1 control, 2 depressed, 3 suicidal)."""
    import mat73
    raw = np.asarray(mat73.loadmat(str(path))["subject_index"]).reshape(-1)
    if not np.isfinite(raw).all() or not np.equal(raw, np.floor(raw)).all():
        raise ValueError("subject_index must contain finite integer group codes.")
    return raw.astype(int)


def load_trial_index(path: Path) -> list:
    """Per-subject trial-index entries (``["Sentiment"]`` of the behavior pickle)."""
    return load_pickle(path)["Sentiment"]
