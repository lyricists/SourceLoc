"""Keep source responses only where scout decoding is significant.

Each group-average source-response map (vertex x time) from
source_response_norm.py is masked with the same group's one-sample
decoding-vs-chance CBP result: a scout's vertices keep their response at the
scout's significant time points and are zero elsewhere. Vertices outside every
DKT scout are zero.

"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
from pathlib import Path

import mat73
import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "Results"
DATA = ROOT.parent.parent / "Data"

VERSION = 1
# Source-response group prefix -> CBP group key.
GROUPS = {"c": "control", "d": "depressed", "s": "suicidal"}
# CBP task -> source-response conditions decoded by that task.
TASK_CONDITIONS = {
    "intention": ("congruent_intention", "incongruent_intention"),
    "sentiment": ("positive_neutral_noBio", "negative_noBio"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=RESULTS / "source_response_norm" / "evoked_baseline-zscore",
    )
    parser.add_argument(
        "--intention-cbp-dir", type=Path, default=RESULTS / "CBP_Chance_scout_con_Int"
    )
    parser.add_argument(
        "--sentiment-cbp-dir",
        type=Path,
        default=RESULTS / "CBP_Chance_scout_sent_noBio",
    )
    parser.add_argument(
        "--atlas", type=Path, default=DATA / "Preprocessed data" / "DKT_scout.mat"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: '<source-dir>_significant' next to the source directory.",
    )
    return parser.parse_args()


def file_signature(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def load_pickle(path: Path):
    with path.open("rb") as file:
        return pickle.load(file)


def atomic_save_array(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as file:
        np.save(file, value)
    os.replace(temporary, path)


def atomic_save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2)
        file.write("\n")
    os.replace(temporary, path)


def clean_label(label: object, index: int) -> str:
    clean = re.sub(r"\s+", " ", str(label)).strip()
    return clean or f"Scout {index + 1}"


def load_atlas(path: Path, n_vertices: int) -> tuple[list[np.ndarray], list[str]]:
    """Return zero-based vertex indices and labels per scout."""
    atlas = mat73.loadmat(str(path))["DKT_scout"]
    raw_labels = atlas.get("Label", atlas.get("Region"))
    raw_vertices = atlas["Vertices"]
    if raw_labels is None or len(raw_labels) != len(raw_vertices):
        raise ValueError("Atlas must contain one label per scout.")

    vertices = []
    for index, values in enumerate(raw_vertices):
        scout_vertices = np.asarray(values, dtype=np.int64).reshape(-1) - 1
        if scout_vertices.size == 0:
            raise ValueError(f"Scout {index + 1} has no vertices.")
        if np.any((scout_vertices < 0) | (scout_vertices >= n_vertices)):
            raise IndexError(f"Scout {index + 1} has an out-of-range vertex index.")
        vertices.append(scout_vertices)
    all_vertices = np.concatenate(vertices)
    if np.unique(all_vertices).size != all_vertices.size:
        raise ValueError("Atlas scouts share vertices; masking would be ambiguous.")
    labels = [clean_label(label, index) for index, label in enumerate(raw_labels)]
    return vertices, labels


def find_group_statistics(cbp_dir: Path, group_key: str) -> Path:
    candidates = sorted(
        (cbp_dir / "spatiotemporal_statistics").glob(f"group_{group_key}_scouts_*.pkl")
    )
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one '{group_key}' statistics pickle in "
            f"{cbp_dir / 'spatiotemporal_statistics'}; found {len(candidates)}."
        )
    return candidates[0]


def load_scout_mask(
    path: Path, times_ms: np.ndarray, atlas_labels: list[str]
) -> tuple[np.ndarray, dict]:
    """Return a time x atlas-scout significance mask from one CBP result."""
    statistics = load_pickle(path)
    cbp_times = np.asarray(statistics["times_ms"], dtype=np.float64)
    if cbp_times.shape != times_ms.shape or not np.allclose(cbp_times, times_ms):
        raise ValueError(f"{path.name}: CBP time vector differs from the source maps.")

    scout_indices = np.asarray(statistics["scout_indices"], dtype=np.intp)
    significant = np.asarray(statistics["significant_mask"], dtype=bool)
    if significant.shape != (times_ms.size, scout_indices.size):
        raise ValueError(
            f"{path.name}: significant_mask has shape {significant.shape}; expected "
            f"{(times_ms.size, scout_indices.size)}."
        )
    if np.any((scout_indices < 0) | (scout_indices >= len(atlas_labels))):
        raise IndexError(f"{path.name}: scout index outside the atlas.")
    cbp_labels = [
        clean_label(label, int(scout))
        for label, scout in zip(statistics["scout_labels"], scout_indices)
    ]
    expected_labels = [atlas_labels[scout] for scout in scout_indices]
    if cbp_labels != expected_labels:
        raise ValueError(f"{path.name}: scout labels do not match the atlas.")

    # Scouts the CBP run did not test stay non-significant.
    mask = np.zeros((times_ms.size, len(atlas_labels)), dtype=bool)
    mask[:, scout_indices] = significant
    info = {
        "statistics": file_signature(path),
        "cluster_alpha": float(statistics["test"]["cluster_alpha"]),
        "n_subjects": int(statistics["n_subjects"]),
        "tested_scout_indices": scout_indices.tolist(),
        "n_significant_scouts": int(mask.any(axis=0).sum()),
        "n_significant_scout_time_points": int(mask.sum()),
    }
    return mask, info


def mask_response(
    response: np.ndarray, scout_vertices: list[np.ndarray], scout_mask: np.ndarray
) -> np.ndarray:
    masked = np.zeros_like(response)
    for scout, vertices in enumerate(scout_vertices):
        times = np.flatnonzero(scout_mask[:, scout])
        if times.size:
            masked[np.ix_(vertices, times)] = response[np.ix_(vertices, times)]
    return masked


def run(args: argparse.Namespace) -> None:
    source_dir = args.source_dir
    output_dir = args.output_dir or source_dir.with_name(
        source_dir.name + "_significant"
    )
    if output_dir.resolve() == source_dir.resolve():
        raise ValueError("Output directory must differ from the source directory.")
    cbp_dirs = {
        "intention": args.intention_cbp_dir,
        "sentiment": args.sentiment_cbp_dir,
    }

    source_metadata = json.loads((source_dir / "metadata.json").read_text())
    times_ms = np.load(source_dir / "times_ms.npy")
    vertex_indices = np.load(source_dir / "vertex_indices_1based.npy")
    n_vertices = int(source_metadata["n_vertices"])
    if times_ms.ndim != 1 or times_ms.size != source_metadata["n_times"]:
        raise ValueError("times_ms.npy does not match the source metadata.")
    if vertex_indices.shape != (n_vertices,):
        raise ValueError(
            "vertex_indices_1based.npy does not match the source metadata."
        )
    scout_vertices, scout_labels = load_atlas(args.atlas, n_vertices)
    print(
        f"{len(scout_labels)} scouts cover {sum(v.size for v in scout_vertices)} of "
        f"{n_vertices} vertices; {times_ms.size} time points.",
        flush=True,
    )

    output_files = {}
    mask_files = {}
    cbp_info = {}
    for group, group_key in GROUPS.items():
        output_files[group] = {}
        mask_files[group] = {}
        cbp_info[group] = {}
        for task, conditions in TASK_CONDITIONS.items():
            statistics_path = find_group_statistics(cbp_dirs[task], group_key)
            scout_mask, info = load_scout_mask(statistics_path, times_ms, scout_labels)
            cbp_info[group][task] = info
            mask_file = f"scout_mask_{group}_{task}.npy"
            atomic_save_array(output_dir / mask_file, scout_mask)
            mask_files[group][task] = mask_file

            for condition in conditions:
                filename = source_metadata["condition_files"][group][condition]
                response = np.load(source_dir / filename)
                if response.shape != (n_vertices, times_ms.size):
                    raise ValueError(
                        f"{filename} has shape {response.shape}; expected "
                        f"{(n_vertices, times_ms.size)}."
                    )
                atomic_save_array(
                    output_dir / filename,
                    mask_response(response, scout_vertices, scout_mask),
                )
                output_files[group][condition] = filename
            print(
                f"{group} ({group_key}) {task}: {info['n_significant_scouts']} "
                f"significant scouts, {info['n_significant_scout_time_points']} "
                "scout-time points",
                flush=True,
            )

    atomic_save_array(output_dir / "times_ms.npy", times_ms)
    atomic_save_array(output_dir / "vertex_indices_1based.npy", vertex_indices)
    atomic_save_json(
        output_dir / "metadata.json",
        {
            "version": VERSION,
            "method": (
                "Group source response kept at each scout's significant "
                "decoding-vs-chance CBP time points (same group); zero at "
                "non-significant time points and at vertices outside every scout."
            ),
            "inputs": {
                "source_metadata": file_signature(source_dir / "metadata.json"),
                "atlas": file_signature(args.atlas),
                "intention_cbp_dir": str(args.intention_cbp_dir.resolve()),
                "sentiment_cbp_dir": str(args.sentiment_cbp_dir.resolve()),
            },
            "source_response_mode": source_metadata["response_mode"],
            "source_normalization": source_metadata["normalization"],
            "group_to_cbp_group": GROUPS,
            "task_conditions": TASK_CONDITIONS,
            "cbp_statistics": cbp_info,
            "condition_files": output_files,
            "condition_file_axes": "vertex x time",
            "scout_mask_files": mask_files,
            "scout_mask_axes": "time x scout",
            "scout_labels": scout_labels,
            "n_vertices": n_vertices,
            "n_vertices_in_scouts": int(sum(v.size for v in scout_vertices)),
            "n_times": int(times_ms.size),
            "times_file": "times_ms.npy",
            "vertex_indices_file": "vertex_indices_1based.npy",
        },
    )
    n_maps = sum(len(files) for files in output_files.values())
    print(f"Saved {n_maps} masked vertex x time group maps to {output_dir}", flush=True)


if __name__ == "__main__":
    run(parse_args())
