"""Keep source responses only where scout decoding is significant.

Each group-average source-response map (vertex x time) from main_source_response.py
is masked with the same group's one-sample decoding-vs-chance CBP result from
main_sentence_stats.py (three-group grouping): a scout's vertices keep their response
at the scout's significant time points and are zero elsewhere. Intention conditions use
the con_Int CBP, sentiment conditions the sent_noBio CBP. These masks require cluster
p-values corrected across Control, Depressed, and Suicidal. Vertices outside every DKT
scout are zero.

The masked maps show the response magnitude where a group's sentences are decodable
above chance. They show neither the pattern that separates the conditions nor a group
difference: each group is masked with its own test against chance.

Example (run from Code/):
    python3 main_source_significant.py --source-dir ../Results/source_response_norm/evoked_baseline-zscore
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mat73
import numpy as np

from utils import io, paths
from utils import source_response as sr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=paths.RESULTS / "source_response_norm" / "evoked_baseline-zscore",
    )
    parser.add_argument(
        "--intention-cbp-dir", type=Path,
        default=paths.RESULTS / "CBP_Chance_scout_con_Int_3group_corrected"
    )
    parser.add_argument(
        "--sentiment-cbp-dir",
        type=Path,
        default=paths.RESULTS / "CBP_Chance_scout_sent_noBio_3group_corrected",
    )
    parser.add_argument(
        "--atlas", type=Path, default=paths.ATLAS
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: '<source-dir>_significant_3group_corrected' next to the source directory.",
    )
    return parser.parse_args()


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
    labels = [io.clean_label(label, index) for index, label in enumerate(raw_labels)]
    return vertices, labels


def run(args: argparse.Namespace) -> None:
    source_dir = args.source_dir
    output_dir = args.output_dir or source_dir.with_name(
        source_dir.name + "_significant_3group_corrected"
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
    for group, group_key in sr.GROUPS.items():
        output_files[group] = {}
        mask_files[group] = {}
        cbp_info[group] = {}
        for task, conditions in sr.TASK_CONDITIONS.items():
            statistics_path = sr.find_group_statistics(cbp_dirs[task], group_key)
            scout_mask, info = sr.load_scout_mask(statistics_path, times_ms, scout_labels)
            cbp_info[group][task] = info
            mask_file = f"scout_mask_{group}_{task}.npy"
            io.save_array(output_dir / mask_file, scout_mask)
            mask_files[group][task] = mask_file

            for condition in conditions:
                filename = source_metadata["condition_files"][group][condition]
                response = np.load(source_dir / filename)
                if response.shape != (n_vertices, times_ms.size):
                    raise ValueError(
                        f"{filename} has shape {response.shape}; expected "
                        f"{(n_vertices, times_ms.size)}."
                    )
                io.save_array(
                    output_dir / filename,
                    sr.mask_response(response, scout_vertices, scout_mask),
                )
                output_files[group][condition] = filename
            print(
                f"{group} ({group_key}) {task}: {info['n_significant_scouts']} "
                f"significant scouts, {info['n_significant_scout_time_points']} "
                "scout-time points",
                flush=True,
            )

    io.save_array(output_dir / "times_ms.npy", times_ms)
    io.save_array(output_dir / "vertex_indices_1based.npy", vertex_indices)
    io.save_json(
        output_dir / "metadata.json",
        {
            "method": (
                "Group source response kept at each scout's significant "
                "decoding-vs-chance CBP time points (same group); zero at "
                "non-significant time points and at vertices outside every scout."
            ),
            "interpretation": (
                "Response magnitude where each group's decoding exceeds chance; not the "
                "discriminating pattern and not a group difference."
            ),
            "inputs": {
                "source_metadata": io.file_signature(source_dir / "metadata.json"),
                "atlas": io.file_signature(args.atlas),
                "intention_cbp_dir": str(args.intention_cbp_dir.resolve()),
                "sentiment_cbp_dir": str(args.sentiment_cbp_dir.resolve()),
            },
            "source_response_mode": source_metadata["response_mode"],
            "source_normalization": source_metadata["normalization"],
            "group_to_cbp_group": sr.GROUPS,
            "task_conditions": sr.TASK_CONDITIONS,
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
