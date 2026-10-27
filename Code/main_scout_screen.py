"""Single-scout group classification screen (EEGNet per DKT scout).

Each subject's intention responses are projected to one scout, reduced to shared scout
PCs, and turned into incongruent - congruent bootstrap averages; see
decoders/subject_classifier.py. The screen's per-scout AUCs select the scouts for
main_group_classifier.py, and its best scout is that script's single-scout reference.

Examples (run from Code/):
    python3 main_scout_screen.py --smoke-test
    python3 main_scout_screen.py --scout all --device cpu --jobs 12
    python3 main_scout_screen.py --scout 13 --permute-seed 1     # one label-permutation run
"""

from __future__ import annotations

import argparse
from pathlib import Path

from decoders import subject_classifier as sc
from utils import io, paths, source


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scout", dest="scouts", nargs="+", help="Zero-based scouts (spaces or commas), or 'all'")
    p.add_argument("--pair", dest="pairs", nargs="+", choices=source.PAIRS, default=["control-vs-depressed+suicidal"])
    p.add_argument("--eeg", type=Path, default=paths.EEG)
    p.add_argument("--tmatrix", type=Path, default=paths.TMATRIX)
    p.add_argument("--atlas", type=Path, default=paths.ATLAS)
    p.add_argument("--subject-index", type=Path, default=paths.SUBJECT_INDEX)
    p.add_argument("--trial-index", type=Path, default=paths.TOI_INDEX)
    p.add_argument("--category", default="Intention", help="Key under each subject in --trial-index; 'none' if absent")
    p.add_argument("--cache-dir", type=Path, default=paths.SUBJECT_CACHE)
    p.add_argument("--output-dir", type=Path, default=paths.RESULTS / "scout_subject_cds_all")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--n-components", type=int, default=32)
    p.add_argument("--window", nargs=2, type=int, default=[0, 800], metavar=("START_MS", "END_MS"),
                   help="Model input window; the end is exclusive")
    p.add_argument("--avg-range", nargs=2, type=int, default=[8, 24], metavar=("LO", "HI"),
                   help="Trials per training bootstrap average, drawn per example")
    p.add_argument("--draws-per-epoch", type=int, default=16, help="New bootstrap averages per subject per epoch")
    p.add_argument("--tta-avg", type=int, default=16, help="Trials per test bootstrap average")
    p.add_argument("--tta-draws", type=int, default=32, help="Test bootstrap averages per subject")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--avg-epochs", type=int, default=10, help="Average the weights of the final N epochs")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3, help="Peak of the one-cycle learning rate")
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--max-shift", type=int, default=3, help="Time-shift augmentation in samples")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--permute-seed", type=int, default=None, help="Shuffle group labels to build a null distribution")
    p.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    p.add_argument("--jobs", type=int, default=1,
                   help="Parallel CPU folds; with more than one, per-epoch metrics go to the history files only")
    p.add_argument("--smoke-test", action="store_true", help="Short run on synthetic data")
    args = p.parse_args()
    if args.smoke_test:
        args.scouts = args.scouts or ["0", "1"]
    elif not args.scouts:
        p.error("--scout is required")
    if [s.lower() for s in args.scouts] == ["all"]:
        args.scouts = "all"
    else:
        try:
            args.scouts = [int(x) for tok in args.scouts for x in tok.split(",") if x.strip()]
        except ValueError:
            p.error("--scout takes zero-based integers or 'all'")
        if not args.scouts or min(args.scouts) < 0 or len(set(args.scouts)) != len(args.scouts):
            p.error("--scout needs distinct nonnegative indices")
    if len(set(args.pairs)) != len(args.pairs):
        p.error("--pair names must be distinct")
    if args.avg_range[0] < 1 or args.avg_range[1] < args.avg_range[0]:
        p.error("--avg-range needs 1 <= LO <= HI")
    if args.window[0] < paths.EPOCH_START_MS or args.window[1] <= args.window[0]:
        p.error("--window must start at or after -200 ms and end after it starts")
    for name in ("n_components", "draws_per_epoch", "tta_avg", "tta_draws", "epochs", "avg_epochs", "batch_size",
                 "folds", "repeats", "jobs"):
        if getattr(args, name) < 1:
            p.error(f"--{name.replace('_', '-')} must be positive")
    if args.avg_epochs > args.epochs:
        p.error("--avg-epochs cannot exceed --epochs")
    if args.jobs > 1 and args.device not in ("auto", "cpu"):
        p.error("--jobs above 1 runs on CPU; drop --device or use --device cpu")
    return args


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        args.cache_dir, args.trial_index = source.build_synthetic_cache()
        args.output_dir, args.category = args.cache_dir.parent / "output", "Intention"
        args.repeats, args.epochs, args.avg_epochs = 1, 3, 2
    meta = source.ensure_cache(args.cache_dir, args.eeg, args.tmatrix, args.atlas, args.subject_index)
    scouts = list(range(len(meta["vertices"]))) if args.scouts == "all" else args.scouts
    if max(scouts) >= len(meta["vertices"]):
        raise ValueError(f"scouts must be between 0 and {len(meta['vertices']) - 1}")
    device = sc.choose_device("cpu" if args.jobs > 1 else args.device)
    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    print(f"Device {device} | {len(scouts)} scout(s) | pairs {', '.join(args.pairs)} | "
          f"{args.repeats} x {args.folds}-fold CV | output {sc.scout_output_root(cfg)}", flush=True)
    sc.scout_output_root(cfg).mkdir(parents=True, exist_ok=True)
    collected = {}
    if args.jobs == 1:
        for pair in args.pairs:
            print(f"\n=== {pair} ===", flush=True)
            for scout in scouts:
                data = sc.ScoutData(cfg, meta, scout, pair)
                for repeat in range(args.repeats):
                    for fold in range(args.folds):
                        r = sc.run_scout_fold(cfg, meta, scout, pair, repeat, fold, device, True, data)
                        print(sc.fold_line(r, with_name=False), flush=True)
                        collected.setdefault((pair, scout), []).append(r)
    else:
        from joblib import Parallel, delayed
        tasks = [(s, p, r, f) for p in args.pairs for s in scouts for r in range(args.repeats) for f in range(args.folds)]
        for r in Parallel(n_jobs=args.jobs, return_as="generator_unordered")(
                delayed(sc.run_scout_fold)(cfg, meta, s, p, rep, f, device, False) for s, p, rep, f in tasks):
            print(sc.fold_line(r), flush=True)
            collected.setdefault((r["pair"], r["scout"]), []).append(r)
    for pair in args.pairs:
        print(f"\n=== {pair} ===", flush=True)
        rows = [sc.summarize_scout(cfg, scout, pair, collected[(pair, scout)]) for scout in scouts]
        io.save_csv(sc.scout_output_root(cfg) / f"{pair}_scouts.csv",
                    [{k: (";".join(map(str, v)) if isinstance(v, list) else v) for k, v in row.items()} for row in rows])
    if args.smoke_test:
        print(f"Smoke-test output: {args.output_dir}")


if __name__ == "__main__":
    main()
