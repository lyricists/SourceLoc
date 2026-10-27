"""Time-resolved sentence decoding within each DKT scout (linear SVM per time point).

See decoders/sentence_decoder.py for the leakage-controlled pipeline. The result holds
``decodeScore`` (scout x time x subject, balanced accuracy 0-1) and ``subIdx`` and is
the input of main_sentence_stats.py and main_decoding_plot.py.

--condition picks the trial-index file and output name of the two analyses:
  con_Int      congruent vs incongruent intention sentences (senIdx_congruency_Intention.pkl)
  sent_noBio   positive/neutral vs negative sentences, no biographical (senIdx_sentiment_noBio.pkl)

Examples (run from Code/):
    python3 main_sentence_decoding.py --condition con_Int
    python3 main_sentence_decoding.py --condition sent_noBio --subjects 0 1 --scouts 13
"""

from __future__ import annotations

import argparse
from pathlib import Path

from decoders.sentence_decoder import FoldPCAScoutDecoder
from utils import paths

CONDITIONS = {"con_Int": paths.INTENTION_INDEX, "sent_noBio": paths.SENTIMENT_INDEX}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--condition", choices=CONDITIONS, default="con_Int")
    parser.add_argument("--eeg", type=Path, default=paths.EEG)
    parser.add_argument("--atlas", type=Path, default=paths.ATLAS)
    parser.add_argument("--tmatrix", type=Path, default=paths.TMATRIX)
    parser.add_argument("--subject-index", type=Path, default=paths.SUBJECT_INDEX)
    parser.add_argument("--sentiment-index", type=Path, default=None,
                        help="Trial-index file; default: the one of --condition")
    parser.add_argument("--output", type=Path, default=None,
                        help="Default: Results/svmDecoding_scout_<condition>.pkl")
    parser.add_argument("--k-fold", type=int, default=5)
    parser.add_argument("--trial-num", type=int, default=125)
    parser.add_argument("--avg-num", type=int, default=12)
    parser.add_argument("--n-pcs", type=int, default=3)
    parser.add_argument("--state", type=int, default=42)
    parser.add_argument("--bootstrap-chunk-size", type=int, default=10)
    parser.add_argument("--pca-transform-trial-chunk", type=int, default=16)
    parser.add_argument("--pca-solver", choices=("randomized", "full"), default="randomized",
                        help="Randomized SVD is the practical default for retaining 3 components from large scouts; "
                             "use full for exact but substantially slower PCA.")
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument("--subjects", type=int, nargs="*", default=None,
                        help="Optional zero-based subject indices; default: every subject.")
    parser.add_argument("--scouts", type=int, nargs="*", default=None,
                        help="Optional zero-based scout indices; default: every scout.")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                        help="Load compatible subject/scout checkpoints when available.")
    args = parser.parse_args()
    args.sentiment_index = args.sentiment_index or CONDITIONS[args.condition]
    args.output = args.output or paths.RESULTS / f"svmDecoding_scout_{args.condition}.pkl"
    return args


def main() -> None:
    FoldPCAScoutDecoder(parse_args()).run()


if __name__ == "__main__":
    main()
