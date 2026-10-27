#!/bin/bash
# ============================================================
# Group (Control vs Depressed + Suicidal) analyses:
#   1. single-scout EEGNet screen (selects scouts, best single scout)
#   2. time-resolved group decoding per scout (ridge, label permutations)
#   3. flat multi-scout EEGNet on scout sets + ablation (random scout sets,
#      all scouts, sensors) and a summary across sets
# Finished folds and maps are reused, so reruns only compute what is missing.
# Every scout set is chosen with the labels of the same subjects (screen AUCs or
# group-decoding clusters), so the set comparisons are optimistic.
# Run from Code/:  ./run_group_classification.sh
# ============================================================

PYTHON_EXEC="python3"
PAIR="control-vs-depressed+suicidal"
JOBS=12                 # parallel CPU workers
THRESHOLD=0.60          # single-scout AUC for scout selection
N_RANDOM=100            # random scout sets per set size in the ablation

${PYTHON_EXEC} main_scout_screen.py --scout all --pair ${PAIR} --device cpu --jobs ${JOBS} || exit 1

${PYTHON_EXEC} main_group_decoding.py --pair ${PAIR} --jobs ${JOBS} || exit 1

# Scout sets (zero-based scout indices; README "Results so far"):
#   timeres3    significant group-decoding clusters (middletemporal R, superiortemporal R, medialorbitofrontal R)
#   timeres6    timeres3 + the three cluster-p 0.07-0.08 trends
#   star5       literature + group-decoding p < 0.10 (no random sets were run for it)
#   top10       the 10 best screen AUCs
#   network10   top10 minus paracentral R and transversetemporal R, plus medial and lateral OFC R
#   selected17  screen AUC > THRESHOLD
# The timeres/star5 indices come from the group-decoding result above and are fixed here;
# update them if that result changes.
run_set() {  # name, then the remaining main_group_classifier.py arguments
  local name=$1; shift
  ${PYTHON_EXEC} main_group_classifier.py --pair ${PAIR} --set-name "${name}" "$@" \
      --device cpu --jobs ${JOBS} || exit 1
}
run_set timeres3 --selected 25 27 57 --n-random ${N_RANDOM}
run_set timeres6 --selected 21 25 27 48 53 57 --n-random ${N_RANDOM}
run_set star5 --selected 21 25 27 48 57 --arms selected all sensor
run_set top10 --selected 13 27 29 36 42 48 50 52 57 61 --n-random ${N_RANDOM}
run_set network10 --selected 13 21 25 27 36 42 48 50 52 57 --n-random ${N_RANDOM}
run_set selected17 --threshold ${THRESHOLD} --n-random ${N_RANDOM}

${PYTHON_EXEC} main_ablation_summary.py --sets timeres3 timeres6 star5 network10 top10 selected17 || exit 1

echo ""
echo "All group analyses finished."
