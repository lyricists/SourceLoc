#!/bin/bash
# ============================================================
# network10 (flat multi-scout EEGNet) for the other TOI categories, every group pair:
# arms network10, all 62 scouts, and the 64 sensors (no random sets).
# The Intention runs are in group_classifier/ (C vs D+S) and group_classifier_<pair>/.
# network10 was chosen from the Intention C vs D+S results, so it is prespecified for
# these categories. There is no single-scout screen for them, so --screen-csv points to
# a file that does not exist and the best-single-scout comparison is skipped (otherwise
# it would compare with the Intention screen's predictions).
# Each category x pair has its own output and projection directory; finished folds are
# reused, so a rerun only computes what is missing.
# Run from Code/:  ./run_toi_network10.sh
# ============================================================

set -o pipefail

PYTHON_EXEC="python3"
RESULTS="../Results/group_classifier_toi"
JOBS=12
NETWORK10="13 21 25 27 36 42 48 50 52 57"

CATEGORIES=("Biography" "Reflection" "Action")
PAIRS=("control-vs-depressed+suicidal" "control-depressed" "control-suicidal" "depressed-suicidal")

mkdir -p "${RESULTS}"
for category in "${CATEGORIES[@]}"; do
  for pair in "${PAIRS[@]}"; do
    out="${RESULTS}/${category}/${pair}"
    echo ""
    echo ">>> ${category} | ${pair}"
    echo "==========================================================="
    ${PYTHON_EXEC} main_group_classifier.py --pair "${pair}" --category "${category}" \
        --set-name network10 --selected ${NETWORK10} --arms selected all sensor \
        --screen-csv "${RESULTS}/no_screen.csv" \
        --output-dir "${out}" --projection-root "${out}" \
        --device cpu --jobs ${JOBS} 2>&1 | tee -a "${RESULTS}/${category}_${pair}.log" || exit 1
  done
done

echo ""
echo "All TOI network10 runs finished."
