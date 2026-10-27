#!/bin/bash
# ============================================================
# Time-resolved sentence decoding per scout -> scout x time CBP vs chance
# -> group time-course plots, for both sentence contrasts.
# Run from Code/:  ./run_sentence_decoding.sh
# ============================================================

PYTHON_EXEC="python3"
RESULTS="../Results"

# Decoding is the expensive step (137 subjects x 62 scouts x 5 folds x 300 SVMs);
# the existing results were computed on the cluster. Set RUN_DECODING=1 to recompute.
RUN_DECODING=${RUN_DECODING:-0}

CONDITIONS=("con_Int" "sent_noBio")
GROUPINGS=("three" "control-patients")
N_PERMUTATIONS=10000

for condition in "${CONDITIONS[@]}"; do
  echo ""
  echo ">>> Condition: ${condition}"
  echo "==========================================================="
  if [ "${RUN_DECODING}" = "1" ]; then
    ${PYTHON_EXEC} main_sentence_decoding.py --condition "${condition}" || exit 1
  fi
  for grouping in "${GROUPINGS[@]}"; do
    ${PYTHON_EXEC} main_sentence_stats.py \
        --input "${RESULTS}/svmDecoding_scout_${condition}.pkl" \
        --grouping "${grouping}" \
        --n-permutations ${N_PERMUTATIONS} || exit 1
  done
  ${PYTHON_EXEC} main_decoding_plot.py --input "${RESULTS}/svmDecoding_scout_${condition}.pkl" || exit 1
done

echo ""
echo "All sentence-decoding statistics and plots finished."
