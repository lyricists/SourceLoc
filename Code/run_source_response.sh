#!/bin/bash
# ============================================================
# Source-response maps per condition and group -> masked by significant
# scout decoding (needs the three-group CBP of run_sentence_decoding.sh).
# Run from Code/:  ./run_source_response.sh
# ============================================================

PYTHON_EXEC="python3"
RESULTS="../Results"

RESPONSE_MODE="evoked"            # evoked | trial-magnitude
NORMALIZATION="baseline-zscore"   # kernel | baseline-zscore (baseline-zscore needs evoked)

${PYTHON_EXEC} main_source_response.py \
    --response-mode ${RESPONSE_MODE} \
    --normalization ${NORMALIZATION} || exit 1

${PYTHON_EXEC} main_source_significant.py \
    --source-dir "${RESULTS}/source_response_norm/${RESPONSE_MODE}_${NORMALIZATION}" \
    --intention-cbp-dir "${RESULTS}/CBP_Chance_scout_con_Int_3group_corrected" \
    --sentiment-cbp-dir "${RESULTS}/CBP_Chance_scout_sent_noBio_3group_corrected" || exit 1

echo ""
echo "Source-response maps and masked maps finished."
