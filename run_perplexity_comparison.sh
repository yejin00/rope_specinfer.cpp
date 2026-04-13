#!/bin/bash

# Perplexity comparison: q4_0 vs q4_0_head
# Logs all output to files

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./perplexity_logs_${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

MODEL="../jongjip/Llama-3.1-8B-Instruct-Q4_0.gguf"
DATASET="../jongjip/executorch/QNN_test/wiki.test.raw"

echo "=== Perplexity Comparison Test ==="
echo "Timestamp: ${TIMESTAMP}"
echo "Model: ${MODEL}"
echo "Dataset: ${DATASET}"
echo "Logs will be saved to: ${LOG_DIR}"
echo ""

# Test 1: Per-block (q4_0)
echo "[1/2] Running per-block (q4_0)..."
./llama-perplexity-head \
    -m "${MODEL}" \
    -f "${DATASET}" \
    -ctk q4_0 \
    -ctv q4_0 \
    -fa on \
    -b 128 \
    --chunks 100 \
    2>&1 | tee "${LOG_DIR}/perplexity_q4_0.log"

echo ""
echo "[1/2] Completed. Log: ${LOG_DIR}/perplexity_q4_0.log"
echo ""

# Test 2: Per-head (q4_0_head)
echo "[2/2] Running per-head (q4_0_head)..."
./llama-perplexity-head \
    -m "${MODEL}" \
    -f "${DATASET}" \
    -ctk q4_0_head \
    -ctv q4_0_head \
    -fa on \
    -b 128 \
    --chunks 100 \
    2>&1 | tee "${LOG_DIR}/perplexity_q4_0_head.log"

echo ""
echo "[2/2] Completed. Log: ${LOG_DIR}/perplexity_q4_0_head.log"
echo ""

# Extract final perplexity values
echo "=== Results Summary ===" | tee "${LOG_DIR}/summary.txt"
echo "" | tee -a "${LOG_DIR}/summary.txt"
echo "Per-block (q4_0):" | tee -a "${LOG_DIR}/summary.txt"
grep -E "Final estimate:|perplexity:" "${LOG_DIR}/perplexity_q4_0.log" | tail -1 | tee -a "${LOG_DIR}/summary.txt"
echo "" | tee -a "${LOG_DIR}/summary.txt"
echo "Per-head (q4_0_head):" | tee -a "${LOG_DIR}/summary.txt"
grep -E "Final estimate:|perplexity:" "${LOG_DIR}/perplexity_q4_0_head.log" | tail -1 | tee -a "${LOG_DIR}/summary.txt"
echo "" | tee -a "${LOG_DIR}/summary.txt"
echo "All logs saved to: ${LOG_DIR}" | tee -a "${LOG_DIR}/summary.txt"
