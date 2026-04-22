#!/usr/bin/env bash

set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT_DEFAULT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
REPO_ROOT="${REPO_ROOT:-${REPO_ROOT_DEFAULT}}"
GSM8K_PATH="${GSM8K_PATH:-${REPO_ROOT_DEFAULT}/gsm8k_train.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/yejin/gsm8k}"

MODEL_BASE="${MODEL_BASE:-/data/yejin/Llama-3.1-8B-Instruct.gguf}"
MODEL_RPN="${MODEL_RPN:-/data/yejin/Llama-3.1-8B-Instruct_rotated_rpn.gguf}"

NUM_SAMPLES="${NUM_SAMPLES:-200}"
CTX_SIZE="${CTX_SIZE:-4096}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
UBATCH_SIZE="${UBATCH_SIZE:-4096}"
N_GPU_LAYERS="${N_GPU_LAYERS:-99}"
FLASH_ATTN="${FLASH_ATTN:-on}"
PROMPT_FORMAT="${PROMPT_FORMAT:-chat-template}"

STOP_ON_ERROR=0

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Runs the 7 GSM8K few-shot evaluation presets sequentially.

Options:
  --python PATH            Python executable to use
  --repo-root PATH         Repo root containing tools/eval_gsm8k_fewshot.py
  --gsm8k-path PATH        GSM8K JSONL/JSON path
  --output-root PATH       Root directory for all outputs and logs
  --model-base PATH        Base instruct model path
  --model-rpn PATH         Rotated RPN model path
  --num-samples N          Number of evaluation samples (default: ${NUM_SAMPLES})
  --ctx-size N             Context size (default: ${CTX_SIZE})
  --batch-size N           Batch size (default: ${BATCH_SIZE})
  --ubatch-size N          Ubatch size (default: ${UBATCH_SIZE})
  --n-gpu-layers N         GPU layers (default: ${N_GPU_LAYERS})
  --flash-attn MODE        Flash attention mode: on/off/auto (default: ${FLASH_ATTN})
  --prompt-format FORMAT   Prompt format (default: ${PROMPT_FORMAT})
  --stop-on-error          Stop immediately when one run fails
  -h, --help               Show this help

Environment overrides are also supported for:
  PYTHON_BIN, REPO_ROOT, GSM8K_PATH, OUTPUT_ROOT,
  MODEL_BASE, MODEL_RPN, NUM_SAMPLES, CTX_SIZE,
  BATCH_SIZE, UBATCH_SIZE, N_GPU_LAYERS, FLASH_ATTN, PROMPT_FORMAT
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --python)
            PYTHON_BIN="$2"
            shift 2
            ;;
        --repo-root)
            REPO_ROOT="$2"
            shift 2
            ;;
        --gsm8k-path)
            GSM8K_PATH="$2"
            shift 2
            ;;
        --output-root)
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --model-base)
            MODEL_BASE="$2"
            shift 2
            ;;
        --model-rpn)
            MODEL_RPN="$2"
            shift 2
            ;;
        --num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --ctx-size)
            CTX_SIZE="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --ubatch-size)
            UBATCH_SIZE="$2"
            shift 2
            ;;
        --n-gpu-layers)
            N_GPU_LAYERS="$2"
            shift 2
            ;;
        --flash-attn)
            FLASH_ATTN="$2"
            shift 2
            ;;
        --prompt-format)
            PROMPT_FORMAT="$2"
            shift 2
            ;;
        --stop-on-error)
            STOP_ON_ERROR=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

mkdir -p "${OUTPUT_ROOT}"
LOG_DIR="${OUTPUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY_PATH="${OUTPUT_ROOT}/run_summary_${TIMESTAMP}.txt"

declare -a RESULTS=()

run_eval() {
    local name="$1"
    local model="$2"
    local output_dir="$3"
    shift 3
    
    mkdir -p "${output_dir}"

    local archive_log_path="${LOG_DIR}/${TIMESTAMP}_${name}.log"
    local run_log_path="${output_dir}/run.log"
    local command_path="${output_dir}/run_command.sh"
    local command_txt_path="${output_dir}/run_command.txt"
    local -a cmd=(
        "${PYTHON_BIN}" "${REPO_ROOT}/tools/eval_gsm8k_fewshot.py"
        --model "${model}"
        --repo-root "${REPO_ROOT}"
        --gsm8k-path "${GSM8K_PATH}"
        --shots 4 8 16
        --num-samples "${NUM_SAMPLES}"
        --output-dir "${output_dir}"
        --prompt-format "${PROMPT_FORMAT}"
        --ctx-size "${CTX_SIZE}"
        --batch-size "${BATCH_SIZE}"
        --ubatch-size "${UBATCH_SIZE}"
        --n-gpu-layers "${N_GPU_LAYERS}"
        --flash-attn "${FLASH_ATTN}"
    )
    cmd+=("$@")

    {
        echo "#!/usr/bin/env bash"
        echo "set -euo pipefail"
        printf '%q' "${cmd[0]}"
        local i
        for (( i = 1; i < ${#cmd[@]}; ++i )); do
            printf ' %q' "${cmd[i]}"
        done
        printf '\n'
    } > "${command_path}"
    chmod +x "${command_path}"

    printf '%q' "${cmd[0]}" > "${command_txt_path}"
    local j
    for (( j = 1; j < ${#cmd[@]}; ++j )); do
        printf ' %q' "${cmd[j]}" >> "${command_txt_path}"
    done
    printf '\n' >> "${command_txt_path}"

    {
        echo "============================================================"
        echo "[$(date '+%F %T')] Starting ${name}"
        printf 'Command:'
        printf ' %q' "${cmd[@]}"
        printf '\n'
        echo "Run log: ${run_log_path}"
        echo "Archive log: ${archive_log_path}"
        echo "Command script: ${command_path}"
        echo "Output dir: ${output_dir}"
        echo "============================================================"
    } | tee -a "${SUMMARY_PATH}"

    local -a stream_prefix=()
    if command -v stdbuf >/dev/null 2>&1; then
        stream_prefix=(stdbuf -oL -eL)
    fi

    if env PYTHONUNBUFFERED=1 "${stream_prefix[@]}" "${cmd[@]}" 2>&1 | tee "${run_log_path}" "${archive_log_path}"; then
        RESULTS+=("OK   ${name}  ${output_dir}")
        echo "[$(date '+%F %T')] Finished ${name}: OK" | tee -a "${SUMMARY_PATH}"
    else
        local status=$?
        RESULTS+=("FAIL ${name}  exit=${status}  ${output_dir}")
        echo "[$(date '+%F %T')] Finished ${name}: FAIL (exit=${status})" | tee -a "${SUMMARY_PATH}"
        if [[ "${STOP_ON_ERROR}" -eq 1 ]]; then
            echo "Stopping early because --stop-on-error was set." | tee -a "${SUMMARY_PATH}"
            exit "${status}"
        fi
    fi

    echo | tee -a "${SUMMARY_PATH}"
}

{
    echo "GSM8K few-shot suite"
    echo "Started: $(date '+%F %T')"
    echo "Repo root: ${REPO_ROOT}"
    echo "GSM8K path: ${GSM8K_PATH}"
    echo "Output root: ${OUTPUT_ROOT}"
    echo "Model base: ${MODEL_BASE}"
    echo "Model RPN: ${MODEL_RPN}"
    echo "Common args: shots=4 8 16 num_samples=${NUM_SAMPLES} ctx=${CTX_SIZE} batch=${BATCH_SIZE} ubatch=${UBATCH_SIZE} ngl=${N_GPU_LAYERS} flash_attn=${FLASH_ATTN} prompt_format=${PROMPT_FORMAT}"
    echo
} | tee "${SUMMARY_PATH}"

run_eval \
    "fp16" \
    "${MODEL_BASE}" \
    "${OUTPUT_ROOT}/fp16"

run_eval \
    "q3_0_head" \
    "${MODEL_BASE}" \
    "${OUTPUT_ROOT}/q3_0_head" \
    --ctk q3_0_head \
    --ctv q3_0_head

run_eval \
    "q3_0_head_rotated_rpn" \
    "${MODEL_RPN}" \
    "${OUTPUT_ROOT}/q3_0_head_rotated_rpn" \
    --ctk q3_0_head \
    --ctv q3_0_head

run_eval \
    "hadamard" \
    "${MODEL_BASE}" \
    "${OUTPUT_ROOT}/hadamard" \
    --hadamard \
    --hadamard-seed 0 \
    --hadamard-granularity head \
    --ctk q3_0_head \
    --ctv q3_0_head

run_eval \
    "kvtuner_359" \
    "${MODEL_BASE}" \
    "${OUTPUT_ROOT}/kvtuner_359" \
    --ctk q4_0_head \
    --ctv q4_0_head \
    --kv-layer-v-types "0:q8_0_head,1-4,7,13,18-19,22-25,27,29,31:q2_0_head"

run_eval \
    "kvtuner_359_rotated_rpn" \
    "${MODEL_RPN}" \
    "${OUTPUT_ROOT}/kvtuner_359_rotated_rpn" \
    --ctk q4_0_head \
    --ctv q4_0_head \
    --kv-layer-v-types "31:q8_0_head,4,6-11,13,15-20,22:q2_0_head"

run_eval \
    "kvtuner_3bit_rotated_rpn" \
    "${MODEL_RPN}" \
    "${OUTPUT_ROOT}/kvtuner_3bit_rotated_rpn" \
    --ctk q2_0_q4_0_head \
    --ctv q2_0_head \
    --kv-layer-k-types "0-9:q4_0_head,12:q4_0_head,14:q4_0_head,16:q4_0_head,21:q4_0_head,23-25:q4_0_head,27-28:q4_0_head,31:q4_0_head" \
    --kv-layer-v-types "0-2:q4_0_head,21:q4_0_head,25:q4_0_head,31:q4_0_head"

{
    echo "Summary:"
    for result in "${RESULTS[@]}"; do
        echo "  ${result}"
    done
    echo
    echo "Finished: $(date '+%F %T')"
} | tee -a "${SUMMARY_PATH}"

echo "Wrote suite summary to ${SUMMARY_PATH}"
