#!/usr/bin/env bash

set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT_DEFAULT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
REPO_ROOT="${REPO_ROOT:-${REPO_ROOT_DEFAULT}}"
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-${REPO_ROOT_DEFAULT}/build-cuda/bin/llama-server}"
EVAL_SCRIPT="${EVAL_SCRIPT:-${REPO_ROOT_DEFAULT}/tools/eval_longbench_e.py}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/yejin/longbench_e_rerun}"

MODEL_BASE="${MODEL_BASE:-/data/yejin/Llama-3.1-8B-Instruct.gguf}"
MODEL_RPN="${MODEL_RPN:-/data/yejin/Llama-3.1-8B-Instruct_rotated_rpn.gguf}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
SERVER_CTX_SIZE="${SERVER_CTX_SIZE:-65536}"
N_GPU_LAYERS="${N_GPU_LAYERS:-99}"
FLASH_ATTN="${FLASH_ATTN:-on}"

FORMAT_MODE="${FORMAT_MODE:-chat}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-You are a careful assistant. Follow the task instruction exactly and answer concisely.}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-1.0}"
SEED="${SEED:-42}"
TIMEOUT="${TIMEOUT:-600}"
READY_TIMEOUT="${READY_TIMEOUT:-120}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
HF_REPO="${HF_REPO:-THUDM/LongBench}"
API_KEY="${API_KEY:-}"
VERBOSE="${VERBOSE:-1}"
STOP_ON_ERROR=0

CURRENT_SERVER_PID=""

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Runs the 7 LongBench-E presets sequentially. For each preset:
1. start llama-server with the requested model/KV config
2. wait for /health
3. run tools/eval_longbench_e.py
4. stop the server

Options:
  --python PATH              Python executable to use
  --repo-root PATH           Repo root containing llama-server and eval_longbench_e.py
  --server-bin PATH          Explicit llama-server binary path
  --eval-script PATH         Explicit tools/eval_longbench_e.py path
  --output-root PATH         Root directory for outputs and logs
  --model-base PATH          Base instruct model path
  --model-rpn PATH           Rotated RPN model path
  --host HOST                Host for llama-server (default: ${HOST})
  --port PORT                Port for llama-server (default: ${PORT})
  --server-ctx-size N        llama-server -c value (default: ${SERVER_CTX_SIZE})
  --n-gpu-layers N           GPU layers for llama-server (default: ${N_GPU_LAYERS})
  --flash-attn MODE          Flash attention mode for llama-server (default: ${FLASH_ATTN})
  --format MODE              chat|completion (default: ${FORMAT_MODE})
  --system-prompt TEXT       System prompt for chat mode
  --temperature X            Generation temperature (default: ${TEMPERATURE})
  --top-p X                  Generation top-p (default: ${TOP_P})
  --seed N                   Base generation seed (default: ${SEED})
  --timeout SEC              HTTP timeout for eval script (default: ${TIMEOUT})
  --ready-timeout SEC        Server readiness timeout (default: ${READY_TIMEOUT})
  --max-samples N            Optional per-task sample limit for debugging
  --hf-repo NAME             HF dataset repo (default: ${HF_REPO})
  --api-key KEY              Optional llama-server API key
  --no-verbose               Disable sample-level progress logs
  --stop-on-error            Stop immediately when one run fails
  -h, --help                 Show this help

Environment overrides are also supported for the variables defined at the top of this script.
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
        --server-bin)
            LLAMA_SERVER_BIN="$2"
            shift 2
            ;;
        --eval-script)
            EVAL_SCRIPT="$2"
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
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --server-ctx-size)
            SERVER_CTX_SIZE="$2"
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
        --format)
            FORMAT_MODE="$2"
            shift 2
            ;;
        --system-prompt)
            SYSTEM_PROMPT="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --top-p)
            TOP_P="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --ready-timeout)
            READY_TIMEOUT="$2"
            shift 2
            ;;
        --max-samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --hf-repo)
            HF_REPO="$2"
            shift 2
            ;;
        --api-key)
            API_KEY="$2"
            shift 2
            ;;
        --no-verbose)
            VERBOSE=0
            shift
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

BASE_URL="http://${HOST}:${PORT}"

mkdir -p "${OUTPUT_ROOT}"
LOG_DIR="${OUTPUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY_PATH="${OUTPUT_ROOT}/run_summary_${TIMESTAMP}.txt"

declare -a RESULTS=()

stop_server() {
    if [[ -n "${CURRENT_SERVER_PID}" ]] && kill -0 "${CURRENT_SERVER_PID}" 2>/dev/null; then
        kill "${CURRENT_SERVER_PID}" 2>/dev/null || true
        local _i
        for _i in $(seq 1 20); do
            if ! kill -0 "${CURRENT_SERVER_PID}" 2>/dev/null; then
                break
            fi
            sleep 0.5
        done
        if kill -0 "${CURRENT_SERVER_PID}" 2>/dev/null; then
            kill -9 "${CURRENT_SERVER_PID}" 2>/dev/null || true
        fi
        wait "${CURRENT_SERVER_PID}" 2>/dev/null || true
    fi
    CURRENT_SERVER_PID=""
}

cleanup() {
    stop_server
}

trap cleanup EXIT INT TERM

wait_for_server() {
    local base_url="$1"
    local pid="$2"
    local timeout="$3"

    "${PYTHON_BIN}" - "${base_url}" "${pid}" "${timeout}" <<'PY'
import os
import sys
import time
import urllib.request

base_url = sys.argv[1].rstrip("/")
pid = int(sys.argv[2])
timeout = float(sys.argv[3])
deadline = time.time() + timeout
last_error = None

while time.time() < deadline:
    try:
        os.kill(pid, 0)
    except OSError:
        print("llama-server exited before becoming ready", file=sys.stderr)
        raise SystemExit(1)

    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=2.0) as response:
            if 200 <= response.status < 300:
                req = urllib.request.Request(
                    f"{base_url}/tokenize",
                    data=b'{"content":"","add_special":true}',
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=2.0) as tokenize_response:
                    if 200 <= tokenize_response.status < 300:
                        raise SystemExit(0)
    except Exception as exc:
        last_error = exc
        time.sleep(0.5)

print(f"timed out waiting for {base_url}/health: {last_error}", file=sys.stderr)
raise SystemExit(1)
PY
}

write_command_script() {
    local path="$1"
    shift
    local -a cmd=("$@")

    {
        echo "#!/usr/bin/env bash"
        echo "set -euo pipefail"
        printf '%q' "${cmd[0]}"
        local i
        for (( i = 1; i < ${#cmd[@]}; ++i )); do
            printf ' %q' "${cmd[i]}"
        done
        printf '\n'
    } > "${path}"
    chmod +x "${path}"
}

write_command_txt() {
    local path="$1"
    shift
    local -a cmd=("$@")

    printf '%q' "${cmd[0]}" > "${path}"
    local i
    for (( i = 1; i < ${#cmd[@]}; ++i )); do
        printf ' %q' "${cmd[i]}" >> "${path}"
    done
    printf '\n' >> "${path}"
}

extract_json_summary() {
    local json_path="$1"
    "${PYTHON_BIN}" - "${json_path}" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as handle:
    payload = json.load(handle)
overall = payload.get("overall_average")
categories = payload.get("category_average", {})
parts = [f"overall={overall:.4f}" if overall is not None else "overall=nan"]
for key in sorted(categories):
    parts.append(f"{key}={categories[key]:.4f}")
print(" ".join(parts))
PY
}

run_eval() {
    local name="$1"
    local model="$2"
    local output_dir="$3"
    shift 3

    mkdir -p "${output_dir}"

    local run_log_path="${output_dir}/run.log"
    local server_log_path="${output_dir}/server.log"
    local archive_run_log_path="${LOG_DIR}/${TIMESTAMP}_${name}.log"
    local archive_server_log_path="${LOG_DIR}/${TIMESTAMP}_${name}_server.log"
    local run_command_path="${output_dir}/run_command.sh"
    local run_command_txt_path="${output_dir}/run_command.txt"
    local server_command_path="${output_dir}/server_command.sh"
    local server_command_txt_path="${output_dir}/server_command.txt"

    local -a server_cmd=(
        "${LLAMA_SERVER_BIN}"
        -m "${model}"
        -c "${SERVER_CTX_SIZE}"
        -ngl "${N_GPU_LAYERS}"
        -fa "${FLASH_ATTN}"
        --host "${HOST}"
        --port "${PORT}"
    )
    server_cmd+=("$@")

    local -a eval_cmd=(
        "${PYTHON_BIN}" "${EVAL_SCRIPT}"
        --base-url "${BASE_URL}"
        --output-dir "${output_dir}"
        --setting-name "${name}"
        --format "${FORMAT_MODE}"
        --system-prompt "${SYSTEM_PROMPT}"
        --temperature "${TEMPERATURE}"
        --top-p "${TOP_P}"
        --seed "${SEED}"
        --timeout "${TIMEOUT}"
        --ready-timeout "${READY_TIMEOUT}"
        --hf-repo "${HF_REPO}"
    )
    if [[ -n "${API_KEY}" ]]; then
        eval_cmd+=(--api-key "${API_KEY}")
    fi
    if [[ "${MAX_SAMPLES}" -gt 0 ]]; then
        eval_cmd+=(--max-samples "${MAX_SAMPLES}")
    fi
    if [[ "${VERBOSE}" -eq 1 ]]; then
        eval_cmd+=(--verbose)
    fi

    write_command_script "${server_command_path}" "${server_cmd[@]}"
    write_command_script "${run_command_path}" "${eval_cmd[@]}"
    write_command_txt "${server_command_txt_path}" "${server_cmd[@]}"
    write_command_txt "${run_command_txt_path}" "${eval_cmd[@]}"

    {
        echo "============================================================"
        echo "[$(date '+%F %T')] Starting ${name}"
        printf 'Server command:'
        printf ' %q' "${server_cmd[@]}"
        printf '\n'
        printf 'LongBench-E command:'
        printf ' %q' "${eval_cmd[@]}"
        printf '\n'
        echo "Run log: ${run_log_path}"
        echo "Server log: ${server_log_path}"
        echo "Archive run log: ${archive_run_log_path}"
        echo "Archive server log: ${archive_server_log_path}"
        echo "Output dir: ${output_dir}"
        echo "============================================================"
    } | tee -a "${SUMMARY_PATH}" "${run_log_path}"

    local -a stream_prefix=()
    if command -v stdbuf >/dev/null 2>&1; then
        stream_prefix=(stdbuf -oL -eL)
    fi

    stop_server

    if [[ ${#stream_prefix[@]} -gt 0 ]]; then
        "${stream_prefix[@]}" "${server_cmd[@]}" > >(tee "${server_log_path}" "${archive_server_log_path}") 2>&1 &
    else
        "${server_cmd[@]}" > >(tee "${server_log_path}" "${archive_server_log_path}") 2>&1 &
    fi
    CURRENT_SERVER_PID=$!

    if ! wait_for_server "${BASE_URL}" "${CURRENT_SERVER_PID}" "${READY_TIMEOUT}"; then
        RESULTS+=("FAIL ${name}  server_start_failed  ${output_dir}")
        echo "[$(date '+%F %T')] Finished ${name}: FAIL (server did not become ready)" | tee -a "${SUMMARY_PATH}" "${run_log_path}"
        stop_server
        if [[ "${STOP_ON_ERROR}" -eq 1 ]]; then
            echo "Stopping early because --stop-on-error was set." | tee -a "${SUMMARY_PATH}" "${run_log_path}"
            exit 1
        fi
        echo | tee -a "${SUMMARY_PATH}" "${run_log_path}"
        return
    fi

    local status=0
    if env PYTHONUNBUFFERED=1 "${stream_prefix[@]}" "${eval_cmd[@]}" 2>&1 | tee -a "${run_log_path}" "${archive_run_log_path}"; then
        local json_path="${output_dir}/summary.json"
        local summary_metrics=""
        if [[ -f "${json_path}" ]]; then
            summary_metrics="$(extract_json_summary "${json_path}")"
        fi
        RESULTS+=("OK   ${name}  ${summary_metrics}  ${output_dir}")
        echo "[$(date '+%F %T')] Finished ${name}: OK ${summary_metrics}" | tee -a "${SUMMARY_PATH}" "${run_log_path}"
    else
        status=$?
        RESULTS+=("FAIL ${name}  exit=${status}  ${output_dir}")
        echo "[$(date '+%F %T')] Finished ${name}: FAIL (exit=${status})" | tee -a "${SUMMARY_PATH}" "${run_log_path}"
    fi

    stop_server

    if [[ "${status}" -ne 0 && "${STOP_ON_ERROR}" -eq 1 ]]; then
        echo "Stopping early because --stop-on-error was set." | tee -a "${SUMMARY_PATH}" "${run_log_path}"
        exit "${status}"
    fi

    echo | tee -a "${SUMMARY_PATH}" "${run_log_path}"
}

{
    echo "LongBench-E suite"
    echo "Started: $(date '+%F %T')"
    echo "Repo root: ${REPO_ROOT}"
    echo "llama-server: ${LLAMA_SERVER_BIN}"
    echo "eval_longbench_e.py: ${EVAL_SCRIPT}"
    echo "Output root: ${OUTPUT_ROOT}"
    echo "Base URL: ${BASE_URL}"
    echo "Model base: ${MODEL_BASE}"
    echo "Model RPN: ${MODEL_RPN}"
    echo "Server args: ctx=${SERVER_CTX_SIZE} ngl=${N_GPU_LAYERS} flash_attn=${FLASH_ATTN}"
    echo "Eval args: format=${FORMAT_MODE} temperature=${TEMPERATURE} top_p=${TOP_P} seed=${SEED} timeout=${TIMEOUT} ready_timeout=${READY_TIMEOUT} hf_repo=${HF_REPO} max_samples=${MAX_SAMPLES} verbose=${VERBOSE}"
    echo "System prompt: ${SYSTEM_PROMPT}"
    echo
} | tee "${SUMMARY_PATH}"

run_eval \
    "fp16" \
    "${MODEL_BASE}" \
    "${OUTPUT_ROOT}/fp16"

# run_eval \
#     "q3_0_head" \
#     "${MODEL_BASE}" \
#     "${OUTPUT_ROOT}/q3_0_head" \
#     -ctk q3_0_head \
#     -ctv q3_0_head
#
# run_eval \
#     "q3_0_head_rotated_rpn" \
#     "${MODEL_RPN}" \
#     "${OUTPUT_ROOT}/q3_0_head_rotated_rpn" \
#     -ctk q3_0_head \
#     -ctv q3_0_head
#
# run_eval \
#     "hadamard" \
#     "${MODEL_BASE}" \
#     "${OUTPUT_ROOT}/hadamard" \
#     --hadamard \
#     --hadamard-seed 0 \
#     --hadamard-granularity head \
#     -ctk q3_0_head \
#     -ctv q3_0_head
#
# run_eval \
#     "kvtuner_359" \
#     "${MODEL_BASE}" \
#     "${OUTPUT_ROOT}/kvtuner_359" \
#     -ctk q4_0_head \
#     -ctv q4_0_head \
#     --kv-layer-v-types "0:q8_0_head,1-4,7,13,18-19,22-25,27,29,31:q2_0_head"
#
# run_eval \
#     "kvtuner_359_rotated_rpn" \
#     "${MODEL_RPN}" \
#     "${OUTPUT_ROOT}/kvtuner_359_rotated_rpn" \
#     -ctk q4_0_head \
#     -ctv q4_0_head \
#     --kv-layer-v-types "31:q8_0_head,4,6-11,13,15-20,22:q2_0_head"
#
run_eval \
    "kvtuner_3bit_rotated_rpn" \
    "${MODEL_RPN}" \
    "${OUTPUT_ROOT}/kvtuner_3bit_rotated_rpn" \
    -ctk q2_0_q4_0_head \
    -ctv q2_0_head \
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
