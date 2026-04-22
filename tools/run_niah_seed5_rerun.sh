#!/usr/bin/env bash

set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT_DEFAULT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
REPO_ROOT="${REPO_ROOT:-${REPO_ROOT_DEFAULT}}"
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-${REPO_ROOT_DEFAULT}/build-cuda/bin/llama-server}"
NIAH_SCRIPT="${NIAH_SCRIPT:-${REPO_ROOT_DEFAULT}/scripts/niah.py}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/yejin/needle_seed5_rerun}"

MODEL_RPN="${MODEL_RPN:-/data/yejin/Llama-3.1-8B-Instruct_rotated_rpn.gguf}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
SERVER_CTX_SIZE="${SERVER_CTX_SIZE:-65536}"
N_GPU_LAYERS="${N_GPU_LAYERS:-99}"
FLASH_ATTN="${FLASH_ATTN:-on}"

CTX_LENS="${CTX_LENS:-51200,65536}"
DEPTHS="${DEPTHS:-0.0,0.11,0.22,0.33,0.44,0.56,0.67,0.78,0.89,1.0}"
SEEDS="${SEEDS:-0,1,2,3,4}"
NEEDLE_KIND="${NEEDLE_KIND:-passkey}"
DIGITS="${DIGITS:-8}"
FORMAT_MODE="${FORMAT_MODE:-chat}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-8}"
TEMPERATURE="${TEMPERATURE:-0}"
TIMEOUT="${TIMEOUT:-600}"
READY_TIMEOUT="${READY_TIMEOUT:-120}"
FILLER_FILE="${FILLER_FILE:-/home/yjkim00/rope_specinfer.cpp/tools/niah_numeric_filler.txt}"
VERBOSE="${VERBOSE:-1}"

STOP_ON_ERROR=0
CURRENT_SERVER_PID=""

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Reruns only the two failed seed-5 NIAH settings:
  - kvtuner_359_rotated_rpn
  - kvtuner_3bit_rotated_rpn

Defaults:
  output-root   ${OUTPUT_ROOT}
  ctx-lens      ${CTX_LENS}
  timeout       ${TIMEOUT}

Options:
  --python PATH
  --repo-root PATH
  --server-bin PATH
  --niah-script PATH
  --output-root PATH
  --model-rpn PATH
  --host HOST
  --port PORT
  --server-ctx-size N
  --n-gpu-layers N
  --flash-attn MODE
  --ctx-lens LIST
  --depths LIST
  --seeds LIST
  --needle-kind KIND
  --digits N
  --format MODE
  --system-prompt TEXT
  --max-output-tokens N
  --temperature X
  --timeout SEC
  --ready-timeout SEC
  --filler-file PATH
  --no-verbose
  --stop-on-error
  -h, --help
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
        --niah-script)
            NIAH_SCRIPT="$2"
            shift 2
            ;;
        --output-root)
            OUTPUT_ROOT="$2"
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
        --ctx-lens)
            CTX_LENS="$2"
            shift 2
            ;;
        --depths)
            DEPTHS="$2"
            shift 2
            ;;
        --seeds)
            SEEDS="$2"
            shift 2
            ;;
        --needle-kind)
            NEEDLE_KIND="$2"
            shift 2
            ;;
        --digits)
            DIGITS="$2"
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
        --max-output-tokens)
            MAX_OUTPUT_TOKENS="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
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
        --filler-file)
            FILLER_FILE="$2"
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

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    payload = json.load(handle)
summary = payload.get("summary", [])
if not summary:
    print("mean_acc=nan rows=0")
    raise SystemExit(0)

mean_acc = sum(row["accuracy"] for row in summary) / len(summary)
best = max(summary, key=lambda row: row["accuracy"])
print(
    f"mean_acc={mean_acc:.4f} rows={len(summary)} "
    f"best_ctx={best['ctx_len']} best_depth={best['requested_depth']:.2f} "
    f"best_acc={best['accuracy']:.4f}"
)
PY
}

run_eval() {
    local name="$1"
    local output_dir="$2"
    shift 2

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
        -m "${MODEL_RPN}"
        -c "${SERVER_CTX_SIZE}"
        -ngl "${N_GPU_LAYERS}"
        -fa "${FLASH_ATTN}"
        --host "${HOST}"
        --port "${PORT}"
    )
    server_cmd+=("$@")

    local -a niah_cmd=(
        "${PYTHON_BIN}" "${NIAH_SCRIPT}"
        --base-url "${BASE_URL}"
        --ctx-lens "${CTX_LENS}"
        --depths "${DEPTHS}"
        --seeds "${SEEDS}"
        --needle-kind "${NEEDLE_KIND}"
        --format "${FORMAT_MODE}"
        --system-prompt "${SYSTEM_PROMPT}"
        --filler-file "${FILLER_FILE}"
        --max-output-tokens "${MAX_OUTPUT_TOKENS}"
        --temperature "${TEMPERATURE}"
        --timeout "${TIMEOUT}"
        --ready-timeout "${READY_TIMEOUT}"
        --output-dir "${output_dir}"
        --output-stem "${name}"
    )
    if [[ "${NEEDLE_KIND}" == "passkey" ]]; then
        niah_cmd+=(--digits "${DIGITS}")
    fi
    if [[ "${VERBOSE}" -eq 1 ]]; then
        niah_cmd+=(--verbose)
    fi

    write_command_script "${server_command_path}" "${server_cmd[@]}"
    write_command_script "${run_command_path}" "${niah_cmd[@]}"
    write_command_txt "${server_command_txt_path}" "${server_cmd[@]}"
    write_command_txt "${run_command_txt_path}" "${niah_cmd[@]}"

    {
        echo "============================================================"
        echo "[$(date '+%F %T')] Starting ${name}"
        printf 'Server command:'
        printf ' %q' "${server_cmd[@]}"
        printf '\n'
        printf 'NIAH command:'
        printf ' %q' "${niah_cmd[@]}"
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
    if env PYTHONUNBUFFERED=1 "${stream_prefix[@]}" "${niah_cmd[@]}" 2>&1 | tee -a "${run_log_path}" "${archive_run_log_path}"; then
        local json_path="${output_dir}/${name}.json"
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
    echo "NIAH rerun suite"
    echo "Started: $(date '+%F %T')"
    echo "Repo root: ${REPO_ROOT}"
    echo "llama-server: ${LLAMA_SERVER_BIN}"
    echo "niah.py: ${NIAH_SCRIPT}"
    echo "Output root: ${OUTPUT_ROOT}"
    echo "Base URL: ${BASE_URL}"
    echo "Model RPN: ${MODEL_RPN}"
    echo "Server args: ctx=${SERVER_CTX_SIZE} ngl=${N_GPU_LAYERS} flash_attn=${FLASH_ATTN}"
    echo "NIAH args: ctx_lens=${CTX_LENS} depths=${DEPTHS} seeds=${SEEDS} needle_kind=${NEEDLE_KIND} format=${FORMAT_MODE} filler_file=${FILLER_FILE} max_output_tokens=${MAX_OUTPUT_TOKENS} temperature=${TEMPERATURE} timeout=${TIMEOUT} verbose=${VERBOSE}"
    echo "System prompt: ${SYSTEM_PROMPT}"
    echo
} | tee "${SUMMARY_PATH}"

run_eval \
    "kvtuner_359_rotated_rpn" \
    "${OUTPUT_ROOT}/kvtuner_359_rotated_rpn" \
    -ctk q4_0_head \
    -ctv q4_0_head \
    --kv-layer-v-types "31:q8_0_head,4,6-11,13,15-20,22:q2_0_head"

run_eval \
    "kvtuner_3bit_rotated_rpn" \
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

echo "Wrote rerun summary to ${SUMMARY_PATH}"
