#!/usr/bin/env bash

set -u
set -o pipefail

MODEL="/data/yejin/Llama-3.1-8B-Instruct_rotated_rpn.gguf"
DATA="/home/yjkim00/rope_specinfer.cpp/data/wikitext-2-raw-v1.test.raw"
BIN="./build-cuda/bin/llama-perplexity"

LOG_DIR="./logs_kv_ranked_centered_avg3bit"
mkdir -p "$LOG_DIR"

COMMON_ARGS=(
  -m "$MODEL"
  -f "$DATA"
  --ctx-size 4096
  -b 4096
  -ub 4096
  -ngl 99
  --ppl-full-chunk
  -fa on
  -ctk q2_0_q4_0_head
  -ctv q2_0_head
)

run_exp() {
  local name="$1"
  local k_types="$2"
  local v_types="$3"
  local log_file="${LOG_DIR}/${name}.log"

  echo "============================================================" | tee -a "$LOG_DIR/_summary.log"
  echo "[$(date '+%F %T')] START $name" | tee -a "$LOG_DIR/_summary.log"
  echo "K_TYPES: $k_types" | tee -a "$LOG_DIR/_summary.log"
  echo "V_TYPES: $v_types" | tee -a "$LOG_DIR/_summary.log"
  echo "LOG: $log_file" | tee -a "$LOG_DIR/_summary.log"

  {
    echo "[$(date '+%F %T')] START $name"
    echo "CMD: $BIN ${COMMON_ARGS[*]} --kv-layer-k-types \"$k_types\" --kv-layer-v-types \"$v_types\""
    echo
    "$BIN" \
      "${COMMON_ARGS[@]}" \
      --kv-layer-k-types "$k_types" \
      --kv-layer-v-types "$v_types"
    status=$?
    echo
    echo "[$(date '+%F %T')] END $name (exit=$status)"
    exit $status
  } 2>&1 | tee "$log_file"

  cmd_status=${PIPESTATUS[0]}

  if [ "$cmd_status" -eq 0 ]; then
    echo "[$(date '+%F %T')] DONE  $name" | tee -a "$LOG_DIR/_summary.log"
  else
    echo "[$(date '+%F %T')] FAIL  $name (exit=$cmd_status)" | tee -a "$LOG_DIR/_summary.log"
  fi

  echo | tee -a "$LOG_DIR/_summary.log"
}

# K ranking
# 1,0,31,25,4,6,23,21,24,3,5,7,2,28,12,14,27,9,8,16,29,19,15,22,30,17,20,18,26,11,10,13

# V ranking
# 31,0,25,2,1,21,24,26,29,27,23,3,28,30,12,14,5,4,16,18,20,6,9,7,11,19,8,22,15,10,13,17

# centered sweep: (14,9) (16,8) (18,7) (20,6) (22,5) (24,4) (26,3)

# [1] K=14, V=9
run_exp "01_K14_V9" \
  "1:q4_0_head,0:q4_0_head,31:q4_0_head,25:q4_0_head,4:q4_0_head,6:q4_0_head,23:q4_0_head,21:q4_0_head,24:q4_0_head,3:q4_0_head,5:q4_0_head,7:q4_0_head,2:q4_0_head,28:q4_0_head" \
  "31:q4_0_head,0:q4_0_head,25:q4_0_head,2:q4_0_head,1:q4_0_head,21:q4_0_head,24:q4_0_head,26:q4_0_head,29:q4_0_head"

# [2] K=16, V=8
run_exp "02_K16_V8" \
  "1:q4_0_head,0:q4_0_head,31:q4_0_head,25:q4_0_head,4:q4_0_head,6:q4_0_head,23:q4_0_head,21:q4_0_head,24:q4_0_head,3:q4_0_head,5:q4_0_head,7:q4_0_head,2:q4_0_head,28:q4_0_head,12:q4_0_head,14:q4_0_head" \
  "31:q4_0_head,0:q4_0_head,25:q4_0_head,2:q4_0_head,1:q4_0_head,21:q4_0_head,24:q4_0_head,26:q4_0_head"

# [3] K=18, V=7
run_exp "03_K18_V7" \
  "1:q4_0_head,0:q4_0_head,31:q4_0_head,25:q4_0_head,4:q4_0_head,6:q4_0_head,23:q4_0_head,21:q4_0_head,24:q4_0_head,3:q4_0_head,5:q4_0_head,7:q4_0_head,2:q4_0_head,28:q4_0_head,12:q4_0_head,14:q4_0_head,27:q4_0_head,9:q4_0_head" \
  "31:q4_0_head,0:q4_0_head,25:q4_0_head,2:q4_0_head,1:q4_0_head,21:q4_0_head,24:q4_0_head"

# [4] K=20, V=6  <-- current best, 기존 커맨드와 동일
run_exp "04_K20_V6" \
  "1:q4_0_head,0:q4_0_head,31:q4_0_head,25:q4_0_head,4:q4_0_head,6:q4_0_head,23:q4_0_head,21:q4_0_head,24:q4_0_head,3:q4_0_head,5:q4_0_head,7:q4_0_head,2:q4_0_head,28:q4_0_head,12:q4_0_head,14:q4_0_head,27:q4_0_head,9:q4_0_head,8:q4_0_head,16:q4_0_head" \
  "31:q4_0_head,0:q4_0_head,25:q4_0_head,2:q4_0_head,1:q4_0_head,21:q4_0_head"

# [5] K=22, V=5
run_exp "05_K22_V5" \
  "1:q4_0_head,0:q4_0_head,31:q4_0_head,25:q4_0_head,4:q4_0_head,6:q4_0_head,23:q4_0_head,21:q4_0_head,24:q4_0_head,3:q4_0_head,5:q4_0_head,7:q4_0_head,2:q4_0_head,28:q4_0_head,12:q4_0_head,14:q4_0_head,27:q4_0_head,9:q4_0_head,8:q4_0_head,16:q4_0_head,29:q4_0_head,19:q4_0_head" \
  "31:q4_0_head,0:q4_0_head,25:q4_0_head,2:q4_0_head,1:q4_0_head"

# [6] K=24, V=4
run_exp "06_K24_V4" \
  "1:q4_0_head,0:q4_0_head,31:q4_0_head,25:q4_0_head,4:q4_0_head,6:q4_0_head,23:q4_0_head,21:q4_0_head,24:q4_0_head,3:q4_0_head,5:q4_0_head,7:q4_0_head,2:q4_0_head,28:q4_0_head,12:q4_0_head,14:q4_0_head,27:q4_0_head,9:q4_0_head,8:q4_0_head,16:q4_0_head,29:q4_0_head,19:q4_0_head,15:q4_0_head,22:q4_0_head" \
  "31:q4_0_head,0:q4_0_head,25:q4_0_head,2:q4_0_head"

# [7] K=26, V=3
run_exp "07_K26_V3" \
  "1:q4_0_head,0:q4_0_head,31:q4_0_head,25:q4_0_head,4:q4_0_head,6:q4_0_head,23:q4_0_head,21:q4_0_head,24:q4_0_head,3:q4_0_head,5:q4_0_head,7:q4_0_head,2:q4_0_head,28:q4_0_head,12:q4_0_head,14:q4_0_head,27:q4_0_head,9:q4_0_head,8:q4_0_head,16:q4_0_head,29:q4_0_head,19:q4_0_head,15:q4_0_head,22:q4_0_head,30:q4_0_head,17:q4_0_head" \
  "31:q4_0_head,0:q4_0_head,25:q4_0_head"

echo "All runs finished. Check logs in: $LOG_DIR"
