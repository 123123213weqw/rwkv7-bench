#!/bin/bash
# 一键跑一组评测。
#
# 用法:
#   BENCH_ROOT=/path/to/rwkvbench ./run.sh [profile] [out_name] [max_tokens] [extra args...]
#
# 例:
#   ./run.sh rwkv-demo q4_demo 8192
#   ./run.sh rwkv-demo q8_official 500 --no-think
#   ./run.sh legacy q4_legacy 8192
set -eu

BENCH_ROOT=${BENCH_ROOT:-$(cd "$(dirname "$0")" && pwd)}
PROFILE=${1:-rwkv-demo}
OUT=${2:-run}
MAXTOK=${3:-8192}
shift 3 2>/dev/null || shift $#

BASE_URL=${BASE_URL:-http://127.0.0.1:18860}
MODEL=${MODEL:-rwkv29-g1j-q4}
BANK=${BANK:-$BENCH_ROOT/bench/samples/iq_bank_v3_sample.json}

mkdir -p "$BENCH_ROOT/logs"
LOG="$BENCH_ROOT/logs/eval_${OUT}.log"

echo "[run] profile=$PROFILE out=$OUT max_tokens=$MAXTOK $(date)" | tee -a "$LOG"
python3 "$BENCH_ROOT/bench/run_eval_rwkv.py" \
  --bank "$BANK" \
  --profile "$PROFILE" \
  --base-url "$BASE_URL" \
  --model "$MODEL" \
  --out "results_${OUT}.json" \
  --max-tokens "$MAXTOK" \
  --tier all \
  --resume "$@" 2>&1 | tee -a "$LOG"
echo "[run] done rc=$? $(date)" | tee -a "$LOG"
