#!/bin/bash
# 对照组服务：不设任何采样 CLI 默认值，走 llama.cpp 原味默认。
# 配合 bench/run_eval_rwkv.py --profile legacy 复现「原 v3 协议」
# （只下发 temperature + top_p 0.95，其余由服务端默认决定）。
set -eu

BENCH_ROOT=${BENCH_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
MODEL=${MODEL:-$BENCH_ROOT/models/rwkv7-g1j-2.9b-Q4_K_M.gguf}
LLAMA_BIN=${LLAMA_BIN:-$BENCH_ROOT/llama-rwkv-pr1/build/bin/llama-server}
PORT=${PORT:-18861}
ALIAS=${ALIAS:-rwkv29-g1j-q4}

exec "$LLAMA_BIN" \
  -m "$MODEL" \
  --alias "$ALIAS" \
  --host 127.0.0.1 --port "$PORT" \
  -ngl 99 -c 49152 -b 2048 -ub 512 --parallel 1 \
  --jinja
