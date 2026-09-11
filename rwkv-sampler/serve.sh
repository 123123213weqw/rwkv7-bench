#!/bin/bash
# 启动 llama-server（需先应用 rwkv-sampler/ 下的补丁并编译）
#
# 环境变量：
#   BENCH_ROOT  仓库/模型/日志的根目录（默认脚本所在仓库的上一级）
#   MODEL       gguf 路径
#   LLAMA_BIN   编译出的 llama-server 路径
#   PORT        端口（默认 18860）
#   ALIAS       模型别名（默认 rwkv29-g1j-q4）
#
# 采样协议主要通过请求体下发（见 bench/run_eval_rwkv.py --profile rwkv-demo）。
# 这里给的默认值仅用于手动 curl 调试，与 RWKV 官方 demo 一致。
set -eu

BENCH_ROOT=${BENCH_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
MODEL=${MODEL:-$BENCH_ROOT/models/rwkv7-g1j-2.9b-Q4_K_M.gguf}
LLAMA_BIN=${LLAMA_BIN:-$BENCH_ROOT/llama-rwkv-pr1/build/bin/llama-server}
PORT=${PORT:-18860}
ALIAS=${ALIAS:-rwkv29-g1j-q4}

exec "$LLAMA_BIN" \
  -m "$MODEL" \
  --alias "$ALIAS" \
  --host 127.0.0.1 --port "$PORT" \
  -ngl 99 -c 49152 -b 2048 -ub 512 --parallel 1 \
  --jinja \
  --temp 1.0 --top-p 0.5 --top-k 500 --min-p 0.0 \
  --repeat-penalty 1.0 --repeat-last-n 0
