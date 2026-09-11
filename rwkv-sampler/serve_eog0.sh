#!/bin/bash
# 与 serve.sh 相同，但把 EOG 限制为只有 token id 0，
# 对齐 RWKV 官方 demo 的 `if token == 0: stop`。
#
# 背景：RWKV 的 gguf 自带 tokenizer.ggml.eot_token_id = 261（'\n\n'），
# llama.cpp 会把它也当成停止符，而官方 demo 不会。
# 用 --override-kv 把 eot_token_id 也指向 0，使 EOG == {0}。
#
# 注意：实测该模型几乎不输出空行，所以这条口径差异在 iq_bank_v3 上
#       不产生可观测差异（见 docs/notes_official_vs_thinking.md）。
set -eu

BENCH_ROOT=${BENCH_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
MODEL=${MODEL:-$BENCH_ROOT/models/rwkv7-g1j-2.9b-Q4_K_M.gguf}
LLAMA_BIN=${LLAMA_BIN:-$BENCH_ROOT/llama-rwkv-pr1/build/bin/llama-server}
PORT=${PORT:-18862}
ALIAS=${ALIAS:-rwkv29-g1j-q4}

exec "$LLAMA_BIN" \
  -m "$MODEL" \
  --alias "$ALIAS" \
  --host 127.0.0.1 --port "$PORT" \
  --override-kv tokenizer.ggml.eot_token_id=int:0 \
  -ngl 99 -c 49152 -b 2048 -ub 512 --parallel 1 \
  --jinja \
  --temp 1.0 --top-p 0.5 --top-k 500 --min-p 0.0 \
  --repeat-penalty 1.0 --repeat-last-n 0
