#!/bin/bash
# 编译打过补丁的 llama.cpp（补丁见本目录 patch_llama_rwkv_penalties.py）
#
# 用法：
#   1. git clone https://github.com/ggml-org/llama.cpp llama-rwkv-pr1 && cd llama-rwkv-pr1
#   2. git checkout <PR1 对应的提交>          # 需支持 RWKV7 架构
#   3. python3 /path/to/patch_llama_rwkv_penalties.py .
#      python3 /path/to/patch_task_maps.py .
#   4. 本脚本
set -eu

LLAMA_SRC=${LLAMA_SRC:-$(cd "$(dirname "$0")/.." && pwd)/llama-rwkv-pr1}
CONDA_ENV=${CONDA_ENV:-}
JOBS=${JOBS:-$(nproc)}

cd "$LLAMA_SRC"
if [ -n "$CONDA_ENV" ]; then
  export PATH="$CONDA_ENV/bin:$PATH"
  export CUDAToolkit_ROOT="$CONDA_ENV"
fi

echo "[1/2] cmake configure $(date)"
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH:-70} \
  -DGGML_CUDA=${GGML_CUDA:-ON} \
  -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_GRAPHS=ON \
  -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DGGML_NATIVE=ON

echo "[2/2] cmake build $(date)"
cmake --build build -j "$JOBS" --target llama-server llama-cli
echo "[build] done rc=$? $(date)"
