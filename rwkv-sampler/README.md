# llama.cpp 的 RWKV 官方解码补丁

## 问题

llama.cpp 原生采样器**无法表达** RWKV 官方 Gradio demo 的解码语义。四处差异：

| 差异 | llama.cpp 原生 | RWKV 官方 demo |
|---|---|---|
| prompt 是否计入罚项 | `init_sampler()` 先把整个 prompt `accept()` 一遍 → **prompt 也被罚** | 计数器在 prefill 后清零 → **prompt 不计罚** |
| 频次罚是否有衰减 | 无 | `count *= 0.99` |
| 罚项在采样链的位置 | 链**尾部**（top-k/top-p 之后） | 链**头部**（作用于原始 logits） |
| 罚项形式 | multiply/divide `repeat_penalty` | **减去** `count_penalty*count + presence` |

官方语义（`app.py`）：

```python
if alpha_frequency: logits.sub_(occurrence_count, alpha=alpha_frequency)  # count 罚 ×0.1
if alpha_presence:  logits.sub_(occurrence_presence)                      # presence 罚 1.0
sampled = sample_logits_batch_cuda(logits, temperature=1.0, top_p=0.5, k=500)
if penalty_decay != 1: occurrence_count.mul_(penalty_decay)               # ×0.99
occurrence_count[batch_rows, sampled_tensor] += 1
if alpha_presence: occurrence_presence[batch_rows, sampled_tensor] = alpha_presence
```

## 做法

新增采样器 `llama_sampler_init_rwkv_penalties(n_vocab, count_penalty, presence_penalty, penalty_decay)`：

- **链最前端**插入（能看到原始 logits）
- `started` 标志位：第一次 `apply()` 之前的所有 `accept()`（即 prompt）**全部忽略**
- `count` / `presence` 浮点向量 + `touched` 列表做稀疏衰减，避免每步遍历整个词表
- `apply()` 里 `logit -= count_penalty*count[id] + presence[id]`，并置 `sorted=false`

新增请求字段：`rwkv_count_penalty` / `rwkv_presence_penalty` / `rwkv_penalty_decay`

## 用法

```bash
git clone https://github.com/ggml-org/llama.cpp llama-rwkv-pr1
cd llama-rwkv-pr1 && git checkout <含 RWKV7 支持的提交>

python3 /path/to/patch_llama_rwkv_penalties.py .   # 改 include/llama.h, src/llama-sampler.cpp,
                                                   # common/common.h, common/sampling.cpp,
                                                   # tools/server/server-schema.cpp
python3 /path/to/patch_task_maps.py .              # 改 tools/server/server-task.cpp

LLAMA_SRC=$PWD bash /path/to/build_patched.sh
BENCH_ROOT=/data/rwkvbench bash serve.sh
```

跑评测时用 `--profile rwkv-demo` 下发官方协议（见 `bench/run_eval_rwkv.py`）。

## 验证（不要跳过）

| 测试 | 方法 | 预期 |
|---|---|---|
| 罚项生效 | **完全固定** temp/top_p/top_k，只切罚项 | 输出长度与内容都变化 |
| **prompt 未被计罚** | greedy `top_k=1`，prompt `"1 "*40` | 首个生成字符与无罚项时**相同**，之后才分岔 |
| 衰减生效 | 同 seed，`decay 0.99` vs `1.0` | 输出不同 |

现成脚本在 [`../experiments/verify_sampler/`](../experiments/verify_sampler/)：

```bash
python3 ../experiments/verify_sampler/test_penalty_isolated.py
python3 ../experiments/verify_sampler/test_prompt_only2.py
```

> ⚠️ **坑**：llama.cpp server 的 `/v1/completions` 返回的 `logprobs` 是**采样前的原始 logits**
> （同一 token 在不同罚项下数值完全不变）。**用它验证罚项是无效的**，必须用行为测试。

## 附带的两个口径偏差

1. **EOG 多了一个 `\n\n`**：RWKV 的 gguf 自带 `tokenizer.ggml.eot_token_id = 261`（`'\n\n'`），
   llama.cpp 会把它也当停止符，官方 demo 只认 `token == 0`。用
   `--override-kv tokenizer.ggml.eot_token_id=int:0` 可对齐（见 `serve_eog0.sh`）。
   **实测该模型几乎不输出空行，所以在 iq_bank_v3 上不产生可观测差异**。

2. **官方默认关闭思维链**：官方构造的生成前缀是 `Assistant: <think></think`，而 llama.cpp
   走 `--jinja` 模板只给出裸 `Assistant:`（等于**开着**思维链）。用
   `run_eval_rwkv.py --no-think`（下发 `chat_template_kwargs.enable_thinking=false`）对齐。
   **这是本项目最大的一个混杂因素**，见根目录 README §2.1。

## `token[0]` 是否被正确处理

是。gguf `eos_token_id = 0`，实测正常结束的题 `finish_reason=stop` 且末 token id 恰为 0。
另外词表对齐无误：GGUF 的 `token[1]='\x00'` 与官方 vocab 第 1 行 `1 '\x00'` 严格对应。
验证脚本：`../experiments/diagnostics/diag_stop_token.py`
