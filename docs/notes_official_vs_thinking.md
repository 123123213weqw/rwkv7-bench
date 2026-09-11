# RWKV7-G1j-2.9B-Q4_K_M × iq_bank_v3：官方 demo 解码复现 + 截断成因排查（结论修订版）

> 本文件替换此前的结论。上一版把「48.3」当成官方解码成绩，**是错的**——见 §4。

## 1. 任务
用 RWKV 官方 Gradio demo 的解码方法测 G1j-2.9B-Q4：
Temperature 1 / Top P 0.5 / Presence Penalty 1 / Count Penalty 0.1 / Penalty Decay 0.99
（`BlinkDL/RWKV-Gradio-2/app.py`）

## 2. 官方采样器语义（逐行核对 app.py）
```python
if alpha_frequency: logits.sub_(occurrence_count, alpha=alpha_frequency)   # -= 0.1*count
if alpha_presence:  logits.sub_(occurrence_presence)                       # -= 1.0（出现过的）
sampled = sample_logits_batch_cuda(logits, 1.0, 0.5, k=500)                # topk500→softmax→topp
if penalty_decay != 1: occurrence_count.mul_(penalty_decay)                # 0.99
occurrence_count[rows, sampled] += 1
occurrence_presence[rows, sampled] = alpha_presence
...
if token == 0: finished[b] = True                                          # ← 唯一的停止条件
```
另外两条**同样重要、但容易被忽略**的官方设定：
```python
gen_limit = 1000
token_count = gr.Slider(10, gen_limit, step=10, value=500)   # Max Tokens 默认 500
def f(instruction): return f"User: {instruction}\n\nAssistant: <think></think"   # 关思考
```
即官方 demo = **非思考模式 + 默认 500 token 预算**。

## 3. llama.cpp 原生表达不了 → 打补丁（这部分仍然有效）
| 官方语义 | llama.cpp 原生 | 处理 |
|---|---|---|
| 罚项只对**生成** token 计数 | `init_sampler()` 先把整个 prompt accept 一遍（server-context.cpp:349） | `started` 标志：首次 `apply()` 前的 accept 全部忽略 |
| count 每步 ×0.99 衰减 | frequency penalty 无衰减 | 自研 sampler 内实现 |
| 罚项作用在**原始 logits**（top-k/p 之前） | 原生 penalties 在链尾 | 自建 sampler 插在链**最前端** |

补丁：`patch_llama_rwkv_penalties.py`（`src/llama-sampler.cpp` / `include/llama.h` / `common/common.h` /
`common/sampling.cpp` / `tools/server/server-schema.cpp` / `tools/server/server-task.cpp`），
新增请求字段 `rwkv_count_penalty` / `rwkv_presence_penalty` / `rwkv_penalty_decay`。

## 4. 关于「token[0] 停止」和「为什么会截断这么多」
### 4.1 token[0]：**没问题，已实测**
`tokenizer.ggml.eos_token_id = 0`，llama.cpp 把 id 0 放进 EOG 集合。
抽样实测每题正常结束都是 `finish_reason=stop` 且**末 token id=0**（Q13/14/16/18/61 全部 id=0）。
词表对齐也无误：GGUF `token[1]='\x00'` ↔ 官方 vocab 第 1 行 `1 '\x00'`。

### 4.2 顺手排除一个"假主因"：id 261 (`\n\n`) 被当 EOG
gguf 自带 `eot_token_id = 261`，llama.cpp 把它也插进 EOG，而官方只认 id 0 —— 这是个真实口径偏差。
但实测（EOG={0,261} vs `--override-kv tokenizer.ggml.eot_token_id=int:0` 强制 EOG={0}，同 seed×3）
**输出逐字节相同、空行数都是 0** → **不触发，不是截断原因**。

### 4.3 真正主因：**思考模式没关**
harness 走 `--jinja` 模板，生成前缀只有裸 `Assistant:`（= 开思考）；
官方是 `Assistant: <think></think`（= 关思考）。同题同采样参数对比：

| 题 | 裸 `Assistant:`（我跑的） | `<think></think`（官方格式） |
|---|---|---|
| Q01 agent | **8192 截断** | **327 tok, stop**，答案完整 |
| Q21 math | **8192 截断**（卡在"3×4=12,加进位1,得13"） | **56 tok, stop**，`最终答案：1184` |
| Q36 ruozhiba | **8192 截断** | **299 tok, stop** |

### 4.4 次因：预算用的是 8192 而非官方的 500
### 4.5 两个方法论教训
* 113 道截断题的 `completion_tokens` **精确全是 8192** → 全是撞 `max_tokens`，无一提早停。
* 我最初用 8-gram 重复度查"死循环"，只抓到 1 道 —— **该指标漏检严重**：模型是"换着说法绕圈子"
  （Q41 逐条枚举日志`不是。【0257】…不是。`、Q46 反复纠结四舍五入位数、Q21 反复重算同一段乘法），
  而非逐字重复。上一版据此得出的"提升来自消除死循环"不成立。

## 5. 四种条件全量 200 题实测
| 条件 | 总分 | agent | coding | math | dialog | ruozhi | longctx | other | 截断率 | 耗时 |
|---|---|---|---|---|---|---|---|---|---|---|
| 旧基线 8192/思考/旧协议 | 41.2 | 38.2% | 25.0% | 40.0% | 48.8% | 62.5% | 21.7% | 72.5% | 164/200 | 230m |
| demo采样 8192/思考 | 48.3 | 38.5% | 40.0% | 45.0% | 59.6% | 70.0% | 39.0% | 67.5% | 113/200 | 214m |
| demo采样 500/思考 | 37.9 | 42.8% | **1.2%** | 27.5% | 62.5% | 75.0% | 27.2% | 70.8% | 177/200 | 19m |
| **demo采样 500/不思考（=官方）** | **44.2** | 35.6% | **41.2%** | 30.0% | 59.6% | **85.0%** | 18.7% | 65.0% | **90/200** | 14m |

* **真·官方条件的成绩是 44.2**，不是 48.3；相对旧基线 41.2 只有 **+3.0**。
* 截断率 164/200 → **90/200**。
* 逐题（官方 vs 旧基线）：赢 39、输 32、平 129。
* ⚠️ **不能简单说"官方解码更好"**：在**两边都没被截断的 27 题**上，旧基线 56.5 vs 官方 40.2
  —— 官方条件在 math(30.0%) 和 longctx(18.7%) 上明显吃亏，因为 500 token 不够它在长材料上展开。
  它是"另一个工作点"，不是全面更优。

## 6. 交付物
本地 `workspace\bench_rwkv\`：
* `results_v3_rwkv7_g1j_2p9b_q4_official.json`（官方条件 44.2，推荐引用）
* `results_v3_rwkv7_g1j_2p9b_q4_rwkvdemo.json`（8192/思考 48.3，非官方）
* `results_v3_rwkv7_g1j_2p9b_q4_demo500.json`（500/思考 37.9）
* `diag_truncation.py` / `diag_multiturn.py` / `diag_stop_token.py` / `test_eog_261.py`（排查脚本）
* `compare_four.py`（四条件对比）、`report\iq_v3_report.html`
服务器 `$BENCH_ROOT/`。服务已全部停止。
