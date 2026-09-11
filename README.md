# RWKV7 解码保真与量化损伤研究

针对 **RWKV7-G1j-2.9B** 在 llama.cpp 上的**解码口径保真**与**量化档位副作用**做的实测研究，
附带一套可复用的中文端侧模型评测框架（`iq_bank_v3`）。

核心是两条可复现的结论：

| # | 结论 | 关键证据 |
|---|---|---|
| **A** | llama.cpp 原生采样器**无法表达** RWKV 官方 demo 的解码语义，需要新增一个采样器 | 补齐后 Q4 从 41.2 → 44.2（官方口径），截断 82% → 45% |
| **B** | **Q4_K_M 量化档破坏了这个模型的终止行为** —— 同模型换 Q8_0，同口径 **44.2 → 64.1（+19.9）**，截断 **90/200 → 39/200** | 12 道 Q4 全部跑满 8192 拿 0 分的题，Q8 有 9 道在 18 秒内正常收尾并答对 |

结论 B 是更重要的那条：**它说明"低分"未必是模型能力问题，可能是量化档坏了**。

---

## 1. 为什么需要改采样器

RWKV 官方 Gradio demo（`BlinkDL/RWKV-Gradio`）的解码流程是：

```python
# 1) 罚项作用在【原始 logits】上，位于 top-k / top-p 之前
if alpha_frequency: logits.sub_(occurrence_count, alpha=alpha_frequency)   # count 罚 ×0.1
if alpha_presence:  logits.sub_(occurrence_presence)                       # presence 罚 1.0

# 2) 采样
sampled = sample_logits_batch_cuda(logits, temperature=1.0, top_p=0.5, k=500)

# 3) 采样后更新计数器（带衰减）
if penalty_decay != 1: occurrence_count.mul_(penalty_decay)               # ×0.99
occurrence_count[batch_rows, sampled_tensor] += 1
if alpha_presence: occurrence_presence[batch_rows, sampled_tensor] = alpha_presence
```

即 **Temperature 1 / Top P 0.5 / Presence Penalty 1 / Count Penalty 0.1 / Penalty Decay 0.99**，
且生成前缀为 `Assistant: <think></think`（**默认关闭思维链**）。

llama.cpp 原生的 `repeat_penalty` / `presence_penalty` / `frequency_penalty` **四处都对不上**：

| 差异 | llama.cpp 原生 | RWKV 官方 demo |
|---|---|---|
| prompt 是否计入罚项 | `init_sampler()` 会把整个 prompt 先 `accept()` 一遍 → **prompt 也被罚** | 计数器在 prefill 后清零 → **prompt 不计罚** |
| 频次罚是否有衰减 | 无 | `count *= 0.99` |
| 罚项在采样链的位置 | 链**尾部**（top-k/top-p 之后） | 链**头部**（作用于原始 logits） |
| 罚项形式 | multiply/divide `repeat_penalty` | **减去** `count_penalty*count + presence` |

因此本项目新增了采样器 `llama_sampler_init_rwkv_penalties()`：

- 链最前端插入（能看到原始 logits）
- `started` 标志位：第一次 `apply()` 之前的所有 `accept()`（即 prompt）**全部忽略**
- `count` / `presence` 浮点向量 + `touched` 列表做稀疏衰减，避免每步遍历整个词表

补丁见 [`rwkv-sampler/`](rwkv-sampler/)，新增请求字段：
`rwkv_count_penalty` / `rwkv_presence_penalty` / `rwkv_penalty_decay`。

### 补丁的验证

| 测试 | 方法 | 结果 |
|---|---|---|
| 参数生效 | `/props` 回声 | 三个新字段出现 ✔ |
| 罚项确实起作用 | **完全固定** temp/top_p/top_k，只切罚项 | presence 从 1 提到 50 → 输出从 210 tok 变成 83 tok 且内容完全不同 ✔ |
| **prompt 未被计罚**（关键） | greedy `top_k=1`，prompt `"1 "*40` | 无罚项输出 `1 1 1 1`；presence=1000 输出 `1 1\n0` —— **第一个字符仍相同**（prompt 未计罚），第 3 步才分岔（生成 token 被罚）✔ |
| 衰减生效 | 同 seed，`decay 0.99` vs `1.0` | 输出不同 ✔ |

> ⚠️ **一个坑**：llama.cpp server 的 `/v1/completions` 返回的 `logprobs` 是**采样前的原始 logits**，
> 用它验证罚项是无效的（本项目一开始就被误导过）。必须用**行为测试**。

---

## 2. 结果

### 2.1 解码口径对照（同模型 Q4_K_M、同题库各 200 题）

| 解码口径 | 二进制 | max_tokens | 思维链 | **总分** | agent | coding | math | dialogue | 急转弯 | 长上下文 | 其他 | 截断 |
|---|---|--:|:--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 默认协议（原 v3） | 旧 | 8192 | 开 | **41.2** | 38.2 | 25.0 | 40.0 | 48.8 | 62.5 | 21.7 | 72.5 | 164/200 |
| 官方 demo 采样 | 本补丁 | 8192 | 开 | **48.3** | 38.5 | 40.0 | 45.0 | 59.6 | 70.0 | 39.0 | 67.5 | 113/200 |
| **官方 demo 采样（完整对齐）** | 本补丁 | **500** | **关** | **44.2** | 35.6 | 41.2 | 30.0 | 59.6 | 85.0 | 18.7 | 65.0 | 90/200 |
| 官方 demo 采样 | 本补丁 | 8192 | 关 | **43.4** | 35.2 | 52.5 | 32.5 | 52.5 | 50.0 | 23.3 | 67.5 | 74/200 |
| 官方 demo 采样 | 本补丁 | 500 | 开 | **37.9** | 42.8 | **1.2** | 27.5 | 62.5 | 75.0 | 27.2 | 70.8 | 177/200 |

"官方口径" = temp 1 / top_p 0.5 / top_k 500 / presence 1 / count 0.1 / decay 0.99
\+ 生成前缀 `Assistant: <think></think`（关思维链）+ max_tokens 500（官方滑块默认值）。

**读法**：

- **思维链是主导变量**。同为 500 预算，关掉思维链使总分 37.9 → 44.2、截断 177/200 → 90/200。
  开着思维链在 500 token 内 **coding 只有 1.2%** —— 全花在思考上，代码还没开始写。
- 48.3（8192/开）比 44.2 高，**但那 4.1 分是"多给 16 倍 token 预算"买来的**，不是解码更正确。

### 2.2 加预算没用（重要反直觉结果）

固定"关思维链"，只扫 `max_tokens`（同一份 200 题结果按不同阈值统计）：

| max_tokens | 500 | 1024 | 2048 | 4096 | 8192 |
|---|--:|--:|--:|--:|--:|
| 截断题数 | 85 | 78 | 76 | **76** | 74 |
| 截断率 | 42.5% | 39.0% | 38.0% | **38.0%** | 37.0% |

生成 token 分布是**强双峰**：`<500` 有 115 题（58%），`=8192` 有 74 题（37%），中间地带几乎为空（2k~4k 恰好 0 题）。

**并且总分反而下降**：500/关 = 44.2，8192/关 = 43.4（耗时 14m → 117m，急转弯 85.0 → 50.0）。
因为跑得越久，模型越可能在答案后**又补一段**，而判分取最后一行 `最终答案：`，于是取到后面那个错答案。

> 结论：约 37% 的题"不吐 EOS"是**模型/权重层面的行为**，不是评测预算能修的 —— 而这正好指向结论 B。

### 2.3 量化档位：Q4_K_M vs Q8_0（同口径 500/关思维链）

| 量化 | **总分** | agent | coding | math | dialogue | 急转弯 | 长上下文 | 其他 | 截断 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Q4_K_M | 44.2 | 35.6 | 41.2 | 30.0 | 59.6 | 85.0 | 18.7 | 65.0 | 90/200 |
| **Q8_0** | **64.1** | 46.4 | **80.0** | **57.5** | 70.4 | 85.0 | 30.0 | 87.5 | **39/200** |

**+19.9 分**。逐题：Q8 赢 50、输 15、平 135。

截断按类别（Q4 → Q8）：`coding 15/40 → 0/40` · `math 20/40 → 0/40` · `longctx 12/20 → 0/20` ·
`agent 17/40 → 15/40` · `dialogue 6/20 → 8/20` · `急转弯 12/20 → 11/20` · `other 8/20 → 5/20`。

**有唯一答案的类别（coding / math / longctx）截断直接归零**，剩下的全是开放式闲聊类。

### 2.4 隔离实验：同一批题，只换量化

从 Q4 结果里取出**跑满 8192 且得 0 分**的 12 道题，用**完全相同的协议**在 Q8 上重跑：

| 题 | Q4 | **Q8** | Q8 耗时 |
|---|---|---|---|
| Q116 coding | 8192 截断 → 0 | **PASS** | 1.0s |
| Q120 coding | 8192 截断 → 0 | **PASS** | 6.5s |
| Q122 math | 8192 截断 → 0 | **1.00** | 9.6s |
| Q128 math | 8192 截断 → 0 | **1.00** | 13.6s |
| Q136 急转弯 | 8192 截断 → 0 | **1.00** | 12.9s |
| Q137 急转弯 | 8192 截断 → 0 | **1.00** | 17.6s |
| Q162 coding | 8192 截断 → 0 | **PASS** | 11.9s |
| Q165 coding | 8192 截断 → 0 | **PASS** | 12.0s |
| Q166 coding | 8192 截断 → 0 | **PASS** | 2.8s |
| Q177 math | 8192 截断 → 0 | **1.00** | 105.1s |
| Q117 coding | 0 | 0（UTF-8 语法错） | 105.2s |
| Q170 coding | 0 | 0（IndexError） | 105.2s |
| **合计** | **0/12** | **9/12** | |

这些题 Q4 全部跑满预算、Q8 大多一两秒就正常收尾 —— **Q4 档下模型失去了输出终止符的能力**。

**机理推测**（未直接验证）：RWKV 的 `wkv` 状态是跨数千步的递归累加，权重中的量化误差会随递推**逐步累积放大**；
Transformer 每层重算、不累积，所以对 Q4 更耐受。这与"同一份榜单里 Qwen/Qwen3.5 的 Q4 行都正常、
唯独 RWKV 的 Q4 行塌了"一致。

---

## 3. 排查过程中确认的几件事（避免重复踩坑）

| 疑点 | 实测结论 |
|---|---|
| **token 0 是否被当停止符** | **正常**。gguf 的 `eos_token_id = 0`，llama.cpp 已放入 EOG 集合；实测正常结束的题 `finish_reason=stop` 且末 token id=0 |
| 词表是否错位 | **没有**。gguf `token[1]='\x00'` 与官方 `rwkv_vocab_v20230424.txt` 第 1 行 `1 '\x00'` 严格对应；官方 HF tokenizer 注释明确 `id 0` 是 eot |
| `\n\n`(id 261) 被当 EOG | **是偏差但不触发**。gguf 自带 `eot_token_id=261`，llama.cpp 也把它当停止符，而官方只认 `token==0`；但强制 EOG={0} 后**同 seed 三次输出逐字节相同**（该模型不输出空行）。见 `rwkv-sampler/serve_eog0.sh` |
| 是不是"死循环" | **不是逐字重复**。用 8-gram 检测只抓到 1/113 道；实际是"换着说法绕圈子"（逐条枚举日志、反复纠结保留几位小数）。**用 n-gram 判循环会严重漏检** |
| `logprobs` 能否验证采样器 | **不能**。返回的是采样前原始 logits（对同一 token 恒定）。必须用行为测试 |

---

## 4. 仓库结构

```
bench/                       评测框架
  run_eval.py                原版 harness
  run_eval_rwkv.py           本项目扩展版（--profile / --no-think / 补丁参数）
  build_bank_v3.py           题库生成
  build_report.py            报告生成
  agg_v3.py                  汇总
  samples/iq_bank_v3_sample.json   示例子集（17 题，覆盖全部 7 种 grader）
rwkv-sampler/                llama.cpp 补丁与部署
  patch_llama_rwkv_penalties.py   新增 rwkv_penalties 采样器
  patch_task_maps.py              server-task.cpp 缩进适配
  build_patched.sh / serve*.sh
experiments/
  verify_sampler/            采样器行为验证（隔离测试 / prompt 计罚测试 / EOG 测试）
  diagnostics/               截断归因（终止 token、多轮续写、预算扫描、量化对比）
  analysis/                  统计与制表
results/rwkv7_g1j_2p9b/      本次全部结果 JSON（含 sampling 与 chat_template_kwargs 元信息）
docs/                        详细笔记与表格
report/                      汇总报告 HTML
```

---

## 5. 复现

### 5.1 评测框架

```bash
# 起服务（需先打补丁编译，见 rwkv-sampler/）
BENCH_ROOT=/path/to/rwkvbench MODEL=$BENCH_ROOT/models/xxx.gguf ./rwkv-sampler/serve.sh &

# 跑示例题库
python3 bench/run_eval_rwkv.py \
  --bank bench/samples/iq_bank_v3_sample.json \
  --profile rwkv-demo --no-think \
  --base-url http://127.0.0.1:18860 --model rwkv29-g1j-q4 \
  --out out.json --max-tokens 500 --tier all
```

`--profile` 三档：

| profile | 下发内容 | 用途 |
|---|---|---|
| `default` | top_p 0.95 + 关闭所有罚项 | 与历史榜单口径一致 |
| `rwkv-demo` | temp 1 / top_p 0.5 / top_k 500 / presence 1 / count 0.1 / decay 0.99 | RWKV 官方 demo |
| `legacy` | **只**下发 temperature + top_p 0.95，其余走服务端默认 | 复现旧协议 |

`--no-think` 注入 `chat_template_kwargs={"enable_thinking": false}`，
对齐官方前缀 `Assistant: <think></think`。

### 5.2 采样器验证（不依赖完整题库）

```bash
python3 experiments/verify_sampler/test_penalty_isolated.py   # 固定 temp/top_p，只切罚项
python3 experiments/verify_sampler/test_prompt_only2.py       # greedy 验证 prompt 不计罚
python3 experiments/verify_sampler/test_eog_261.py            # EOG 口径差异
```

---

## 6. 评测框架简介（iq_bank_v3）

- **200 题**，7 类别，加权总分 = `Σ(类权重 × 类得分率) × 100`
  权重：`agent/coding/math` 各 0.2，`dialogue/ruozhiba/longctx/other` 各 0.1
- **难度分层**：easy 100（双平行卷 A/B，可估计噪声）/ medium 50 / hard 50，`difficulty` 1–10
- **7 种判分器**：`keys`（关键词组命中率）、`exact`（取最后一行 `最终答案：`）、`num` / `num_range`、
  `tool`（多轮工具回显，0.5×调用 + 0.5×终答）、`code`（**执行隐藏测试**，需 `PASS`）、`jsonspec`
- 结果 JSON 会记录 `sampling` 与 `chat_template_kwargs`，便于口径追溯

> **题库只提供示例子集**（`bench/samples/`，17 题覆盖全部判分器）。
> 完整 200 题不公开 —— 避免被用于针对性训练（基准测试的防刷分要求）。
> schema 与完整版完全一致，跑通流程不受影响。

---

## 7. 局限与未竟事项

1. **Q4 的 gguf 是社区量化**。G1j 这一代 HF 上**只有 `shoumenchougou` 一家**发布 GGUF，
   官方 `RWKV/RWKV7-G1j-2.9B-20260831` 只有 safetensors、`BlinkDL/rwkv7-g1` 只有 `.pth`。
   因此结论 B 严格来说是"**这份社区 Q4 量化**损伤了终止行为"。
   已排除转换脚本问题（同仓库 Q8 正常），但**未排除该仓库 Q4 量化参数选得差**。
   → 待办：从官方 safetensors 自己转一份 Q4_K_M 交叉验证。
2. **种子未固定**（与原榜单一致，单次成绩）。表内 <2 分的分差不应过度解读。
3. **样本量**：隔离实验 n=12 题，方向性明确但置信区间宽。
4. `results/` 中的 JSON **不含题目原文**（只有模型输出与判分），可直接公开。

---

## 8. 数据来源

| 项 | 值 |
|---|---|
| 权重 | `RWKV/RWKV7-G1j-2.9B-20260831`（官方 safetensors 5.90 GB）/ `BlinkDL/rwkv7-g1`（.pth） |
| GGUF | `shoumenchougou/RWKV7-G1j-2.9B-GGUF`（社区，llama.cpp RWKV 支持 PR 指向的合集） |
| Q4_K_M | 1,919,047,616 B · sha256 `68c3a49dfc8a6dc34032d7c89adf64542ec9c81229a1333591163717bc3ab7b5` |
| Q8_0 | 3,258,603,456 B · sha256 `3004002e250af500592a2e7814cb0e0ab54dafa22a556853540ef36923709577` |

## 9. 许可

代码 MIT。评测题目（`bench/samples/`）与报告中的成绩数据版权归原作者，仅供研究参考。
