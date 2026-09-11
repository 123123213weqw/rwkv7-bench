# RWKV7-G1j-2.9B-Q4_K_M × iq_bank_v3：改用 RWKV 官方 demo 解码

## 1. 任务
在服务器（2×V100-32G）上用 **RWKV 官方 Gradio demo 的解码方法**重测 G1j-2.9B-Q4：
Temperature 1 / Top P 0.5 / Presence Penalty 1 / Count Penalty 0.1 / Penalty Decay 0.99
（参照 https://huggingface.co/spaces/BlinkDL/RWKV-Gradio-2/blob/main/app.py ）

## 2. 官方采样器语义（app.py，逐行核对）
```python
# 每次生成调用：预填后 occurrence_* 清零 → prompt token 不计罚
if alpha_frequency: logits.sub_(occurrence_count, alpha=alpha_frequency)   # -= 0.1 * count
if alpha_presence:  logits.sub_(occurrence_presence)                        # -= 1.0（出现过的 token）
sampled = sample_logits_batch_cuda(logits, temperature=1.0, top_p=0.5, k=500)  # topk500→softmax→topp
if penalty_decay != 1: occurrence_count.mul_(penalty_decay)                 # 0.99 衰减
occurrence_count[batch_rows, sampled] += 1
if alpha_presence: occurrence_presence[batch_rows, sampled] = alpha_presence
```

## 3. 为什么必须打补丁（llama.cpp 原生表达不了）
| 官方语义 | llama.cpp 原生 | 差异 |
|---|---|---|
| presence/count penalty **只对生成 token** 计罚 | `penalties` sampler 在 `init_sampler()` 时先把**整个 prompt** accept 一遍（server-context.cpp:349） | 原生会把 prompt 里出现过的 token 全部扣 1.0 / 按次数扣 |
| count penalty 每步 ×0.99 衰减 | `frequency_penalty` 是窗口内线性计数，无衰减 | 无衰减 ⇒ 罚项随时间无限增长 |
| 罚项作用在 **原始 logits**，在 top-k/top-p 之前 | 原生 `penalties` 在链尾（top-p 之后） | 截断发生在罚项之前，语义不同 |
| 罚项 = 减去 `0.1*count + 1.0` | `repeat_penalty` 是乘除式 | 形式不同 |

## 4. 补丁内容（`patch_llama_rwkv_penalties.py`，作用于 llama-rwkv-pr1，构建副本在
`$BENCH_ROOT/llama-rwkv-pr1`，原件未改动）
| 文件 | 改动 |
|---|---|
| `src/llama-sampler.cpp` | 新增 `llama_sampler_rwkv_penalties`（apply/accept/reset/clone/free/name + init） |
| `include/llama.h` | 声明 `llama_sampler_init_rwkv_penalties(n_vocab, count, presence, decay)` |
| `common/common.h` | 新增 `rwkv_count_penalty / rwkv_presence_penalty / rwkv_penalty_decay` |
| `common/sampling.cpp` | 在采样链**最前端**插入该 sampler（top-k 之前，看到原始 logits） |
| `tools/server/server-schema.cpp` | 三个请求字段 `rwkv_count_penalty` / `rwkv_presence_penalty` / `rwkv_penalty_decay` |
| `tools/server/server-task.cpp` | 两处参数回声表 |

关键实现点：
* `started` 标志：第一次 `apply()` 之前的 `accept()`（即 `init_sampler()` 灌入的 prompt token）**全部忽略** →
  只统计生成 token，与官方“预填后计数器清零”一致。
* `accept(t)`：先把所有 touched 的 count ×0.99，再 `count[t]+=1`、`presence[t]=presence_penalty`（与官方同序）。
* `apply()`：`logit[id] -= count_penalty*count[id] + presence[id]`，并置 `sorted=false`。

## 5. 验证（已做）
1. `/props` 的 `default_generation_settings.params` 回声里出现三个新字段 → schema 生效。
2. **罚项确实作用在采样上（决定性）**：`test_penalty_isolated.py`，完全固定 temp 1.0 / top_p 0.5 / top_k 500，
   只切换 rwkv 罚项、同 seed：
   | 组 | 输出 | 长度 |
   |---|---|---|
   | A 无罚项 | `31 32 33 34 …` | 210 tok |
   | B demo 罚项(count .1 / presence 1.0 / decay .99) | 文本与 A 不同 | 225 tok |
   | C presence=50（夸张） | 与 A 完全不同 | **83 tok** |
   → 若补丁未生效，C 不可能与 A 不同且短一半。
3. 行为对比（同 seed，/v1/completions，400 token）：关罚项输出退化成短语死循环，开罚项不循环。
4. chat 层 A/B（真实题目，max_tokens=1024，字符级 8-gram 指标）：
   | 题 | 默认协议(t=.2/top_p=.95/无罚项) | RWKV demo 解码 |
   |---|---|---|
   | Q01 agent | 最长重复 64，去重率 0.125（死循环） | 最长重复 5，去重率 0.622 |
   | Q21 math  | 最长重复 11，去重率 0.347 | 最长重复 5，去重率 0.795 |
   | Q61 coding| 3 / 0.921 | 3 / 0.917 |
5. decay 0.99 vs 1.0（同 seed）输出不同 → 新字段确实进入了采样器。

### 一个必须说明的验证陷阱
`/v1/completions` 返回的 `logprobs` 是**采样前的原始 logits**（`北` 的 logprob 恒为 -0.7287293672561646，
与罚项开/关无关）。所以用「首步 logprobs 是否一致」去证明「prompt token 未计罚」是**无效证据**
（`test_prompt_tokens.py` / `test_penalty_semantics.py` 均因此得出空的差异集）。
prompt 不计罚这一点由**决定性行为测试**证明（`test_prompt_only2.py`，greedy `top_k=1` ⇒ 采样即 argmax，与 seed 无关；
prompt = `"1 " * 40`，「1」出现 40 次，第一步 argmax 必为「1」）：

| 组 | 输出 |
|---|---|
| A 无罚项 | `1 1 1 1` |
| B `presence=1000` | `1 1\n0` ← **第 1 个字符与 A 相同**（→ prompt token 没被扣），第 3 步起才分岔（→ 生成 token 被扣） |
| C 官方 demo 参数 | `1 1 1 1` |

即：罚项从第 2 个 token 起才可能生效 —— 与官方「预填后计数器清零」一致。

## 6. 模型来源
user 上没有 G1j 任何权重（只有 g1g/g1i）；huggingface.co 直连不通，走 hf-mirror.com 下载
`shoumenchougou/RWKV7-G1j-2.9B-GGUF : rwkv7-g1j-2.9b-Q4_K_M.gguf`（1,919,047,616 B，sha 见下载日志）。
ModelScope 的 `Blink_DL/rwkv7-g1` 只有 .pth（`rwkv7-g1j-2.9b-20260831-ctx16384.pth`，备份用）。

## 7. 运行
* 服务：`bin/serve_g1j.sh` → llama-server（编译自补丁版，CUDA sm70），port 18860，
  `-c 49152 --jinja --parallel 1`，CLI 默认即 demo 协议。
* 评测：`python3 bench/run_eval_rwkv.py --bank iq_bank_v3.json --profile rwkv-demo --base-url http://127.0.0.1:18860
  --model rwkv29-g1j-q4 --out results_v3_rwkv7_g1j_2p9b_q4_rwkvdemo.json --max-tokens 8192 --tier all --resume`
  （harness 在 `chat()` 里下发全部采样参数；`sampling` 字段写入结果 JSON）
* 结果：`bench/results_v3_rwkv7_g1j_2p9b_q4_rwkvdemo.json`，日志 `logs/eval_g1j_q4_rwkvdemo.log`

## 8. 结果
| | 旧协议（temp .2 / top_p .95 / 无罚项，旧二进制） | **RWKV 官方 demo 解码（本次）** | Δ |
|---|---|---|---|
| 加权总分 | 41.2 | **48.3** | **+7.1** |
| agent | 38.2% | 38.5% | +0.2 |
| coding | 25.0% | 40.0% | **+15.0** |
| math | 40.0% | 45.0% | +5.0 |
| dialogue | 48.8% | 59.6% | +10.8 |
| ruozhiba | 62.5% | 70.0% | +7.5 |
| longctx | 21.7% | 39.0% | **+17.3** |
| other | 72.5% | 67.5% | −5.0 |
| easy / medium / hard | 47.5 / 27.8 / 24.6 | **52.4 / 40.3 / 36.3** | +5.0 / +12.5 / +11.7 |
| 撞 8192 上限 | 164/200 | **113/200** | −51 |
| 总耗时 | 230 min | 214 min | −16 |

逐题：新协议赢 36 题、旧协议赢 24 题、其余 140 题相同。
提升集中在 coding/longctx —— 正是旧协议下反复「死循环到 8192 截断」的重灾区。
入榜位置：总榜第 23/26（RWKV 家族第 3），见 `report\iq_v3_report.html`。

## 9. 对照实验（把「采样改动」与「二进制/模型文件差异」分开）
同一补丁二进制 + **旧协议**（`--profile legacy`，只下发 temperature + top_p 0.95，其余用服务端默认）
另跑两组子集（服务 `serve_g1j_plain.sh`，port 18861）：

**A 组 = agent 40 题**（Q01-10 / Q51-60 / Q101-110 / Q151-160）：
| | 旧基线 | demo 解码 | 对照(新二进制+旧协议) |
|---|---|---|---|
| 子集加权 | 7.7 | 7.7 | 8.2 |
| agent 得分率 | 38.2% | 38.5% | 41.1% |
| 撞 8192 上限 | 33/40 | **23/40** | 34/40 |
| 用时 | 46 min | 35 min | 46 min |

→ agent 上三种组合差异在噪声内（n=40，stderr≈±7.7%），与全量 +0.2 一致；
但**截断率**：同二进制下 demo 23/40 vs 旧协议 34/40 → 采样改动确实显著减少「跑飞」。

**B 组 = coding+math 20 题**（Q11-20 coding / Q21-30 math），同一批题三方对比：
| | 旧基线(旧二进制+旧协议) | demo 解码(新二进制) | 对照(新二进制+旧协议) |
|---|---|---|---|
| coding | 50.0% | **55.0%** | 20.0% |
| math | 40.0% | **40.0%** | 30.0% |
| 撞 8192 上限 | 13/20 | **9/20** | 14/20 |

→ **同二进制同模型文件**下，把默认采样换成 RWKV 官方 demo 解码：coding 20%→55%、math 30%→40%，
截断 14/20→9/20。这是把「采样改动」单独隔离出来的干净对比。

⚠️ 需要声明的偏差：**「新二进制 + 旧协议」这一列低于历史旧基线**（coding 20% vs 50%），
说明我复现的 legacy 协议 ≠ 上一批服务端的真实配置（上一批 llama-server 的启动参数已不可考，
大概率不是 llama.cpp 纯默认，例如 top_k/min_p 不同）。因此：
* 「+7.1 / coding +15」是**相对历史记录**的差值，含二进制与启动参数差异，不能 100% 归因于采样；
* 但「同二进制下 demo 采样 ≫ 默认采样」这一结论是受控的。

## 10. 口径与注意事项
* 与上一批（`results_v3_rwkv7_g1j_2p9b_q4.json`，总分 41.2）相比，本次改的是采样协议，
  但**推理二进制不同**（上一批的 llama-server 版本未知）→ 差值不完全可归因于采样，故另跑一次
  同二进制 + 默认协议的对照（`results_v3_rwkv7_g1j_2p9b_q4_legacy40.json`，40 题子集）。
* gguf 来自社区量化（shoumenchougou），与上一批是否同一份文件无法确认（本地/服务器都无上一批的 gguf）。
* 未固定 seed（与上一批一致，服务器默认随机）。
* 工具类题目（28 题）每轮工具调用都是一次独立请求 → 罚项状态按轮重置；官方 app 的单次生成调用同样重置，语义一致。
