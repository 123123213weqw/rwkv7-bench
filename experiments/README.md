# experiments — 验证与诊断脚本

脚本按用途分三组。**所有脚本都曾在真实服务器上跑通并产出结论**，不是示意代码。

## verify_sampler/ — 证明补丁语义正确

| 脚本 | 验证什么 | 期望结果 |
|---|---|---|
| `test_penalty_isolated.py` | **固定** temp/top_p/top_k/seed，只切换罚项 | presence=50 时输出从 210 tok 变 83 tok 且内容完全不同 → 罚项确实生效 |
| `test_prompt_only2.py` | **prompt 是否被计罚**（最关键的一条） | greedy `top_k=1`、prompt `"1 "*40`：无罚项 `1 1 1 1`；presence=1000 → `1 1\n0`（**首字符相同** ⇒ prompt 未计罚；第 3 步分岔 ⇒ 生成 token 被罚） |
| `test_eog_261.py` | `\n\n`(id 261) 被当 EOG 是否提前截断 | 同 seed 三种：EOG={0,261} 与 EOG={0} 输出**逐字节相同** → 该口径差异在本模型上不触发 |

## diagnostics/ — 定位"为什么截断这么多"

按排查顺序排列，记录了一次完整的误判与纠正过程：

| 脚本 | 作用 |
|---|---|
| `diag_stop_token.py` | 打印每题 `finish_reason` + **终止 token id**。查出 token id 0 确实触发停止（排除用户对 token[0] 的疑虑），并发现 id=261 也停 |
| `diag_truncation.py` | 按类别/重复度分层截断题。**注意**：用 8-gram 判"死循环"会**严重漏检**（本模型是"换着说法绕圈子"而非逐字重复），本项目一开始就栽在这里 |
| `diag_multiturn.py` | 对比 `Assistant:`（裸）vs `Assistant: <think></think`（官方格式）。**这是截断的真正主因**：同题裸格式 8192 截断，官方格式 56~327 tok 就收尾 |
| `diag_budget.py` | 预算扫描 + token 分布。发现分布**强双峰**（58% 的题 <500 tok 收尾，37% 一路跑满），证明"加预算"治不了截断 |
| `diag_why.py` | 2.9B-Q4 与 1.5B-Q8 逐题对比，发现"2.9B 未截断时反而更强"（0.705 vs 0.671）→ 指向量化问题 |
| `diag_think.py` | 统计截断题里 `<think>`/`</think>` 出现率 |

## analysis/ — 出表与报告

| 脚本 | 作用 |
|---|---|
| `trunc_all.py` | 所有运行的截断次数汇总（按各自 max_tokens 判定） |
| `compare_four.py` | 四种解码口径的完整对照 + 逐题胜负 |
| `final_q4_q8.py` | **Q4 vs Q8 官方口径对照**（核心结论 B 的出处） |
| `make_tables.py` | 生成 `docs/table_full.md` / `.csv` |
| `patch_q8_official.py` | 修补被 OOM 污染的 5 题（`HARNESS ERROR` → 实跑成绩） |
| `merge_into_report.py` | 把新结果并入报告 HTML（含方法论声明更新） |
| `verify_report.py` | 校验合并后 HTML 的 DATA 数组可解析、结构完整 |

## 复现顺序建议

```
1. 起服务（rwkv-sampler/serve.sh）→ 跑 run_eval_rwkv.py --profile rwkv-demo
2. verify_sampler/* 确认补丁语义
3. diagnostics/diag_multiturn.py 观察思维链格式的影响
4. 换 Q8 权重重跑 → final_q4_q8.py 出对照
```

> ⚠️ 这些脚本里写死了 `BENCH_ROOT` 指向的目录结构，路径已脱敏为环境变量/占位符，
> 迁移到别的机器时请先检查路径常量（见各脚本头部的 `D =` / `HERE =`）。
