# 评测框架（iq_bank_v3）

一套面向**中文端侧小模型**的能力基准，200 题，7 类，按难度分 3 档。

## 结构

| 文件 | 说明 |
|---|---|
| `run_eval.py` | 原始 harness（本项目基于它改） |
| `run_eval_rwkv.py` | **本项目使用的版本**：新增 `--profile`、`--no-think` 与 RWKV 罚项 CLI |
| `build_bank_v3.py` | 题库构建脚本（生成 `iq_bank_v3.json`） |
| `build_report.py` | 生成 HTML 报告 |
| `agg_v3.py` | 汇总统计 |
| `samples/iq_bank_v3_sample.json` | **示例子集：17 题，覆盖全部 7 种判分器** |

> 完整 200 题题库未公开，避免被用于针对性训练。schema 完全一致，示例可直接跑通。

## 题库构成

| 类别 | 题数 | 权重 |
|---|--:|--:|
| agent | 40 | 0.2 |
| coding | 40 | 0.2 |
| math | 40 | 0.2 |
| dialogue | 20 | 0.1 |
| ruozhiba（脑筋急转弯） | 20 | 0.1 |
| longctx（长上下文） | 20 | 0.1 |
| other | 20 | 0.1 |

难度分档：easy 100（Q01-50 + Q101-150 双平行卷）、medium 50（Q51-100）、hard 50（Q151-200）。
总分 = `Σ(类权重 × 类内得分率) × 100`。

## 判分器类型

| type | 判据 |
|---|---|
| `keys` | 关键词组命中率（每组命中任一即可） |
| `exact` | 末行 `最终答案：X` 精确匹配 |
| `num` / `num_range` | 数值 + 容差 |
| `tool` | 多轮工具调用回显，`0.5×调用分 + 0.5×最终分` |
| `code` | 执行隐藏测试，需输出 `PASS` |
| `jsonspec` | JSON 结构校验 |

## 运行

```bash
# 1) 起服务（见 ../rwkv-sampler/）
BENCH_ROOT=/data/rwkvbench bash ../rwkv-sampler/serve.sh &

# 2) 跑评测
python3 run_eval_rwkv.py \
  --bank samples/iq_bank_v3_sample.json \
  --profile rwkv-demo --no-think \
  --base-url http://127.0.0.1:18860 \
  --model rwkv29-g1j-q4 \
  --out results.json \
  --max-tokens 500 --tier all
```

## `--profile` 说明

| profile | 下发内容 |
|---|---|
| `default` | top_p 0.95 + 显式关闭所有罚项（llama.cpp 原生默认行为） |
| `legacy` | **只**下发 temperature + top_p 0.95，其余走服务端启动参数（复现原 v3 协议） |
| `rwkv-demo` | temp 1 / top_p 0.5 / top_k 500 / count 0.1 / presence 1 / decay 0.99 |

`--no-think` 额外下发 `chat_template_kwargs={"enable_thinking": false}`，对齐 RWKV 官方 demo
的 `Assistant: <think></think` 生成前缀。

## 已知问题（本项目未修）

- `--bank` 默认值是 v1，不是 v3 —— 必须显式传 `--bank`
- `build_report.py` 的 `NAMES` 有拼写错误（`minicpm5_*` vs 实际 `mincpm5_*`）
- `agg_v3.py` 里硬编码了一份模型列表
