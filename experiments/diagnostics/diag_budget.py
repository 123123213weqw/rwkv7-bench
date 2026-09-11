#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""预算提升后的诊断：为什么 8192/关思考 只把截断从 45% 降到 37%，分数还略降。"""
import json, io, collections

D = r"C:\Users\31046\AppData\Roaming\com.xlang.xharness\workspace\bench_rwkv"
R = D + r"\report"
S500 = {x["id"]: x for x in json.load(io.open(R + r"\results_v3_rwkv7_g1j_2p9b_q4_official.json", encoding="utf-8"))["results"]}
S8K = {x["id"]: x for x in json.load(io.open(D + r"\results_v3_rwkv7_g1j_2p9b_q4_nothink8192.json", encoding="utf-8"))["results"]}

def tk(x):
    return (x.get("usage") or {}).get("completion_tokens", 0)

# 1) 双峰分布：8192/关 下 token 分布
print("=== 8192/关思考 的 token 分布（看是否双峰）===")
h = collections.Counter()
for q, x in S8K.items():
    t = tk(x)
    b = "<500" if t < 500 else ("500-1k" if t < 1000 else ("1k-2k" if t < 2000 else ("2k-4k" if t < 4000 else ("4k-8k" if t < 8192 else "=8192"))))
    h[b] += 1
for b in ["<500", "500-1k", "1k-2k", "2k-4k", "4k-8k", "=8192"]:
    print("  %-7s %3d" % (b, h[b]))
n_short = sum(1 for q, x in S8K.items() if tk(x) < 500)
print("  <500 token 的题: %d/200 (%.0f%%)  → 大多数题其实很短" % (n_short, 100.0*n_short/200))

# 2) 截断题与不截断题的得分
tr = [x for x in S8K.values() if tk(x) >= 8192]
nt = [x for x in S8K.values() if tk(x) < 8192]
print("\n=== 8192/关思考：截断 vs 未截断 的平均分 ===")
print("  截断 %3d 题 均分 %.3f" % (len(tr), sum(min(1.0, x["score"]) for x in tr)/len(tr)))
print("  未截断 %3d 题 均分 %.3f" % (len(nt), sum(min(1.0, x["score"]) for x in nt)/len(nt)))

# 3) 预算提升的得失
print("\n=== 500/关 -> 8192/关 逐题变化 ===")
win = lose = tie = 0
lost_list = []
for q in sorted(S500):
    a, b = min(1.0, S500[q]["score"]), min(1.0, S8K[q]["score"])
    if b > a: win += 1
    elif b < a: lose += 1; lost_list.append((q, S500[q]["cat"], a, b, tk(S500[q]), tk(S8K[q])))
    else: tie += 1
print("  变大 %d | 变小 %d | 相同 %d" % (win, lose, tie))
print("\n  预算变大后『分数下降』的题（前 12）:")
for q, c, a, b, t1, t2 in lost_list[:12]:
    print("    %-5s %-8s %.2f -> %.2f   (tok %d -> %d)" % (q, c, a, b, t1, t2))

# 4) 4096 够不够：有多少题在 4096 前就自然结束
print("\n=== 若把预算设为 4096，截断会是多少 ===")
for cap in (500, 1024, 2048, 4096, 8192):
    n = sum(1 for x in S8K.values() if tk(x) >= cap)
    print("  cap=%-5d 截断 %3d/200 = %4.1f%%" % (cap, n, 100.0*n/200))
