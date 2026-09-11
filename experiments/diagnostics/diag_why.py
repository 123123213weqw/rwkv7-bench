#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断 RWKV7-G1j-2.9B-Q4 为什么分数低：与同协议下的 1.5B/1.5B-g1i 对比。

三个结果都在「旧协议」下跑（temp0.2/top_p0.95/8192），可直接比。
看点：
  1. 截断率差异（终止失败）
  2. 截断题 vs 未截断题的得分
  3. 输出长度分布
  4. 是否输出 </think>（思考型痕迹）
  5. 逐题：1.5B 拿满而 2.9B 拿零的题
"""
import io, json, os, collections, re

D = r"C:\Users\31046\AppData\Roaming\com.xlang.xharness\workspace\bench_rwkv"
E = r"C:\Users\31046\AppData\Roaming\com.xlang.xharness\workspace\bench_inspect\extract"
CAP = 8192

RUNS = [
    ("RWKV7-G1j-1.5B Q8", "results_v3_rwkv7_g1j_1p5b_q8.json"),
    ("RWKV7-g1i-1.5B Q8", "results_v3_rwkv7_g1i_1p5b_q8.json"),
    ("RWKV7-G1j-2.9B Q4", "results_v3_rwkv7_g1j_2p9b_q4.json"),
]


def find(fn):
    for base in (D, os.path.join(D, "report"), E):
        p = os.path.join(base, fn)
        if os.path.exists(p):
            return p
    return None


def load(fn):
    p = find(fn)
    if not p:
        return None
    d = json.load(io.open(p, encoding="utf-8"))
    return {x["id"]: x for x in d["results"]}


loaded = []
for name, fn in RUNS:
    r = load(fn)
    if r:
        loaded.append((name, r))
        print("载入 %-22s %d 题" % (name, len(r)))
    else:
        print("!! 缺文件", fn)
print()

# 统一字段
k0 = list(loaded[0][1].values())[0]
print("字段:", ", ".join(sorted(k0.keys())))
print()


def tok(x):
    return (x.get("usage") or {}).get("completion_tokens", 0)


print("=" * 96)
print("【1】总体：截断率与得分")
print("=" * 96)
print("%-22s %6s %6s %8s %8s %8s" % ("模型", "截断", "占比", "截断题均分", "非截断均分", "整体均分"))
for name, R in loaded:
    tr = [x for x in R.values() if tok(x) >= CAP]
    nt = [x for x in R.values() if tok(x) < CAP]
    m = lambda L: (sum(min(1.0, x["score"]) for x in L) / len(L)) if L else 0.0
    print("%-22s %6d %5.1f%% %8.3f %10.3f %8.3f" %
          (name, len(tr), 100.0 * len(tr) / len(R), m(tr), m(nt), m(list(R.values()))))

print()
print("=" * 96)
print("【2】输出长度分布（token）")
print("=" * 96)
BINS = [(0, 200), (200, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, 99999)]
print("%-22s %s" % ("模型", "".join("%9s" % ("%d-%d" % b if b[1] < 9999 else "=8192") for b in BINS)))
for name, R in loaded:
    row = []
    for lo, hi in BINS:
        c = sum(1 for x in R.values() if lo <= tok(x) < hi)
        row.append("%9d" % c)
    print("%-22s %s" % (name, "".join(row)))

print()
print("=" * 96)
print("【3】思考链痕迹（head/answer 里出现 </think> 的比例）")
print("=" * 96)
for name, R in loaded:
    c = 0
    for x in R.values():
        t = (x.get("head") or "") + (x.get("answer") or "")
        if "</think>" in t or "<think>" in t:
            c += 1
    print("%-22s %d/%d = %.0f%%" % (name, c, len(R), 100.0 * c / len(R)))

print()
print("=" * 96)
print("【4】按类别得分率")
print("=" * 96)
CATS = ["agent", "coding", "math", "dialogue", "ruozhiba", "longctx", "other"]
print("%-22s %s" % ("模型", "".join("%9s" % c for c in CATS)))
for name, R in loaded:
    row = []
    for c in CATS:
        v = [min(1.0, x["score"]) for x in R.values() if x["cat"] == c]
        row.append("%8.1f%%" % (sum(v) / len(v) * 100) if v else "%9s" % "-")
    print("%-22s %s" % (name, "".join(row)))

print()
print("=" * 96)
print("【5】2.9B-Q4 vs 1.5B：逐题差异（重点看 1.5B=1.0 且 2.9B=0.0）")
print("=" * 96)
if len(loaded) >= 3:
    n15, R15 = loaded[0]
    n29, R29 = loaded[2]
    win, lose, tie = [], [], 0
    for q in sorted(R15):
        if q not in R29:
            continue
        a, b = min(1.0, R15[q]["score"]), min(1.0, R29[q]["score"])
        if abs(a - b) < 1e-9:
            tie += 1
        elif b > a:
            win.append((q, R29[q]["cat"], a, b))
        else:
            lose.append((q, R29[q]["cat"], a, b))
    print("2.9B 更好 %d 题 | 1.5B 更好 %d 题 | 相同 %d 题" % (len(win), len(lose), tie))
    print()
    print("1.5B 拿满分而 2.9B 拿 0 的题（前 15）:")
    bad = [(q, c, a, b) for q, c, a, b in lose if a >= 1.0 and b == 0.0]
    for q, c, a, b in bad[:15]:
        x = R29[q]
        print("  %-5s %-9s tok=%-5d | %s" % (q, c, tok(x), (x.get("head") or "")[:70].replace("\n", " ")))
    print("  ... 共 %d 题" % len(bad))
    print()
    print("按类别统计「2.9B 输」的题数:")
    cnt = collections.Counter(c for _, c, _, _ in lose)
    tot = collections.Counter(x["cat"] for x in R29.values())
    for c in CATS:
        print("  %-9s %2d/%2d" % (c, cnt.get(c, 0), tot.get(c, 0)))
