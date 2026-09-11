#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""四种条件汇总对比：总榜 + 分类别 + 截断率 + 耗时。"""
import json, io, os, collections

HERE = os.path.dirname(os.path.abspath(__file__))
W = {"agent": .2, "coding": .2, "math": .2, "dialogue": .1, "ruozhiba": .1, "longctx": .1, "other": .1}

RUNS = [
    ("旧基线 8192/思考/旧协议", "report/results_v3_rwkv7_g1j_2p9b_q4.json"),
    ("demo采样 8192/思考",      "results_v3_rwkv7_g1j_2p9b_q4_rwkvdemo.json"),
    ("demo采样 500/思考",       "results_v3_rwkv7_g1j_2p9b_q4_demo500.json"),
    ("demo采样 500/不思考(官方)", "results_v3_rwkv7_g1j_2p9b_q4_official.json"),
]


def load(rel):
    p = os.path.join(HERE, rel)
    if not os.path.exists(p):
        return None, None
    d = json.load(io.open(p, encoding="utf-8"))
    return d, {x["id"]: x for x in d["results"]}


def score(rows):
    per = collections.defaultdict(list)
    for x in rows.values():
        per[x["cat"]].append(min(1.0, x["score"]))
    tot = sum(W[c] * sum(v) / len(v) for c, v in per.items()) * 100
    return tot, {c: sum(v) / len(v) * 100 for c, v in per.items()}


print("=" * 96)
print("%-28s %6s %7s %7s %7s %7s %7s %7s %7s  %8s %7s" %
      ("条件", "总分", "agent", "coding", "math", "dialog", "ruozhi", "longctx", "other", "截断率", "耗时"))
print("=" * 96)
for name, rel in RUNS:
    d, rows = load(rel)
    if not rows:
        print("%-28s (缺失)" % name)
        continue
    tot, cats = score(rows)
    cap = None
    sr = d.get("sampling") or {}
    mt = 8192
    if "500" in name:
        mt = 500
    tr = sum(1 for x in rows.values() if (x.get("usage") or {}).get("completion_tokens", 0) >= mt)
    mins = sum(x.get("secs", 0) for x in rows.values()) / 60
    print("%-28s %6.1f %6.1f%% %6.1f%% %6.1f%% %6.1f%% %6.1f%% %6.1f%% %6.1f%%  %4d/%d %5.0fm" %
          (name, tot, cats.get("agent", 0), cats.get("coding", 0), cats.get("math", 0),
           cats.get("dialogue", 0), cats.get("ruozhiba", 0), cats.get("longctx", 0), cats.get("other", 0),
           tr, len(rows), mins))

print()
base = load("report/results_v3_rwkv7_g1j_2p9b_q4.json")[1]
off = load("results_v3_rwkv7_g1j_2p9b_q4_official.json")[1]
if base and off:
    print("逐题（官方条件 vs 旧基线，200 题可比）:")
    w = [q for q in base if min(1.0, off[q]["score"]) > min(1.0, base[q]["score"]) + 1e-9]
    l = [q for q in base if min(1.0, base[q]["score"]) > min(1.0, off[q]["score"]) + 1e-9]
    print("  官方条件更好 %d 题 | 旧基线更好 %d 题 | 相同 %d 题" % (len(w), len(l), 200 - len(w) - len(l)))
    print("  官方条件提升最大的 8 题:")
    for q in sorted(w, key=lambda q: -(min(1.0, off[q]["score"]) - min(1.0, base[q]["score"])))[:8]:
        print("    %-5s %-9s 旧=%.2f 官方=%.2f" % (q, base[q]["cat"], base[q]["score"], off[q]["score"]))
    print("  旧基线领先最大的 8 题:")
    for q in sorted(l, key=lambda q: -(min(1.0, base[q]["score"]) - min(1.0, off[q]["score"])))[:8]:
        print("    %-5s %-9s 旧=%.2f 官方=%.2f" % (q, base[q]["cat"], base[q]["score"], off[q]["score"]))
    # 不截断题上的对比（排除预算影响）
    def clean(rows, mt):
        return {q: x for q, x in rows.items() if (x.get("usage") or {}).get("completion_tokens", 0) < mt}
    cb, co = clean(base, 8192), clean(off, 500)
    both = set(cb) & set(co)
    if both:
        def mr(rows, ids):
            per = collections.defaultdict(list)
            for q in ids:
                per[rows[q]["cat"]].append(min(1.0, rows[q]["score"]))
            return sum(W[c] * sum(v) / len(v) for c, v in per.items()) * 100
        print()
        print("  仅看两边都没被截断的 %d 题（排除预算差异）:" % len(both))
        print("    旧基线 %.1f  |  官方条件 %.1f" % (mr(cb, both), mr(co, both)))
