#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v3 中期聚合：全部已完成模型 raw 总榜 + 分类别榜 + 分层"""
import json, os, io

HERE = os.path.dirname(os.path.abspath(__file__))
W = {"agent": .2, "coding": .2, "math": .2, "dialogue": .1, "ruozhiba": .1, "longctx": .1, "other": .1}
BANK = json.load(io.open(os.path.join(HERE, "iq_bank_v3.json"), encoding="utf-8"))
DIFF = {q["id"]: q["difficulty"] for q in BANK["questions"]}
print("tiers 定义:", BANK["meta"].get("tiers"))

MODELS = [
    ("Qwen3.8-27B-Unc Q5",        "results_v3_qwen38_27b_q5.json"),
    ("Ornith-1.5-35B-Heretic IQ", "results_v3_ornith35b_heretic.json"),
    ("Qwen3.5-9B-MTP Q4",         "results_v3_qwen35_9b_mtp_q4.json"),
    ("Qwen3.5-4B Q4",             "results_v3_qwen35_4b_q4.json"),
    ("Ornith-1.5-9B Q4",          "results_v3_ornith9b_q4.json"),
    ("Spark-X2.5-4B Q8",          "results_v3_spark_x25_4b_q8.json"),
    ("gemma-4-E4B Q4",            "results_v3_gemma4_e4b_q4.json"),
    ("Ling-3.0-tiny Q4",          "results_v3_ling3_q4_kvq8.json"),
    ("LFM2.5-2.6B Q8",            "results_v3_lfm25_2p6b_q8.json"),
    ("gemma-4-E2B Q8",            "results_v3_gemma4_e2b_q8.json"),
    ("Spark-X2.5-1.7B Q8",        "results_v3_spark_x25_1p7b_q8.json"),
    ("MiniCPM5-2B Q8",            "results_v3_mincpm5_2b_q8.json"),
    ("Qwen3.5-2B Q8",             "results_v3_qwen35_2b_q8.json"),
    ("MiniCPM5-1B Q8",            "results_v3_mincpm5_1b_q8.json"),
    ("LFM2.5-8B-A1B Q8",          "results_v3_lfm25_8b_a1b_q8.json"),
    ("Qwen3.5-0.8B Q8",           "results_v3_qwen35_08b_q8.json"),
    ("RWKV7-g1i-1.5B Q8*",        "results_v3_rwkv7_g1i_1p5b_q8.json"),
    ("1B-Fable5 Q8*",             "results_v3_mincpm1b_fable5_q8.json"),
    ("1B-Agentic-DPO Q8*",        "results_v3_mincpm1b_agentic_q8.json"),
    ("1B-anti-hallu Q8*†",        "results_v3_mincpm1b_antihallu_q8_partial.json"),
]

rows = []
for name, f in MODELS:
    p = os.path.join(HERE, f)
    if not os.path.exists(p):
        print("缺文件:", f); continue
    r = json.load(io.open(p, encoding="utf-8"))["results"]
    percat, te = {}, {"e": [0,0], "m": [0,0], "h": [0,0]}
    for x in r:
        s = min(1.0, x["score"])
        percat.setdefault(x["cat"], []).append(s)
        d = DIFF[x["id"]]
        k = "e" if d <= 5 else ("m" if d <= 8 else "h")
        te[k][0] += s; te[k][1] += 1
    tot = sum(W[c] * sum(v)/len(v) for c, v in percat.items()) * 100
    nq = len(r)
    mins = sum(x.get("secs", 0) for x in r) / 60
    rows.append((name, tot, percat, nq, mins, te))

rows.sort(key=lambda t: -t[1])
print(f"\n===== v3 中期总榜（raw，*待更新/部分，†仅40题确证） =====")
print(f"{'#':<3}{'模型':<26}{'总分':>6}  {'题数':>4}{'耗时':>6}  easy  med   hard")
for i, (n, t, pc, nq, mi, te) in enumerate(rows, 1):
    e = te['e'][0]/max(te['e'][1],1)*100; m = te['m'][0]/max(te['m'][1],1)*100; h = te['h'][0]/max(te['h'][1],1)*100
    print(f"{i:<3}{n:<26}{t:>6.1f}  {nq:>4}{mi:>5.0f}m  {e:5.1f} {m:5.1f} {h:5.1f}")

print("\n===== 分类别冠军 =====")
for c in W:
    best = sorted(((n, pc.get(c, [0])) for n, _, pc, _, _, _ in rows), key=lambda t: -sum(t[1])/max(len(t[1]),1))
    line = "  ".join(f"{n} {sum(v)/max(len(v),1)*100:.0f}%" for n, v in best[:3])
    print(f"  {c:<9} {line}")
