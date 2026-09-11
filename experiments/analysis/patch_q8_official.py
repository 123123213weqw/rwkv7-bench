#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 Q8 官方口径跑测里 5 道 HARNESS ERROR 题（Q02-Q06，因第二个 server OOM 导致连接被拒）
用单独重跑的结果替换，输出修补后的结果文件。"""
import io, json, os

D = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(D, "results_v3_rwkv7_g1j_2p9b_q8_official.json")
FIX = os.path.join(D, "results_q8_official_fix.json")
OUT = os.path.join(D, "results_v3_rwkv7_g1j_2p9b_q8_official_fixed.json")

d = json.load(io.open(SRC, encoding="utf-8"))
rows = d["results"]
fix = {x["id"]: x for x in json.load(io.open(FIX, encoding="utf-8"))["results"]}

n = 0
for i, x in enumerate(rows):
    if "HARNESS ERROR" in (x.get("note") or "") and x["id"] in fix:
        old = (x["score"], (x.get("note") or "")[:30])
        rows[i] = fix[x["id"]]
        n += 1
        print("  %s: score %.2f (%s) -> %.2f (%s)" % (
            x["id"], old[0], old[1], rows[i]["score"], (rows[i].get("note") or "")[:40]))
print("替换 %d 题" % n)
d["note_fix"] = "Q02-Q06 原为 HARNESS ERROR（第二个 server OOM 连接被拒），已用单独重跑结果替换"
json.dump(d, io.open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("写出:", os.path.basename(OUT))
