#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成完整表格：总榜（全部模型）+ RWKV G1j-2.9B-Q4 各解码口径对照。

输出 table_full.md / table_full.csv，并打印到终端。
"""
import json, io, os, glob, collections, csv

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "report")
W = {"agent": .2, "coding": .2, "math": .2, "dialogue": .1, "ruozhiba": .1, "longctx": .1, "other": .1}
CATS = ["agent", "coding", "math", "dialogue", "ruozhiba", "longctx", "other"]

BANK = json.load(io.open(os.path.join(R, "iq_bank_v3.json"), encoding="utf-8"))
DIFF = {q["id"]: q["difficulty"] for q in BANK["questions"]}

# 文件名 -> 显示名
NAME = {
    "qwen38_27b_q5": "Qwen3.8-27B-Unc Q5",
    "ornith35b_heretic": "Ornith-1.5-35B-Heretic IQ",
    "qwen35_35b_a3b_q5": "Qwen3.5-35B-A3B Q5",
    "qwen35_9b_mtp_q4": "Qwen3.5-9B-MTP Q4",
    "qwen35_4b_q4": "Qwen3.5-4B Q4",
    "ornith9b_q4": "Ornith-1.5-9B Q4",
    "spark_x25_4b_q8": "Spark-X2.5-4B Q8",
    "gemma4_e4b_q4": "gemma-4-E4B Q4",
    "ling3_q4_kvq8": "Ling-3.0-tiny Q4",
    "lfm25_2p6b_q8": "LFM2.5-2.6B Q8",
    "gemma4_e2b_q8": "gemma-4-E2B Q8",
    "spark_x25_1p7b_q8": "Spark-X2.5-1.7B Q8",
    "mincpm5_2b_q8": "MiniCPM5-2B Q8",
    "qwen35_2b_q8": "Qwen3.5-2B Q8",
    "mincpm5_1b_q8": "MiniCPM5-1B Q8",
    "lfm25_8b_a1b_q8": "LFM2.5-8B-A1B Q8",
    "qwen35_08b_q8": "Qwen3.5-0.8B Q8",
    "mincpm1b_fable5_q8": "1B-Fable5 Q8",
    "rwkv7_g1j_1p5b_q8": "RWKV7-G1j-1.5B Q8",
    "rwkv7_g1i_1p5b_q8": "RWKV7-g1i-1.5B Q8",
    "rwkv7_g1j_2p9b_q4_rwkvdemo": "RWKV7-G1j-2.9B Q4 · demo采样/8192/思考",
    "rwkv7_g1j_2p9b_q4_official": "RWKV7-G1j-2.9B Q4 · 官方口径/500/关思考",
    "rwkv7_g1j_2p9b_q4_nothink8192": "RWKV7-G1j-2.9B Q4 · 关思考/8192",
    "rwkv7_g1j_2p9b_q4": "RWKV7-G1j-2.9B Q4 · 旧协议/8192/思考",
    "rwkv7_g1j_2p9b_q4_demo500": "RWKV7-G1j-2.9B Q4 · demo采样/500/思考",
    "mincpm1b_agentic_q8": "1B-Agentic-DPO Q8",
    "mincpm1b_antihallu_q8_partial": "1B-anti-hallu Q8",
    "glm_53": "GLM-5.3 API参照",
    "glm_53_flash": "GLM-5.3-Flash API参照",
}
# 备注标记
FLAG = {
    "qwen35_35b_a3b_q5": "部分",
    "glm_53": "部分",
    "glm_53_flash": "部分",
    "mincpm1b_antihallu_q8_partial": "部分",
    "rwkv7_g1j_2p9b_q4_rwkvdemo": "★新",
    "rwkv7_g1j_2p9b_q4_official": "★新",
}

# 文件名 -> 解码口径描述（仅本次实测）
PROTO = {
    "rwkv7_g1j_2p9b_q4":              ("默认协议", "8192", "开", "旧二进制"),
    "rwkv7_g1j_2p9b_q4_rwkvdemo":     ("官方demo采样", "8192", "开", "本补丁"),
    "rwkv7_g1j_2p9b_q4_official":     ("官方demo采样", "500", "关", "本补丁"),
    "rwkv7_g1j_2p9b_q4_nothink8192":  ("官方demo采样", "8192", "关", "本补丁"),
    "rwkv7_g1j_2p9b_q4_demo500":      ("官方demo采样", "500", "开", "本补丁"),
    "rwkv7_g1j_2p9b_q4_legacy40":     ("默认协议", "8192", "开", "本补丁"),
    "rwkv7_g1j_2p9b_q4_legacy_cm20":  ("默认协议", "8192", "开", "本补丁"),
}


BUDGET = {  # 文件名 -> 该次运行的 max_tokens（截断判定必须用各自预算）
    "rwkv7_g1j_2p9b_q4_official": 500,
    "rwkv7_g1j_2p9b_q4_demo500": 500,
}


def score(rows, key=""):
    per = collections.defaultdict(list)
    te = {"e": [0.0, 0], "m": [0.0, 0], "h": [0.0, 0]}
    for x in rows:
        s = min(1.0, x["score"])
        per[x["cat"]].append(s)
        d = DIFF.get(x["id"], 5)
        k = "e" if d <= 5 else ("m" if d <= 8 else "h")
        te[k][0] += s
        te[k][1] += 1
    tot = sum(W[c] * sum(v) / len(v) for c, v in per.items()) * 100
    cats = {c: (sum(per[c]) / len(per[c]) * 100 if per.get(c) else None) for c in CATS}
    tier = {k: (v[0] / v[1] * 100 if v[1] else None) for k, v in te.items()}
    cap = BUDGET.get(key, 8192)
    n_trunc = sum(1 for x in rows if (x.get("usage") or {}).get("completion_tokens", 0) >= cap)
    mins = sum(x.get("secs", 0) for x in rows) / 60
    return tot, cats, tier, n_trunc, mins


def main():
    recs = []
    for p in sorted(glob.glob(os.path.join(R, "results_v3_*.json"))):
        key = os.path.basename(p)[len("results_v3_"):-len(".json")]
        d = json.load(io.open(p, encoding="utf-8"))
        rows = d["results"]
        tot, cats, tier, ntr, mins = score(rows, key)
        recs.append(dict(key=key, name=NAME.get(key, key), n=len(rows), tot=tot, cats=cats,
                         tier=tier, trunc=ntr, mins=mins, flag=FLAG.get(key, ""),
                         proto=PROTO.get(key)))
    recs.sort(key=lambda r: -r["tot"])

    def f(v, w=6, p=1):
        return ("%*.1f" % (w, v)) if v is not None else " " * (w - 1) + "—"

    # ---------------- 表 1：总榜 ----------------
    lines = []
    lines.append("## 表 1  端侧模型能力基准 v3（iq_bank_v3，200 题加权总分）")
    lines.append("")
    lines.append("| # | 模型 | 总分 | agent | coding | math | dialogue | 脑筋急转弯 | 长上下文 | 其他 | easy | med | hard | 题数 | 截断 | 耗时 |")
    lines.append("|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for i, r in enumerate(recs, 1):
        c = r["cats"]
        t = r["tier"]
        nm = r["name"] + ((" †" + r["flag"]) if r["flag"] and r["flag"] != "★新" else "")
        if r["flag"] == "★新":
            nm = "**" + r["name"] + "**"
        lines.append("| %d | %s | **%.1f** | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %d | %d | %.0fm |" % (
            i, nm, r["tot"],
            f(c["agent"]), f(c["coding"]), f(c["math"]), f(c["dialogue"]),
            f(c["ruozhiba"]), f(c["longctx"]), f(c["other"]),
            f(t["e"]), f(t["m"]), f(t["h"]), r["n"], r["trunc"], r["mins"]))
    lines.append("")
    lines.append("说明：截断 = `completion_tokens ≥ 该次运行的 max_tokens`（8192 档按 8192 判，500 档按 500 判）。带 † 的行题数不足 200，总分不可直接比较。")
    lines.append("")

    # ---------------- 表 2：G1j-2.9B-Q4 各口径 ----------------
    lines.append("## 表 2  RWKV7-G1j-2.9B-Q4_K_M 各解码口径对照（同一题库、各 200 题）")
    lines.append("")
    lines.append("| 解码口径 | 二进制 | max_tokens | 思考 | 总分 | agent | coding | math | dialogue | 急转弯 | 长上下文 | 其他 | easy | med | hard | 截断 | 耗时 |")
    lines.append("|---|---|--:|:--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    order = ["rwkv7_g1j_2p9b_q4", "rwkv7_g1j_2p9b_q4_rwkvdemo",
             "rwkv7_g1j_2p9b_q4_official", "rwkv7_g1j_2p9b_q4_nothink8192",
             "rwkv7_g1j_2p9b_q4_demo500"]
    extra = {}
    p500 = os.path.join(HERE, "results_v3_rwkv7_g1j_2p9b_q4_demo500.json")
    if os.path.exists(p500):
        d = json.load(io.open(p500, encoding="utf-8"))
        tot, cats, tier, ntr, mins = score(d["results"], "rwkv7_g1j_2p9b_q4_demo500")
        extra["rwkv7_g1j_2p9b_q4_demo500"] = dict(n=len(d["results"]), tot=tot, cats=cats,
                                                  tier=tier, trunc=ntr, mins=mins)
    byk = {r["key"]: r for r in recs}
    byk.update(extra)
    for k in order:
        r = byk.get(k)
        if not r:
            continue
        proto, mt, think, binr = PROTO[k]
        c, t = r["cats"], r["tier"]
        lines.append("| %s | %s | %s | %s | **%.1f** | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %d/%d | %.0fm |" % (
            proto, binr, mt, think, r["tot"],
            f(c["agent"]), f(c["coding"]), f(c["math"]), f(c["dialogue"]),
            f(c["ruozhiba"]), f(c["longctx"]), f(c["other"]),
            f(t["e"]), f(t["m"]), f(t["h"]), r["trunc"], r["n"], r["mins"]))
    lines.append("")
    lines.append("官方 = BlinkDL/RWKV-Gradio 默认：temp 1 / top_p 0.5 / top_k 500 / presence 1 / count 0.1 / decay 0.99，"
                 "生成前缀 `Assistant: <think></think`（关思考），Max Tokens 滑块默认 500。")
    lines.append("")

    # ---- 表 3：受控子集（同二进制、同题，只切采样） ----
    lines.append("## 表 3  受控对比：同二进制、同一批题，只切换采样协议")
    lines.append("")
    lines.append("| 子集 | 题数 | 条件（同题、同二进制可比） | agent | coding | math | 撞各自上限 |")
    lines.append("|---|--:|---|--:|--:|--:|--:|")

    def sub_rows(key, ids, label):
        out = []
        for cond, path, cap in label:
            p = os.path.join(HERE, os.path.basename(path))
            if not os.path.exists(p):
                continue
            rr = {x["id"]: x for x in json.load(io.open(p, encoding="utf-8"))["results"]}
            sel = [rr[q] for q in ids if q in rr]
            if not sel:
                continue
            per = collections.defaultdict(list)
            for x in sel:
                per[x["cat"]].append(min(1.0, x["score"]))
            tot = sum(W[c] * sum(v) / len(v) for c, v in per.items()) * 100
            ntr = sum(1 for x in sel if (x.get("usage") or {}).get("completion_tokens", 0) >= cap)
            out.append((cond, tot, per, len(sel), ntr))
        return out

    def find(p):
        for base in (R, HERE):
            q = os.path.join(base, os.path.basename(p))
            if os.path.exists(q):
                return q
        return None

    def sub_rows(ids, conds):
        out = []
        for cond, path, cap in conds:
            p = find(path)
            if not p:
                continue
            rr = {x["id"]: x for x in json.load(io.open(p, encoding="utf-8"))["results"]}
            sel = [rr[q] for q in ids if q in rr]
            if not sel:
                continue
            per = collections.defaultdict(list)
            for x in sel:
                per[x["cat"]].append(min(1.0, x["score"]))
            ntr = sum(1 for x in sel if (x.get("usage") or {}).get("completion_tokens", 0) >= cap)
            out.append((cond, per, len(sel), ntr))
        return out

    AGENT40 = ["Q%02d" % i for i in range(1, 11)] + ["Q%02d" % i for i in range(51, 61)] + \
              ["Q%02d" % i for i in range(101, 111)] + ["Q%02d" % i for i in range(151, 161)]
    CM20 = ["Q%02d" % i for i in range(11, 31)]
    for ids, cap_lab in ((AGENT40, "agent 40 题"), (CM20, "coding+math 20 题")):
        conds = [
            ("旧基线（旧二进制+旧协议）", os.path.join(R, "results_v3_rwkv7_g1j_2p9b_q4.json"), 8192),
            ("官方demo采样（本补丁）", os.path.join(R, "results_v3_rwkv7_g1j_2p9b_q4_rwkvdemo.json"), 8192),
        ]
        for suffix in ("legacy40", "legacy_cm20"):
            pp = "results_v3_rwkv7_g1j_2p9b_q4_%s.json" % suffix
            if find(pp):
                conds.append(("对照：旧协议（本补丁二进制）", pp, 8192))
        rows3 = sub_rows(ids, conds)
        for cond, per, n, ntr in rows3:
            def g(c):
                return ("%5.1f%%" % (sum(per[c]) / len(per[c]) * 100)) if per.get(c) else "    —"
            lines.append("| %s | %d | %s | %s | %s | %s | %d/%d |" %
                         (cap_lab, n, cond, g("agent"), g("coding"), g("math"), ntr, n))
        lines.append("")
    lines.append("> 表 3 说明：`旧基线` 用的是上一批的旧二进制；`对照` 用**本补丁二进制**但走旧协议，"
                 "用于把「采样改动」与「二进制/启动参数差异」分开。")
    lines.append("")

    # ---- 表 4：预算扫描（用 8192/关思考 那一次的数据反推不同 cap 下的截断） ----
    p8192 = find("results_v3_rwkv7_g1j_2p9b_q4_nothink8192.json")
    if p8192:
        rows8 = json.load(io.open(p8192, encoding="utf-8"))["results"]
        lines.append("## 表 4  预算扫描：关思考时把 max_tokens 加大能救回多少截断")
        lines.append("")
        lines.append("| max_tokens | 截断题数 | 截断率 | 说明 |")
        lines.append("|--:|--:|--:|---|")
        prev = None
        for cap in (500, 1024, 2048, 4096, 8192):
            ntr = sum(1 for x in rows8 if (x.get("usage") or {}).get("completion_tokens", 0) >= cap)
            note = ""
            if prev is not None and cap > 500:
                note = "比上一档少 %d 题" % (prev - ntr) if prev - ntr else "**无改善**"
            lines.append("| %d | %d/200 | %.1f%% | %s |" % (cap, ntr, 100.0 * ntr / len(rows8), note))
            prev = ntr
        tk = [(x.get("usage") or {}).get("completion_tokens", 0) for x in rows8]
        lo = sum(1 for t in tk if t < 500)
        hi = sum(1 for t in tk if t >= 8192)
        mid = len(tk) - lo - hi
        lines.append("")
        lines.append("token 分布呈**强双峰**：`<500` 的 %d 题（%.0f%%）、`=8192` 的 %d 题（%.0f%%）、"
                     "中间(500~8191)只有 %d 题。" % (lo, 100.0 * lo / len(tk), hi, 100.0 * hi / len(tk), mid))
        lines.append("")
        lines.append("> 结论：截断**不是预算不足**，而是约 37% 的题模型**根本不吐 EOS**（一条道走到黑）。"
                     "把预算从 500 提到 8192，截断只从 45.0% 降到 37.0%，且 8192 之后已饱和；"
                     "代价是耗时 14m → 117m，总分反而 44.2 → 43.4。")
        lines.append("")

    txt = "\n".join(lines)
    io.open(os.path.join(HERE, "table_full.md"), "w", encoding="utf-8").write(txt)

    # CSV
    with io.open(os.path.join(HERE, "table_full.csv"), "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "key", "name", "total", "agent", "coding", "math", "dialogue",
                    "ruozhiba", "longctx", "other", "easy", "med", "hard", "n", "truncated", "minutes"])
        for i, r in enumerate(recs, 1):
            c, t = r["cats"], r["tier"]
            w.writerow([i, r["key"], r["name"], round(r["tot"], 2)] +
                       [(round(c[x], 2) if c[x] is not None else "") for x in CATS] +
                       [(round(t[x], 2) if t[x] is not None else "") for x in ("e", "m", "h")] +
                       [r["n"], r["trunc"], round(r["mins"])])

    print(txt)
    print("written: table_full.md / table_full.csv")


if __name__ == "__main__":
    main()
