#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把本轮 RWKV7-G1j-2.9B-Q4 的 4 组解码口径结果并入用户的新版报告 HTML。

做法：
  1. 读入用户附件（36 模型的 DATA 数组）
  2. 用结果 JSON 现场算出每组的 total / cats / easy / med / hard（与 build_report 同口径）
  3. 追加为新条目（保留原有 41.2 冻结行不动）
  4. 更新偏差声明与脚注
  5. 写到 workspace 下的新文件
"""
import io, json, os, re, collections, sys

ATT = (r"\\?\C:\Users\31046\AppData\Roaming\com.xlang.xharness\state\attachments"
       r"\5f61056db1d442bac038161aa62be839d8fdac3c7007e6f042febf55ee15375b"
       r"\3d606488f06c3a4a343e79ecb8136e8bff4ef9588c4ed0dd84e7f5a3bedf11b0\image")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "iq_v3_report_v4_rwkv.html")

BANK = json.load(io.open(os.path.join(HERE, "report", "iq_bank_v3.json"), encoding="utf-8"))
DIFF = {q["id"]: q["difficulty"] for q in BANK["questions"]}
W = {"agent": .2, "coding": .2, "math": .2, "dialogue": .1,
     "ruozhiba": .1, "longctx": .1, "other": .1}

# 本轮 4 组：文件 -> (显示名, 颜色, 耗时分钟, 图例后缀)
RUNS = [
    ("results_v3_rwkv7_g1j_2p9b_q4_rwkvdemo.json",
     "RWKV7-G1j-2.9B Q4 · demo采样/8192/思考", "#f0a000", 214),
    ("results_v3_rwkv7_g1j_2p9b_q4_official.json",
     "RWKV7-G1j-2.9B Q4 · 官方demo/500/关思考", "#20c997", 14),
    ("results_v3_rwkv7_g1j_2p9b_q4_nothink8192.json",
     "RWKV7-G1j-2.9B Q4 · demo采样/8192/关思考", "#b06cf0", 117),
    ("results_v3_rwkv7_g1j_2p9b_q4_demo500.json",
     "RWKV7-G1j-2.9B Q4 · demo采样/500/思考", "#e05c8a", 19),
    # Q8：换无损量化后的**同口径**对照（官方 demo 解码 + 关思考 + 500 tok），与 Q4 官方行可直接比
    ("results_v3_rwkv7_g1j_2p9b_q8_official_fixed.json",
     "RWKV7-G1j-2.9B Q8 · 官方demo/500/关思考", "#00b3b3", 15),
]


def stats(path):
    rows = json.load(io.open(path, encoding="utf-8"))["results"]
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
    cats = {c: round(sum(per[c]) / len(per[c]) * 100, 1) for c in per}
    tier = {k: round(v[0] / v[1] * 100, 1) for k, v in te.items()}
    return round(tot, 1), cats, tier, len(rows)


def main():
    html = io.open(ATT, encoding="utf-8").read()

    m = re.search(r"const DATA = (\[.*?\]);\n", html, re.S)
    if not m:
        sys.exit("找不到 DATA 数组")
    data = json.loads(m.group(1))
    print("原有模型数:", len(data))

    new = []
    for fn, name, color, mins in RUNS:
        p = os.path.join(HERE, fn)
        if not os.path.exists(p):
            print("  跳过(缺文件):", fn)
            continue
        tot, cats, tier, nq = stats(p)
        if mins is None:  # 从日志累计耗时推导
            lg = os.path.join(HERE, "eval_%s.log" % fn.split("results_v3_")[-1].replace(".json", ""))
            if not os.path.exists(lg):
                lg = os.path.join(HERE, "..", "logs", os.path.basename(lg))
            secs = 0.0
            if os.path.exists(lg):
                for ln in io.open(lg, encoding="utf-8", errors="replace"):
                    mm = re.search(r"([\d.]+)s\s*$", ln.rstrip())
                    if mm and ln.startswith("["):
                        secs += float(mm.group(1))
            mins = int(round(secs / 60.0)) if secs else 0
            print("    (耗时由日志推导: %.0f s -> %d min)" % (secs, mins))
        new.append({"name": name, "color": color, "total": tot, "cats": cats,
                    "ref": False, "totalC": tot, "easy": tier["e"], "med": tier["m"],
                    "hard": tier["h"], "mins": mins, "nq": nq})
        print("  + %-40s total=%5.1f  nq=%d" % (name, tot, nq))

    data.extend(new)

    # 写回 DATA（保持单行紧凑格式）
    blob = json.dumps(data, ensure_ascii=False, separators=(", ", ": "))
    html = html[:m.start(1)] + blob + html[m.end(1):]

    # 更新偏差声明 ①
    old_note = ("① RWKV-World 官方推荐 temp 0.8 + repeat_penalty 1.15，与本协议不同——统一低温下 RWKV 存在逐字重复循环风险，"
                "<b>其成绩应视为下界</b>")
    new_note = ("① RWKV-World 官方 demo 的解码方式与本协议不同（temp 1 · top_p 0.5 · presence 1 · count 0.1 · decay 0.99，"
                "并关闭思维链）；本协议统一低温下 RWKV 存在重复循环风险，故表中 RWKV 行<b>应视为下界</b>。"
                "经补充实测 RWKV7-G1j-2.9B Q4 在官方解码下为 <b>44.2 分（500/关思考）</b>、48.3 分（8192/开思考），"
                "均高于本协议下的 41.2，<b>下界判断成立</b>；解码口径差异见表中 · 官方demo/demo采样 各行<br>"
                "② <b>RWKV7-G1j-2.9B 的 Q4_K_M 档属量化受损，不适用本报告「≥5bpw 质量高原区」的档位假设</b>："
                "同模型换 <b>Q8_0</b>、同官方解码口径复测，总分 <b>44.2 → 64.1（+19.9）</b>，撞 500 token 上限的题数 "
                "<b>90/200 → 39/200</b>，coding 41.2% → 80.0%、math 30.0% → 57.5%。"
                "Q4 档下 82% 的题无法输出终止符而跑满预算，属<b>量化破坏了终止行为</b>，非模型能力差异；"
                "该模型应以表中 <b>· Q8 · 官方demo</b> 一行为准")
    if old_note in html:
        html = html.replace(old_note, new_note)
        print("偏差声明 ① 已更新")
    else:
        print("!! 偏差声明未匹配")

    # 脚注
    old_fn = "RWKV-g1j 与 Qwen3.5/3.6-35B 原版测完后重新生成本报告。"
    new_fn = ("RWKV-g1j 与 Qwen3.5/3.6-35B 原版测完后重新生成本报告。"
              "RWKV7-G1j-2.9B Q4 另有 4 行解码口径对照（· demo采样 / · 官方demo），"
              "为同一题目、各 200 题的独立跑测，原始成绩 41.2 行保持不变。")
    if old_fn in html:
        html = html.replace(old_fn, new_fn)
        print("脚注已更新")

    io.open(OUT, "w", encoding="utf-8").write(html)
    print("\n已写出: %s  (%d bytes, %d 模型)" % (OUT, len(html.encode("utf-8")), len(data)))


if __name__ == "__main__":
    main()
