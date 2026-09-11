#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""截断成因诊断：模型是不是在「自问自答续写多轮对话」，以及 <think></think 格式的影响。

对抽样题目做完整生成（max_tokens 大），统计：
  - 生成 token 数 / finish_reason
  - 文本里 "Assistant:" / "User:" / "</think>" 出现次数（>1 说明在续写新轮次）
  - 尾部 120 字
对比两种 generation prompt：
  bare      = harness 现在的做法（模板只加 'Assistant:'）
  nothink   = chat_template_kwargs {"enable_thinking": false} → 模板加 'Assistant: <think>\\n</think>'
              （与官方 demo 的 'Assistant: <think></think' 一致）
"""
import json, io, sys, os, re, urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:18860"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "rwkv29-g1j-q4"
HERE = os.path.dirname(os.path.abspath(__file__))
BANK = json.load(io.open(os.path.join(HERE, "..", "bench", "iq_bank_v3.json"), encoding="utf-8"))["questions"]
OP = urllib.request.build_opener(urllib.request.ProxyHandler({}))

SAMPLING = {"temperature": 1.0, "top_p": 0.5, "top_k": 500, "min_p": 0.0,
            "repeat_last_n": 0, "repeat_penalty": 1.0,
            "presence_penalty": 0.0, "frequency_penalty": 0.0,
            "rwkv_count_penalty": 0.1, "rwkv_presence_penalty": 1.0, "rwkv_penalty_decay": 0.99}

# 每个类别抽 1-2 题
WANT = ["agent", "coding", "math", "dialogue", "ruozhiba", "longctx", "other"]
PICK = []
seen = {}
for q in BANK:
    c = q["cat"]
    if seen.get(c, 0) < 1:
        PICK.append(q)
        seen[c] = seen.get(c, 0) + 1
PICK = PICK[:8]


def gen(q, extra=None, n=8192):
    content = ((q.get("payload", "") + "\n\n--------\n\n") if q.get("payload") else "") + q["prompt"]
    body = {"model": MODEL, "messages": [{"role": "user", "content": content}],
            "max_tokens": n, "stream": False}
    body.update(SAMPLING)
    if extra:
        body.update(extra)
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with OP.open(req, timeout=1800) as r:
        d = json.loads(r.read().decode("utf-8"))
    ch = d["choices"][0]
    msg = ch["message"]
    text = (msg.get("content") or "") + (msg.get("reasoning_content") or "")
    return text, ch.get("finish_reason"), d.get("usage", {})


def probe(text):
    return dict(asst=text.count("Assistant:"), user=text.count("User:"),
                think_close=text.count("</think>"), think_open=text.count("<think>"),
                nn=text.count("\n\n"))


print("模型续写多轮对话 / 格式敏感性诊断")
print("=" * 86)
for q in PICK:
    ref = gen(q)
    dis = gen(q, {"chat_template_kwargs": {"enable_thinking": False}})
    print("\n%s %-9s" % (q["id"], q["cat"]))
    for tag, (t, f, u) in (("bare  ", ref), ("nothink", dis)):
        p = probe(t)
        print("  %s finish=%-6s tok=%-5s | Assistant:x%d User:x%d </think>x%d 空行x%d" %
              (tag, f, u.get("completion_tokens"), p["asst"], p["user"], p["think_close"], p["nn"]))
        print("        尾部: %r" % t[-110:])
