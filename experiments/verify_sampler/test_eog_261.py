#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检测 llama.cpp 把 id=261('\\n\\n') 当 EOG 是否会提前截断（官方 demo 只认 id=0）。

对比两台服务：
  A = 默认（EOG={0, 261}）
  B = 限制 EOG={0}（--override-kv tokenizer.ggml.eot_token_id=int:0）
同一 prompt、同 seed、发几次，看是否在第一个空行处被切断。
"""
import json
import sys
import urllib.request

A = "http://127.0.0.1:18860"
B = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:18862"
MODEL = "rwkv29-g1j-q4"
OP = urllib.request.build_opener(urllib.request.ProxyHandler({}))

SAMPLING = {"temperature": 0.2, "top_p": 0.5, "top_k": 500, "min_p": 0.0,
            "repeat_last_n": 0, "repeat_penalty": 1.0,
            "presence_penalty": 0.0, "frequency_penalty": 0.0,
            "rwkv_count_penalty": 0.1, "rwkv_presence_penalty": 1.0, "rwkv_penalty_decay": 0.99}

# 明确要求输出含空行的结构化内容（markdown 列表 + 段落）
PROMPT = ("User: 请用 markdown 写一份包含 3 个小节的清单：每节先一个标题，"
          "然后是一个含 2 个要点的无序列表，小节之间用空行分隔。不要省略。\n\nAssistant: <think></think")


def run(base, seed, n=700):
    body = {"model": MODEL, "prompt": PROMPT, "max_tokens": n, "stream": False, "seed": seed}
    body.update(SAMPLING)
    req = urllib.request.Request(base + "/v1/completions",
                                 data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with OP.open(req, timeout=900) as r:
        d = json.loads(r.read().decode("utf-8"))
    c = d["choices"][0]
    return c["text"], c["finish_reason"], d.get("usage", {})


print("prompt 要求输出带空行的 markdown（每节之间有空行）\n")
for seed in (1, 2, 3):
    ta, fa, ua = run(A, seed)
    tb, fb, ub = run(B, seed)
    print("seed=%d" % seed)
    print("  A(EOG含'\\n\\n') finish=%-6s tokens=%-5s 空行数=%d 字符=%d" %
          (fa, ua.get("completion_tokens"), ta.count("\n\n"), len(ta)))
    print("  B(EOG仅id0)   finish=%-6s tokens=%-5s 空行数=%d 字符=%d" %
          (fb, ub.get("completion_tokens"), tb.count("\n\n"), len(tb)))
    print("  A 尾部: %r" % ta[-70:])
    print("  B 尾部: %r" % tb[-70:])
    print()
