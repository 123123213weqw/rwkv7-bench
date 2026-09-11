#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断：llama.cpp 到底停在哪个 token 上？

背景：官方 RWKV demo 只在 `token == 0` 时停止；本 GGUF 还声明了 eot_token_id=261('\n\n')，
llama.cpp 把两者都放进 EOG，于是模型合法输出「空行」时会**提前停止**。

本脚本用 /v1/chat/completions + logprobs 拿到每个生成 token 的 id，
报告：finish_reason、终止 token id、被截断处的文本尾部。
"""
import json
import sys
import io
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:18860"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "rwkv29-g1j-q4"
BANK = sys.argv[3] if len(sys.argv) > 3 else "$BENCH_ROOT/bench/iq_bank_v3.json"
IDS = sys.argv[4].split(",") if len(sys.argv) > 4 else ["Q13", "Q14", "Q15", "Q16", "Q18", "Q61"]

OP = urllib.request.build_opener(urllib.request.ProxyHandler({}))
bank = json.load(io.open(BANK, encoding="utf-8"))
QS = {q["id"]: q for q in bank["questions"]}

SAMPLING = {"temperature": 1.0, "top_p": 0.5, "top_k": 500, "min_p": 0.0,
            "repeat_last_n": 0, "repeat_penalty": 1.0,
            "presence_penalty": 0.0, "frequency_penalty": 0.0,
            "rwkv_count_penalty": 0.1, "rwkv_presence_penalty": 1.0, "rwkv_penalty_decay": 0.99}


def run(q, max_tokens=8192):
    content = ((q.get("payload", "") + "\n\n--------\n\n") if q.get("payload") else "") + q["prompt"]
    body = {"model": MODEL, "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens, "stream": False, "logprobs": True, "top_logprobs": 1}
    body.update(SAMPLING)
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with OP.open(req, timeout=1200) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    print("每题报告：finish_reason / 生成 token 数 / 终止 token id / 终止 token 文本")
    print("（官方 demo 只认 id=0 停止；id=261 是 llama.cpp 额外加的 '\\n\\n'）\n")
    tally = {}
    for qid in IDS:
        q = QS.get(qid)
        if not q:
            print("  %s 不存在" % qid)
            continue
        d = run(q)
        ch = d["choices"][0]
        lp = (ch.get("logprobs") or {}).get("content") or []
        text = ch.get("text") or ch.get("message", {}).get("content") or ""
        last_id = lp[-1]["id"] if lp else None
        last_tok = lp[-1]["token"] if lp else None
        finish = ch.get("finish_reason")
        usage = d.get("usage", {})
        tally[(finish, last_id)] = tally.get((finish, last_id), 0) + 1
        print("  %-5s %-9s finish=%-6s tokens=%-5s 末 token id=%s %r" %
              (qid, q["cat"], finish, usage.get("completion_tokens"), last_id, last_tok))
        print("      尾部 80 字: %r" % text[-80:])

    print("\n=== 汇总 (finish_reason, 终止 token id) ===")
    for (f, i), n in sorted(tally.items(), key=lambda kv: -kv[1]):
        print("  %-8s id=%-6s  %d 题" % (f, i, n))
    print("\n判读：")
    print("  finish=stop 且 id=0   -> 与官方 demo 一致（正常 EOT）")
    print("  finish=stop 且 id=261 -> llama.cpp 额外把 '\\n\\n' 当 EOT，**官方不会停** -> 提前截断")
    print("  finish=length         -> 撞 max_tokens")


if __name__ == "__main__":
    main()
