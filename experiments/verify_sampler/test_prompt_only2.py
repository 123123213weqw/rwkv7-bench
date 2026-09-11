#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""判定性测试（greedy 版）：prompt token 是否被计入罚项。

做法：top_k=1 → 采样就是 argmax，完全确定性、与 seed 无关。
prompt 里让某 token 大量重复（' 1' / '1'），其第一步 argmax 必然是该 token。
 - 若 prompt token 被计罚：把 presence_penalty 拉到 1000，第一步 argmax 会被强行改掉。
 - 若只在生成后计罚（官方语义）：第一步没有任何 token 在计数器里，penalty 再大也不影响第一步。
"""
import json
import sys
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:18860"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "rwkv29-g1j-q4"
OP = urllib.request.build_opener(urllib.request.ProxyHandler({}))

# token 多次重复，模型第一步几乎必定续写同样的 token
PROMPT = "1 " * 40


def run(**over):
    body = {"model": MODEL, "prompt": PROMPT, "max_tokens": 4, "stream": False,
            "temperature": 1.0, "top_p": 1.0, "top_k": 1, "min_p": 0.0,
            "repeat_last_n": 0, "repeat_penalty": 1.0,
            "presence_penalty": 0.0, "frequency_penalty": 0.0,
            "rwkv_count_penalty": 0.0, "rwkv_presence_penalty": 0.0, "rwkv_penalty_decay": 1.0}
    body.update(over)
    req = urllib.request.Request(BASE + "/v1/completions",
                                 data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with OP.open(req, timeout=600) as r:
        d = json.loads(r.read().decode("utf-8"))
    return d["choices"][0]["text"], d["usage"]


print("prompt = %r  (token '1'/' 1' 出现 40 次,  greedy top_k=1)\n" % PROMPT[:40])

A, ua = run()
B, ub = run(rwkv_presence_penalty=1000.0, rwkv_penalty_decay=0.99)
C, uc = run(rwkv_presence_penalty=1.0, rwkv_count_penalty=0.1, rwkv_penalty_decay=0.99)

print("  A 无罚项          : %r  %s" % (A, ua))
print("  B presence=1000   : %r  %s" % (B, ub))
print("  C 官方 demo 参数   : %r  %s" % (C, uc))

same = (A[:1] == B[:1]) and A[:1] != ""
print("\n判定:")
print("  B 的第 1 个生成字符与 A 相同 -> %s" % same)
print("  结论:", "PASS：罚项不作用于 prompt token（计数器在第一次采样后才开始记）"
      if same else "FAIL：prompt token 被计罚")
