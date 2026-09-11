#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""隔离测试：只改 RWKV 罚项（temp/top_p/top_k 完全一致），HARD-ON 用极大罚项做判定性验证

A: 基础采样 temp1.0 / top_p0.5 / top_k500，无任何罚项
B: 同上 + rwkv_count 0.1 / rwkv_presence 1.0 / decay 0.99   （官方 demo 值）
C: 同上 + rwkv_presence 50.0（夸张值）—— 若补丁真的生效，重复 token 会被彻底压死，
   输出必须与 A 截然不同（这是判定性证据，不受采样噪声影响）
"""
import json
import sys
import urllib.request
from collections import Counter

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:18860"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "rwkv29-g1j-q4"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

BASE_S = {"temperature": 1.0, "top_p": 0.5, "top_k": 500, "min_p": 0.0,
          "repeat_last_n": 0, "repeat_penalty": 1.0,
          "presence_penalty": 0.0, "frequency_penalty": 0.0,
          "rwkv_count_penalty": 0.0, "rwkv_presence_penalty": 0.0, "rwkv_penalty_decay": 0.99}
B_ = dict(BASE_S, rwkv_count_penalty=0.1, rwkv_presence_penalty=1.0)
C_ = dict(BASE_S, rwkv_count_penalty=0.1, rwkv_presence_penalty=50.0)

PROMPT = "请从 1 开始一直往上数：1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30"


def post(path, body):
    req = urllib.request.Request(BASE + path,
                                 data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with OPENER.open(req, timeout=900) as r:
        return json.loads(r.read().decode("utf-8"))


def gen(sampling, n=350, prompt=PROMPT, seed=999):
    body = {"model": MODEL, "prompt": prompt, "max_tokens": n, "stream": False, "seed": seed}
    body.update(sampling)
    r = post("/v1/completions", body)
    return r["choices"][0]["text"], r.get("usage", {}), r["choices"][0].get("finish_reason")


def stats(t):
    toks = t.split()
    if len(toks) < 8:
        return dict(n=len(toks), distinct=1.0, maxrep=0)
    g = Counter(tuple(toks[i:i + 4]) for i in range(len(toks) - 3))
    return dict(n=len(toks), distinct=round(len(g) / (len(toks) - 3), 3), maxrep=max(g.values()))


def show(tag, t, u):
    print("  %-8s %s | 前 150 字: %s" % (tag, stats(t), t[:150].replace("\n", "\\n")))


def main():
    print("== A 无罚项 ==")
    ta, ua, fa = gen(BASE_S)
    show("A", ta, ua)
    print("== B 官方 demo 罚项 (count .1 / presence 1.0 / decay .99) ==")
    tb, ub, fb = gen(B_)
    show("B", tb, ub)
    print("== C 夸张 presence 50（判定性）==")
    tc, uc, fc = gen(C_)
    show("C", tc, uc)

    print("\n== 判定 ==")
    print("  A vs B 文本是否不同      :", ta != tb)
    print("  A vs C 文本是否不同      :", ta != tc, " (presence=50 必须不同，否则补丁未生效)")
    print("  C 的 4-gram 最大重复次数 :", stats(tc)["maxrep"], " (A 为 %d)" % stats(ta)["maxrep"])
    print("  C 是否出现连续重复 token :", stats(tc)["maxrep"] > 3)

    print("\n== logprobs 原始结构（确认字段位置）==")
    body = dict(BASE_S, model=MODEL, prompt="中国的首都是", max_tokens=1, logprobs=5, seed=7)
    r = post("/v1/completions", body)
    print("  choice keys:", list(r["choices"][0].keys()))
    lp = r["choices"][0].get("logprobs")
    print("  logprobs:", json.dumps(lp, ensure_ascii=False)[:400])


if __name__ == "__main__":
    main()
