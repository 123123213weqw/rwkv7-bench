#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断截断成因：把 200 题按 finish 情况分层，看「死循环」与「真长」各占多少。

用法: python diag_truncation.py <results.json> [<results.json> ...]
"""
import json, io, sys, os, re, collections

CAP = 8192


def rep_profile(x):
    """用 head+answer 估计重复度（harness 只留了前 300 字 + 最终答案行）"""
    t = (x.get("head") or "") + " " + (x.get("answer") or "")
    toks = re.findall(r"\S+", t)
    if len(toks) < 12:
        return 0, 1.0
    g = collections.Counter(tuple(toks[i:i + 8]) for i in range(len(toks) - 7))
    top = g.most_common(1)[0][1] if g else 0
    return top, len(g) / max(1, len(toks) - 7)


def main():
    for path in sys.argv[1:]:
        d = json.load(io.open(path, encoding="utf-8"))
        rows = d["results"]
        print("=" * 78)
        print(os.path.basename(path), " n=%d" % len(rows))
        s = d.get("sampling")
        if s:
            print("  sampling:", json.dumps(s, ensure_ascii=False))

        trunc = [x for x in rows if (x.get("usage") or {}).get("completion_tokens", 0) >= CAP]
        norm = [x for x in rows if (x.get("usage") or {}).get("completion_tokens", 0) < CAP]
        print("  撞上限(>=%d): %d/%d (%.0f%%)" % (CAP, len(trunc), len(rows), 100.0 * len(trunc) / len(rows)))

        # 分层：截断题里的重复度
        hi = [x for x in trunc if rep_profile(x)[0] >= 3]
        lo = [x for x in trunc if rep_profile(x)[0] < 3]
        print("    截断题里 8-gram 重复>=3 的（疑似死循环）: %d" % len(hi))
        print("    截断题里 重复<3 的（疑似真长/未收尾）  : %d" % len(lo))

        # 按类别
        print("  按类别（截断数/总数）:")
        bycat = collections.defaultdict(lambda: [0, 0])
        for x in rows:
            bycat[x["cat"]][1] += 1
        for x in trunc:
            bycat[x["cat"]][0] += 1
        for c in sorted(bycat):
            a, b = bycat[c]
            print("    %-9s %2d/%2d  (%.0f%%)" % (c, a, b, 100.0 * a / b))

        # 分数对比
        def mean(v):
            return sum(v) / len(v) if v else 0.0
        print("  平均分: 截断题 %.3f | 未截断题 %.3f" %
              (mean([min(1.0, x["score"]) for x in trunc]), mean([min(1.0, x["score"]) for x in norm])))
        print("  token 分布: %s" % collections.Counter(
            min(8, (x.get("usage") or {}).get("completion_tokens", 0) // 1024) for x in rows).most_common())

        # 最典型的死循环样例
        print("  截断且重复最高的 5 题:")
        for x in sorted(trunc, key=lambda z: -rep_profile(z)[0])[:5]:
            print("    %-5s %-9s rep=%d score=%.2f | %s" %
                  (x["id"], x["cat"], rep_profile(x)[0], min(1.0, x["score"]),
                   (x.get("head") or "")[:90].replace("\n", " ")))


if __name__ == "__main__":
    main()
