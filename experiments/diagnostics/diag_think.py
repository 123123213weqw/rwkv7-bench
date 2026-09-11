import json, io, collections
p = r"C:\Users\31046\AppData\Roaming\com.xlang.xharness\workspace\bench_rwkv\results_v3_rwkv7_g1j_2p9b_q4_rwkvdemo.json"
rows = json.load(io.open(p, encoding="utf-8"))["results"]
tr = [x for x in rows if (x.get("usage") or {}).get("completion_tokens", 0) >= 8192]
print("截断题 %d 道" % len(tr))
c = collections.Counter()
for x in tr:
    h = (x.get("head") or "")
    c["head含</think>"] += ("</think>" in h)
    c["head含Assistant:"] += ("Assistant:" in h)
    c["head含<think"] += ("<think" in h)
print(dict(c))
# 非截断题对照
nt = [x for x in rows if (x.get("usage") or {}).get("completion_tokens", 0) < 8192]
c2 = collections.Counter()
for x in nt:
    h = (x.get("head") or "")
    c2["head含</think>"] += ("</think>" in h)
    c2["head含Assistant:"] += ("Assistant:" in h)
print("非截断题 %d 道: %s" % (len(nt), dict(c2)))
print()
print("截断题的 token 数分布:", collections.Counter(
    (x["usage"]["completion_tokens"] // 1024) * 1024 for x in rows if x.get("usage",{}).get("completion_tokens",0) >= 4096).most_common(6))
