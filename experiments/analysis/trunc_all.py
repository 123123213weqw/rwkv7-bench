import json, io, collections
D = r"C:\Users\31046\AppData\Roaming\com.xlang.xharness\workspace\bench_rwkv"
CATS = ["agent","coding","math","dialogue","ruozhiba","longctx","other"]
RUNS = [
    ("Q4 · 旧协议/8192/思考",   "report\\results_v3_rwkv7_g1j_2p9b_q4.json", 8192),
    ("Q4 · demo/8192/思考",     "results_v3_rwkv7_g1j_2p9b_q4_rwkvdemo.json", 8192),
    ("Q4 · demo/8192/关思考",   "results_v3_rwkv7_g1j_2p9b_q4_nothink8192.json", 8192),
    ("Q4 · 官方/500/关思考",     "results_v3_rwkv7_g1j_2p9b_q4_official.json", 500),
    ("Q4 · demo/500/思考",      "results_v3_rwkv7_g1j_2p9b_q4_demo500.json", 500),
    ("★Q8 · 官方/500/关思考",    "results_v3_rwkv7_g1j_2p9b_q8_official_fixed.json", 500),
]
print("=" * 108)
print("%-24s %6s %8s   %s" % ("运行", "预算", "截断", "按类别"))
print("=" * 108)
for name, fn, cap in RUNS:
    try:
        R = json.load(io.open(D + "\\" + fn, encoding="utf-8"))["results"]
    except Exception as e:
        print("%-24s  缺文件" % name); continue
    tr = [x for x in R if (x.get("usage") or {}).get("completion_tokens", 0) >= cap]
    bc = collections.defaultdict(lambda: [0,0])
    for x in R: bc[x["cat"]][1] += 1
    for x in tr: bc[x["cat"]][0] += 1
    cat = " ".join("%s%d/%d" % (c[:4], bc[c][0], bc[c][1]) for c in CATS)
    print("%-24s %6d %4d/%-3d %5.1f%%   %s" % (name, cap, len(tr), len(R), 100.0*len(tr)/len(R), cat))
print()
print("Q4→Q8（同官方口径 500/关思考）逐类别截断对比:")
a = json.load(io.open(D + r"\results_v3_rwkv7_g1j_2p9b_q4_official.json", encoding="utf-8"))["results"]
b = json.load(io.open(D + r"\results_v3_rwkv7_g1j_2p9b_q8_official_fixed.json", encoding="utf-8"))["results"]
print("  %-10s %8s %8s %8s" % ("类别", "Q4", "Q8", "变化"))
for c in CATS:
    x = sum(1 for r in a if r["cat"]==c and (r.get("usage") or {}).get("completion_tokens",0)>=500)
    y = sum(1 for r in b if r["cat"]==c and (r.get("usage") or {}).get("completion_tokens",0)>=500)
    n = sum(1 for r in a if r["cat"]==c)
    print("  %-10s %4d/%-3d %4d/%-3d  %+d" % (c, x, n, y, n, y-x))
