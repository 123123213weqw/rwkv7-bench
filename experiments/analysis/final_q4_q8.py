import json, io, collections
D = r"C:\Users\31046\AppData\Roaming\com.xlang.xharness\workspace\bench_rwkv"
W = {"agent":.2,"coding":.2,"math":.2,"dialogue":.1,"ruozhiba":.1,"longctx":.1,"other":.1}
CATS = ["agent","coding","math","dialogue","ruozhiba","longctx","other"]
BANK = json.load(io.open(D + r"\report\iq_bank_v3.json", encoding="utf-8"))
DIFF = {q["id"]: q["difficulty"] for q in BANK["questions"]}

def load(fn):
    return json.load(io.open(D + "\\" + fn, encoding="utf-8"))["results"]

runs = [
    ("Q4_K_M", "results_v3_rwkv7_g1j_2p9b_q4_official.json"),
    ("Q8_0", "results_v3_rwkv7_g1j_2p9b_q8_official_fixed.json"),
]
print("=" * 100)
print("官方 demo 解码口径 · 同协议同预算（temp1/top_p0.5/top_k500/罚项 + 关思考 + 500 tok）")
print("=" * 100)
print("%-8s %7s  %s  %6s %6s %6s   %s" % ("量化", "总分", "".join("%9s"%c for c in CATS), "easy","med","hard","截断"))
res = {}
for name, fn in runs:
    R = load(fn)
    per = collections.defaultdict(list); te = {"e":[0.,0],"m":[0.,0],"h":[0.,0]}
    for x in R:
        s = min(1.0, x["score"]); per[x["cat"]].append(s)
        dv = DIFF.get(x["id"],5); k = "e" if dv<=5 else ("m" if dv<=8 else "h")
        te[k][0]+=s; te[k][1]+=1
    tot = sum(W[c]*sum(v)/len(v) for c,v in per.items())*100
    tr = sum(1 for x in R if (x.get("usage") or {}).get("completion_tokens",0) >= 500)
    res[name] = (tot, per, te)
    print("%-8s %7.1f  %s  %6.1f %6.1f %6.1f   %d/%d" % (
        name, tot, "".join("%8.1f%%" % (sum(per[c])/len(per[c])*100) for c in CATS),
        te["e"][0]/te["e"][1]*100, te["m"][0]/te["m"][1]*100, te["h"][0]/te["h"][1]*100, tr, len(R)))
print()
d = res["Q8_0"][0] - res["Q4_K_M"][0]
print("总分差: Q8 %.1f - Q4 %.1f = %+.1f" % (res["Q8_0"][0], res["Q4_K_M"][0], d))
print()
print("类别差（Q8 - Q4）:")
for c in CATS:
    a = sum(res["Q4_K_M"][1][c])/len(res["Q4_K_M"][1][c])*100
    b = sum(res["Q8_0"][1][c])/len(res["Q8_0"][1][c])*100
    print("  %-9s %5.1f%% -> %5.1f%%  (%+.1f)" % (c, a, b, b-a))
