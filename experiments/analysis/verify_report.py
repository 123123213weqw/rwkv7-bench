import io, json, re
p = r"C:\Users\31046\AppData\Roaming\com.xlang.xharness\workspace\bench_rwkv\iq_v3_report_v4_rwkv.html"
h = io.open(p, encoding="utf-8").read()
m = re.search(r"const DATA = (\[.*?\]);\n", h, re.S)
d = json.loads(m.group(1))
print("DATA 解析 OK, 模型数 =", len(d))
print()
print("尾部 5 行:")
for x in d[-5:]:
    print("  %-42s total=%5.1f totalC=%5.1f nq=%d mins=%d" % (x["name"], x["total"], x["totalC"], x["nq"], x["mins"]))
print()
# 结构完整性
for t in ("<html", "</html>", "<script>", "</script>", 'id="board"', 'id="refboard"', 'id="cmp"', 'id="radar"', 'id="cks"'):
    print("  %-14s x%d" % (t, h.count(t)))
print()
print("已更新偏差声明:", "44.2 分（500/关思考）" in h)
print("已更新脚注    :", "4 行解码口径对照" in h)
print("旧 41.2 行保留:", any(x["name"] == "RWKV7-G1j-2.9B Q4" and x["total"] == 41.2 for x in d))
