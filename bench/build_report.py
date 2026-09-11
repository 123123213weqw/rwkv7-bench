#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""iq_bank_v3 交互式 HTML 报告生成器：7 类别环形图（勾选模型环比）+ 总榜表"""
import json, os, io

HERE = os.path.dirname(os.path.abspath(__file__))
W = {"agent": .2, "coding": .2, "math": .2, "dialogue": .1, "ruozhiba": .1, "longctx": .1, "other": .1}
CATS_ZH = {"agent": "Agent 能力", "coding": "编码能力", "math": "数学能力", "dialogue": "对话能力",
           "ruozhiba": "脑筋急转弯", "longctx": "长上下文", "other": "其他"}
BANK = json.load(io.open(os.path.join(HERE, "iq_bank_v3.json"), encoding="utf-8"))
DIFF = {q["id"]: q["difficulty"] for q in BANK["questions"]}

NAMES = {  # 结果文件名(去 results_v3_ 前缀) -> (显示名, 颜色); 未列出的自动命名入榜
    "qwen38_27b_q5.json":            ("Qwen3.8-27B-Unc Q5",     "#e6194b"),
    "ornith35b_heretic.json":        ("Ornith-35B-Heretic Q6",  "#3cb44b"),
    "qwen35_4b_q4.json":             ("Qwen3.5-4B Q4",          "#ffe119"),
    "qwen35_9b_mtp_q4.json":         ("Qwen3.5-9B-MTP Q4",      "#0082c8"),
    "ornith9b_q4.json":              ("Ornith-1.5-9B Q4",       "#f58231"),
    "spark_x25_4b_q8.json":          ("Spark-X2.5-4B Q8",       "#911eb4"),
    "gemma4_e4b_q4.json":            ("gemma-4-E4B Q4",         "#46f0f0"),
    "ling3_q4_kvq8.json":            ("Ling-3.0-tiny Q4",       "#f032e6"),
    "lfm25_2p6b_q8.json":            ("LFM2.5-2.6B Q8",         "#bcf60c"),
    "gemma4_e2b_q8.json":            ("gemma-4-E2B Q8",         "#fabebe"),
    "spark_x25_1p7b_q8.json":        ("Spark-X2.5-1.7B Q8",     "#008080"),
    "minicpm5_2b_q8.json":           ("MiniCPM5-2B Q8",         "#e6beff"),
    "qwen35_2b_q8.json":             ("Qwen3.5-2B Q8",          "#9a6324"),
    "minicpm5_1b_q8.json":           ("MiniCPM5-1B Q8",         "#800000"),
    "lfm25_8b_a1b_q8.json":          ("LFM2.5-8B-A1B Q8",       "#808000"),
    "qwen35_08b_q8.json":            ("Qwen3.5-0.8B Q8",        "#000075"),
    "minicpm1b_fable5_q8.json":      ("1B-Fable5 Q8",           "#a9a9a9"),
    "rwkv7_g1i_1p5b_q8.json":        ("RWKV7-g1i-1.5B Q8",      "#fffac8"),
    "minicpm1b_agentic_q8.json":     ("1B-Agentic-DPO Q8",      "#4363d8"),
    "minicpm1b_antihallu_q8_partial.json": ("1B-anti-hallu Q8 †", "#ffd8b1"),
    # —— 队列中：文件出现即自动入榜 ——
    "rwkv7_g1j_2p9b_q4.json":        ("RWKV7-G1j-2.9B Q4",      "#aaaaff"),
    "rwkv7_g1j_1p5b_q8.json":        ("RWKV7-G1j-1.5B Q8",      "#ddaaa0"),
    "qwen35_35b_a3b_q5.json":        ("Qwen3.5-35B-A3B Q5",     "#ff6666"),
    "qwen36_35b_a3b_q5.json":        ("Qwen3.6-35B-A3B Q5",     "#66ff99"),
    "ling3_tiny_q8.json":            ("Ling-3.0-tiny Q8",       "#ff99cc"),
    "qwen35_4b_q8.json":             ("Qwen3.5-4B Q8",          "#99ff66"),
    "gemma4_e4b_q8.json":            ("gemma-4-E4B Q8",         "#66ccff"),
    "ornith9b_q8.json":              ("Ornith-1.5-9B Q8",       "#ffaa33"),
    "ornith9b_official_q4.json":     ("Ornith-9B官方Q4",        "#cc8800"),
    "qwen35_9b_q8.json":             ("Qwen3.5-9B Q8",          "#3399ff"),
    "minicpm4_8b_q4.json":           ("MiniCPM4-8B Q4",         "#cc66ff"),
    "ornith35b_orig_q4.json":        ("Ornith-35B原版 Q4",      "#33cc99"),
    "qwen38_27b_q8.json":            ("Qwen3.8-27B-Unc Q8",     "#ff4444"),
    "ornith35b_orig_q8.json":        ("Ornith-35B原版 Q8",      "#00cc77"),
    "glm_53.json":                   ("GLM-5.3 API 参照",        "#ffffff"),
    "glm_53_flash.json":             ("GLM-5.3-Flash API 参照",  "#c0c0c0"),
}
import glob, time
FILES = sorted(glob.glob(os.path.join(HERE, "results_v3_*.json")))
data = []
for p in FILES:
    key = os.path.basename(p)[len("results_v3_"):]
    nm = NAMES.get(key, (key.replace(".json", "").replace("_", " "), "#888888"))
    name, color = nm[0], nm[1]
    try:
        r = json.load(io.open(p, encoding="utf-8"))["results"]
    except Exception:
        continue
    # 进行中的评测（题数未满且文件 5 分钟内仍在更新）不入榜
    if len(r) < 200 and time.time() - os.path.getmtime(p) < 300:
        continue
    percat = {}
    te = {"e": [0.0, 0], "m": [0.0, 0], "h": [0.0, 0]}
    for x in r:
        s = min(1.0, x["score"])
        percat.setdefault(x["cat"], []).append(s)
        k = "e" if DIFF[x["id"]] <= 5 else ("m" if DIFF[x["id"]] <= 8 else "h")
        te[k][0] += s; te[k][1] += 1
    cats = {c: round(sum(v) / len(v) * 100, 1) for c, v in percat.items()}
    total = sum(W[c] * sum(v) / len(v) for c, v in percat.items()) * 100
    data.append(dict(name=name, color=color, total=round(total, 1), cats=cats, ref="API 参照" in name,
                     easy=round(te["e"][0] / max(te["e"][1], 1) * 100, 1),
                     med=round(te["m"][0] / max(te["m"][1], 1) * 100, 1),
                     hard=round(te["h"][0] / max(te["h"][1], 1) * 100, 1),
                     mins=round(sum(x.get("secs", 0) for x in r) / 60), nq=len(r)))
data.sort(key=lambda d: -d["total"])
DATA_JSON = json.dumps(data, ensure_ascii=False)

html = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>端侧模型能力基准测试v3（xLeaves Benchmark v3）</title>
<style>
 body{font-family:"Segoe UI","Microsoft YaHei",sans-serif;margin:0;background:#0f1419;color:#e6e6e6;padding:24px}
 h1{font-size:22px} h2{font-size:17px;margin:28px 0 12px;border-left:4px solid #4f9cf9;padding-left:10px}
 .meta{color:#9aa4b2;font-size:13px}
 .panel{display:flex;flex-wrap:wrap;gap:6px;margin:10px 0;background:#161d26;padding:12px;border-radius:10px}
 .ck{display:flex;align-items:center;gap:6px;padding:4px 10px;border-radius:14px;background:#1f2833;font-size:13px;cursor:pointer;user-select:none}
 .ck input{accent-color:#4f9cf9}
 .ck .dot{width:10px;height:10px;border-radius:50%}
 .ck.off{opacity:.38}
 .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px}
 .card{background:#161d26;border-radius:12px;padding:12px;text-align:center}
 .card h3{margin:4px 0 8px;font-size:14px;color:#cfd8e3}
 table{border-collapse:collapse;width:100%;font-size:13px;background:#161d26;border-radius:10px;overflow:hidden}
 th,td{padding:7px 10px;border-bottom:1px solid #232d3a;text-align:center}
 th{background:#1c2634;color:#9fc3f7;cursor:pointer;white-space:nowrap}
 td.l{text-align:left;white-space:nowrap}
 tr:hover td{background:#1c2634}
 .bar{height:8px;border-radius:4px;background:#2a3648;position:relative;min-width:70px}
 .bar i{position:absolute;left:0;top:0;bottom:0;border-radius:4px}
 .legend{font-size:12px;color:#9aa4b2;margin-top:6px}
 .rank1{color:#ffd700;font-weight:600}.rank2{color:#c0c0c0}.rank3{color:#cd7f32}
</style>
</head>
<body>
<h1>🧠 端侧模型能力基准测试v3（xLeaves Benchmark v3）</h1>
<div class="meta" id="meta"></div>
<details open style="background:#161d26;padding:12px 14px;border-radius:10px;margin:10px 0;font-size:13px;line-height:1.8">
<summary style="cursor:pointer;color:#9fc3f7;font-size:14px;font-weight:600">📋 测试参数说明（点击折叠）</summary>
<div style="color:#c2ccd8">
<b>统一协议</b>：temperature <b>0.2</b>（对话类 0.7）· top_p 0.95 · repeat_penalty 未启用（默认 1.0）· max_tokens <b>8192</b> · KV 缓存统一 <b>q8_0</b> · 上下文窗口 49152 · OpenAI tools 协议（--jinja）<br>
<b>理由</b>：单一变量原则——所有模型同题库、同采样、同 token 预算，使分数差异只反映模型能力本身。温度取全池交集的低区间（多数参测模型官方推荐 0.2-0.7），高温会引入采样噪声、让排名在重跑间漂移；8192 为思考型模型与普通模型的可用折中（v1 实测 4096 会截断思考链）。<br>
<b>已知偏差声明</b>：① RWKV-World 官方推荐采样为 temp 0.8 + repeat_penalty 1.15，与本协议不同——统一低温下 RWKV 存在逐字重复循环风险，<b>其成绩应视为下界</b>；② 个别思考型微调变体（如 1B-Agentic-DPO）在 8192 预算内思考链可能无法终止，属协议约束下的真实行为；③ 量化档随模型略有差异（Q4/Q5/Q6/Q8，表中已标注），均为 ≥5bpw 质量高原区档位。
</div></details>
<h2>模型勾选（雷达图环比）</h2>
<div class="panel" id="cks"></div>
<div class="card" style="max-width:760px;margin:0 auto"><h3>七维智力雷达</h3><div id="radar"></div>
<div class="legend">每层多边形 = 一个勾选模型 · 悬停查看数值 · 建议同时对比 ≤6 个模型</div></div>
<h2>勾选模型对比表</h2>
<table id="cmp"></table>
<h2>参照组（点击表头排序）</h2>
<p class="meta" style="margin:-6px 0 10px">题库仅针对端侧模型能力范围构建，参照组成绩仅用于参考端侧模型距离云端模型的能力差距。</p>
<table id="refboard"></table>
<h2>完整总榜（点击表头排序）</h2>
<table id="board"></table>
<p class="meta">雷达图：7 轴=7 类别得分率；鼠标悬停看数值。†=仅40题确证；RWKV-g1j 与 Qwen3.5/3.6-35B 原版测完后重新生成本报告。</p>
<script>
const DATA = __DATA__;
const CATS = ["agent","coding","math","dialogue","longctx","ruozhiba","other"];
const CATS_ZH = {agent:"Agent 能力",coding:"编码能力",math:"数学能力",dialogue:"对话能力",longctx:"长上下文",ruozhiba:"脑筋急转弯",other:"其他"};
const W = {agent:.2,coding:.2,math:.2,dialogue:.1,ruozhiba:.1,longctx:.1,other:.1};
const sel = new Set(DATA.slice(0,6).map(d=>d.name));
document.getElementById("meta").textContent =
  `xLeaves Benchmark v3 · 端侧模型能力基准 · 200 题 · 7 类配权 20/20/20/10/10/10/10 · 难度 1-10 · KV 统一 q8_0 · temp 0.2（对话 0.7）· max_tokens 8192 · 报告更新于 ${new Date().toLocaleString("zh-CN")}`;

function renderCks(){
  const el=document.getElementById("cks");
  el.innerHTML=DATA.map(d=>`<label class="ck ${sel.has(d.name)?"":"off"}">
    <input type="checkbox" ${sel.has(d.name)?"checked":""} data-n="${d.name}">
    <span class="dot" style="background:${d.color}"></span>${d.name}</label>`).join("");
  el.querySelectorAll("input").forEach(i=>i.onchange=()=>{i.checked?sel.add(i.dataset.n):sel.delete(i.dataset.n);renderAll();});
}
function radar(){
  const list=DATA.filter(d=>sel.has(d.name));
  const CX=380,CY=340,R=250,N=CATS.length;
  const pt=(i,v)=>{const a=-Math.PI/2+i*2*Math.PI/N;return [CX+Math.cos(a)*R*v/100,CY+Math.sin(a)*R*v/100];};
  let s=`<svg viewBox="0 0 760 700" style="max-width:100%">`;
  for(let g=1;g<=5;g++){
    const v=g*20;
    s+=`<polygon points="${CATS.map((c,i)=>pt(i,v).map(x=>x.toFixed(1)).join(",")).join(" ")}" fill="none" stroke="#263140" stroke-width="1"/>`;
    s+=`<text x="${CX+6}" y="${CY-R*v/100+4}" font-size="11" fill="#5c6b7d">${v}</text>`;
  }
  CATS.forEach((c,i)=>{
    const [x,y]=pt(i,100);
    s+=`<line x1="${CX}" y1="${CY}" x2="${x.toFixed(1)}" y2="${y.toFixed(1)}" stroke="#263140" stroke-width="1"/>`;
    const [lx,ly]=pt(i,121);
    s+=`<text x="${lx.toFixed(1)}" y="${ly.toFixed(1)}" text-anchor="middle" dominant-baseline="middle" font-size="14" fill="#cfd8e3" font-weight="600">${CATS_ZH[c]}<tspan fill="#66788c" font-size="11" font-weight="400"> ${W[c]*100}%</tspan></text>`;
  });
  list.forEach(d=>{
    const ps=CATS.map((c,i)=>pt(i,d.cats[c]||0));
    s+=`<polygon points="${ps.map(p=>p.map(x=>x.toFixed(1)).join(",")).join(" ")}" fill="${d.color}" fill-opacity="0.10" stroke="${d.color}" stroke-width="2.2"><title>${d.name}（总分 ${d.total}）
${CATS.map(c=>CATS_ZH[c]+" "+(d.cats[c]??0).toFixed(1)+"%").join(" · ")}</title></polygon>`;
    ps.forEach((p,i)=>{s+=`<circle cx="${p[0].toFixed(1)}" cy="${p[1].toFixed(1)}" r="3.4" fill="${d.color}"><title>${d.name} · ${CATS_ZH[CATS[i]]} ${(d.cats[CATS[i]]??0).toFixed(1)}%</title></circle>`;});
  });
  s+=`<text x="${CX}" y="${CY+4}" text-anchor="middle" font-size="11" fill="#46586c">0</text></svg>`;
  return s;
}
function renderRadar(){document.getElementById("radar").innerHTML=radar();}
function renderCmp(){
  const list=DATA.filter(d=>sel.has(d.name));
  let h="<tr><td class='l'>模型</td>"+CATS.map(c=>`<th>${CATS_ZH[c]}</th>`).join("")+"<th>加权总分</th></tr>";
  list.forEach(d=>{
    h+=`<tr><td class="l"><span class="dot" style="display:inline-block;width:9px;height:9px;border-radius:50%;background:${d.color};margin-right:5px"></span>${d.name}</td>`
      +CATS.map(c=>`<td>${(d.cats[c]??0).toFixed(1)}</td>`).join("")+`<td><b>${d.total.toFixed(1)}</b></td></tr>`;
  });
  document.getElementById("cmp").innerHTML=h;
}
let sortKey="total",asc=false,refMode=false;
function renderBoard(){
  const cols=[["name","模型",d=>d.name,"l"],["total","总分",d=>d.total],["easy","easy",d=>d.easy],["med","med",d=>d.med],["hard","hard",d=>d.hard],["mins","耗时m",d=>d.mins],["nq","题数",d=>d.nq]].concat(CATS.map(c=>[c,CATS_ZH[c],d=>d.cats[c]??0]));
  const arr=DATA.filter(d=>d.ref===refMode).sort((a,b)=>{const f=cols.find(c=>c[0]===sortKey)[2];const x=f(a),y=f(b);return(typeof x==="string"?x.localeCompare(y):x-y)*(asc?1:-1);});
  let h="<tr>"+cols.map(c=>`<th data-k="${c[0]}">${c[1]}${sortKey===c[0]?(asc?" ▲":" ▼"):""}</th>`).join("")+"</tr>";
  arr.forEach((d,i)=>{
    const rk=i+1,cls=rk===1?"rank1":rk===2?"rank2":rk===3?"rank3":"";
    h+=`<tr><td class="l"><span class="${cls}" style="margin-right:6px">${rk}</span>${d.name}</td><td><b>${d.total.toFixed(1)}</b></td>
    <td>${d.easy}</td><td>${d.med}</td><td>${d.hard}</td><td>${d.mins}</td><td>${d.nq}</td>`
    +CATS.map(c=>{const v=d.cats[c]??0;return `<td><div class="bar" title="${v}"><i style="width:${v}%;background:${d.color}"></i></div><span style="font-size:11px;color:#9aa4b2">${v.toFixed(0)}</span></td>`;}).join("")+"</tr>";
  });
  const b=document.getElementById(refMode?"refboard":"board");b.innerHTML=h;
  b.querySelectorAll("th").forEach(t=>t.onclick=()=>{const k=t.dataset.k;if(sortKey===k)asc=!asc;else{sortKey=k;asc=typeof cols.find(c=>c[0]===k)[2]({name:""})==="string";}renderBoard();renderRef();});
}
function renderRef(){const keep=refMode,ka=asc,ks=sortKey;refMode=true;renderBoard();refMode=keep;asc=ka;sortKey=ks;}
function renderAll(){renderCks();renderRadar();renderCmp();renderBoard();renderRef();}
renderAll();
</script>
</body>
</html>"""
html = html.replace("__DATA__", DATA_JSON)
out = os.path.join(HERE, "iq_v3_report.html")
io.open(out, "w", encoding="utf-8").write(html)
print("written", out, len(html), "bytes,", len(data), "models")
