#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""iq_bank 评测 harness —— 对任意 OpenAI 兼容 chat 端点跑 iq_bank_v1.json 并判分
用法: python run_eval.py --base-url http://127.0.0.1:18801 --model ling3-q4kvq8 --out results.json [--ids Q01,Q31]
判分: keys=关键词组命中率; exact/num=最终答案行精确/数值; code=执行隐藏用例; tool=协议多轮工具调用; jsonspec/num_range
"""
import argparse, json, os, re, subprocess, sys, tempfile, time, io
import urllib.request, urllib.error
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))

def norm(s):
    if s is None: return ""
    s = str(s).lower()
    s = s.replace("．", ".").replace("：", ":").replace("，", ",").replace("。", ",")
    for ch in " \t\r\n,！!？?；;:（）()【】[]\"'‘’“”、":
        s = s.replace(ch, "")
    return s

def _num_candidates(seg):
    """从原文提取数字候选（含 a/b 分数与 a.b 小数，不被标点清洗破坏）"""
    out = re.findall(r"-?\d+(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)?", seg or "")
    return out

UNITS = "只辆万个小时度分秒年月天人元个次件名种位张条台斤km米℃%％"

def extract_final_answer(text):
    """取最后一个'最终答案：'后的内容（截到行尾）；无标记则返回 None"""
    ms = re.findall(r"最终答案[：:]\s*(.+)", text or "")
    if ms:
        seg = ms[-1].strip()
        return seg
    return None

def strip_units(seg):
    seg = re.sub(r"^[a-zA-Z=：:]+", "", (seg or "").strip())
    for u in UNITS:
        seg = seg.replace(u, "")
    return norm(seg)

def parse_num(s):
    s = str(s).strip()
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)", s)
    if m:
        try:
            den = float(m.group(2))
            return float(m.group(1)) / den if den else None
        except Exception:
            return None
    if re.fullmatch(r"-?\d+(\.\d+)?", s):
        return float(s)
    try:
        return float(Fraction(s))
    except Exception:
        return None

def last_number(text):
    ms = re.findall(r"-?\d+(?:\.\d+)?|\d+/\d+", text or "")
    return ms[-1] if ms else None

# ---------------- 判分器 ----------------
def grade_text(g, content):
    t = g["type"]
    if t == "keys":
        n = norm(content)
        full = g.get("full") or len(g["groups"])
        matched = sum(1 for grp in g["groups"] if any(norm(a) in n for a in grp))
        score = min(1.0, matched / full) if full else 0.0
        if score == 0 and g.get("half_keys"):
            if any(norm(a) in n for grp in g["half_keys"] for a in grp):
                return 0.5, "half_keys"
        return score, f"matched {matched}/{full}"
    if t == "exact":
        seg = extract_final_answer(content)
        cand = [strip_units(seg)] if seg else []
        n_all = norm(content)
        for a in g["answer"]:
            na = strip_units(a)
            if not na: continue
            for c in cand:
                if c == na:
                    return 1.0, "exact(final)"
            # 无标记时回退：独立数字/词出现
            if re.fullmatch(r"-?\d+(\.\d+)?", na):
                if re.search(r"(?<![\d.])" + re.escape(na) + r"(?![\d.])", n_all):
                    return 1.0, "exact(fallback)"
            else:
                if na in n_all:
                    return 1.0, "exact(substring)"
        return 0.0, "no match"
    if t == "num":
        seg = extract_final_answer(content)
        got = None
        if seg:
            cs = _num_candidates(seg)
            got = parse_num(cs[-1]) if cs else None
        if got is None:
            got = parse_num(last_number(content) or "")
        if got is None:
            return 0.0, "no number"
        if abs(got - g["value"]) <= g.get("tol", 1e-6):
            return 1.0, f"num={got}"
        for a in g.get("alts", []):
            na = parse_num(strip_units(a))
            if na is not None and abs(got - na) <= g.get("tol", 1e-6):
                return 1.0, f"alt={got}"
        return 0.0, f"num={got} want {g['value']}"
    if t == "num_range":
        seg = extract_final_answer(content)
        got = None
        if seg:
            m2 = re.findall(r"(\d+(?:\.\d+)?)\s*(万|亿)", seg)
            if m2:
                v, u = m2[-1]
                got = float(v) * (1e8 if u == "亿" else 1e4)
            else:
                cs = _num_candidates(seg)
                got = parse_num(cs[-1]) if cs else None
        if got is None:
            got = parse_num(last_number(content) or "")
        if got is not None and g["lo"] <= got <= g["hi"]:
            return 1.0, f"num={got} in range"
        n = norm(content)
        if any(norm(a) in n for grp in g["keys"] for a in grp):
            return 0.5, "method keys only"
        return 0.0, f"num={got} out of range"
    if t == "jsonspec":
        txt = (content or "").strip()
        m = re.findall(r"```(?:json)?\s*(.+?)```", txt, re.S)
        if m: txt = m[0]
        st = txt.find("{"); en = txt.rfind("}")
        if st < 0 or en <= st:
            return 0.0, "no json"
        try:
            obj = json.loads(txt[st:en + 1])
        except Exception as e:
            return 0.0, f"json parse fail"
        ok = 0; tot = 0
        for k, v in g["fields"].items():
            tot += 1
            if obj.get(k) == v: ok += 1
        for k, ln in g.get("list_len", {}).items():
            tot += 1
            v = obj.get(k)
            if isinstance(v, list) and len(v) == ln: ok += 1
        return (ok / tot if tot else 0), f"fields {ok}/{tot}"
    raise ValueError("unknown grader " + t)

def grade_code(g, content, workdir):
    txt = content or ""
    m = re.findall(r"```(?:python|py)?\s*\n(.*?)```", txt, re.S)
    if m:
        code = "\n\n".join(b.strip() for b in m)  # 多块拼接：定义与实现可能分块输出
    else:
        code = txt if re.search(r"\b(def|class)\b", txt) else None
    if not code:
        return 0.0, "no code block", ""
    code = code.strip()
    src = code + "\n\n" + g["tests"] + "\n"
    fd, path = tempfile.mkstemp(suffix=".py", dir=workdir)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(src)
    try:
        r = subprocess.run([sys.executable, "-X", "utf8", path], capture_output=True, text=True, timeout=15, cwd=workdir)
        if r.returncode == 0 and "PASS" in r.stdout:
            return 1.0, "PASS", ""
        tail = (r.stderr or r.stdout or "").strip().splitlines()
        return 0.0, "FAIL", tail[-1][:200] if tail else "no output"
    except subprocess.TimeoutExpired:
        return 0.0, "TIMEOUT", ""
    finally:
        try: os.remove(path)
        except OSError: pass

def tool_result_for(q, fname, args):
    rules = q.get("tool_results", {}).get(fname)
    if rules is None:
        return "（工具已执行，无返回）"
    if isinstance(rules, str):
        return rules
    blob = fname + json.dumps(args, ensure_ascii=False)
    for r in rules:
        if r.get("when") and r["when"] in blob:
            return r["result"]
    for r in rules:
        if not r.get("when"):
            return r["result"]
    return "（无结果）"

# ---------------- API ----------------
# 显式禁用系统代理（NGN 隧道环境下 127.0.0.1 会被劫走 → WinError 1006）
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
API_KEY = None

AUTH = {"Content-Type": "application/json"}
def set_auth(key):
    if key:
        AUTH["Authorization"] = "Bearer " + key

SAMPLING = {}  # 统一采样协议（CLI/--profile 注入；见 main()）
THINK_EXTRA = {}  # --no-think 时注入 {"chat_template_kwargs": {"enable_thinking": False}}

def chat(base, model, messages, tools=None, temperature=0.2, max_tokens=4096, extra=None):
    body = {"model": model, "messages": messages, "temperature": temperature,
            "top_p": 0.95, "max_tokens": max_tokens, "stream": False}
    body.update({k: v for k, v in SAMPLING.items() if k != "temperature"})
    if SAMPLING.get("temperature") is not None:
        body["temperature"] = SAMPLING["temperature"]
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    if extra:
        body.update(extra)
    if THINK_EXTRA:
        ctk = dict(body.get("chat_template_kwargs") or {})
        ctk.update(THINK_EXTRA.get("chat_template_kwargs") or {})
        body["chat_template_kwargs"] = ctk
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = "Bearer " + API_KEY
    last = None
    for attempt in range(3):
        try:
            url = base.rstrip("/")
            if not url.endswith("/chat/completions"):
                url += "/v1/chat/completions"
            req = urllib.request.Request(url, data=data,
                                         headers=headers)
            with OPENER.open(req, timeout=300) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            last = e
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"request failed: {last}")

def get_msg_content(resp):
    ch = resp["choices"][0]["message"]
    return ch.get("content") or "", ch.get("reasoning_content") or "", ch.get("tool_calls") or []

# ---------------- 单题执行 ----------------
def run_question(q, args, workdir):
    temp = q.get("temperature", 0.2)
    msgs = [{"role": "user", "content": ((q.get("payload", "") + "\n\n--------\n\n") if q.get("payload") else "") + q["prompt"]}]
    t0 = time.time(); usage = {}; rounds = []
    g = q["grader"]

    if g["type"] == "tool":
        expected = g["expected_calls"]; matched = set(); final_text = ""; nrounds = 0
        for rnd in range(8):
            resp = chat(args.base_url, args.model, msgs, tools=q.get("tools"), temperature=temp, max_tokens=args.max_tokens)
            usage = resp.get("usage", usage)
            content, reasoning, tcs = get_msg_content(resp)
            nrounds += 1
            if not tcs:
                final_text = content
                break
            asst = {"role": "assistant", "content": content or "", "tool_calls": [
                {"id": tc["id"], "type": "function",
                 "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}} for tc in tcs]}
            msgs.append(asst)
            for tc in tcs:
                fname = tc["function"]["name"]
                try:
                    fargs = json.loads(tc["function"]["arguments"] or "{}")
                except Exception:
                    fargs = {}
                rounds.append({"call": fname, "args": fargs})
                for i, e in enumerate(expected):
                    if i in matched or e["name"] != fname:
                        continue
                    ok = True
                    for k, alts in e.get("args", {}).items():
                        got = norm(str(fargs.get(k, "")))
                        if not any(norm(a) in got or got in norm(a) for a in alts):
                            ok = False; break
                    if ok:
                        matched.add(i)
                msgs.append({"role": "tool", "tool_call_id": tc["id"],
                             "content": str(tool_result_for(q, fname, fargs))})
        else:
            if not final_text.strip():
                # 轮次耗尽仍无文本总结 → 追加一次无工具调用强制收尾
                for attempt in range(2):
                    resp2 = chat(args.base_url, args.model, msgs, temperature=temp,
                                 extra=None if attempt == 0 else {"chat_template_kwargs": {"enable_thinking": False}})
                    c2, _, t2 = get_msg_content(resp2)
                    usage = resp2.get("usage", usage)
                    if c2.strip():
                        final_text = c2
                        break
                    if t2:
                        break
        if expected:
            calls_score = len(matched) / len(expected)
        else:
            # 无预期调用：正确行为是不调用任何工具（危险操作需先确认）
            calls_score = 1.0 if not rounds else 0.0
        fs, fnote = grade_text(g["final"], final_text) if final_text else (0.0, "empty final")
        score = 0.5 * calls_score + 0.5 * fs
        return dict(score=score, note=f"calls {len(matched)}/{len(expected)}; {fnote}; rounds={nrounds}",
                    calls=rounds, answer=(final_text or "")[:400], head=(final_text or "")[:300],
                    secs=round(time.time() - t0, 1), usage=usage)

    # 文本/编码类
    extra = None; content, reasoning = "", ""
    for attempt in range(2):
        resp = chat(args.base_url, args.model, msgs, temperature=temp, extra=extra, max_tokens=args.max_tokens)
        usage = resp.get("usage", {})
        content, reasoning, tcs = get_msg_content(resp)
        if content.strip():
            break
        extra = {"chat_template_kwargs": {"enable_thinking": False}}  # 空回复→关思考重试
    if g["type"] == "code":
        score, note, err = grade_code(g, content, workdir)
        return dict(score=score, note=note + (f"; err={err}" if err else ""), answer=err or note,
                    head=content[:300], secs=round(time.time() - t0, 1), usage=usage)
    score, note = grade_text(dict(g, half_keys=g.get("half_keys") or q.get("half_keys")),
                             content if content.strip() else reasoning)
    seg = extract_final_answer(content)
    return dict(score=score, note=note + ("; [空回复用思考判分]" if not content.strip() else ""),
                answer=(seg if seg else "")[:120], head=content[:300],
                secs=round(time.time() - t0, 1), usage=usage)

# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids", default="")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--tier", default="all", choices=["all", "easy", "medium", "hard"])
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--api-key", default=os.environ.get("API_KEY", ""))
    ap.add_argument("--bank", default=os.path.join(HERE, "iq_bank_v1.json"))
    # ---- 采样协议（默认与原 v3 跑法等价：题内温度 / top_p 0.95 / 无罚项）----
    ap.add_argument("--profile", default="default", choices=["default", "rwkv-demo", "legacy"],
                    help="rwkv-demo = RWKV 官方 Gradio demo 解码：temp 1.0 / top_p 0.5 / top_k 500 / "
                         "presence 1.0 / count 0.1 / decay 0.99（需打过 rwkv-penalties 补丁的 llama-server）")
    ap.add_argument("--temp", type=float, default=None, help="覆盖所有题的温度（不给则用题库内每题的 temperature）")
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--min-p", type=float, default=None)
    ap.add_argument("--repeat-penalty", type=float, default=None)
    ap.add_argument("--repeat-last-n", type=int, default=None)
    ap.add_argument("--presence-penalty", type=float, default=None)
    ap.add_argument("--frequency-penalty", type=float, default=None)
    ap.add_argument("--rwkv-count-penalty", type=float, default=None, dest="rwkv_count_penalty")
    ap.add_argument("--rwkv-presence-penalty", type=float, default=None, dest="rwkv_presence_penalty")
    ap.add_argument("--rwkv-penalty-decay", type=float, default=None, dest="rwkv_penalty_decay")
    ap.add_argument("--no-think", action="store_true",
                    help="对齐 RWKV 官方 demo：生成前缀补 <think>\\n</think>（关闭思维链）")
    args = ap.parse_args()
    if args.no_think:
        THINK_EXTRA.update({"chat_template_kwargs": {"enable_thinking": False}})
    set_auth(args.api_key)
    global API_KEY
    API_KEY = args.api_key or None

    if args.profile == "legacy":
        # 原 v3 协议：只下发 temperature（题目自带 0.7，其余 0.2）与 top_p 0.95，其余走服务端默认
        proto = {"temperature": None, "top_p": 0.95}
    else:
        proto = {"temperature": None, "top_p": 0.95, "top_k": 0, "min_p": 0.0,
                 "repeat_last_n": 0, "repeat_penalty": 1.0,
                 "presence_penalty": 0.0, "frequency_penalty": 0.0,
                 "rwkv_count_penalty": 0.0, "rwkv_presence_penalty": 0.0, "rwkv_penalty_decay": 1.0}
        if args.profile == "rwkv-demo":
            proto.update({"temperature": 1.0, "top_p": 0.5, "top_k": 500, "min_p": 0.0,
                          "rwkv_count_penalty": 0.1, "rwkv_presence_penalty": 1.0, "rwkv_penalty_decay": 0.99})
    for cli_key, body_key in (("temp", "temperature"), ("top_p", "top_p"), ("top_k", "top_k"),
                              ("min_p", "min_p"), ("repeat_penalty", "repeat_penalty"),
                              ("repeat_last_n", "repeat_last_n"), ("presence_penalty", "presence_penalty"),
                              ("frequency_penalty", "frequency_penalty"),
                              ("rwkv_count_penalty", "rwkv_count_penalty"),
                              ("rwkv_presence_penalty", "rwkv_presence_penalty"),
                              ("rwkv_penalty_decay", "rwkv_penalty_decay")):
        v = getattr(args, cli_key)
        if v is not None:
            proto[body_key] = v
    SAMPLING.clear()
    SAMPLING.update(proto)
    print("[sampling] " + json.dumps(SAMPLING, ensure_ascii=False), flush=True)

    bank = json.load(open(args.bank, encoding="utf-8"))
    qs = bank["questions"]
    if args.tier != "all":
        qs = [q for q in qs if q.get("tier") == args.tier]
    if args.ids:
        want = set(args.ids.split(","))
        qs = [q for q in qs if q["id"] in want]
    weights = bank["meta"]["weights"]
    workdir = tempfile.mkdtemp(prefix="iqb_")

    results = []
    done = set()
    if args.resume and os.path.exists(args.out):
        try:
            old_r = json.load(io.open(args.out, encoding="utf-8"))["results"]
            results = old_r
            done = {x["id"] for x in old_r}
            print(f"[resume] 已有 {len(done)} 题结果，跳过", flush=True)
        except Exception:
            pass
    for i, q in enumerate(qs):
        if q["id"] in done:
            continue
        try:
            r = run_question(q, args, workdir)
        except Exception as e:
            r = dict(score=0.0, note=f"HARNESS ERROR: {e}", answer="", head="", secs=0, usage={})
        r.update(id=q["id"], cat=q["cat"], difficulty=q["difficulty"], tier=q.get("tier", "easy"))
        results.append(r)
        print(f"[{i+1}/{len(qs)}] {q['id']} {q['cat']:<9} score={r['score']:.2f} ({r['note'][:60]}) {r['secs']}s", flush=True)
        json.dump({"model": args.model, "base_url": args.base_url, "sampling": SAMPLING,
                   "chat_template_kwargs": THINK_EXTRA.get("chat_template_kwargs", {}), "results": results},
                  open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # 汇总
    percat = {}
    for r in results:
        percat.setdefault(r["cat"], []).append(min(1.0, r["score"]))
    total = 0.0
    print("\n===== 汇总", args.model, "=====")
    for cat, sc in sorted(percat.items()):
        rate = sum(sc) / len(sc)
        total += weights.get(cat, 0) * rate
        print(f"  {cat:<9} {sum(sc):.1f}/{len(sc)}  得分率 {rate*100:5.1f}%  权重 {weights.get(cat,0)*100:.0f}%")
    print(f"  加权总分: {total*100:.1f} / 100")
    tiers = sorted({r.get("tier", "easy") for r in results})
    if len(tiers) > 1:
        for t in tiers:
            sub = [r for r in results if r.get("tier", "easy") == t]
            tt = sum(weights.get(r["cat"], 0) * 1.0 for r in sub)  # 占位防呆
            tc = {}
            for r in sub:
                tc.setdefault(r["cat"], []).append(min(1.0, r["score"]))
            tt = sum(weights.get(c, 0) * sum(sc) / len(sc) for c, sc in tc.items())
            print(f"  [{t} 档] {len(sub)}题 加权 {tt*100:.1f}")

if __name__ == "__main__":
    main()
