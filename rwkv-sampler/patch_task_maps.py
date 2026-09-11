import io
p = "$BENCH_ROOT/llama-rwkv-pr1/tools/server/server-task.cpp"
s = io.open(p, encoding="utf-8").read()
orig = s
for ind in ("            ", "        "):
    old = ind + "{\"presence_penalty\",          sampling.penalty_present},\n" + ind + "{\"frequency_penalty\",         sampling.penalty_freq},\n"
    new = old + (ind + "{\"rwkv_count_penalty\",        sampling.rwkv_count_penalty},\n"
                     + ind + "{\"rwkv_presence_penalty\",     sampling.rwkv_presence_penalty},\n"
                     + ind + "{\"rwkv_penalty_decay\",        sampling.rwkv_penalty_decay},\n")
    if old in s:
        s = s.replace(old, new)
        print("patched map with indent %d" % len(ind))
    else:
        print("skip indent %d" % len(ind))
if s != orig:
    if not __import__("os").path.exists(p + ".prepatch"):
        io.open(p + ".prepatch", "w", encoding="utf-8").write(orig)
    io.open(p, "w", encoding="utf-8", newline="").write(s)
    print("written")
else:
    print("NO CHANGE")
