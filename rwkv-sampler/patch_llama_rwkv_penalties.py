#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Patch llama.cpp (llama-rwkv-pr1) with an exact RWKV sampler, matching BlinkDL's official
RWKV-Gradio demo decoding loop:

    # per generated token, on the raw logits, before top-k / top-p
    logits -= count_penalty * occurrence_count      # occurrence_count is decayed every step
    logits -= occurrence_presence                   # presence_penalty, applied once per seen token
    # after sampling token t
    occurrence_count *= penalty_decay
    occurrence_count[t] += 1
    occurrence_presence[t] = presence_penalty

Reference: RWKV-Gradio-{2,3}/app.py  (sample_logits_batch_cuda + generate loop)

Request params added to the OpenAI-compatible server:
    rwkv_count_penalty, rwkv_presence_penalty, rwkv_penalty_decay

Penalties are inserted at the HEAD of the sampler chain (raw logits) and only count tokens
sampled after the first apply() of the sampler instance -> prompt tokens are never penalized.
"""
import io
import os
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "$LLAMA_SRC"

FILES = {
    "llama_h":        os.path.join(ROOT, "include/llama.h"),
    "sampler_cpp":    os.path.join(ROOT, "src/llama-sampler.cpp"),
    "common_h":       os.path.join(ROOT, "common/common.h"),
    "sampling_cpp":   os.path.join(ROOT, "common/sampling.cpp"),
    "schema_cpp":     os.path.join(ROOT, "tools/server/server-schema.cpp"),
    "task_cpp":       os.path.join(ROOT, "tools/server/server-task.cpp"),
}


def read(p):
    with io.open(p, encoding="utf-8") as f:
        return f.read()


def write(p, s):
    with io.open(p, "w", encoding="utf-8", newline="") as f:
        f.write(s)


def backup(p):
    b = p + ".prepatch"
    if not os.path.exists(b):
        with io.open(p, "rb") as f:
            data = f.read()
        with io.open(b, "wb") as f:
            f.write(data)


def sub(path, old, new, count=1, tag=""):
    s = read(path)
    n = s.count(old)
    if n != count:
        raise SystemExit("ANCHOR MISMATCH (%s): found %d, expected %d in %s\n---\n%s\n---" % (tag, n, count, path, old[:400]))
    backup(path)
    write(path, s.replace(old, new, count))
    print("patched %-22s %s" % (tag, os.path.relpath(path, ROOT)))


# --------------------------------------------------------------------------------------
# 1) public declaration
# --------------------------------------------------------------------------------------
DECL_ANCHOR = """    LLAMA_API struct llama_sampler * llama_sampler_init_penalties(
                             int32_t   penalty_last_n,   // last n tokens to penalize (0 = disable penalty, -1 = context size)
                               float   penalty_repeat,   // 1.0 = disabled
                               float   penalty_freq,     // 0.0 = disabled
                               float   penalty_present); // 0.0 = disabled
"""

DECL_NEW = DECL_ANCHOR + """
    /// RWKV penalties: presence penalty + count penalty with per-step decay, as used by the
    /// official RWKV Gradio demos (BlinkDL/RWKV-Gradio-*). Applied to the raw logits before
    /// top-k/top-p, and only to tokens generated after this sampler's first apply()
    /// (i.e. prompt tokens are never penalized):
    ///
    ///     logits[t] -= count_penalty * count[t] + presence[t]
    ///     after sampling t:   count *= penalty_decay;  count[t] += 1;  presence[t] = presence_penalty
    ///
    LLAMA_API struct llama_sampler * llama_sampler_init_rwkv_penalties(
                             int32_t   n_vocab,
                               float   count_penalty,     // 0.0 = disabled (alpha_frequency)
                               float   presence_penalty,  // 0.0 = disabled (alpha_presence)
                               float   penalty_decay);    // 1.0 = no decay
"""

sub(FILES["llama_h"], DECL_ANCHOR, DECL_NEW, 1, "llama.h decl")

# --------------------------------------------------------------------------------------
# 2) sampler implementation
# --------------------------------------------------------------------------------------
IMPL_ANCHOR = """// top-n-sigma

struct llama_sampler_top_n_sigma {
"""

IMPL_NEW = """// rwkv penalties (presence penalty + count penalty with decay, as in the official RWKV demos)

struct llama_sampler_rwkv_penalties {
    const int32_t n_vocab;
    const float   count_penalty;
    const float   presence_penalty;
    const float   penalty_decay;

    std::vector<float>       count;     // decayed occurrence count per token
    std::vector<float>       presence;  // presence penalty per token (0 or presence_penalty)
    std::vector<llama_token> touched;   // tokens with count > 0

    bool started    = false;            // set by the first apply(); before that, accepts are prefill
    int  n_accepted = 0;
};

static const char * llama_sampler_rwkv_penalties_name(const struct llama_sampler * /*smpl*/) {
    return "rwkv-penalties";
}

static void llama_sampler_rwkv_penalties_accept(struct llama_sampler * smpl, llama_token token) {
    auto * ctx = (llama_sampler_rwkv_penalties *) smpl->ctx;

    // tokens accepted before the first apply() are prompt/prefill tokens -> not penalized
    if (!ctx->started) {
        return;
    }

    if (token < 0 || token >= ctx->n_vocab) {
        return;
    }

    if (ctx->penalty_decay != 1.0f) {
        for (const auto t : ctx->touched) {
            ctx->count[t] *= ctx->penalty_decay;
        }
    }

    if (ctx->count[token] == 0.0f) {
        ctx->touched.push_back(token);
    }
    ctx->count[token] += 1.0f;

    if (ctx->presence_penalty != 0.0f) {
        ctx->presence[token] = ctx->presence_penalty;
    }

    ctx->n_accepted++;
}

static void llama_sampler_rwkv_penalties_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
    auto * ctx = (llama_sampler_rwkv_penalties *) smpl->ctx;

    ctx->started = true;

    if (ctx->count_penalty == 0.0f && ctx->presence_penalty == 0.0f) {
        return;
    }

    for (size_t i = 0; i < cur_p->size; ++i) {
        const llama_token id = cur_p->data[i].id;
        if (id < 0 || id >= ctx->n_vocab) {
            continue;
        }
        const float penalty = ctx->count_penalty * ctx->count[id] + ctx->presence[id];
        if (penalty != 0.0f) {
            cur_p->data[i].logit -= penalty;
        }
    }

    cur_p->sorted = false;
}

static void llama_sampler_rwkv_penalties_reset(struct llama_sampler * smpl) {
    auto * ctx = (llama_sampler_rwkv_penalties *) smpl->ctx;

    std::fill(ctx->count.begin(),    ctx->count.end(),    0.0f);
    std::fill(ctx->presence.begin(), ctx->presence.end(), 0.0f);
    ctx->touched.clear();
    ctx->started    = false;
    ctx->n_accepted = 0;
}

static struct llama_sampler * llama_sampler_rwkv_penalties_clone(const struct llama_sampler * smpl) {
    const auto * ctx = (const llama_sampler_rwkv_penalties *) smpl->ctx;

    auto * result = llama_sampler_init_rwkv_penalties(
            ctx->n_vocab,
            ctx->count_penalty,
            ctx->presence_penalty,
            ctx->penalty_decay);

    auto * result_ctx = (llama_sampler_rwkv_penalties *) result->ctx;

    result_ctx->count      = ctx->count;
    result_ctx->presence   = ctx->presence;
    result_ctx->touched    = ctx->touched;
    result_ctx->started    = ctx->started;
    result_ctx->n_accepted = ctx->n_accepted;

    return result;
}

static void llama_sampler_rwkv_penalties_free(struct llama_sampler * smpl) {
    delete (llama_sampler_rwkv_penalties *) smpl->ctx;
}

static struct llama_sampler_i llama_sampler_rwkv_penalties_i = {
    /* .name              = */ llama_sampler_rwkv_penalties_name,
    /* .accept            = */ llama_sampler_rwkv_penalties_accept,
    /* .apply             = */ llama_sampler_rwkv_penalties_apply,
    /* .reset             = */ llama_sampler_rwkv_penalties_reset,
    /* .clone             = */ llama_sampler_rwkv_penalties_clone,
    /* .free              = */ llama_sampler_rwkv_penalties_free,
    /* .backend_init      = */ nullptr,
    /* .backend_accept    = */ nullptr,
    /* .backend_apply     = */ nullptr,
    /* .backend_set_input = */ nullptr,
};

struct llama_sampler * llama_sampler_init_rwkv_penalties(
        int32_t n_vocab,
        float   count_penalty,
        float   presence_penalty,
        float   penalty_decay) {
    if (n_vocab <= 0 || (count_penalty == 0.0f && presence_penalty == 0.0f)) {
        return llama_sampler_init_empty("?rwkv-penalties");
    }

    return llama_sampler_init(
        /* .iface = */ &llama_sampler_rwkv_penalties_i,
        /* .ctx   = */ new llama_sampler_rwkv_penalties {
            /* .n_vocab          = */ n_vocab,
            /* .count_penalty    = */ count_penalty,
            /* .presence_penalty = */ presence_penalty,
            /* .penalty_decay    = */ penalty_decay,
            /* .count            = */ std::vector<float>(n_vocab, 0.0f),
            /* .presence         = */ std::vector<float>(n_vocab, 0.0f),
            /* .touched          = */ {},
            /* .started          = */ false,
            /* .n_accepted       = */ 0,
        }
    );
}

// top-n-sigma

struct llama_sampler_top_n_sigma {
"""

sub(FILES["sampler_cpp"], IMPL_ANCHOR, IMPL_NEW, 1, "llama-sampler.cpp")

# --------------------------------------------------------------------------------------
# 3) params
# --------------------------------------------------------------------------------------
PARAMS_ANCHOR = "    float   penalty_present    = 0.00f;  // 0.0 = disabled\n"
PARAMS_NEW = PARAMS_ANCHOR + \
    "    float   rwkv_count_penalty    = 0.0f;  // 0.0 = disabled; RWKV count penalty (alpha_frequency)\n" \
    "    float   rwkv_presence_penalty = 0.0f;  // 0.0 = disabled; RWKV presence penalty (alpha_presence)\n" \
    "    float   rwkv_penalty_decay    = 1.0f;  // per-step decay of the RWKV count penalty (1.0 = no decay)\n"

sub(FILES["common_h"], PARAMS_ANCHOR, PARAMS_NEW, 1, "common.h params")

# --------------------------------------------------------------------------------------
# 4) chain construction (head of the chain: raw logits)
# --------------------------------------------------------------------------------------
CHAIN_ANCHOR = """    if (params.mirostat == 0) {

        bool use_adaptive_p = false; // see below

        for (const auto & cnstr : params.samplers) {
"""

CHAIN_NEW = """    if (params.mirostat == 0) {

        bool use_adaptive_p = false; // see below

        // RWKV penalties must see the raw logits, before top-k/top-p truncation
        if (params.rwkv_count_penalty != 0.0f || params.rwkv_presence_penalty != 0.0f) {
            samplers.push_back(llama_sampler_init_rwkv_penalties(
                    (int32_t) llama_vocab_n_tokens(vocab),
                    params.rwkv_count_penalty,
                    params.rwkv_presence_penalty,
                    params.rwkv_penalty_decay));
        }

        for (const auto & cnstr : params.samplers) {
"""

sub(FILES["sampling_cpp"], CHAIN_ANCHOR, CHAIN_NEW, 1, "sampling.cpp chain")

# --------------------------------------------------------------------------------------
# 5) server schema (OpenAI-compatible request fields)
# --------------------------------------------------------------------------------------
SCHEMA_ANCHOR = """    add((new field_num("presence_penalty", params.sampling.penalty_present))
        ->set_desc("Repeat alpha presence penalty (0 = disabled)"));
"""

SCHEMA_NEW = SCHEMA_ANCHOR + """
    add((new field_num("rwkv_count_penalty", params.sampling.rwkv_count_penalty))
        ->set_desc("RWKV count penalty (decayed per step), applied to generated tokens before top-k/top-p (0 = disabled)"));

    add((new field_num("rwkv_presence_penalty", params.sampling.rwkv_presence_penalty))
        ->set_desc("RWKV presence penalty, applied to generated tokens before top-k/top-p (0 = disabled)"));

    add((new field_num("rwkv_penalty_decay", params.sampling.rwkv_penalty_decay))
        ->set_desc("Per-step decay of the RWKV count penalty (1.0 = no decay)"));
"""

sub(FILES["schema_cpp"], SCHEMA_ANCHOR, SCHEMA_NEW, 1, "server-schema.cpp")

# --------------------------------------------------------------------------------------
# 6) server task param maps (2 places: chat completions + completions)
# --------------------------------------------------------------------------------------
TASK_ANCHOR = """            {"presence_penalty",          sampling.penalty_present},
            {"frequency_penalty",         sampling.penalty_freq},
"""

TASK_NEW = TASK_ANCHOR + \
    """            {"rwkv_count_penalty",        sampling.rwkv_count_penalty},
            {"rwkv_presence_penalty",     sampling.rwkv_presence_penalty},
            {"rwkv_penalty_decay",        sampling.rwkv_penalty_decay},
"""

sub(FILES["task_cpp"], TASK_ANCHOR, TASK_NEW, 2, "server-task.cpp maps")

print("OK: all patches applied")
