#include "server-classifier.h"

#include "common.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "log.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <regex>
#include <stdexcept>

// Mirrors tinyjev's pointer family (tinyjev/families/pointer.py) and System One answer shapes (tinyjev/agent.py),
// which follow Kev (Apache-2.0, Jared Palmer); verified against that Python implementation on OpenDecision cases.

#define CLS_INF(fmt, ...) LOG_INF("classifier: " fmt, __VA_ARGS__)

// ---------------------------------------------------------------------------------------------------------------
// text rendering: state / instructions / option descriptions may be strings, numbers, lists or objects

static std::string lstrip(const std::string & s) {
    size_t i = 0;
    while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\t' || s[i] == '\r')) { i++; }
    return s.substr(i);
}

static std::string render(const json & v, int indent = 0) {
    const std::string pad(2 * indent, ' ');
    if (v.is_null())    { return ""; }
    if (v.is_string())  { return v.get<std::string>(); }
    if (v.is_boolean()) { return v.get<bool>() ? "True" : "False"; }   // Python str(bool)
    if (v.is_number())  { return v.dump(); }
    std::string out;
    if (v.is_array()) {
        for (size_t i = 0; i < v.size(); i++) {
            if (i) { out += "\n"; }
            out += pad + "- " + lstrip(render(v.at(i), indent + 1));
        }
        return out;
    }
    bool first = true;
    for (const auto & [k, x] : v.items()) {
        if (!first) { out += "\n"; }
        first = false;
        if (x.is_object() || x.is_array()) {
            out += pad + k + ":\n" + render(x, indent + 1);
        } else {
            out += pad + k + ": " + render(x);
        }
    }
    return out;
}

static std::string replace_all(std::string s, const std::string & from, const std::string & to) {
    for (size_t p = 0; (p = s.find(from, p)) != std::string::npos; p += to.size()) { s.replace(p, from.size(), to); }
    return s;
}

// ---------------------------------------------------------------------------------------------------------------
// config + head loading

static void load_config(const std::string & path, classifier_config & c) {
    if (path.empty()) { return; }
    std::ifstream f(path, std::ios::binary);
    if (!f) { throw std::runtime_error("cannot open --classifier-config " + path); }
    std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const json j = json::parse(text);
    auto str = [&](const json & o, const char * k, std::string & dst) { if (o.contains(k) && o.at(k).is_string()) { dst = o.at(k).get<std::string>(); } };
    auto num = [&](const json & o, const char * k, auto & dst) { if (o.contains(k) && o.at(k).is_number()) { dst = o.at(k).get<typename std::decay<decltype(dst)>::type>(); } };
    str(j, "name", c.name);
    str(j, "family", c.family);
    num(j, "max_state", c.max_state);
    num(j, "max_branch", c.max_branch);
    num(j, "max_options", c.max_options);
    if (j.contains("head")) { num(j.at("head"), "head_dim", c.head_dim); num(j.at("head"), "temperature", c.temperature); }
    if (j.contains("layout")) {
        const json & l = j.at("layout");
        if (l.contains("tokens")) {
            const json & t = l.at("tokens");
            str(t, "state", c.tok_state); str(t, "question", c.tok_question); str(t, "option", c.tok_option);
            str(t, "close", c.tok_close); str(t, "decide", c.tok_decide);
        }
        if (l.contains("escape_special") && l.at("escape_special").is_boolean()) { c.escape_special = l.at("escape_special").get<bool>(); }
        str(l, "option_format", c.option_format);
        if (l.contains("boolean_labels")) { str(l.at("boolean_labels"), "false", c.label_false); str(l.at("boolean_labels"), "true", c.label_true); }
    }
    if (c.family != "pointer" && c.family != "kev") {
        throw std::runtime_error("unsupported classifier family '" + c.family + "' (supported: pointer)");
    }
}

// minimal safetensors reader: u64 header size, JSON header {name: {dtype, shape, data_offsets}}, raw little-endian data
static void load_head(const std::string & path, std::vector<float> & wq, std::vector<float> & bq,
                      std::vector<float> & wk, std::vector<float> & bk, int & head_dim, int n_embd) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { throw std::runtime_error("cannot open --classifier-head " + path); }
    uint64_t hlen = 0;
    f.read(reinterpret_cast<char *>(&hlen), 8);
    if (!f || hlen == 0 || hlen > (1u << 24)) { throw std::runtime_error("not a safetensors file: " + path); }
    std::string htext(hlen, '\0');
    f.read(htext.data(), (std::streamsize) hlen);
    const json header = json::parse(htext);
    const std::streamoff base = 8 + (std::streamoff) hlen;

    auto tensor = [&](const std::string & name, std::vector<int64_t> & shape) {
        if (!header.contains(name)) { throw std::runtime_error("classifier head is missing tensor '" + name + "'"); }
        const json & t = header.at(name);
        const std::string dtype = t.at("dtype").get<std::string>();
        shape.clear();
        int64_t n = 1;
        for (size_t i = 0; i < t.at("shape").size(); i++) { shape.push_back(t.at("shape").at(i).get<int64_t>()); n *= shape.back(); }
        const uint64_t a = t.at("data_offsets").at(0).get<uint64_t>(), b = t.at("data_offsets").at(1).get<uint64_t>();
        std::vector<uint8_t> raw(b - a);
        f.seekg(base + (std::streamoff) a);
        f.read(reinterpret_cast<char *>(raw.data()), (std::streamsize) raw.size());
        std::vector<float> out((size_t) n);
        if (dtype == "F32" && raw.size() == (size_t) n * 4) {
            std::memcpy(out.data(), raw.data(), raw.size());
        } else if (dtype == "F16" && raw.size() == (size_t) n * 2) {
            for (int64_t i = 0; i < n; i++) { ggml_fp16_t h; std::memcpy(&h, raw.data() + 2 * i, 2); out[i] = ggml_fp16_to_fp32(h); }
        } else if (dtype == "BF16" && raw.size() == (size_t) n * 2) {
            for (int64_t i = 0; i < n; i++) { uint32_t u = (uint32_t) (raw[2 * i] | (raw[2 * i + 1] << 8)) << 16; std::memcpy(&out[i], &u, 4); }
        } else {
            throw std::runtime_error("classifier head tensor '" + name + "': unsupported dtype " + dtype);
        }
        return out;
    };
    std::vector<int64_t> s_wq, s_bq, s_wk, s_bk;
    wq = tensor("q.weight", s_wq); bq = tensor("q.bias", s_bq);
    wk = tensor("k.weight", s_wk); bk = tensor("k.bias", s_bk);
    if (s_wq.size() != 2 || s_wq != s_wk || s_wq[1] != n_embd || s_bq.size() != 1 || s_bq[0] != s_wq[0] || s_bk != s_bq) {
        throw std::runtime_error(string_format("classifier head shapes do not match the backbone (n_embd %d): q.weight [%lld, %lld]",
                                               n_embd, (long long) s_wq[0], (long long) (s_wq.size() > 1 ? s_wq[1] : 0)));
    }
    if (head_dim == 0) { head_dim = (int) s_wq[0]; }
}

// ---------------------------------------------------------------------------------------------------------------
// graph capture: the post-norm hidden state of the output rows

static bool classifier_eval_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    const bool is_norm = std::strcmp(ggml_get_name(t), "result_norm") == 0;
    if (ask) { return is_norm; }
    if (!is_norm) { return true; }
    auto * self = static_cast<server_classifier *>(user_data);
    if (t->type != GGML_TYPE_F32) { return true; }
    const int64_t rows = ggml_nrows(t);
    const size_t off = self->captured.size();
    self->captured.resize(off + (size_t) (rows * t->ne[0]));
    ggml_backend_tensor_get(t, self->captured.data() + off, 0, ggml_nbytes(t));
    self->captured_rows += rows;
    return true;
}

// ---------------------------------------------------------------------------------------------------------------

bool server_classifier::init(llama_model * m, const common_params & params, std::string & err) {
    try {
        model = m;
        load_config(params.classifier_config, cfg);
        n_embd = llama_model_n_embd(model);
        load_head(params.classifier_head, wq, bq, wk, bk, cfg.head_dim, n_embd);

        const llama_vocab * vocab = llama_model_get_vocab(model);
        auto special = [&](const std::string & s) {
            auto t = common_tokenize(vocab, s, false, true);
            if (t.size() != 1) { throw std::runtime_error("delimiter '" + s + "' is not a single token of this model's vocabulary"); }
            return t[0];
        };
        id_state = special(cfg.tok_state); id_question = special(cfg.tok_question); id_option = special(cfg.tok_option);
        id_close = special(cfg.tok_close); id_decide = special(cfg.tok_decide);

        llama_context_params cp = common_context_params_to_llama(params);
        n_ctx = params.n_ctx > 0 ? params.n_ctx : 4096;   // model default (32k) would be a needlessly large KV cache
        cp.n_ctx      = n_ctx;
        cp.n_batch    = n_ctx;
        cp.n_ubatch   = std::min<uint32_t>(cp.n_ubatch, (uint32_t) n_ctx);
        cp.n_seq_max  = 1;
        cp.n_outputs_max         = 0;   // one output per option + <decide>; the server's generation caps do not apply
        cp.n_outputs_max_per_seq = 0;
        cp.embeddings = false;   // outputs only for the rows the head reads (see header)
        cp.cb_eval           = classifier_eval_cb;
        cp.cb_eval_user_data = this;
        lctx = llama_init_from_model(model, cp);
        if (!lctx) { throw std::runtime_error("failed to create the classifier context"); }
        CLS_INF("%s ready: family %s, head_dim %d, temperature %.4g, n_ctx %d, n_embd %d\n", cfg.name.c_str(), cfg.family.c_str(),
                cfg.head_dim, cfg.temperature, n_ctx, n_embd);
        return true;
    } catch (const std::exception & e) {
        err = e.what();
        return false;
    }
}

server_classifier::~server_classifier() {
    if (lctx) { llama_free(lctx); }
}

std::vector<llama_token> server_classifier::user_tokens(const std::string & text) const {
    std::string s = text;
    if (cfg.escape_special) {
        static const std::regex re("<\\|([A-Za-z0-9_]+)\\|>");
        s = std::regex_replace(s, re, "<\xC2\xA6$1\xC2\xA6>");   // "<|x|>" -> "<¦x¦>"
    }
    return common_tokenize(llama_model_get_vocab(model), s, false, false);
}

// decode `toks` as positions start.. of sequence 0 and return the captured hidden rows for the sorted positions `want`
std::vector<float> server_classifier::decode_chunk(const std::vector<llama_token> & toks, int start, const std::vector<int> & want) {
    llama_batch batch = llama_batch_init((int32_t) toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); i++) {
        const int pos = start + (int) i;
        common_batch_add(batch, toks[i], pos, { 0 }, std::binary_search(want.begin(), want.end(), pos));
    }
    captured.clear();
    captured_rows = 0;
    const int rc = llama_decode(lctx, batch);
    llama_batch_free(batch);
    if (rc != 0) { throw std::runtime_error(string_format("llama_decode failed (%d)", rc)); }
    if (captured_rows != (int64_t) want.size()) {
        throw std::runtime_error(string_format("captured %lld hidden rows, expected %zu", (long long) captured_rows, want.size()));
    }
    return captured;
}

json server_classifier::classify(const json & body) {
    const auto t0 = std::chrono::steady_clock::now();
    if (!body.is_object() || !body.contains("state") || !body.contains("questions")) {
        throw std::invalid_argument("request needs {\"state\", \"questions\"}");
    }
    const json & qs = body.at("questions");
    if (!qs.is_object() || qs.size() == 0) { throw std::invalid_argument("questions must be a non-empty object"); }

    // --- encode (pointer family) ---
    std::vector<llama_token> prefix = { id_state };
    { auto st = user_tokens(render(body.at("state"))); prefix.insert(prefix.end(), st.begin(), st.end()); }
    if ((int) prefix.size() > cfg.max_state) { throw std::invalid_argument(string_format("state exceeds %d tokens", cfg.max_state)); }

    auto option_text = [&](const std::string & name, const json & desc) {
        if (desc.is_null() || (desc.is_string() && desc.get<std::string>().empty())) { return name; }
        return replace_all(replace_all(cfg.option_format, "{name}", name), "{desc}", render(desc));
    };
    std::vector<encoded_question> enc;
    for (const auto & [qid, q] : qs.items()) {
        if (!q.is_object()) { throw std::invalid_argument("question '" + qid + "' must be an object"); }
        std::string type = q.contains("type") && q.at("type").is_string() ? q.at("type").get<std::string>() : "";
        if (type == "noul") { type = "boolean"; }
        encoded_question e;
        e.id = qid; e.type = type;
        const json crit = q.contains("criteria") ? q.at("criteria") : json();
        if (type == "boolean") {
            if (!crit.is_null() && !crit.is_object()) { throw std::invalid_argument(qid + ": boolean criteria may only hold false/true"); }
            e.keys = { "false", "true" };
            e.opts = { option_text(cfg.label_false, crit.is_object() && crit.contains("false") ? crit.at("false") : json()),
                       option_text(cfg.label_true,  crit.is_object() && crit.contains("true")  ? crit.at("true")  : json()) };
        } else if (type == "choice") {
            if (!crit.is_object() || crit.size() < 1 || (int) crit.size() > cfg.max_options) {
                throw std::invalid_argument(string_format("%s: choice criteria must hold 1..%d options", qid.c_str(), cfg.max_options));
            }
            for (const auto & [k, v] : crit.items()) { e.keys.push_back(k); e.opts.push_back(option_text(k, v)); }
        } else if (type == "score") {
            if (!crit.is_array() || crit.size() < 2 || (int) crit.size() > cfg.max_options) {
                throw std::invalid_argument(string_format("%s: score criteria must be a list of 2..%d levels", qid.c_str(), cfg.max_options));
            }
            for (size_t i = 0; i < crit.size(); i++) { e.keys.push_back(std::to_string(i)); e.opts.push_back(render(crit.at(i))); }
        } else {
            throw std::invalid_argument(qid + ": unsupported question type '" + type + "' (choice, noul, score)");
        }
        e.row.push_back(id_question);
        { auto it = user_tokens(render(q.contains("instructions") ? q.at("instructions") : json())); e.row.insert(e.row.end(), it.begin(), it.end()); }
        for (const auto & o : e.opts) {
            e.row.push_back(id_option);
            auto ot = user_tokens(o);
            e.row.insert(e.row.end(), ot.begin(), ot.end());
            e.row.push_back(id_close);
            e.close_pos.push_back((int) (prefix.size() + e.row.size() - 1));
        }
        e.row.push_back(id_decide);
        e.decide = (int) (prefix.size() + e.row.size() - 1);
        if ((int) e.row.size() > cfg.max_branch - (int) prefix.size()) {
            throw std::invalid_argument(string_format("%s: question row too long (%zu tokens)", qid.c_str(), e.row.size()));
        }
        if ((int) (prefix.size() + e.row.size()) > n_ctx) {
            throw std::invalid_argument(string_format("%s: state + question is %zu tokens, over the classifier context of %d",
                                                      qid.c_str(), prefix.size() + e.row.size(), n_ctx));
        }
        enc.push_back(std::move(e));
    }

    // --- backbone: one pass for a single question; otherwise the state prefix once, then each row continues from it ---
    json answers = json::object();
    std::lock_guard<std::mutex> lock(mtx);
    llama_memory_t mem = llama_get_memory(lctx);
    llama_memory_clear(mem, true);
    const bool shared = enc.size() > 1;
    if (shared) {
        decode_chunk(prefix, 0, { (int) prefix.size() - 1 });   // one output keeps llama_decode on its ordinary path
    }
    const float scale = 1.0f / std::sqrt((float) cfg.head_dim) / (float) cfg.temperature;
    for (auto & e : enc) {
        std::vector<int> want = e.close_pos;
        want.push_back(e.decide);
        std::sort(want.begin(), want.end());
        std::vector<float> h;
        if (shared) {
            llama_memory_seq_rm(mem, 0, (llama_pos) prefix.size(), -1);
            h = decode_chunk(e.row, (int) prefix.size(), want);
        } else {
            std::vector<llama_token> all = prefix;
            all.insert(all.end(), e.row.begin(), e.row.end());
            h = decode_chunk(all, 0, want);
        }
        auto row_of = [&](int pos) { return h.data() + (size_t) (std::lower_bound(want.begin(), want.end(), pos) - want.begin()) * n_embd; };

        // --- head: query from <decide>, one key per <close> ---
        const int dp = cfg.head_dim;
        std::vector<float> qv(dp);
        const float * hd = row_of(e.decide);
        for (int d = 0; d < dp; d++) {
            double acc = bq[d];
            const float * w = wq.data() + (size_t) d * n_embd;
            for (int k = 0; k < n_embd; k++) { acc += (double) w[k] * hd[k]; }
            qv[d] = (float) acc;
        }
        std::vector<double> z(e.close_pos.size());
        for (size_t i = 0; i < e.close_pos.size(); i++) {
            const float * hc = row_of(e.close_pos[i]);
            double dot = 0;
            for (int d = 0; d < dp; d++) {
                double acc = bk[d];
                const float * w = wk.data() + (size_t) d * n_embd;
                for (int k = 0; k < n_embd; k++) { acc += (double) w[k] * hc[k]; }
                dot += acc * qv[d];
            }
            z[i] = dot * scale;
        }
        const double zmax = *std::max_element(z.begin(), z.end());
        double sum = 0;
        for (auto & v : z) { v = std::exp(v - zmax); sum += v; }
        std::vector<double> p(z.size());
        for (size_t i = 0; i < z.size(); i++) { p[i] = z[i] / sum; }

        // --- answer, System One shape ---
        auto r2 = [](double x) { return std::round(x * 100.0) / 100.0; };
        auto r4 = [](double x) { return std::round(x * 10000.0) / 10000.0; };
        json probs = json::object();
        for (size_t i = 0; i < p.size(); i++) { probs[e.keys[i]] = p[i]; }
        json a = json::object();
        if (e.type == "boolean") {
            a["type"] = "noul";
            a["noul"] = r4(p[1]);
        } else if (e.type == "choice") {
            const size_t best = (size_t) (std::max_element(p.begin(), p.end()) - p.begin());
            const double K = (double) p.size();
            a["type"] = "choice";
            a["choice"] = e.keys[best];
            a["confidence"] = p.size() == 1 ? 1.0 : r2((p[best] - 1.0 / K) / (1.0 - 1.0 / K));
            a["probabilities"] = probs;
        } else {
            const size_t mode = (size_t) (std::max_element(p.begin(), p.end()) - p.begin());
            double score = 0, spread = 0;
            for (size_t i = 0; i < p.size(); i++) { score += (double) i * p[i]; spread += p[i] * std::fabs((double) i - (double) mode); }
            json legend = json::object();
            for (size_t i = 0; i < e.opts.size(); i++) { legend[e.keys[i]] = e.opts[i]; }
            a["type"] = "score";
            a["score"] = r4(score);
            a["legend"] = legend;
            a["probabilities"] = probs;
            a["confidence"] = r2(1.0 - spread / (double) (p.size() - 1));
        }
        answers[e.id] = a;
    }

    json res = json::object();
    res["model"] = body.contains("model") && body.at("model").is_string() ? body.at("model").get<std::string>() : cfg.name;
    res["answers"] = answers;
    res["latency_ms"] = std::round(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() * 100.0) / 100.0;
    return res;
}
