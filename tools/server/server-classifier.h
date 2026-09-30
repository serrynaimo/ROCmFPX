#pragma once

// Jev-style decision classifier ("pointer" family, e.g. TinyJev / Kev): typed questions about a text answered with
// calibrated probabilities in the TypeSafe System One shape, without generating text.
//
// The backbone is the model llama-server already loaded (-m, any backend: CPU with -ngl 0 or a GPU). On top of it:
//   --classifier-head   FILE  the decision head, a safetensors file with q.weight/q.bias/k.weight/k.bias
//   --classifier-config FILE  JSON with everything that is prompt layout rather than weights: the delimiter tokens,
//                             how options and yes/no labels are written, limits, temperature. A tinyjev.json manifest
//                             works as-is; unset keys take the pointer-family defaults (see classifier_config below).
//
// Per question the prompt is  <state> text  <question> instructions  (<option> text <close>)*  <decide>  and the
// head scores the final normed hidden state at <decide> (query) against each <close> (keys):
//     z_i = (W_k h_close_i + b_k) . (W_q h_decide + b_q) / sqrt(head_dim) / temperature
// Only those rows are requested as outputs; the post-norm hidden state is captured from the graph ("result_norm")
// with embeddings mode off, because embeddings mode would force every token through the last layer and lm_head.

#include "server-common.h"

#include <memory>
#include <mutex>
#include <string>
#include <vector>

struct llama_model;
struct llama_context;
struct common_params;

struct classifier_config {
    std::string name        = "classifier";
    std::string family      = "pointer";
    int         head_dim    = 0;       // 0 = from the head weights
    double      temperature = 1.0;     // divides the head logits (the model's fitted temperature)
    int         max_state   = 8192;    // tokens of state (incl. the <state> delimiter)
    int         max_branch  = 8192;    // tokens of one question row
    int         max_options = 255;
    // delimiter tokens (must each be a single token of the backbone's vocabulary)
    std::string tok_state    = "<|fim_prefix|>";
    std::string tok_question = "<|fim_middle|>";
    std::string tok_option   = "<|box_start|>";
    std::string tok_close    = "<|box_end|>";
    std::string tok_decide   = "<|fim_suffix|>";
    // user text can never forge a delimiter: "<|name|>" is rewritten to "<¦name¦>" before tokenizing
    bool        escape_special = true;
    // how a choice/boolean option is written when it has a description ("{name}" / "{desc}" placeholders)
    std::string option_format = "{name}: {desc}";
    std::string label_false   = "no";
    std::string label_true    = "yes";
};

struct server_classifier {
    // loads config + head and creates a private llama_context on `model` (shares the weights, own small KV cache)
    // returns false and sets `err` on failure
    bool init(llama_model * model, const common_params & params, std::string & err);
    ~server_classifier();

    // System One request {state, questions, model?} -> {model, answers, latency_ms}
    // throws std::invalid_argument for bad requests (-> 422); thread-safe (requests are serialized)
    json classify(const json & body);

    const classifier_config & config() const { return cfg; }

private:
    struct encoded_question {
        std::string id, type;                 // type: boolean | choice | score
        std::vector<std::string> keys, opts;  // answer keys, option texts
        std::vector<llama_token> row;         // question row (after the shared state prefix)
        int decide = 0;                       // absolute position of <decide>
        std::vector<int> close_pos;           // absolute positions of each <close>
    };

    std::vector<llama_token> user_tokens(const std::string & text) const;
    std::vector<float> hidden_rows(const std::vector<llama_token> & prefix, const std::vector<encoded_question> & qs,
                                   std::vector<std::vector<int>> & out_positions);
    std::vector<float> decode_chunk(const std::vector<llama_token> & toks, int start, const std::vector<int> & want);

    classifier_config cfg;
    llama_model   * model = nullptr;
    llama_context * lctx  = nullptr;
    int n_embd = 0, n_ctx = 0;
    llama_token id_state = -1, id_question = -1, id_option = -1, id_close = -1, id_decide = -1;
    std::vector<float> wq, bq, wk, bk;  // [head_dim, n_embd] row-major, [head_dim]
    std::mutex mtx;

public:
    // graph callback state (filled by the eval callback during llama_decode)
    std::vector<float> captured;
    int64_t captured_rows = 0;
};
