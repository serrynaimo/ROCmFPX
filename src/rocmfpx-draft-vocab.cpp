// SPDX-License-Identifier: MIT
#include "rocmfpx-draft-vocab.h"
#include "rocmfpx-draft-select.h"
#include "llama-graph.h"
#include "llama-model.h"
#include "ggml-backend.h"
#include <cmath>
#include <cstdlib>
#include <mutex>
#include <numeric>
#include <unordered_map>

namespace rocmfpx {
struct draft_vocabulary::storage {
    const int vocabulary;
    const int budget;
    mutable std::mutex mutex;
    bool driver_claimed = false;
    draft_selector selector;
    // [TAG_DRAFT_VOCAB_SEQ] the candidates of one sequence; a sequence that was never updated has the first rows
    struct candidates {
        std::vector<int32_t> projection;
        std::vector<int64_t> scatter;
    };
    mutable std::vector<candidates> seqs;
    storage(int n, int k) : vocabulary(n), budget(k) {}
    candidates & of(llama_seq_id seq) const {
        const size_t i = seq < 0 ? 0 : (size_t) seq;
        while (seqs.size() <= i) {
            candidates c;
            c.projection.resize(budget);
            c.scatter.resize(budget);
            std::iota(c.projection.begin(), c.projection.end(), 0);
            std::iota(c.scatter.begin(), c.scatter.end(), 0);
            seqs.push_back(std::move(c));
        }
        return seqs[i];
    }
};

draft_vocabulary::draft_vocabulary(int n, int k) {
    if (n <= 0 || k <= 0 || k > n) throw std::invalid_argument("invalid draft vocabulary dimensions");
    data_ = std::make_unique<storage>(n, k);
}
draft_vocabulary::~draft_vocabulary() = default;
int draft_vocabulary::size() const { return data_->budget; }
int draft_vocabulary::vocabulary_size() const { return data_->vocabulary; }
std::shared_ptr<void> draft_vocabulary::claim_driver() {
    struct lease_state {
        std::shared_ptr<draft_vocabulary> owner;
        bool active = false;
        explicit lease_state(std::shared_ptr<draft_vocabulary> value) : owner(std::move(value)) {}
        ~lease_state() {
            if (active) {
                std::lock_guard<std::mutex> guard(owner->data_->mutex);
                owner->data_->driver_claimed = false;
            }
        }
    };
    // Allocate before locking: even allocation failure must not run a deleter
    // that tries to acquire a mutex already held by this thread.
    auto lease = std::make_shared<lease_state>(shared_from_this());
    std::lock_guard<std::mutex> lock(data_->mutex);
    if (data_->driver_claimed) throw std::runtime_error("native draft vocabulary supports one driver per model");
    data_->driver_claimed = true;
    lease->active = true;
    return lease;
}
void draft_vocabulary::update(const float * scores, int n, llama_token previous, llama_seq_id seq) {
    if (!scores || n != data_->vocabulary) throw std::invalid_argument("draft vocabulary score shape mismatch");
    std::lock_guard<std::mutex> lock(data_->mutex);
    auto & cand = data_->of(seq);
    cand.projection = data_->selector.select(scores, n, data_->budget, previous);
    cand.scatter.assign(cand.projection.begin(), cand.projection.end());
}
bool draft_vocabulary::unique_max(const float * scores, llama_token & token, llama_seq_id seq) const {
    if (!scores) return false;
    std::lock_guard<std::mutex> lock(data_->mutex);
    const auto & projection = data_->of(seq).projection;
    int best = projection.front();
    bool single = true;
    if (std::isnan(scores[best])) return false;
    for (size_t i = 1; i < projection.size(); ++i) {
        const int id = projection[i];
        if (std::isnan(scores[id])) return false;
        if (scores[id] > scores[best]) { best = id; single = true; }
        else if (scores[id] == scores[best]) single = false;
    }
    if (!single || !std::isfinite(scores[best])) return false;
    token = best;
    return true;
}
void draft_vocabulary::upload(ggml_tensor * projection, ggml_tensor * scatter, llama_seq_id seq) const {
    std::lock_guard<std::mutex> lock(data_->mutex);
    const auto & cand = data_->of(seq);
    ggml_backend_tensor_set(projection, cand.projection.data(), 0, cand.projection.size() * sizeof(int32_t));
    ggml_backend_tensor_set(scatter, cand.scatter.data(), 0, cand.scatter.size() * sizeof(int64_t));
}
std::shared_ptr<draft_vocabulary> draft_vocabulary_for(const llama_model * model) {
    // [TAG_DRAFT_VOCAB_SEQ] the Qwen3.x MTP head projects through the same full LM head
    if (!model || (model->arch != LLM_ARCH_QWEN4EXP && model->arch != LLM_ARCH_QWEN35 && model->arch != LLM_ARCH_QWEN35MOE)) return {};
    const char * setting = std::getenv("ROCMFPX_DRAFT_VOCAB");
    if (!setting || !*setting) return {};
    char * end = nullptr;
    const long budget = std::strtol(setting, &end, 10);
    const int vocabulary = model->vocab.n_tokens();
    if (*end || budget < 1 || budget > vocabulary) {
        throw std::invalid_argument("ROCMFPX_DRAFT_VOCAB must be between 1 and the vocabulary size");
    }
    // Weak entries do not extend a model's lifetime or retain stale candidate
    // buffers when a model is unloaded. A driver and its graphs are the owners.
    static std::mutex mutex;
    static std::unordered_map<const llama_model *, std::weak_ptr<draft_vocabulary>> registry;
    std::lock_guard<std::mutex> lock(mutex);
    for (auto it = registry.begin(); it != registry.end();) {
        if (it->second.expired()) it = registry.erase(it);
        else ++it;
    }
    auto state = registry[model].lock();
    if (!state) {
        state = std::make_shared<draft_vocabulary>(vocabulary, int(budget));
        registry[model] = state;
    } else if (state->size() != budget) {
        throw std::invalid_argument("cannot change draft vocabulary budget while graphs are alive");
    }
    return state;
}

namespace {
class vocabulary_input final : public llm_graph_input_i {
public:
    vocabulary_input(ggml_context * ctx, std::shared_ptr<draft_vocabulary> owner) : owner_(std::move(owner)) {
        rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, owner_->size());
        destinations = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, owner_->size());
        ggml_set_name(rows, "rocmfpx_draft_rows");
        ggml_set_name(destinations, "rocmfpx_draft_destinations");
        ggml_set_input(rows);
        ggml_set_input(destinations);
    }
    // the graph has one row, so one sequence: its candidates ([TAG_DRAFT_VOCAB_SEQ])
    void set_input(const llama_ubatch * ubatch) override {
        const llama_seq_id seq = ubatch && ubatch->n_tokens > 0 && ubatch->seq_id && ubatch->seq_id[0] ? ubatch->seq_id[0][0] : 0;
        owner_->upload(rows, destinations, seq);
    }
    bool can_reuse(const llm_graph_params & params) override { return params.ubatch.n_tokens == 1; }
    ggml_tensor * rows;
    ggml_tensor * destinations;
private:
    std::shared_ptr<draft_vocabulary> owner_;
};
}

draft_projection build_draft_projection(ggml_context * ctx,
        std::shared_ptr<draft_vocabulary> vocabulary, ggml_tensor * weights, ggml_tensor * hidden) {
    if (!vocabulary || weights->ne[1] != vocabulary->vocabulary_size() ||
        weights->ne[0] != hidden->ne[0] || !ggml_is_matrix(weights) ||
        ggml_nrows(hidden) != 1 || !ggml_is_contiguous(weights) || !ggml_is_contiguous(hidden)) {
        throw std::invalid_argument("unsupported draft projection dimensions");
    }
    auto input = std::make_unique<vocabulary_input>(ctx, vocabulary);
    // ggml's indirect matmul treats each vocabulary row as a 1-row matrix.
    // Its output is already [1, budget, 1], the source shape for row scattering.
    auto * matrices = ggml_reshape_3d(ctx, weights, weights->ne[0], 1, weights->ne[1]);
    auto * values = ggml_mul_mat_id(ctx, matrices, hidden, input->rows);
    auto * masked = ggml_fill(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, weights->ne[1]), -INFINITY);
    auto * expanded = ggml_set_rows(ctx, masked, values, input->destinations);
    auto * logits = ggml_reshape_1d(ctx, expanded, weights->ne[1]);
    ggml_set_name(logits, "rocmfpx_draft_logits");
    return {logits, std::move(input)};
}
}
