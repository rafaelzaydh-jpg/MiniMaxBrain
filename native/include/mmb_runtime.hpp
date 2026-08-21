#pragma once

#include "mmb_ggml_bridge.hpp"
#include "mmb_llama_loader.hpp"
#include "mmb_pager.hpp"

#include "llama.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace mmb {

struct ChatMessage {
    std::string role;
    std::string content;
};

struct GenerationParams {
    int32_t max_tokens = 128;
    float temperature = 0.7f;
    float top_p = 0.9f;
    int32_t top_k = 40;
    uint32_t seed = LLAMA_DEFAULT_SEED;
};

using StreamCallback = bool (*)(const char * data, size_t bytes, void * user_data);

class MMBLlamaRuntime {
public:
    MMBLlamaRuntime(
        const std::string & model_dir,
        uint64_t expert_cache_bytes,
        bool verify_sha256,
        uint32_t n_ctx,
        int32_t n_threads);
    ~MMBLlamaRuntime();

    MMBLlamaRuntime(const MMBLlamaRuntime &) = delete;
    MMBLlamaRuntime & operator=(const MMBLlamaRuntime &) = delete;

    // Returns false only when the callback asked to cancel generation.
    bool chat(
        const std::vector<ChatMessage> & messages,
        const GenerationParams & params,
        StreamCallback callback,
        void * user_data);

    MMBPagerSnapshot stats() const;

private:
    void cleanup() noexcept;
    std::string apply_chat_template(const std::vector<ChatMessage> & messages) const;
    std::vector<llama_token> tokenize(const std::string & text) const;
    std::string token_piece(llama_token token) const;
    llama_sampler * create_sampler(const GenerationParams & params) const;

    std::shared_ptr<MMBModel> mmb_model_;
    std::unique_ptr<MMBPager> pager_;
    std::unique_ptr<MMBGGMLProviderRegistration> provider_;
    llama_model * model_ = nullptr;
    llama_context * ctx_ = nullptr;
    const llama_vocab * vocab_ = nullptr;
    bool backend_initialized_ = false;
    uint32_t n_ctx_ = 0;
    mutable std::mutex generation_mutex_;
};

} // namespace mmb
