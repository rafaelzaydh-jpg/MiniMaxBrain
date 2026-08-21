#include "mmb_runtime.hpp"

#include <algorithm>
#include <climits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mmb {

MMBLlamaRuntime::MMBLlamaRuntime(
        const std::string & model_dir,
        uint64_t expert_cache_bytes,
        bool verify_sha256,
        uint32_t n_ctx,
        int32_t n_threads)
    : n_ctx_(n_ctx) {
    if (model_dir.empty()) {
        throw std::runtime_error("model_dir is empty");
    }
    if (expert_cache_bytes == 0) {
        throw std::runtime_error("expert_cache_bytes must be positive");
    }
    if (n_ctx < 128) {
        throw std::runtime_error("n_ctx must be >= 128");
    }
    if (n_threads <= 0) {
        throw std::runtime_error("n_threads must be positive");
    }

    llama_backend_init();
    backend_initialized_ = true;

    try {
        mmb_model_ = std::make_shared<MMBModel>(model_dir);

        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = 0;
        model_params.check_tensors = false;
        model_params.load_mode = LLAMA_LOAD_MODE_NONE;

        model_ = load_llama_model_from_mmb(*mmb_model_, verify_sha256, model_params);
        if (!model_) {
            throw std::runtime_error("MMB llama model load returned null");
        }

        pager_ = std::make_unique<MMBPager>(
            mmb_model_, expert_cache_bytes, verify_sha256);
        provider_ = std::make_unique<MMBGGMLProviderRegistration>(*pager_);

        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = n_ctx;
        ctx_params.n_batch = n_ctx;
        ctx_params.n_ubatch = std::min<uint32_t>(512u, n_ctx);
        ctx_params.n_threads = n_threads;
        ctx_params.n_threads_batch = n_threads;
        ctx_params.n_seq_max = 1;
        ctx_params.offload_kqv = false;

        ctx_ = llama_init_from_model(model_, ctx_params);
        if (!ctx_) {
            throw std::runtime_error("llama_init_from_model failed for MMB runtime");
        }

        vocab_ = llama_model_get_vocab(model_);
        if (!vocab_) {
            throw std::runtime_error("MMB model has no vocabulary");
        }

        if (!llama_model_chat_template(model_, nullptr)) {
            throw std::runtime_error(
                "MMB model has no default chat template; direct chat requires a model chat template");
        }
    } catch (...) {
        cleanup();
        throw;
    }
}

MMBLlamaRuntime::~MMBLlamaRuntime() {
    cleanup();
}

void MMBLlamaRuntime::cleanup() noexcept {
    if (ctx_) {
        llama_free(ctx_);
        ctx_ = nullptr;
    }

    // The provider owns a raw pager pointer. Unregister it before destroying
    // either the pager or the model.
    provider_.reset();
    pager_.reset();

    if (model_) {
        llama_model_free(model_);
        model_ = nullptr;
    }
    vocab_ = nullptr;
    mmb_model_.reset();

    if (backend_initialized_) {
        llama_backend_free();
        backend_initialized_ = false;
    }
}

std::string MMBLlamaRuntime::apply_chat_template(
        const std::vector<ChatMessage> & messages) const {
    if (messages.empty()) {
        throw std::runtime_error("messages must not be empty");
    }

    const char * tmpl = llama_model_chat_template(model_, nullptr);
    if (!tmpl) {
        throw std::runtime_error("model chat template is unavailable");
    }

    std::vector<llama_chat_message> chat;
    chat.reserve(messages.size());
    for (const auto & message : messages) {
        if (message.role != "system" &&
            message.role != "user" &&
            message.role != "assistant") {
            throw std::runtime_error("invalid chat role: " + message.role);
        }
        chat.push_back({message.role.c_str(), message.content.c_str()});
    }

    const int32_t required = llama_chat_apply_template(
        tmpl, chat.data(), chat.size(), true, nullptr, 0);
    if (required < 0) {
        throw std::runtime_error(
            "llama.cpp does not support the model's chat template in the direct runtime");
    }

    std::vector<char> buffer(static_cast<size_t>(required) + 1u, '\0');
    const int32_t written = llama_chat_apply_template(
        tmpl,
        chat.data(),
        chat.size(),
        true,
        buffer.data(),
        static_cast<int32_t>(buffer.size()));
    if (written < 0) {
        throw std::runtime_error("llama_chat_apply_template failed");
    }
    if (written > static_cast<int32_t>(buffer.size())) {
        throw std::runtime_error("chat template size changed between sizing and formatting");
    }

    return std::string(buffer.data(), static_cast<size_t>(written));
}

std::vector<llama_token> MMBLlamaRuntime::tokenize(const std::string & text) const {
    if (text.size() > static_cast<size_t>(INT32_MAX)) {
        throw std::runtime_error("formatted prompt is too large to tokenize");
    }

    int32_t count = llama_tokenize(
        vocab_,
        text.data(),
        static_cast<int32_t>(text.size()),
        nullptr,
        0,
        true,
        true);

    if (count == INT32_MIN) {
        throw std::runtime_error("formatted prompt is too large to tokenize");
    }
    if (count < 0) {
        count = -count;
    }

    std::vector<llama_token> tokens(static_cast<size_t>(count));
    count = llama_tokenize(
        vocab_,
        text.data(),
        static_cast<int32_t>(text.size()),
        tokens.data(),
        static_cast<int32_t>(tokens.size()),
        true,
        true);

    if (count < 0) {
        throw std::runtime_error("llama_tokenize failed");
    }
    tokens.resize(static_cast<size_t>(count));
    return tokens;
}

std::string MMBLlamaRuntime::token_piece(llama_token token) const {
    std::vector<char> buffer(64);
    int32_t size = llama_token_to_piece(
        vocab_,
        token,
        buffer.data(),
        static_cast<int32_t>(buffer.size()),
        0,
        true);

    if (size < 0) {
        buffer.resize(static_cast<size_t>(-size));
        size = llama_token_to_piece(
            vocab_,
            token,
            buffer.data(),
            static_cast<int32_t>(buffer.size()),
            0,
            false);
    }
    if (size < 0) {
        throw std::runtime_error("llama_token_to_piece failed");
    }

    return std::string(buffer.data(), static_cast<size_t>(size));
}

llama_sampler * MMBLlamaRuntime::create_sampler(
        const GenerationParams & params) const {
    if (params.temperature <= 0.0f) {
        llama_sampler * greedy = llama_sampler_init_greedy();
        if (!greedy) {
            throw std::runtime_error("llama_sampler_init_greedy failed");
        }
        return greedy;
    }

    llama_sampler * chain = llama_sampler_chain_init(
        llama_sampler_chain_default_params());
    if (!chain) {
        throw std::runtime_error("llama_sampler_chain_init failed");
    }

    try {
        if (params.top_k > 0) {
            llama_sampler_chain_add(chain, llama_sampler_init_top_k(params.top_k));
        }
        if (params.top_p < 1.0f) {
            llama_sampler_chain_add(chain, llama_sampler_init_top_p(params.top_p, 1));
        }
        llama_sampler_chain_add(chain, llama_sampler_init_temp(params.temperature));
        llama_sampler_chain_add(chain, llama_sampler_init_dist(params.seed));
        return chain;
    } catch (...) {
        llama_sampler_free(chain);
        throw;
    }
}

bool MMBLlamaRuntime::chat(
        const std::vector<ChatMessage> & messages,
        const GenerationParams & params,
        StreamCallback callback,
        void * user_data) {
    if (!callback) {
        throw std::runtime_error("stream callback is null");
    }
    if (params.max_tokens <= 0) {
        throw std::runtime_error("max_tokens must be positive");
    }
    if (params.temperature < 0.0f) {
        throw std::runtime_error("temperature must be >= 0");
    }
    if (params.top_p < 0.0f || params.top_p > 1.0f) {
        throw std::runtime_error("top_p must be in [0, 1]");
    }
    if (params.top_k < 0) {
        throw std::runtime_error("top_k must be >= 0");
    }

    std::lock_guard<std::mutex> lock(generation_mutex_);

    // Reuse the expensive model/context allocation, but rebuild the conversation
    // from the full message history on every request. This keeps the public chat
    // contract simple and avoids stale KV state across unrelated callers.
    llama_memory_clear(llama_get_memory(ctx_), true);

    const std::string prompt = apply_chat_template(messages);
    std::vector<llama_token> prompt_tokens = tokenize(prompt);
    if (prompt_tokens.empty()) {
        throw std::runtime_error("formatted chat tokenized to an empty sequence");
    }

    const uint64_t required_tokens =
        static_cast<uint64_t>(prompt_tokens.size()) +
        static_cast<uint64_t>(params.max_tokens);
    if (required_tokens > static_cast<uint64_t>(n_ctx_)) {
        throw std::runtime_error(
            "formatted chat + requested output exceeds context size");
    }

    llama_sampler * sampler = create_sampler(params);
    try {
        llama_batch batch = llama_batch_get_one(
            prompt_tokens.data(), static_cast<int32_t>(prompt_tokens.size()));
        const int32_t prompt_rc = llama_decode(ctx_, batch);
        if (prompt_rc != 0) {
            throw std::runtime_error(
                "llama_decode failed on chat prompt with code " +
                std::to_string(prompt_rc));
        }

        for (int32_t i = 0; i < params.max_tokens; ++i) {
            const llama_token token = llama_sampler_sample(sampler, ctx_, -1);
            llama_sampler_accept(sampler, token);

            if (llama_vocab_is_eog(vocab_, token)) {
                llama_sampler_free(sampler);
                return true;
            }

            const std::string piece = token_piece(token);
            if (!piece.empty() && !callback(piece.data(), piece.size(), user_data)) {
                llama_sampler_free(sampler);
                return false;
            }

            llama_token next = token;
            batch = llama_batch_get_one(&next, 1);
            const int32_t rc = llama_decode(ctx_, batch);
            if (rc != 0) {
                throw std::runtime_error(
                    "llama_decode failed during generation with code " +
                    std::to_string(rc));
            }
        }

        llama_sampler_free(sampler);
        return true;
    } catch (...) {
        llama_sampler_free(sampler);
        throw;
    }
}

MMBPagerSnapshot MMBLlamaRuntime::stats() const {
    if (!pager_) {
        return {};
    }
    return pager_->snapshot();
}

} // namespace mmb
