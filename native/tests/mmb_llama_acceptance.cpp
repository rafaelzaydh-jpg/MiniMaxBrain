#include "mmb_ggml_bridge.hpp"
#include "mmb_model.hpp"
#include "mmb_llama_loader.hpp"
#include "mmb_pager.hpp"

#include "llama.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

struct Args {
    std::string gguf;
    std::string bundle;
    std::string prompt;
    uint64_t cache_bytes = 4ull * 1024 * 1024 * 1024;
    int32_t n_ctx = 2048;
    int32_t n_threads = std::max(1u, std::thread::hardware_concurrency());
    int32_t n_tokens = 32;
    bool verify_sha256 = true;
};

uint64_t parse_u64(const std::string & value, const char * name) {
    size_t pos = 0;
    unsigned long long parsed = 0;
    try {
        parsed = std::stoull(value, &pos, 10);
    } catch (...) {
        throw std::runtime_error(std::string("invalid ") + name + ": " + value);
    }
    if (pos != value.size()) {
        throw std::runtime_error(std::string("invalid ") + name + ": " + value);
    }
    return static_cast<uint64_t>(parsed);
}

int32_t parse_i32(const std::string & value, const char * name) {
    const uint64_t parsed = parse_u64(value, name);
    if (parsed > static_cast<uint64_t>(INT32_MAX)) {
        throw std::runtime_error(std::string(name) + " is too large");
    }
    return static_cast<int32_t>(parsed);
}

Args parse_args(int argc, char ** argv) {
    Args out;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&](const char * name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return argv[++i];
        };

        if (arg == "--gguf") out.gguf = next("--gguf");
        else if (arg == "--bundle") out.bundle = next("--bundle");
        else if (arg == "--prompt") out.prompt = next("--prompt");
        else if (arg == "--cache-bytes") out.cache_bytes = parse_u64(next("--cache-bytes"), "cache bytes");
        else if (arg == "--ctx") out.n_ctx = parse_i32(next("--ctx"), "context size");
        else if (arg == "--threads") out.n_threads = parse_i32(next("--threads"), "threads");
        else if (arg == "--tokens") out.n_tokens = parse_i32(next("--tokens"), "token count");
        else if (arg == "--no-verify") out.verify_sha256 = false;
        else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: mmb_llama_acceptance --gguf MODEL.gguf --bundle MODEL_MMB "
                << "[--prompt TEXT] [--tokens 32] [--cache-bytes N] [--ctx 2048] [--threads N]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (out.gguf.empty()) throw std::runtime_error("--gguf is required");
    if (out.bundle.empty()) throw std::runtime_error("--bundle is required");
    if (out.prompt.empty()) out.prompt = "Hello";
    if (out.cache_bytes == 0) throw std::runtime_error("--cache-bytes must be positive");
    if (out.n_ctx <= 0 || out.n_threads <= 0 || out.n_tokens <= 0) {
        throw std::runtime_error("ctx, threads and tokens must be positive");
    }
    return out;
}

std::vector<llama_token> tokenize(
        const llama_vocab * vocab,
        const std::string & text) {
    int32_t count = llama_tokenize(
        vocab,
        text.data(),
        static_cast<int32_t>(text.size()),
        nullptr,
        0,
        true,
        true);

    if (count == INT32_MIN) {
        throw std::runtime_error("prompt is too large to tokenize");
    }

    if (count < 0) count = -count;
    std::vector<llama_token> tokens(static_cast<size_t>(count));

    count = llama_tokenize(
        vocab,
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

std::string token_piece(const llama_vocab * vocab, llama_token token) {
    std::vector<char> buf(64);
    int32_t size = llama_token_to_piece(
        vocab, token, buf.data(), static_cast<int32_t>(buf.size()), 0, true);
    if (size < 0) {
        buf.resize(static_cast<size_t>(-size));
        size = llama_token_to_piece(
            vocab, token, buf.data(), static_cast<int32_t>(buf.size()), 0, true);
    }
    if (size < 0) {
        throw std::runtime_error("llama_token_to_piece failed");
    }
    return std::string(buf.data(), static_cast<size_t>(size));
}

struct Generation {
    std::vector<llama_token> tokens;
    std::string text;
};

Generation generate(
        llama_model * model,
        const Args & args) {
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = static_cast<uint32_t>(args.n_ctx);
    ctx_params.n_batch = static_cast<uint32_t>(args.n_ctx);
    ctx_params.n_ubatch = std::min<uint32_t>(512, static_cast<uint32_t>(args.n_ctx));
    ctx_params.n_threads = args.n_threads;
    ctx_params.n_threads_batch = args.n_threads;
    ctx_params.n_seq_max = 1;
    ctx_params.offload_kqv = false;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        throw std::runtime_error("llama_init_from_model failed");
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    if (!vocab) {
        llama_free(ctx);
        throw std::runtime_error("model has no vocabulary");
    }

    llama_sampler * sampler = llama_sampler_init_greedy();
    if (!sampler) {
        llama_free(ctx);
        throw std::runtime_error("llama_sampler_init_greedy failed");
    }

    try {
        std::vector<llama_token> prompt_tokens = tokenize(vocab, args.prompt);
        if (prompt_tokens.empty()) {
            throw std::runtime_error("prompt tokenized to an empty sequence");
        }
        if (prompt_tokens.size() + static_cast<size_t>(args.n_tokens) >
            static_cast<size_t>(args.n_ctx)) {
            throw std::runtime_error("prompt + requested output exceeds context size");
        }

        llama_batch batch = llama_batch_get_one(
            prompt_tokens.data(), static_cast<int32_t>(prompt_tokens.size()));
        const int32_t prompt_rc = llama_decode(ctx, batch);
        if (prompt_rc != 0) {
            throw std::runtime_error("llama_decode failed on prompt with code " + std::to_string(prompt_rc));
        }

        Generation out;
        out.tokens.reserve(static_cast<size_t>(args.n_tokens));

        for (int32_t i = 0; i < args.n_tokens; ++i) {
            const llama_token token = llama_sampler_sample(sampler, ctx, -1);
            out.tokens.push_back(token);
            out.text += token_piece(vocab, token);

            if (llama_vocab_is_eog(vocab, token)) {
                break;
            }

            llama_token next = token;
            batch = llama_batch_get_one(&next, 1);
            const int32_t rc = llama_decode(ctx, batch);
            if (rc != 0) {
                throw std::runtime_error("llama_decode failed during generation with code " + std::to_string(rc));
            }
        }

        llama_sampler_free(sampler);
        llama_free(ctx);
        return out;
    } catch (...) {
        llama_sampler_free(sampler);
        llama_free(ctx);
        throw;
    }
}

std::string token_list(const std::vector<llama_token> & tokens) {
    std::ostringstream out;
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i) out << ',';
        out << tokens[i];
    }
    return out.str();
}

} // namespace

int main(int argc, char ** argv) {
    try {
        const Args args = parse_args(argc, argv);

        llama_backend_init();

        llama_model_params baseline_params = llama_model_default_params();
        baseline_params.n_gpu_layers = 0;
        baseline_params.check_tensors = true;
        baseline_params.load_mode = LLAMA_LOAD_MODE_MMAP;

        llama_model * baseline_model = llama_model_load_from_file(args.gguf.c_str(), baseline_params);
        if (!baseline_model) {
            llama_backend_free();
            throw std::runtime_error("llama_model_load_from_file failed");
        }

        Generation baseline;
        Generation paged;
        llama_model * paged_model = nullptr;

        try {
            baseline = generate(baseline_model, args);
            const int32_t baseline_layer_count = llama_model_n_layer(baseline_model);

            // The baseline exists only to produce reference token IDs. Release it
            // before constructing the MMB model so the acceptance run never keeps
            // two large model mappings/allocations alive at the same time.
            llama_model_free(baseline_model);
            baseline_model = nullptr;

            auto mmb_model = std::make_shared<mmb::MMBModel>(args.bundle);
            if (mmb_model->layer_count() != static_cast<uint32_t>(baseline_layer_count)) {
                throw std::runtime_error("MMB layer count does not match GGUF model");
            }

            llama_model_params paged_params = llama_model_default_params();
            paged_params.n_gpu_layers = 0;
            paged_params.check_tensors = false;
            paged_params.load_mode = LLAMA_LOAD_MODE_NONE;

            // Unlike the previous acceptance path, this model is constructed
            // only from model.mmb-meta.gguf + MMBW core blocks. The original
            // GGUF is not a data source for the paged generation.
            paged_model = mmb::load_llama_model_from_mmb(
                *mmb_model, args.verify_sha256, paged_params);

            if (llama_model_n_layer(paged_model) != baseline_layer_count) {
                throw std::runtime_error("MMB-loaded llama model layer count differs from baseline");
            }

            mmb::MMBPager pager(mmb_model, args.cache_bytes, args.verify_sha256);
            {
                mmb::MMBGGMLProviderRegistration registration(pager);
                paged = generate(paged_model, args);
            }

            const auto stats = pager.snapshot();
            const bool parity = baseline.tokens == paged.tokens;
            const bool used = stats.pager.paged_kernel_invocations > 0;

            std::cout << "baseline_tokens=" << token_list(baseline.tokens) << "\n";
            std::cout << "mmb_tokens=" << token_list(paged.tokens) << "\n";
            std::cout << "token_parity=" << (parity ? "true" : "false") << "\n";
            std::cout << "paged_kernel_used=" << (used ? "true" : "false") << "\n";
            std::cout << "paged_kernel_invocations=" << stats.pager.paged_kernel_invocations << "\n";
            std::cout << "router_requests=" << stats.pager.real_router_requests << "\n";
            std::cout << "cache_hits=" << stats.pager.hits << "\n";
            std::cout << "cache_misses=" << stats.pager.misses << "\n";
            std::cout << "bytes_read=" << stats.pager.bytes_read << "\n";
            std::cout << "resident_bytes=" << stats.cache.resident_bytes << "\n";
            std::cout << "peak_resident_bytes=" << stats.cache.peak_resident_bytes << "\n";
            std::cout << "mmb_source=metadata+core_mmbw+expert_mmbw\n";
            std::cout << "baseline_text_begin\n" << baseline.text << "\nbaseline_text_end\n";
            std::cout << "mmb_text_begin\n" << paged.text << "\nmmb_text_end\n";

            llama_model_free(paged_model);
            paged_model = nullptr;
            llama_backend_free();

            if (!used) {
                std::cerr << "MMB acceptance failed: no MMB-backed MUL_MAT_ID kernel executed\n";
                return 3;
            }
            if (!parity) {
                std::cerr << "MMB acceptance failed: greedy token IDs diverged from GGUF baseline\n";
                return 4;
            }
            return 0;
        } catch (...) {
            if (paged_model) {
                llama_model_free(paged_model);
            }
            if (baseline_model) {
                llama_model_free(baseline_model);
            }
            llama_backend_free();
            throw;
        }
    } catch (const std::exception & exc) {
        std::cerr << "mmb_llama_acceptance: " << exc.what() << "\n";
        return 2;
    }
}
