#include "mmb_llama_loader.hpp"

#include "ggml-backend.h"
#include "ggml-moe-provider.h"
#include "gguf.h"

#include <array>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace mmb {
namespace {

bool ends_with(std::string_view value, std::string_view suffix) {
    return value.size() >= suffix.size() &&
           value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool is_routed_expert_tensor(std::string_view name) {
    return ends_with(name, ".ffn_down_exps.weight") ||
           ends_with(name, ".ffn_gate_exps.weight") ||
           ends_with(name, ".ffn_up_exps.weight");
}

bool is_any_expert_tensor(std::string_view name) {
    return name.find("_exps") != std::string_view::npos;
}

struct LoadState {
    const MMBModel * model = nullptr;
    bool verify_sha256 = true;
    std::string error;
};

void set_error(LoadState & state, const std::string & message) {
    if (state.error.empty()) {
        state.error = message;
    }
}

void set_tensor_data(ggml_tensor * tensor, void * userdata) {
    auto & state = *static_cast<LoadState *>(userdata);
    if (!state.error.empty()) {
        return;
    }

    try {
        if (!tensor || !tensor->name[0]) {
            throw std::runtime_error("llama.cpp requested an unnamed model tensor");
        }

        const std::string name = tensor->name;
        if (is_routed_expert_tensor(name)) {
            if (!tensor->buffer ||
                ggml_backend_buffer_get_type(tensor->buffer) !=
                    ggml_cpu_moe_placeholder_buffer_type()) {
                throw std::runtime_error(
                    "routed expert tensor was not assigned to the MMB placeholder buffer: " + name);
            }
            // Intentionally do not populate expert bytes. The CPU MUL_MAT_ID
            // provider must resolve the selected expert from the pager.
            return;
        }

        if (is_any_expert_tensor(name)) {
            throw std::runtime_error(
                "MMB bundle contains an unsupported expert tensor layout: " + name);
        }

        const MMBCoreTensor & core = state.model->core_tensor(name);
        if (core.region.ggml_type != static_cast<int32_t>(tensor->type)) {
            throw std::runtime_error("core tensor GGML type mismatch: " + name);
        }

        for (uint32_t dim = 0; dim < core.region.n_dims; ++dim) {
            if (tensor->ne[dim] != core.region.ne[dim]) {
                throw std::runtime_error("core tensor shape mismatch: " + name);
            }
        }
        for (uint32_t dim = core.region.n_dims; dim < GGML_MAX_DIMS; ++dim) {
            if (tensor->ne[dim] != 1) {
                throw std::runtime_error("core tensor rank mismatch: " + name);
            }
        }

        const size_t expected = ggml_nbytes(tensor);
        if (core.region.length != expected) {
            throw std::runtime_error("core tensor encoded length mismatch: " + name);
        }

        std::vector<uint8_t> bytes = state.model->read_core(core, state.verify_sha256);
        if (bytes.size() != expected) {
            throw std::runtime_error("short core tensor read: " + name);
        }

        ggml_backend_tensor_set(tensor, bytes.data(), 0, bytes.size());
    } catch (const std::exception & exc) {
        set_error(state, exc.what());
    } catch (...) {
        set_error(state, "unknown error while loading an MMB core tensor");
    }
}

struct GGUFDeleter {
    void operator()(gguf_context * ctx) const {
        if (ctx) {
            gguf_free(ctx);
        }
    }
};

} // namespace

llama_model * load_llama_model_from_mmb(
        const MMBModel & model,
        bool verify_sha256,
        llama_model_params params) {
    if (model.metadata_path().empty()) {
        throw std::runtime_error(
            "MMB bundle does not contain metadata-only GGUF required for native loading");
    }

    gguf_init_params gguf_params{};
    gguf_params.no_alloc = true;
    gguf_params.ctx = nullptr;

    std::unique_ptr<gguf_context, GGUFDeleter> metadata(
        gguf_init_from_file(model.metadata_path().string().c_str(), gguf_params));
    if (!metadata) {
        throw std::runtime_error(
            "failed to parse MMB metadata GGUF: " + model.metadata_path().string());
    }

    ggml_backend_buffer_type_t placeholder = ggml_cpu_moe_placeholder_buffer_type();
    if (!placeholder) {
        throw std::runtime_error("MMB MoE placeholder buffer type is unavailable");
    }

    const llama_model_tensor_buft_override overrides[] = {
        { ".*\\.ffn_(down|gate|up)_exps\\.weight$", placeholder },
        { nullptr, nullptr },
    };

    // This mode has no source model file, therefore llama.cpp must obtain every
    // tensor through set_tensor_data. Expert tensors are deliberately skipped
    // by the callback and remain virtual placeholders.
    params.load_mode = LLAMA_LOAD_MODE_NONE;
    params.tensor_buft_overrides = overrides;
    params.use_extra_bufts = false;

    LoadState state{};
    state.model = &model;
    state.verify_sha256 = verify_sha256;

    llama_model * result = llama_model_init_from_user(
        metadata.get(), set_tensor_data, &state, params);

    if (!state.error.empty()) {
        if (result) {
            llama_model_free(result);
        }
        throw std::runtime_error("MMB tensor load failed: " + state.error);
    }
    if (!result) {
        throw std::runtime_error("llama_model_init_from_user failed for MMB bundle");
    }

    return result;
}

} // namespace mmb
