#include "mmb_ggml_bridge.hpp"

#include "mmb_kernel.hpp"
#include "ggml-moe-provider.h"

#include <cctype>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

namespace mmb {
namespace {

std::mutex g_registration_mutex;
MMBPager * g_registered_pager = nullptr;

struct TensorRoute {
    uint32_t layer = 0;
    TensorRole role = TensorRole::Down;
};

struct ProviderOp {
    std::unique_ptr<MMBKernelLease> lease;
    TensorRole role = TensorRole::Down;
    size_t expected_expert_bytes = 0;
};

TensorRoute parse_tensor_route(const char * name) {
    if (!name || !*name) {
        throw std::runtime_error("GGML MoE tensor has no name");
    }

    const std::string_view value(name);
    const size_t blk = value.find("blk.");
    if (blk == std::string_view::npos) {
        throw std::runtime_error("MMB provider only accepts blk.* MoE tensors");
    }

    size_t pos = blk + 4;
    if (pos >= value.size() || !std::isdigit(static_cast<unsigned char>(value[pos]))) {
        throw std::runtime_error("MMB provider cannot parse layer from tensor name");
    }

    uint64_t layer = 0;
    while (pos < value.size() && std::isdigit(static_cast<unsigned char>(value[pos]))) {
        layer = layer * 10 + uint64_t(value[pos] - '0');
        if (layer > UINT32_MAX) {
            throw std::runtime_error("MMB provider layer index overflows uint32");
        }
        ++pos;
    }

    TensorRole role;
    if (value.find(".ffn_down_exps") != std::string_view::npos) {
        role = TensorRole::Down;
    } else if (value.find(".ffn_gate_exps") != std::string_view::npos) {
        role = TensorRole::Gate;
    } else if (value.find(".ffn_up_exps") != std::string_view::npos) {
        role = TensorRole::Up;
    } else {
        throw std::runtime_error("MMB provider received a non-expert tensor");
    }

    return TensorRoute{static_cast<uint32_t>(layer), role};
}

int provider_begin(
        void * user_data,
        const char * tensor_name,
        const int32_t * expert_ids,
        size_t expert_count,
        size_t expected_expert_bytes,
        void ** op_context) {
    if (op_context) {
        *op_context = nullptr;
    }

    const std::string_view tensor = tensor_name ? std::string_view(tensor_name) : std::string_view();
    const bool is_mmb_expert =
        tensor.find(".ffn_down_exps") != std::string_view::npos ||
        tensor.find(".ffn_gate_exps") != std::string_view::npos ||
        tensor.find(".ffn_up_exps") != std::string_view::npos;
    if (!is_mmb_expert) {
        return 1;
    }

    try {
        if (!user_data) {
            throw std::runtime_error("MMB GGML provider has no pager");
        }
        if (!op_context) {
            throw std::runtime_error("MMB GGML provider op_context is null");
        }
        if (!expert_ids || expert_count == 0) {
            throw std::runtime_error("MMB GGML provider received no router experts");
        }

        auto & pager = *static_cast<MMBPager *>(user_data);
        const TensorRoute route = parse_tensor_route(tensor_name);

        if (route.layer >= pager.model().layer_count()) {
            throw std::runtime_error("GGML router layer is outside MMB model topology");
        }

        auto op = std::make_unique<ProviderOp>();
        op->role = route.role;
        op->expected_expert_bytes = expected_expert_bytes;
        op->lease = std::make_unique<MMBKernelLease>(
            pager,
            route.layer,
            std::span<const int32_t>(expert_ids, expert_count));

        // Validate every selected segment before publishing the operation to GGML.
        for (size_t i = 0; i < expert_count; ++i) {
            const int32_t expert = expert_ids[i];
            if (expert < 0) {
                throw std::runtime_error("GGML router produced a negative expert ID");
            }
            const auto segment = op->lease->segment(
                static_cast<uint32_t>(expert), route.role);
            if (segment.bytes != expected_expert_bytes) {
                throw std::runtime_error(
                    "MMBW expert segment size does not match GGML expert stride");
            }
        }

        *op_context = op.release();
        return 0;
    } catch (...) {
        return -1;
    }
}

const void * provider_get(
        void *,
        void * op_context,
        int32_t expert_id,
        size_t expected_expert_bytes) {
    if (!op_context || expert_id < 0) {
        return nullptr;
    }

    auto & op = *static_cast<ProviderOp *>(op_context);
    if (expected_expert_bytes != op.expected_expert_bytes) {
        return nullptr;
    }

    try {
        const auto segment = op.lease->segment(
            static_cast<uint32_t>(expert_id), op.role);
        if (segment.bytes != expected_expert_bytes) {
            return nullptr;
        }
        return segment.data;
    } catch (...) {
        return nullptr;
    }
}

void provider_end(
        void *,
        void * op_context,
        int success) {
    std::unique_ptr<ProviderOp> op(static_cast<ProviderOp *>(op_context));
    if (!op) {
        return;
    }

    if (success) {
        try {
            op->lease->commit_after_compute();
        } catch (...) {
            // GGML has already completed the op. Destruction still releases an
            // uncommitted lease, while the paged-kernel metric remains false.
        }
    }
}

} // namespace

MMBGGMLProviderRegistration::MMBGGMLProviderRegistration(MMBPager & pager)
    : pager_(&pager) {
    std::lock_guard<std::mutex> lock(g_registration_mutex);
    if (g_registered_pager != nullptr) {
        throw std::runtime_error("only one MMB GGML provider may be active per process");
    }

    ggml_moe_expert_provider provider{};
    provider.user_data = pager_;
    provider.begin = provider_begin;
    provider.get = provider_get;
    provider.end = provider_end;

    if (ggml_cpu_set_moe_expert_provider(&provider) != 0) {
        pager_ = nullptr;
        throw std::runtime_error("GGML rejected the MMB MoE expert provider");
    }

    g_registered_pager = &pager;
}

MMBGGMLProviderRegistration::~MMBGGMLProviderRegistration() {
    std::lock_guard<std::mutex> lock(g_registration_mutex);
    if (pager_ != nullptr && g_registered_pager == pager_) {
        ggml_cpu_set_moe_expert_provider(nullptr);
        g_registered_pager = nullptr;
    }
}

} // namespace mmb
