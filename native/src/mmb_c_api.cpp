#include "mmb.h"
#include "mmb_pager.hpp"
#ifdef MMB_WITH_LLAMA
#include "mmb_runtime.hpp"
#endif
#include <cstring>
#include <memory>
#include <limits>
#include <vector>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>

using namespace mmb;

struct mmb_pager_handle {
    std::shared_ptr<MMBModel> model;
    std::unique_ptr<MMBPager> pager;
};

struct mmb_lease_handle {
    std::unique_ptr<MMBLease> lease;
};

struct mmb_runtime_handle {
#ifdef MMB_WITH_LLAMA
    std::unique_ptr<MMBLlamaRuntime> runtime;
#endif
};


namespace {
thread_local std::string g_error;
std::mutex g_runtime_mutex;
bool g_runtime_active = false;

template<class F>
int guarded(F && fn) {
    try {
        fn();
        g_error.clear();
        return 0;
    } catch (const std::exception & exc) {
        g_error = exc.what();
        return -1;
    } catch (...) {
        g_error = "unknown native MMB error";
        return -1;
    }
}

size_t role_index(mmb_tensor_role role) {
    switch (role) {
        case MMB_TENSOR_DOWN: return 0;
        case MMB_TENSOR_GATE: return 1;
        case MMB_TENSOR_UP: return 2;
        default: throw std::runtime_error("invalid tensor role");
    }
}

void fill_stats(const MMBPagerSnapshot & snapshot, mmb_pager_stats * out) {
    if (!out) throw std::runtime_error("stats output is null");
    const auto & p = snapshot.pager;
    const auto & c = snapshot.cache;
    std::memset(out, 0, sizeof(*out));
    out->cache_hits = p.hits;
    out->cache_misses = p.misses;
    out->bytes_read = p.bytes_read;
    out->resident_bytes = c.resident_bytes;
    out->peak_resident_bytes = c.peak_resident_bytes;
    out->loads = p.loads;
    out->evictions = c.evictions;
    out->real_router_requests = p.real_router_requests;
    out->experts_used = p.experts_used;
    out->acquire_ns = p.acquire_ns;
    out->io_ns = p.io_ns;
    out->paged_experts_used = p.paged_kernel_invocations > 0 ? 1u : 0u;
}
}

extern "C" {

uint32_t mmb_abi_version(void) { return MMB_ABI_VERSION; }
uint32_t mmb_backend_capabilities(void) {
#ifdef MMB_WITH_LLAMA
    return MMB_CAP_PAGER | MMB_CAP_PAGED_MOE_KERNEL | MMB_CAP_NATIVE_RUNTIME;
#else
    return MMB_CAP_PAGER;
#endif
}
const char * mmb_backend_version(void) { return "0.3.1-direct"; }
const char * mmb_last_error(void) { return g_error.c_str(); }

int mmb_pager_open(const char * model_dir, uint64_t cache_capacity_bytes,
                   int verify_sha256, mmb_pager_handle ** out) {
    if (out) *out = nullptr;
    return guarded([&] {
        if (!out) throw std::runtime_error("out handle is null");
        if (!model_dir || !*model_dir) throw std::runtime_error("model_dir is empty");
        auto model = std::make_shared<MMBModel>(model_dir);
        auto handle = std::make_unique<mmb_pager_handle>();
        handle->model = model;
        handle->pager = std::make_unique<MMBPager>(
            std::move(model), cache_capacity_bytes, verify_sha256 != 0
        );
        *out = handle.release();
    });
}

void mmb_pager_close(mmb_pager_handle * pager) {
    delete pager;
}

int mmb_pager_model_info(mmb_pager_handle * pager, uint32_t * layer_count,
                         uint32_t * expert_count, uint32_t * active_experts_per_token) {
    return guarded([&] {
        if (!pager || !pager->model) throw std::runtime_error("pager handle is null");
        if (!layer_count || !expert_count || !active_experts_per_token) {
            throw std::runtime_error("model info output pointer is null");
        }
        *layer_count = pager->model->layer_count();
        *expert_count = pager->model->expert_count();
        *active_experts_per_token = pager->model->active_experts_per_token();
    });
}

int mmb_pager_acquire(mmb_pager_handle * pager, uint32_t layer,
                      const uint32_t * experts, size_t expert_count,
                      int router_request, mmb_lease_handle ** out) {
    if (out) *out = nullptr;
    return guarded([&] {
        if (!pager || !pager->pager) throw std::runtime_error("pager handle is null");
        if (!out) throw std::runtime_error("out lease pointer is null");
        if (!experts || expert_count == 0) throw std::runtime_error("expert list is empty");
        std::span<const uint32_t> ids(experts,expert_count);
        auto lease = std::make_unique<mmb_lease_handle>();
        lease->lease = pager->pager->acquire(layer,ids,router_request != 0);
        *out = lease.release();
    });
}

int mmb_lease_count(mmb_lease_handle * lease, size_t * out_count) {
    return guarded([&] {
        if (!lease || !lease->lease) throw std::runtime_error("lease handle is null");
        if (!out_count) throw std::runtime_error("out_count is null");
        *out_count = lease->lease->entries.size();
    });
}

int mmb_lease_expert_id(mmb_lease_handle * lease, size_t index, uint32_t * out_expert) {
    return guarded([&] {
        if (!lease || !lease->lease) throw std::runtime_error("lease handle is null");
        if (!out_expert) throw std::runtime_error("out_expert is null");
        if (index >= lease->lease->entries.size()) throw std::runtime_error("lease index out of range");
        *out_expert = lease->lease->entries[index]->expert;
    });
}

int mmb_lease_segment(mmb_lease_handle * lease, size_t index,
                      mmb_tensor_role role, mmb_segment_view * out) {
    return guarded([&] {
        if (!lease || !lease->lease) throw std::runtime_error("lease handle is null");
        if (!out) throw std::runtime_error("segment output is null");
        if (index >= lease->lease->entries.size()) throw std::runtime_error("lease index out of range");
        auto * entry = lease->lease->entries[index];
        const auto & region = entry->regions[role_index(role)];
        if (region.block_offset > entry->bytes.size() ||
            region.length > entry->bytes.size() - region.block_offset) {
            throw std::runtime_error("segment view exceeds cached expert block");
        }
        out->data = entry->bytes.data() + static_cast<size_t>(region.block_offset);
        out->bytes = region.length;
        out->ggml_type = region.ggml_type;
        out->n_dims = region.n_dims;
        for (size_t i=0;i<4;++i) out->ne[i] = region.ne[i];
    });
}

int mmb_pager_release(mmb_pager_handle * pager, mmb_lease_handle * lease) {
    int rc = guarded([&] {
        if (!pager || !pager->pager) throw std::runtime_error("pager handle is null");
        if (!lease || !lease->lease) throw std::runtime_error("lease handle is null");
        pager->pager->release(*lease->lease);
    });
    if (rc == 0) delete lease;
    return rc;
}

int mmb_pager_get_stats(mmb_pager_handle * pager, mmb_pager_stats * out) {
    return guarded([&] {
        if (!pager || !pager->pager) throw std::runtime_error("pager handle is null");
        if (!out) throw std::runtime_error("stats output is null");
        fill_stats(pager->pager->snapshot(), out);
    });
}


int mmb_runtime_open(
        const char * model_dir,
        uint64_t expert_cache_bytes,
        int verify_sha256,
        uint32_t n_ctx,
        int32_t n_threads,
        mmb_runtime_handle ** out) {
    if (out) *out = nullptr;
    bool reserved = false;
    const int rc = guarded([&] {
        if (!out) throw std::runtime_error("out runtime handle is null");
        if (!model_dir || !*model_dir) throw std::runtime_error("model_dir is empty");
        if (expert_cache_bytes == 0) throw std::runtime_error("expert_cache_bytes must be positive");
#ifdef MMB_WITH_LLAMA
        {
            std::lock_guard<std::mutex> lock(g_runtime_mutex);
            if (g_runtime_active) {
                throw std::runtime_error("only one native MMB runtime may be active per process");
            }
            g_runtime_active = true;
            reserved = true;
        }

        auto handle = std::make_unique<mmb_runtime_handle>();
        handle->runtime = std::make_unique<MMBLlamaRuntime>(
            model_dir, expert_cache_bytes, verify_sha256 != 0, n_ctx, n_threads);
        *out = handle.release();
#else
        (void) verify_sha256;
        (void) n_ctx;
        (void) n_threads;
        throw std::runtime_error("native MMB runtime was built without llama.cpp");
#endif
    });

    if (rc != 0 && reserved) {
        std::lock_guard<std::mutex> lock(g_runtime_mutex);
        g_runtime_active = false;
    }
    return rc;
}

void mmb_runtime_close(mmb_runtime_handle * runtime) {
    if (!runtime) return;
    delete runtime;
    std::lock_guard<std::mutex> lock(g_runtime_mutex);
    g_runtime_active = false;
}

int mmb_runtime_chat(
        mmb_runtime_handle * runtime,
        const mmb_chat_message * messages,
        size_t message_count,
        const mmb_generation_params * params,
        mmb_stream_callback callback,
        void * user_data) {
    int cancelled = 0;
    const int rc = guarded([&] {
#ifdef MMB_WITH_LLAMA
        if (!runtime || !runtime->runtime) throw std::runtime_error("runtime handle is null");
        if (!messages || message_count == 0) throw std::runtime_error("messages are empty");
        if (!params) throw std::runtime_error("generation params are null");
        if (params->struct_size != sizeof(mmb_generation_params)) {
            throw std::runtime_error("mmb_generation_params size mismatch");
        }
        if (!callback) throw std::runtime_error("stream callback is null");

        std::vector<ChatMessage> native_messages;
        native_messages.reserve(message_count);
        for (size_t i = 0; i < message_count; ++i) {
            if (!messages[i].role || !messages[i].content) {
                throw std::runtime_error("chat message contains a null field");
            }
            native_messages.push_back({messages[i].role, messages[i].content});
        }

        GenerationParams native_params{};
        native_params.max_tokens = params->max_tokens;
        native_params.temperature = params->temperature;
        native_params.top_p = params->top_p;
        native_params.top_k = params->top_k;
        native_params.seed = params->seed;

        struct CallbackState {
            mmb_stream_callback callback;
            void * user_data;
        } state{callback, user_data};

        const bool completed = runtime->runtime->chat(
            native_messages,
            native_params,
            [](const char * data, size_t bytes, void * opaque) -> bool {
                auto & state = *static_cast<CallbackState *>(opaque);
                return state.callback(data, bytes, state.user_data) == 0;
            },
            &state);
        if (!completed) {
            cancelled = 1;
        }
#else
        (void) runtime;
        (void) messages;
        (void) message_count;
        (void) params;
        (void) callback;
        (void) user_data;
        throw std::runtime_error("native MMB runtime was built without llama.cpp");
#endif
    });
    if (rc != 0) return rc;
    return cancelled ? 1 : 0;
}

int mmb_runtime_get_stats(mmb_runtime_handle * runtime, mmb_pager_stats * out) {
    return guarded([&] {
#ifdef MMB_WITH_LLAMA
        if (!runtime || !runtime->runtime) throw std::runtime_error("runtime handle is null");
        fill_stats(runtime->runtime->stats(), out);
#else
        (void) runtime;
        (void) out;
        throw std::runtime_error("native MMB runtime was built without llama.cpp");
#endif
    });
}

} // extern "C"
