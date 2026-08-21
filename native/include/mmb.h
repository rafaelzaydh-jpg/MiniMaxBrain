#pragma once
#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
  #if defined(MMB_BACKEND_BUILD)
    #define MMB_API __declspec(dllexport)
  #else
    #define MMB_API __declspec(dllimport)
  #endif
#else
  #define MMB_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define MMB_ABI_VERSION 3u
#define MMB_CAP_PAGER 0x00000001u
#define MMB_CAP_PAGED_MOE_KERNEL 0x00000002u
#define MMB_CAP_NATIVE_RUNTIME 0x00000004u

typedef struct mmb_pager_handle mmb_pager_handle;
typedef struct mmb_lease_handle mmb_lease_handle;
typedef struct mmb_runtime_handle mmb_runtime_handle;

typedef struct mmb_chat_message {
    const char * role;
    const char * content;
} mmb_chat_message;

typedef struct mmb_generation_params {
    uint32_t struct_size;
    int32_t max_tokens;
    float temperature;
    float top_p;
    int32_t top_k;
    uint32_t seed;
} mmb_generation_params;

typedef int (*mmb_stream_callback)(
    const char * utf8,
    size_t bytes,
    void * user_data
);

typedef enum mmb_tensor_role {
    MMB_TENSOR_DOWN = 0,
    MMB_TENSOR_GATE = 1,
    MMB_TENSOR_UP = 2
} mmb_tensor_role;

typedef struct mmb_segment_view {
    const void * data;
    uint64_t bytes;
    int32_t ggml_type;
    uint32_t n_dims;
    int64_t ne[4];
} mmb_segment_view;

typedef struct mmb_pager_stats {
    uint64_t cache_hits;
    uint64_t cache_misses;
    uint64_t bytes_read;
    uint64_t resident_bytes;
    uint64_t peak_resident_bytes;
    uint64_t loads;
    uint64_t evictions;
    uint64_t real_router_requests;
    uint64_t experts_used;
    uint64_t acquire_ns;
    uint64_t io_ns;
    uint32_t paged_experts_used;
} mmb_pager_stats;

MMB_API uint32_t mmb_abi_version(void);
MMB_API uint32_t mmb_backend_capabilities(void);
MMB_API const char * mmb_backend_version(void);
MMB_API const char * mmb_last_error(void);

MMB_API int mmb_pager_open(
    const char * model_dir,
    uint64_t cache_capacity_bytes,
    int verify_sha256,
    mmb_pager_handle ** out
);

MMB_API void mmb_pager_close(mmb_pager_handle * pager);

MMB_API int mmb_pager_model_info(
    mmb_pager_handle * pager,
    uint32_t * layer_count,
    uint32_t * expert_count,
    uint32_t * active_experts_per_token
);

MMB_API int mmb_pager_acquire(
    mmb_pager_handle * pager,
    uint32_t layer,
    const uint32_t * experts,
    size_t expert_count,
    int router_request,
    mmb_lease_handle ** out
);

MMB_API int mmb_lease_count(mmb_lease_handle * lease, size_t * out_count);
MMB_API int mmb_lease_expert_id(mmb_lease_handle * lease, size_t index, uint32_t * out_expert);
MMB_API int mmb_lease_segment(
    mmb_lease_handle * lease,
    size_t index,
    mmb_tensor_role role,
    mmb_segment_view * out
);

MMB_API int mmb_pager_release(mmb_pager_handle * pager, mmb_lease_handle * lease);
MMB_API int mmb_pager_get_stats(mmb_pager_handle * pager, mmb_pager_stats * out);


MMB_API int mmb_runtime_open(
    const char * model_dir,
    uint64_t expert_cache_bytes,
    int verify_sha256,
    uint32_t n_ctx,
    int32_t n_threads,
    mmb_runtime_handle ** out
);

MMB_API void mmb_runtime_close(mmb_runtime_handle * runtime);

MMB_API int mmb_runtime_chat(
    mmb_runtime_handle * runtime,
    const mmb_chat_message * messages,
    size_t message_count,
    const mmb_generation_params * params,
    mmb_stream_callback callback,
    void * user_data
);

MMB_API int mmb_runtime_get_stats(
    mmb_runtime_handle * runtime,
    mmb_pager_stats * out
);

#ifdef __cplusplus
}
#endif
