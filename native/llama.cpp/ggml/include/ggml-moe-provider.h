#pragma once

#include "ggml-backend.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// MiniMaxBrain extension for the pinned GGML CPU backend.
//
// The provider is invoked by GGML_OP_MUL_MAT_ID after the real router IDs are
// available and before selected expert matrices are consumed. Registration is
// process-global in MMB 0.3 and must not be changed while a graph is running.
struct ggml_moe_expert_provider {
    void * user_data;

    // Return 0 when handled, >0 when this tensor is intentionally not handled,
    // and <0 on provider failure. *op_context must be non-null when handled.
    int (*begin)(
        void * user_data,
        const char * tensor_name,
        const int32_t * expert_ids,
        size_t expert_count,
        size_t expected_expert_bytes,
        void ** op_context);

    // The returned encoded expert matrix must remain valid until end().
    const void * (*get)(
        void * user_data,
        void * op_context,
        int32_t expert_id,
        size_t expected_expert_bytes);

    // Called after every CPU worker has finished consuming provider pointers.
    void (*end)(
        void * user_data,
        void * op_context,
        int success);
};

GGML_BACKEND_API int ggml_cpu_set_moe_expert_provider(
    const struct ggml_moe_expert_provider * provider);

// Buffer type for routed expert tensors whose encoded bytes are supplied by
// the MoE provider at compute time. The buffer reserves virtual address space
// only; it does not commit physical pages for the logical expert tensor.
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_cpu_moe_placeholder_buffer_type(void);

#ifdef __cplusplus
}
#endif
