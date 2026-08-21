#pragma once

#include "mmb_model.hpp"

#include "llama.h"

namespace mmb {

// Loads a llama.cpp model from the metadata-only GGUF embedded in an MMB v0.3
// bundle. Core tensors are populated from MMBW blocks. Routed expert tensors
// are allocated in a virtual-address-only placeholder buffer and are supplied
// by MMBPager through the GGML MoE provider during MUL_MAT_ID.
llama_model * load_llama_model_from_mmb(
    const MMBModel & model,
    bool verify_sha256,
    llama_model_params params);

} // namespace mmb
