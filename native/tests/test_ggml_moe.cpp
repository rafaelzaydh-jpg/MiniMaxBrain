#include "mmb_ggml_bridge.hpp"
#include "mmb_model.hpp"
#include "mmb_pager.hpp"
#include "mmb_sha256.hpp"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

#define MMB_TEST_REQUIRE(expr) \
    do { \
        if (!(expr)) { \
            throw std::runtime_error(std::string("test requirement failed: ") + #expr); \
        } \
    } while (0)

static void trace(const char * stage) {
    std::cerr << "[ggml] " << stage << "\n";
}

static std::vector<uint8_t> bytes_of(const std::vector<float> & values) {
    std::vector<uint8_t> out(values.size() * sizeof(float));
    std::memcpy(out.data(), values.data(), out.size());
    return out;
}

static void append(std::vector<uint8_t> & dst, const std::vector<uint8_t> & src) {
    dst.insert(dst.end(), src.begin(), src.end());
}

static fs::path make_moe_fixture(const std::vector<float> & combined_gate_weights) {
    // combined_gate_weights is [K=2, M=2, experts=2], so one expert slice is 16 B.
    if (combined_gate_weights.size() != 8) {
        throw std::runtime_error("invalid test weight cardinality");
    }

    fs::path dir = fs::temp_directory_path() / "mmb-ggml-moe-test";
    fs::remove_all(dir);
    fs::create_directories(dir);

    std::vector<uint8_t> shard;
    std::vector<std::string> hashes;

    for (size_t expert = 0; expert < 2; ++expert) {
        std::vector<float> gate(
            combined_gate_weights.begin() + expert * 4,
            combined_gate_weights.begin() + (expert + 1) * 4);

        std::vector<uint8_t> block;
        append(block, bytes_of(std::vector<float>(4, 0.0f))); // down
        append(block, bytes_of(gate));                         // gate
        append(block, bytes_of(std::vector<float>(4, 0.0f))); // up
        hashes.push_back(mmb::sha256_hex(block.data(), block.size()));
        append(shard, block);
    }

    {
        std::ofstream out(dir / "model-00000.mmbw", std::ios::binary);
        out.write(reinterpret_cast<const char *>(shard.data()),
                  static_cast<std::streamsize>(shard.size()));
    }

    const std::vector<uint8_t> meta{'G','G','U','F'};
    {
        std::ofstream out(dir / "model.mmb-meta.gguf", std::ios::binary);
        out.write(reinterpret_cast<const char *>(meta.data()),
                  static_cast<std::streamsize>(meta.size()));
    }
    const std::string meta_hash = mmb::sha256_hex(meta.data(), meta.size());

    {
        std::ofstream map(dir / "model.mmb-map.json");
        map << R"({
  "schema_version":"mmb-physical-model-map-v1",
  "model":{
    "id":"ggml-moe-test",
    "architecture":"testmoe",
    "parameter_count":24,
    "quantization":{"name":"f32","bits_per_weight":32},
    "backend_contract":"mmb-raw-ggml-expert-segments-v2",
    "map_revision":"test"
  },
  "blocks":[
    {"id":"expert/0/0","kind":"expert","shard":"model-00000.mmbw","offset":0,"length":48,"sha256":")"
            << hashes[0] << R"(","alignment":1,"layer":0,"expert":0},
    {"id":"expert/0/1","kind":"expert","shard":"model-00000.mmbw","offset":48,"length":48,"sha256":")"
            << hashes[1] << R"(","alignment":1,"layer":0,"expert":1}
  ]
})";
    }

    {
        std::ofstream layout(dir / "model.mmb-layout.json");
        layout << R"({
  "schema_version":"mmb-gguf-moe-layout-v2",
  "backend_contract":"mmb-raw-ggml-expert-segments-v2",
  "metadata_gguf":{"file_name":"model.mmb-meta.gguf","size_bytes":4,"sha256":")"
               << meta_hash << R"("},
  "expert_count":2,
  "active_experts_per_token":1,
  "layer_count":1,
  "core_tensors":[],
  "layers":[{
    "layer":0,
    "block_id_pattern":"expert/0/{expert}",
    "encoded_block_length":48,
    "segments":[
      {"role":"down","tensor":"blk.0.ffn_down_exps.weight","offset":0,"encoded_length":16,"ggml_type":0,"expert_shape":[2,2]},
      {"role":"gate","tensor":"blk.0.ffn_gate_exps.weight","offset":16,"encoded_length":16,"ggml_type":0,"expert_shape":[2,2]},
      {"role":"up","tensor":"blk.0.ffn_up_exps.weight","offset":32,"encoded_length":16,"ggml_type":0,"expert_shape":[2,2]}
    ]
  }]
})";
    }

    return dir;
}

static std::vector<float> run_mul_mat_id(
        const std::vector<float> & weights,
        const std::vector<float> & input,
        int32_t expert_id,
        const char * tensor_name = "blk.0.ffn_gate_exps.weight") {
    ggml_init_params params{};
    params.mem_size = 1024 * 1024;
    params.mem_buffer = nullptr;
    params.no_alloc = true;

    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        throw std::runtime_error("ggml_init failed");
    }

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        ggml_free(ctx);
        throw std::runtime_error("ggml_backend_cpu_init failed");
    }
    ggml_backend_cpu_set_n_threads(backend, 2);

    ggml_tensor * as = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 2, 2);
    ggml_set_name(as, tensor_name);
    ggml_tensor * b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 1, 1);
    ggml_set_name(b, "input");
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, 1);
    ggml_set_name(ids, "router_ids");
    ggml_tensor * out = ggml_mul_mat_id(ctx, as, b, ids);
    ggml_set_name(out, "out");

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        ggml_backend_free(backend);
        ggml_free(ctx);
        throw std::runtime_error("ggml_backend_alloc_ctx_tensors failed");
    }

    ggml_backend_tensor_set(as, weights.data(), 0, weights.size() * sizeof(float));
    ggml_backend_tensor_set(b, input.data(), 0, input.size() * sizeof(float));
    ggml_backend_tensor_set(ids, &expert_id, 0, sizeof(expert_id));

    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buffer);
        ggml_backend_free(backend);
        ggml_free(ctx);
        throw std::runtime_error("GGML MUL_MAT_ID compute failed");
    }

    std::vector<float> result(static_cast<size_t>(ggml_nelements(out)));
    ggml_backend_tensor_get(out, result.data(), 0, result.size() * sizeof(float));

    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return result;
}

static bool close_enough(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::fabs(a[i] - b[i]) > 1e-6f) {
            return false;
        }
    }
    return true;
}

int main() {
    // Expert 0 = identity, expert 1 = diag(2,3).
    const std::vector<float> correct_weights{
        1.0f, 0.0f,
        0.0f, 1.0f,

        2.0f, 0.0f,
        0.0f, 3.0f,
    };
    const std::vector<float> zero_weights(8, 0.0f);
    const std::vector<float> input{3.0f, 4.0f};

    trace("baseline MUL_MAT_ID");
        const auto reference = run_mul_mat_id(correct_weights, input, 1);
    trace("zero-weight control");
        const auto without_provider = run_mul_mat_id(zero_weights, input, 1);
    MMB_TEST_REQUIRE(!close_enough(reference, without_provider));

    trace("create MMB fixture");
        const fs::path fixture = make_moe_fixture(correct_weights);
    try {
        trace("construct MMBModel");
        auto model = std::make_shared<mmb::MMBModel>(fixture);
        trace("construct pager");
        mmb::MMBPager pager(model, 96, true);

        {
            trace("register provider");
        mmb::MMBGGMLProviderRegistration registration(pager);
            trace("MMBW-backed MUL_MAT_ID");
        const auto from_mmbw = run_mul_mat_id(zero_weights, input, 1);
            MMB_TEST_REQUIRE(close_enough(reference, from_mmbw));

            // The provider must ignore unrelated MUL_MAT_ID tensors rather
            // than hijacking every MoE-like operation in the process.
            const auto unrelated = run_mul_mat_id(
                correct_weights, input, 1, "adapter.experts.weight");
            MMB_TEST_REQUIRE(close_enough(reference, unrelated));
        }

        trace("validate metrics");
        const auto snapshot = pager.snapshot();
        MMB_TEST_REQUIRE(snapshot.pager.real_router_requests == 1);
        MMB_TEST_REQUIRE(snapshot.pager.paged_kernel_invocations == 1);
        MMB_TEST_REQUIRE(snapshot.pager.bytes_read == 48);
        MMB_TEST_REQUIRE(snapshot.cache.resident_bytes == 48);

        fs::remove_all(fixture);
        return 0;
    } catch (...) {
        fs::remove_all(fixture);
        throw;
    }
}
