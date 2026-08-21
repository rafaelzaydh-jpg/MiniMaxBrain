#include "mmb.h"
#include "mmb_model.hpp"
#include "mmb_pager.hpp"
#include "mmb_kernel.hpp"
#include "mmb_sha256.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
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
    std::cerr << "[native] " << stage << "\n";
}

static void write_file(const fs::path & path, const std::vector<uint8_t> & bytes) {
    std::ofstream out(path,std::ios::binary);
    if (!out) throw std::runtime_error("cannot create test file");
    out.write(reinterpret_cast<const char *>(bytes.data()),static_cast<std::streamsize>(bytes.size()));
}

static fs::path make_fixture() {
    fs::path dir = fs::temp_directory_path() / "mmb-native-test";
    trace("completed");
        fs::remove_all(dir);
    fs::create_directories(dir);

    // Expert 0: down=1,2 gate=3,4 up=5,6
    // Expert 1: down=7,8 gate=9,10 up=11,12
    std::vector<uint8_t> e0{1,2,3,4,5,6};
    std::vector<uint8_t> e1{7,8,9,10,11,12};
    std::vector<uint8_t> core{13,14};
    std::vector<uint8_t> shard=e0;
    shard.insert(shard.end(),e1.begin(),e1.end());
    shard.insert(shard.end(),core.begin(),core.end());
    write_file(dir/"model-00000.mmbw",shard);
    std::vector<uint8_t> meta_bytes{'G','G','U','F'};
    write_file(dir/"model.mmb-meta.gguf",meta_bytes);

    const std::string h0=mmb::sha256_hex(e0.data(),e0.size());
    const std::string hm=mmb::sha256_hex(meta_bytes.data(),meta_bytes.size());
    const std::string h1=mmb::sha256_hex(e1.data(),e1.size());
    const std::string hc=mmb::sha256_hex(core.data(),core.size());

    std::ofstream map(dir/"model.mmb-map.json");
    map << R"({
      "schema_version":"mmb-physical-model-map-v1",
      "model":{
        "id":"native-test",
        "architecture":"testmoe",
        "parameter_count":12,
        "quantization":{"name":"raw","bits_per_weight":8},
        "backend_contract":"mmb-raw-ggml-expert-segments-v2",
        "map_revision":"test"
      },
      "blocks":[
        {"id":"expert/0/0","kind":"expert","shard":"model-00000.mmbw","offset":0,"length":6,"sha256":")"
        << h0 << R"(","alignment":1,"layer":0,"expert":0},
        {"id":"expert/0/1","kind":"expert","shard":"model-00000.mmbw","offset":6,"length":6,"sha256":")"
        << h1 << R"(","alignment":1,"layer":0,"expert":1},
        {"id":"core/token_embd.weight","kind":"core","shard":"model-00000.mmbw","offset":12,"length":2,"sha256":")"
        << hc << R"(","alignment":1}
      ]
    })";

    std::ofstream layout(dir/"model.mmb-layout.json");
    layout << R"({
      "schema_version":"mmb-gguf-moe-layout-v2",
      "backend_contract":"mmb-raw-ggml-expert-segments-v2",
      "metadata_gguf":{"file_name":"model.mmb-meta.gguf","size_bytes":4,"sha256":")"
      << hm << R"("},
      "expert_count":2,
      "active_experts_per_token":1,
      "layer_count":1,
      "core_tensors":[
        {"block_id":"core/token_embd.weight","tensor":"token_embd.weight","shape":[2],"ggml_type":0,"encoded_length":2}
      ],
      "layers":[{
        "layer":0,
        "block_id_pattern":"expert/0/{expert}",
        "encoded_block_length":6,
        "segments":[
          {"role":"down","tensor":"blk.0.ffn_down_exps.weight","offset":0,"encoded_length":2,"ggml_type":0,"expert_shape":[2]},
          {"role":"gate","tensor":"blk.0.ffn_gate_exps.weight","offset":2,"encoded_length":2,"ggml_type":0,"expert_shape":[2]},
          {"role":"up","tensor":"blk.0.ffn_up_exps.weight","offset":4,"encoded_length":2,"ggml_type":0,"expert_shape":[2]}
        ]
      }]
    })";
    return dir;
}

int main() {
    fs::path dir=make_fixture();
    try {
        trace("construct MMBModel");
        auto model=std::make_shared<mmb::MMBModel>(dir);
        MMB_TEST_REQUIRE(model->layer_count()==1);
        MMB_TEST_REQUIRE(model->expert_count()==2);
        MMB_TEST_REQUIRE(model->core_tensor_count()==1);
        auto core_bytes=model->read_core(model->core_tensor("token_embd.weight"),true);
        MMB_TEST_REQUIRE((core_bytes==std::vector<uint8_t>{13,14}));
        auto bytes=model->read_expert(model->expert(0,1),true);
        MMB_TEST_REQUIRE((bytes==std::vector<uint8_t>{7,8,9,10,11,12}));

        trace("pager/cache checks");
        mmb::MMBPager pager(model,6,true);
        {
            uint32_t ids[]={0,0};
            auto lease=pager.acquire(0,std::span<const uint32_t>(ids,2),true);
            MMB_TEST_REQUIRE(lease->entries.size()==1);
            auto * entry=lease->entries[0];
            MMB_TEST_REQUIRE(entry->bytes.size()==6);
            MMB_TEST_REQUIRE(entry->bytes[entry->regions[1].block_offset]==3);
            pager.release(*lease);
        }
        {
            uint32_t ids[]={1};
            auto lease=pager.acquire(0,std::span<const uint32_t>(ids,1),true);
            MMB_TEST_REQUIRE(pager.cache().stats().evictions==1);
            pager.release(*lease);
        }

        // Active leases protect memory from eviction.
        {
            uint32_t id0[]={0};
            auto lease0=pager.acquire(0,std::span<const uint32_t>(id0,1),false);
            bool blocked=false;
            try {
                uint32_t id1[]={1};
                auto ignored=pager.acquire(0,std::span<const uint32_t>(id1,1),false);
                (void)ignored;
            } catch (const std::exception &) {
                blocked=true;
            }
            MMB_TEST_REQUIRE(blocked);
            pager.release(*lease0);
        }

        // The kernel boundary keeps router-selected experts leased and only
        // marks paged execution after successful compute completion.
        {
            trace("kernel lease check");
        int32_t router_ids[]={1,1,0};
            mmb::MMBPager kernel_pager(model,12,true);
            {
                mmb::MMBKernelLease kernel_lease(
                    kernel_pager,0,std::span<const int32_t>(router_ids,3)
                );
                auto seg=kernel_lease.segment(1,mmb::TensorRole::Gate);
                MMB_TEST_REQUIRE(seg.bytes==2 && seg.data[0]==9 && seg.data[1]==10);
                MMB_TEST_REQUIRE(kernel_pager.stats().paged_kernel_invocations==0);
                kernel_lease.commit_after_compute();
            }
            MMB_TEST_REQUIRE(kernel_pager.stats().real_router_requests==1);
            MMB_TEST_REQUIRE(kernel_pager.stats().paged_kernel_invocations==1);
        }

        // C ABI exposes real segment pointers while a lease is alive.
        trace("C ABI check");
        mmb_pager_handle * c_pager=nullptr;
        MMB_TEST_REQUIRE(mmb_pager_open(dir.string().c_str(),12,1,&c_pager)==0);
        MMB_TEST_REQUIRE(mmb_abi_version()==3);
        MMB_TEST_REQUIRE((mmb_backend_capabilities() & MMB_CAP_PAGER)!=0);
        MMB_TEST_REQUIRE((mmb_backend_capabilities() & MMB_CAP_PAGED_MOE_KERNEL)!=0);
        MMB_TEST_REQUIRE((mmb_backend_capabilities() & MMB_CAP_NATIVE_RUNTIME)!=0);
        uint32_t id=1;
        mmb_lease_handle * c_lease=nullptr;
        MMB_TEST_REQUIRE(mmb_pager_acquire(c_pager,0,&id,1,1,&c_lease)==0);
        mmb_segment_view gate{};
        MMB_TEST_REQUIRE(mmb_lease_segment(c_lease,0,MMB_TENSOR_GATE,&gate)==0);
        MMB_TEST_REQUIRE(gate.bytes==2);
        auto * p=static_cast<const uint8_t *>(gate.data);
        MMB_TEST_REQUIRE(p[0]==9 && p[1]==10);
        mmb_pager_stats stats{};
        MMB_TEST_REQUIRE(mmb_pager_get_stats(c_pager,&stats)==0);
        MMB_TEST_REQUIRE(stats.paged_experts_used==0);
        MMB_TEST_REQUIRE(stats.real_router_requests==1);
        MMB_TEST_REQUIRE(mmb_pager_release(c_pager,c_lease)==0);
        mmb_pager_close(c_pager);

        fs::remove_all(dir);
        std::cout << "mmb_native_tests: ok\n";
        return 0;
    } catch (const std::exception & exc) {
        fs::remove_all(dir);
        std::cerr << "mmb_native_tests: " << exc.what() << "\n";
        return 1;
    }
}
