#pragma once
#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace mmb {

enum class TensorRole : uint32_t {
    Down = 0,
    Gate = 1,
    Up = 2,
};

struct MMBRegion {
    uint32_t shard = 0;
    uint64_t offset = 0;
    uint64_t length = 0;
    int32_t ggml_type = -1;
    std::array<int64_t, 4> ne{1,1,1,1};
    uint32_t n_dims = 0;
    uint64_t block_offset = 0;
};

struct MMBExpert {
    uint32_t layer = 0;
    uint32_t expert = 0;
    uint64_t block_offset = 0;
    uint64_t block_length = 0;
    std::string sha256;
    std::array<MMBRegion, 3> regions{};
};

struct MMBCoreTensor {
    std::string name;
    std::string block_id;
    MMBRegion region{};
    std::string sha256;
};

struct MMBShard {
    std::filesystem::path path;
    uint64_t size = 0;
};

class MMBModel {
public:
    explicit MMBModel(std::filesystem::path model_dir);

    const std::filesystem::path & model_dir() const { return model_dir_; }
    const std::string & model_id() const { return model_id_; }
    const std::string & architecture() const { return architecture_; }
    const std::string & backend_contract() const { return backend_contract_; }
    const std::filesystem::path & metadata_path() const { return metadata_path_; }
    uint32_t layer_count() const { return layer_count_; }
    uint32_t expert_count() const { return expert_count_; }
    uint32_t active_experts_per_token() const { return active_experts_per_token_; }
    size_t core_tensor_count() const { return core_tensors_.size(); }

    const MMBExpert & expert(uint32_t layer, uint32_t expert) const;
    const MMBCoreTensor & core_tensor(const std::string & name) const;
    std::vector<uint8_t> read_expert(const MMBExpert & expert, bool verify_sha256) const;
    std::vector<uint8_t> read_core(const MMBCoreTensor & tensor, bool verify_sha256) const;

private:
    std::filesystem::path model_dir_;
    std::string model_id_;
    std::string architecture_;
    std::string backend_contract_;
    std::filesystem::path metadata_path_;
    uint32_t layer_count_ = 0;
    uint32_t expert_count_ = 0;
    uint32_t active_experts_per_token_ = 0;
    std::vector<MMBShard> shards_;
    std::unordered_map<uint64_t, MMBExpert> experts_;
    std::unordered_map<std::string, MMBCoreTensor> core_tensors_;

    static uint64_t key(uint32_t layer, uint32_t expert) {
        return (uint64_t(layer) << 32) | uint64_t(expert);
    }
};

} // namespace mmb
