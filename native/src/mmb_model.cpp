#include "mmb_model.hpp"
#include "mmb_json.hpp"
#include "mmb_sha256.hpp"

#include <algorithm>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace mmb {
namespace {

const json::object & object_at(const json::object & obj, const char * key) {
    return json::require(obj, key).as_object();
}
const json::array & array_at(const json::object & obj, const char * key) {
    return json::require(obj, key).as_array();
}
std::string string_at(const json::object & obj, const char * key) {
    return json::require(obj, key).as_string();
}
int32_t int32_at(const json::object & obj, const char * key) {
    double d = json::require(obj,key).as_number();
    if (d < std::numeric_limits<int32_t>::min() || d > std::numeric_limits<int32_t>::max() ||
        d != static_cast<double>(static_cast<int32_t>(d))) {
        throw std::runtime_error(std::string("invalid int32 field: ") + key);
    }
    return static_cast<int32_t>(d);
}

TensorRole parse_role(const std::string & role) {
    if (role == "down") return TensorRole::Down;
    if (role == "gate") return TensorRole::Gate;
    if (role == "up") return TensorRole::Up;
    throw std::runtime_error("unknown expert tensor role: " + role);
}

size_t role_index(TensorRole role) {
    return static_cast<size_t>(role);
}

bool path_is_inside(const fs::path & base, const fs::path & candidate) {
    auto rel = candidate.lexically_relative(base);
    if (rel.empty()) return candidate == base;
    auto it = rel.begin();
    return it == rel.end() || *it != "..";
}

std::array<int64_t,4> parse_shape(const json::object & obj, const char * key, uint32_t & n_dims) {
    const auto & arr = array_at(obj,key);
    if (arr.empty() || arr.size() > 4) throw std::runtime_error("tensor shape must have 1..4 dimensions");
    std::array<int64_t,4> out{1,1,1,1};
    n_dims = static_cast<uint32_t>(arr.size());
    for (size_t i=0;i<arr.size();++i) {
        double d = arr[i].as_number();
        if (d < 1 || d > static_cast<double>(std::numeric_limits<int64_t>::max()) ||
            d != static_cast<double>(static_cast<int64_t>(d))) {
            throw std::runtime_error("invalid tensor dimension");
        }
        out[i] = static_cast<int64_t>(d);
    }
    return out;
}

struct BlockRecord {
    uint32_t layer = 0;
    uint32_t expert = 0;
    uint32_t shard = 0;
    uint64_t offset = 0;
    uint64_t length = 0;
    std::string sha256;
};

} // namespace

MMBModel::MMBModel(fs::path model_dir) {
    model_dir_ = fs::weakly_canonical(std::move(model_dir));
    if (!fs::is_directory(model_dir_)) {
        throw std::runtime_error("MMB model directory does not exist: " + model_dir_.string());
    }

    const fs::path map_path = model_dir_ / "model.mmb-map.json";
    const fs::path layout_path = model_dir_ / "model.mmb-layout.json";
    if (!fs::is_regular_file(map_path)) throw std::runtime_error("model.mmb-map.json not found");
    if (!fs::is_regular_file(layout_path)) throw std::runtime_error("model.mmb-layout.json not found");

    auto map_root = json::parse_file(map_path.string()).as_object();
    if (string_at(map_root,"schema_version") != "mmb-physical-model-map-v1") {
        throw std::runtime_error("unsupported model map schema");
    }
    const auto & model_obj = object_at(map_root,"model");
    model_id_ = string_at(model_obj,"id");
    architecture_ = string_at(model_obj,"architecture");
    backend_contract_ = string_at(model_obj,"backend_contract");

    std::unordered_map<std::string,uint32_t> shard_index;
    std::unordered_map<uint64_t,BlockRecord> block_records;
    std::unordered_map<std::string,BlockRecord> core_block_records;
    std::unordered_map<uint32_t,std::vector<std::pair<uint64_t,uint64_t>>> physical_ranges;
    for (const auto & block_value : array_at(map_root,"blocks")) {
        const auto & block = block_value.as_object();
        const std::string kind = string_at(block,"kind");
        if (kind != "expert" && kind != "core") {
            throw std::runtime_error("unsupported physical block kind: " + kind);
        }
        const std::string block_id = string_at(block,"id");
        std::string shard_name = string_at(block,"shard");
        fs::path shard_path = fs::weakly_canonical(model_dir_ / fs::path(shard_name));
        if (!path_is_inside(model_dir_, shard_path)) {
            throw std::runtime_error("shard path escapes model directory: " + shard_name);
        }
        if (!fs::is_regular_file(shard_path)) {
            throw std::runtime_error("shard does not exist: " + shard_path.string());
        }

        uint32_t sid;
        auto sit = shard_index.find(shard_path.string());
        if (sit == shard_index.end()) {
            const auto raw_size = fs::file_size(shard_path);
            if (raw_size > std::numeric_limits<uint64_t>::max()) throw std::runtime_error("invalid shard size");
            const uint64_t size = static_cast<uint64_t>(raw_size);
            sid = static_cast<uint32_t>(shards_.size());
            shard_index.emplace(shard_path.string(),sid);
            shards_.push_back(MMBShard{shard_path,size});
        } else {
            sid = sit->second;
        }

        BlockRecord rec;
        rec.shard = sid;
        rec.offset = json::require_u64(block,"offset");
        rec.length = json::require_u64(block,"length");
        rec.sha256 = string_at(block,"sha256");
        if (rec.length == 0 || rec.offset > shards_[sid].size ||
            rec.length > shards_[sid].size - rec.offset) {
            throw std::runtime_error("physical block extends beyond shard");
        }
        if (rec.sha256.size()!=64 || !std::all_of(rec.sha256.begin(),rec.sha256.end(),[](char c){
                return (c>='0'&&c<='9')||(c>='a'&&c<='f');
            })) {
            throw std::runtime_error("invalid block SHA-256");
        }
        physical_ranges[rec.shard].emplace_back(rec.offset,rec.offset+rec.length);

        if (kind == "expert") {
            rec.layer = json::require_u32(block,"layer");
            rec.expert = json::require_u32(block,"expert");
            auto [_, inserted] = block_records.emplace(key(rec.layer,rec.expert),std::move(rec));
            if (!inserted) throw std::runtime_error("duplicate expert route in model map");
        } else {
            auto [_, inserted] = core_block_records.emplace(block_id,std::move(rec));
            if (!inserted) throw std::runtime_error("duplicate core block id in model map");
        }
    }

    for (auto & [sid,ranges] : physical_ranges) {
        (void)sid;
        std::sort(ranges.begin(),ranges.end());
        for (size_t i=1;i<ranges.size();++i) {
            if (ranges[i].first < ranges[i-1].second) {
                throw std::runtime_error("overlapping physical blocks in MMB shard");
            }
        }
    }

    auto layout_root = json::parse_file(layout_path.string()).as_object();
    std::string layout_schema = string_at(layout_root,"schema_version");
    if (layout_schema != "mmb-gguf-moe-layout-v1" &&
        layout_schema != "mmb-gguf-moe-layout-v2") {
        throw std::runtime_error("unsupported MMB MoE layout schema: " + layout_schema);
    }
    const std::string layout_contract = string_at(layout_root,"backend_contract");
    if (layout_contract != backend_contract_) {
        throw std::runtime_error("layout/backend contract mismatch");
    }
    if (layout_schema == "mmb-gguf-moe-layout-v2") {
        const auto & meta = object_at(layout_root,"metadata_gguf");
        const std::string meta_name = string_at(meta,"file_name");
        fs::path meta_path = fs::weakly_canonical(model_dir_ / fs::path(meta_name));
        if (!path_is_inside(model_dir_,meta_path) || !fs::is_regular_file(meta_path)) {
            throw std::runtime_error("metadata-only GGUF is missing or escapes model directory");
        }
        const uint64_t expected_size = json::require_u64(meta,"size_bytes");
        if (fs::file_size(meta_path) != expected_size) {
            throw std::runtime_error("metadata-only GGUF size mismatch");
        }
        const std::string expected_hash = string_at(meta,"sha256");
        if (expected_hash.size()!=64 || sha256_file(meta_path)!=expected_hash) {
            throw std::runtime_error("metadata-only GGUF SHA-256 mismatch");
        }
        metadata_path_ = meta_path;
    }
    std::set<std::string> seen_core_blocks;
    for (const auto & core_value : array_at(layout_root,"core_tensors")) {
        const auto & core = core_value.as_object();
        const std::string block_id = string_at(core,"block_id");
        const std::string tensor_name = string_at(core,"tensor");
        auto bit = core_block_records.find(block_id);
        if (bit == core_block_records.end()) {
            throw std::runtime_error("core tensor references unknown physical block: " + block_id);
        }
        if (!seen_core_blocks.insert(block_id).second) {
            throw std::runtime_error("duplicate core block in layout: " + block_id);
        }
        const BlockRecord & block = bit->second;
        const uint64_t encoded_length = json::require_u64(core,"encoded_length");
        if (encoded_length == 0 || encoded_length != block.length) {
            throw std::runtime_error("core tensor encoded length does not match physical block: " + tensor_name);
        }

        MMBCoreTensor out;
        out.name = tensor_name;
        out.block_id = block_id;
        out.sha256 = block.sha256;
        out.region.shard = block.shard;
        out.region.offset = block.offset;
        out.region.block_offset = 0;
        out.region.length = block.length;
        out.region.ggml_type = int32_at(core,"ggml_type");
        out.region.ne = parse_shape(core,"shape",out.region.n_dims);
        auto [_, inserted] = core_tensors_.emplace(tensor_name,std::move(out));
        if (!inserted) {
            throw std::runtime_error("duplicate core tensor name in layout: " + tensor_name);
        }
    }
    if (seen_core_blocks.size() != core_block_records.size()) {
        throw std::runtime_error("layout does not account for every physical core block");
    }

    layer_count_ = json::require_u32(layout_root,"layer_count");
    expert_count_ = json::require_u32(layout_root,"expert_count");
    active_experts_per_token_ = json::require_u32(layout_root,"active_experts_per_token");
    if (layer_count_ == 0 || expert_count_ == 0 || active_experts_per_token_ == 0 ||
        active_experts_per_token_ > expert_count_) {
        throw std::runtime_error("invalid MoE topology in layout");
    }
    if (block_records.size() != uint64_t(layer_count_) * uint64_t(expert_count_)) {
        throw std::runtime_error("model map expert route cardinality does not match layout");
    }
    for (const auto & item : block_records) {
        const auto & block = item.second;
        if (block.layer >= layer_count_ || block.expert >= expert_count_) {
            throw std::runtime_error("model map contains expert route outside declared topology");
        }
    }

    std::set<uint32_t> seen_layers;
    for (const auto & layer_value : array_at(layout_root,"layers")) {
        const auto & layer = layer_value.as_object();
        uint32_t layer_id = json::require_u32(layer,"layer");
        if (layer_id >= layer_count_ || !seen_layers.insert(layer_id).second) {
            throw std::runtime_error("invalid or duplicate layer layout");
        }
        uint64_t encoded_block_length = json::require_u64(layer,"encoded_block_length");
        const auto & segments = array_at(layer,"segments");
        if (segments.size() != 3) throw std::runtime_error("each MoE layer must define exactly three expert segments");

        std::array<MMBRegion,3> template_regions{};
        std::array<bool,3> roles_seen{false,false,false};
        for (const auto & segment_value : segments) {
            const auto & segment = segment_value.as_object();
            TensorRole role = parse_role(string_at(segment,"role"));
            size_t idx = role_index(role);
            if (roles_seen[idx]) throw std::runtime_error("duplicate expert role in layer layout");
            roles_seen[idx] = true;
            MMBRegion region;
            region.block_offset = json::require_u64(segment,"offset");
            region.length = json::require_u64(segment,"encoded_length");
            region.ggml_type = int32_at(segment,"ggml_type");
            region.ne = parse_shape(segment,"expert_shape",region.n_dims);
            if (region.length == 0 || region.block_offset > encoded_block_length ||
                region.length > encoded_block_length - region.block_offset) {
                throw std::runtime_error("expert segment exceeds encoded block");
            }
            template_regions[idx] = region;
        }
        if (!std::all_of(roles_seen.begin(),roles_seen.end(),[](bool v){ return v; })) {
            throw std::runtime_error("layer layout does not define down/gate/up exactly once");
        }
        std::vector<std::pair<uint64_t,uint64_t>> segment_ranges;
        for (const auto & region : template_regions) {
            segment_ranges.emplace_back(region.block_offset,region.block_offset+region.length);
        }
        std::sort(segment_ranges.begin(),segment_ranges.end());
        uint64_t cursor=0;
        for (const auto & range : segment_ranges) {
            if (range.first != cursor) throw std::runtime_error("expert segments must exactly cover the encoded block");
            cursor=range.second;
        }
        if (cursor != encoded_block_length) throw std::runtime_error("expert segments do not cover the encoded block");

        for (uint32_t expert_id=0; expert_id<expert_count_; ++expert_id) {
            auto bit = block_records.find(key(layer_id,expert_id));
            if (bit == block_records.end()) {
                throw std::runtime_error("missing expert block for layer=" + std::to_string(layer_id) +
                                         " expert=" + std::to_string(expert_id));
            }
            const BlockRecord & block = bit->second;
            if (block.length != encoded_block_length) {
                throw std::runtime_error("expert block length does not match layer layout");
            }
            MMBExpert out;
            out.layer = layer_id;
            out.expert = expert_id;
            out.block_offset = block.offset;
            out.block_length = block.length;
            out.sha256 = block.sha256;
            out.regions = template_regions;
            for (auto & region : out.regions) {
                region.shard = block.shard;
                region.offset = block.offset + region.block_offset;
            }
            experts_.emplace(key(layer_id,expert_id),std::move(out));
        }
    }

    if (seen_layers.size() != layer_count_) throw std::runtime_error("layout is missing MoE layers");
    if (experts_.size() != uint64_t(layer_count_) * uint64_t(expert_count_)) {
        throw std::runtime_error("expert route cardinality mismatch");
    }
}

const MMBExpert & MMBModel::expert(uint32_t layer, uint32_t expert_id) const {
    auto it = experts_.find(key(layer,expert_id));
    if (it == experts_.end()) {
        throw std::runtime_error("unknown expert route layer=" + std::to_string(layer) +
                                 " expert=" + std::to_string(expert_id));
    }
    return it->second;
}

const MMBCoreTensor & MMBModel::core_tensor(const std::string & name) const {
    auto it = core_tensors_.find(name);
    if (it == core_tensors_.end()) {
        throw std::runtime_error("unknown core tensor: " + name);
    }
    return it->second;
}

std::vector<uint8_t> MMBModel::read_core(const MMBCoreTensor & tensor, bool verify_sha256) const {
    const auto & region = tensor.region;
    if (region.shard >= shards_.size()) throw std::runtime_error("invalid core shard index");
    const auto & shard = shards_[region.shard];
    if (region.length > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw std::runtime_error("core tensor is too large for this process");
    }
    if (region.offset > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max())) {
        throw std::runtime_error("core tensor offset is too large for stream API");
    }

    std::vector<uint8_t> bytes(static_cast<size_t>(region.length));
    std::ifstream in(shard.path,std::ios::binary);
    if (!in) throw std::runtime_error("cannot open shard: " + shard.path.string());
    in.seekg(static_cast<std::streamoff>(region.offset),std::ios::beg);
    if (!in) throw std::runtime_error("cannot seek shard");
    in.read(reinterpret_cast<char *>(bytes.data()),static_cast<std::streamsize>(bytes.size()));
    if (in.gcount() != static_cast<std::streamsize>(bytes.size())) {
        throw std::runtime_error("short read from MMB core tensor");
    }
    if (verify_sha256) {
        const std::string actual = sha256_hex(bytes.data(),bytes.size());
        if (actual != tensor.sha256) {
            throw std::runtime_error("core tensor SHA-256 mismatch: " + tensor.name);
        }
    }
    return bytes;
}

std::vector<uint8_t> MMBModel::read_expert(const MMBExpert & expert_desc, bool verify_sha256) const {
    if (expert_desc.regions.empty()) throw std::runtime_error("invalid expert descriptor");
    uint32_t sid = expert_desc.regions[0].shard;
    if (sid >= shards_.size()) throw std::runtime_error("invalid shard index");
    const auto & shard = shards_[sid];
    if (expert_desc.block_length > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw std::runtime_error("expert block is too large for this process");
    }
    std::vector<uint8_t> bytes(static_cast<size_t>(expert_desc.block_length));
    std::ifstream in(shard.path,std::ios::binary);
    if (!in) throw std::runtime_error("cannot open shard: " + shard.path.string());
    if (expert_desc.block_offset > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max())) {
        throw std::runtime_error("expert offset is too large for stream API");
    }
    in.seekg(static_cast<std::streamoff>(expert_desc.block_offset),std::ios::beg);
    if (!in) throw std::runtime_error("cannot seek shard");
    in.read(reinterpret_cast<char *>(bytes.data()),static_cast<std::streamsize>(bytes.size()));
    if (in.gcount() != static_cast<std::streamsize>(bytes.size())) {
        throw std::runtime_error("short read from MMB shard");
    }
    if (verify_sha256) {
        std::string actual = sha256_hex(bytes.data(),bytes.size());
        if (actual != expert_desc.sha256) {
            throw std::runtime_error("expert SHA-256 mismatch for layer=" +
                                     std::to_string(expert_desc.layer) + " expert=" +
                                     std::to_string(expert_desc.expert));
        }
    }
    return bytes;
}

} // namespace mmb
