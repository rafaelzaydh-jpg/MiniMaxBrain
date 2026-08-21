#pragma once
#include "mmb_model.hpp"
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

namespace mmb {

struct CacheEntry {
    uint64_t key = 0;
    uint32_t layer = 0;
    uint32_t expert = 0;
    std::vector<uint8_t> bytes;
    std::array<MMBRegion,3> regions{};
    uint32_t leases = 0;
    uint64_t last_used = 0;
};

struct CacheStats {
    uint64_t resident_bytes = 0;
    uint64_t peak_resident_bytes = 0;
    uint64_t evictions = 0;
};

class MMBCache {
public:
    explicit MMBCache(uint64_t capacity_bytes);

    CacheEntry * find(uint64_t key);
    CacheEntry & insert(uint64_t key, uint32_t layer, uint32_t expert,
                        std::vector<uint8_t> bytes,
                        const std::array<MMBRegion,3> & regions);
    void acquire(CacheEntry & entry);
    void release(CacheEntry & entry);
    void evict_until(uint64_t required_bytes);

    uint64_t capacity_bytes() const { return capacity_bytes_; }
    const CacheStats & stats() const { return stats_; }

private:
    uint64_t capacity_bytes_;
    uint64_t clock_ = 0;
    CacheStats stats_;
    std::unordered_map<uint64_t,std::unique_ptr<CacheEntry>> entries_;
};

} // namespace mmb
