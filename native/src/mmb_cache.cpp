#include "mmb_cache.hpp"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace mmb {

MMBCache::MMBCache(uint64_t capacity_bytes) : capacity_bytes_(capacity_bytes) {
    if (capacity_bytes_ == 0) throw std::runtime_error("MMB cache capacity must be positive");
}

CacheEntry * MMBCache::find(uint64_t key) {
    auto it = entries_.find(key);
    if (it == entries_.end()) return nullptr;
    it->second->last_used = ++clock_;
    return it->second.get();
}

void MMBCache::evict_until(uint64_t required_bytes) {
    if (required_bytes > capacity_bytes_) {
        throw std::runtime_error("expert block exceeds MMB cache capacity");
    }
    while (stats_.resident_bytes > capacity_bytes_ - required_bytes) {
        auto victim = entries_.end();
        uint64_t oldest = std::numeric_limits<uint64_t>::max();
        for (auto it=entries_.begin(); it!=entries_.end(); ++it) {
            if (it->second->leases == 0 && it->second->last_used < oldest) {
                oldest = it->second->last_used;
                victim = it;
            }
        }
        if (victim == entries_.end()) {
            throw std::runtime_error("MMB cache budget exhausted while all eviction candidates are leased");
        }
        stats_.resident_bytes -= victim->second->bytes.size();
        entries_.erase(victim);
        ++stats_.evictions;
    }
}

CacheEntry & MMBCache::insert(uint64_t key, uint32_t layer, uint32_t expert,
                              std::vector<uint8_t> bytes,
                              const std::array<MMBRegion,3> & regions) {
    if (entries_.find(key) != entries_.end()) throw std::runtime_error("duplicate MMB cache insert");
    evict_until(bytes.size());
    auto entry = std::make_unique<CacheEntry>();
    entry->key = key;
    entry->layer = layer;
    entry->expert = expert;
    entry->bytes = std::move(bytes);
    entry->regions = regions;
    entry->last_used = ++clock_;
    stats_.resident_bytes += entry->bytes.size();
    stats_.peak_resident_bytes = std::max(stats_.peak_resident_bytes,stats_.resident_bytes);
    CacheEntry * raw = entry.get();
    entries_.emplace(key,std::move(entry));
    return *raw;
}

void MMBCache::acquire(CacheEntry & entry) {
    ++entry.leases;
    entry.last_used = ++clock_;
}

void MMBCache::release(CacheEntry & entry) {
    if (entry.leases == 0) throw std::runtime_error("MMB cache lease underflow");
    --entry.leases;
    entry.last_used = ++clock_;
}

} // namespace mmb
