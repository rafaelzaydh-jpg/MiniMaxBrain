#pragma once
#include "mmb_cache.hpp"
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

namespace mmb {

struct PagerStats {
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t bytes_read = 0;
    uint64_t loads = 0;
    uint64_t real_router_requests = 0;
    uint64_t experts_used = 0;
    uint64_t acquire_ns = 0;
    uint64_t io_ns = 0;
    uint64_t paged_kernel_invocations = 0;
};

struct MMBLease {
    uint64_t id = 0;
    std::vector<CacheEntry *> entries;
    bool released = false;
};

struct MMBPagerSnapshot {
    PagerStats pager{};
    CacheStats cache{};
};

class MMBPager {
public:
    MMBPager(std::shared_ptr<MMBModel> model, uint64_t cache_capacity_bytes,
             bool verify_sha256);

    std::unique_ptr<MMBLease> acquire(uint32_t layer, std::span<const uint32_t> experts,
                                      bool router_request = false);
    void release(MMBLease & lease);

    // Called only by a real GGML integration after a kernel has completed
    // using MMB-backed segment pointers. It is intentionally not exposed
    // through the public C/Python ABI.
    void record_paged_kernel();

    const MMBModel & model() const { return *model_; }
    const MMBCache & cache() const { return cache_; }
    const PagerStats & stats() const { return stats_; } // tests/internal code under single-threaded ownership
    MMBPagerSnapshot snapshot() const;

private:
    static uint64_t key(uint32_t layer, uint32_t expert) {
        return (uint64_t(layer)<<32)|uint64_t(expert);
    }

    std::shared_ptr<MMBModel> model_;
    MMBCache cache_;
    bool verify_sha256_;
    uint64_t next_lease_id_ = 1;
    PagerStats stats_;
    mutable std::mutex mutex_;
};

} // namespace mmb
