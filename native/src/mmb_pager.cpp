#include "mmb_pager.hpp"
#include <algorithm>
#include <set>
#include <stdexcept>

namespace mmb {

MMBPager::MMBPager(std::shared_ptr<MMBModel> model, uint64_t cache_capacity_bytes,
                   bool verify_sha256)
    : model_(std::move(model)), cache_(cache_capacity_bytes),
      verify_sha256_(verify_sha256) {
    if (!model_) throw std::runtime_error("MMBPager requires a model");
}

std::unique_ptr<MMBLease> MMBPager::acquire(uint32_t layer,
                                            std::span<const uint32_t> experts,
                                            bool router_request) {
    const auto acquire_started = std::chrono::steady_clock::now();
    if (experts.empty()) throw std::runtime_error("MMBPager acquire requires at least one expert");

    std::vector<uint32_t> unique(experts.begin(),experts.end());
    std::sort(unique.begin(),unique.end());
    unique.erase(std::unique(unique.begin(),unique.end()),unique.end());

    std::lock_guard<std::mutex> lock(mutex_);
    auto lease = std::make_unique<MMBLease>();
    lease->id = next_lease_id_++;
    lease->entries.reserve(unique.size());

    try {
        if (router_request) ++stats_.real_router_requests;
        for (uint32_t expert_id : unique) {
            const auto & descriptor = model_->expert(layer,expert_id);
            const uint64_t route_key = key(layer,expert_id);
            CacheEntry * entry = cache_.find(route_key);
            if (entry) {
                ++stats_.hits;
            } else {
                ++stats_.misses;
                const auto io_started = std::chrono::steady_clock::now();
                auto bytes = model_->read_expert(descriptor,verify_sha256_);
                const auto io_done = std::chrono::steady_clock::now();
                stats_.io_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(io_done-io_started).count();
                stats_.bytes_read += bytes.size();
                ++stats_.loads;
                entry = &cache_.insert(route_key,layer,expert_id,std::move(bytes),descriptor.regions);
            }
            cache_.acquire(*entry);
            lease->entries.push_back(entry);
            ++stats_.experts_used;
        }
    } catch (...) {
        for (auto * entry : lease->entries) {
            try { cache_.release(*entry); } catch (...) {}
        }
        lease->entries.clear();
        throw;
    }

    const auto done = std::chrono::steady_clock::now();
    stats_.acquire_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(done-acquire_started).count();
    return lease;
}

void MMBPager::release(MMBLease & lease) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (lease.released) throw std::runtime_error("MMB lease already released");
    for (auto * entry : lease.entries) cache_.release(*entry);
    lease.entries.clear();
    lease.released = true;
}

void MMBPager::record_paged_kernel() {
    std::lock_guard<std::mutex> lock(mutex_);
    ++stats_.paged_kernel_invocations;
}

MMBPagerSnapshot MMBPager::snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return MMBPagerSnapshot{stats_,cache_.stats()};
}

} // namespace mmb
