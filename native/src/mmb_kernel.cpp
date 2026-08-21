#include "mmb_kernel.hpp"
#include <limits>
#include <stdexcept>
#include <vector>

namespace mmb {

MMBKernelLease::MMBKernelLease(MMBPager & pager, uint32_t layer,
                               std::span<const int32_t> router_ids)
    : pager_(&pager) {
    if (router_ids.empty()) throw std::runtime_error("router produced no expert IDs");
    std::vector<uint32_t> ids;
    ids.reserve(router_ids.size());
    for (int32_t raw : router_ids) {
        if (raw < 0) throw std::runtime_error("router produced a negative expert ID");
        ids.push_back(static_cast<uint32_t>(raw));
    }
    lease_ = pager.acquire(layer,ids,true);
}

MMBKernelLease::~MMBKernelLease() {
    if (pager_ && lease_ && !lease_->released) {
        try { pager_->release(*lease_); } catch (...) {}
    }
}

KernelSegment MMBKernelLease::segment(uint32_t expert_id, TensorRole role) const {
    if (!lease_) throw std::runtime_error("kernel lease is not active");
    const size_t role_idx = static_cast<size_t>(role);
    if (role_idx >= 3) throw std::runtime_error("invalid expert tensor role");
    for (auto * entry : lease_->entries) {
        if (entry->expert != expert_id) continue;
        const auto & region = entry->regions[role_idx];
        if (region.block_offset > entry->bytes.size() ||
            region.length > entry->bytes.size() - region.block_offset) {
            throw std::runtime_error("MMB kernel segment exceeds cached expert block");
        }
        KernelSegment out;
        out.data = entry->bytes.data() + static_cast<size_t>(region.block_offset);
        out.bytes = region.length;
        out.ggml_type = region.ggml_type;
        out.n_dims = region.n_dims;
        out.ne = region.ne;
        return out;
    }
    throw std::runtime_error("requested expert is not part of the active router lease");
}

void MMBKernelLease::commit_after_compute() {
    if (!pager_ || !lease_ || lease_->released) throw std::runtime_error("kernel lease is not active");
    if (committed_) throw std::runtime_error("kernel lease already committed");
    pager_->record_paged_kernel();
    pager_->release(*lease_);
    committed_ = true;
}

} // namespace mmb
