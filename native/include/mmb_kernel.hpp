#pragma once
#include "mmb_pager.hpp"
#include <cstdint>
#include <span>

namespace mmb {

struct KernelSegment {
    const uint8_t * data = nullptr;
    uint64_t bytes = 0;
    int32_t ggml_type = -1;
    uint32_t n_dims = 0;
    std::array<int64_t,4> ne{1,1,1,1};
};

// RAII boundary intended for the GGML MUL_MAT_ID integration.
// It keeps all selected experts leased for the entire kernel and records
// paged execution only when commit_after_compute() is called.
class MMBKernelLease {
public:
    MMBKernelLease(MMBPager & pager, uint32_t layer,
                   std::span<const int32_t> router_ids);
    ~MMBKernelLease();

    MMBKernelLease(const MMBKernelLease &) = delete;
    MMBKernelLease & operator=(const MMBKernelLease &) = delete;

    KernelSegment segment(uint32_t expert, TensorRole role) const;
    void commit_after_compute();

private:
    MMBPager * pager_ = nullptr;
    std::unique_ptr<MMBLease> lease_;
    bool committed_ = false;
};

} // namespace mmb
