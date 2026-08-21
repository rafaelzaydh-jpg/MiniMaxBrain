#pragma once

#include "mmb_pager.hpp"

namespace mmb {

// Registers one MMBPager as the process-wide CPU MoE expert provider used by
// the pinned GGML build. The first integration intentionally supports one
// active provider per process; construction fails if another is active.
class MMBGGMLProviderRegistration {
public:
    explicit MMBGGMLProviderRegistration(MMBPager & pager);
    ~MMBGGMLProviderRegistration();

    MMBGGMLProviderRegistration(const MMBGGMLProviderRegistration &) = delete;
    MMBGGMLProviderRegistration & operator=(const MMBGGMLProviderRegistration &) = delete;

private:
    MMBPager * pager_ = nullptr;
};

} // namespace mmb
