#ifndef TT_GRAPH_PATCH_H
#define TT_GRAPH_PATCH_H

#include <cstdint>

namespace tt {

struct IterParams {
    uint32_t epoch = 0;
    uint32_t seed = 0;
    uint32_t flags = 0;
    uint32_t reserved = 0;
};

struct RecorderGraphControl {
    uint64_t region_mask = ~0ull;
    uint32_t snapshot_period = 0;
    uint32_t flags = 0;
    const uint32_t* region_bitmap = nullptr;
    uint32_t bitmap_words = 0;
    uint32_t reserved0 = 0;
};

static constexpr uint32_t kGraphControlStampsEnabled = 1u << 0;

} // namespace tt

#endif // TT_GRAPH_PATCH_H
