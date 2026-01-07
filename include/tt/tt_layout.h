#ifndef TT_LAYOUT_H
#define TT_LAYOUT_H

#include <cstdint>

namespace tt {

static constexpr uint32_t kChunkMagic = 0x54545243u;
static constexpr uint16_t kChunkVersion = 1u;
static constexpr uint16_t kChunkHeaderBytes = 32u;
static constexpr uint32_t kChunkTypeSnapshot = 1u;

struct alignas(64) ControlBlock {
    uint64_t write_pos;
    uint32_t epoch_id;
    uint32_t epoch_index_pos;
    uint32_t ring_bytes;
    uint32_t epoch_capacity;
    uint32_t region_capacity;
    uint32_t flags;
    uint8_t padding[32];
};

struct alignas(32) TrackedRegion {
    uint64_t base_ptr;
    uint32_t size_bytes;
    uint32_t region_id;
    uint32_t options;
    uint32_t full_snapshot_period;
    uint64_t user_tag;
};

struct alignas(32) EpochRecord {
    uint32_t epoch_id;
    uint32_t chunk_count;
    uint32_t region_count;
    uint32_t reserved0;
    uint32_t ring_offset;
    uint32_t reserved1;
    uint64_t timestamp;
};

struct alignas(16) ChunkHeader {
    uint32_t magic;
    uint16_t version;
    uint16_t header_bytes;
    uint32_t epoch_id;
    uint32_t region_id;
    uint32_t chunk_type;
    uint32_t payload_bytes;
    uint32_t uncompressed_bytes;
    uint32_t flags;
};

static_assert(sizeof(ControlBlock) == 64, "ControlBlock must be 64 bytes");
static_assert(alignof(ControlBlock) == 64, "ControlBlock must be 64-byte aligned");

static_assert(sizeof(TrackedRegion) == 32, "TrackedRegion must be 32 bytes");
static_assert(alignof(TrackedRegion) == 32, "TrackedRegion must be 32-byte aligned");

static_assert(sizeof(EpochRecord) == 32, "EpochRecord must be 32 bytes");
static_assert(alignof(EpochRecord) == 32, "EpochRecord must be 32-byte aligned");

static_assert(sizeof(ChunkHeader) == 32, "ChunkHeader must be 32 bytes");
static_assert(alignof(ChunkHeader) == 16, "ChunkHeader must be 16-byte aligned");

} // namespace tt

#endif // TT_LAYOUT_H
