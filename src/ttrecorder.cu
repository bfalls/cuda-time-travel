#include "tt/ttrecorder.h"

#include <cstring>
#include <vector>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <locale>
#include <sstream>

#include "tt/tt_graph.h"
#include "tt/tt_layout.h"

namespace tt {

struct DeviceEpochBegin {
    uint32_t epoch_id;
    uint32_t epoch_index_pos;
    uint64_t write_pos_before;
};

namespace {

static constexpr uint32_t kRingAlignment = 32u;
static constexpr uint32_t kStampsPerEpoch = 3u;
static constexpr uint32_t kStampEpochStart = 0u;
static constexpr uint32_t kStampAfterRegions = 1u;
static constexpr uint32_t kStampEpochEnd = 2u;
static constexpr uint32_t kHashThreads = 256u;
static constexpr uint64_t kHashOffset = 14695981039346656037ull;
static constexpr uint64_t kHashPrime = 1099511628211ull;

__device__ inline bool graph_control_stamps_enabled(const RecorderGraphControl* control) {
    if (!control) {
        return true;
    }
    return (control->flags & kGraphControlStampsEnabled) != 0u;
}

__device__ inline bool graph_control_region_enabled(const RecorderGraphControl* control, uint32_t region_id) {
    if (!control) {
        return true;
    }
    if (control->region_bitmap && control->bitmap_words > 0u) {
        const uint32_t word = region_id / 32u;
        if (word >= control->bitmap_words) {
            return true;
        }
        const uint32_t bit = region_id % 32u;
        const uint32_t mask = 1u << bit;
        return (control->region_bitmap[word] & mask) != 0u;
    }
    if (region_id >= 64u) {
        return true;
    }
    const uint64_t mask = 1ull << region_id;
    return (control->region_mask & mask) != 0ull;
}

__device__ inline bool graph_control_snapshot_allowed(const RecorderGraphControl* control, uint32_t epoch_id) {
    if (!control || control->snapshot_period == 0u) {
        return true;
    }
    return (epoch_id % control->snapshot_period) == 0u;
}

__host__ __device__ inline uint32_t align_up(uint32_t value) {
    return (value + (kRingAlignment - 1u)) & ~(kRingAlignment - 1u);
}

__global__ void reserve_stamp_range_kernel(uint32_t* counter, uint32_t* out_base, uint32_t count) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (!counter || !out_base) {
            return;
        }
        const uint32_t base = atomicAdd(counter, count);
        *out_base = base;
    }
}

__global__ void reserve_stamp_range_controlled_kernel(const RecorderGraphControl* control,
    uint32_t* counter,
    uint32_t* out_base,
    uint32_t count) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (!graph_control_stamps_enabled(control)) {
            return;
        }
        if (!counter || !out_base) {
            return;
        }
        const uint32_t base = atomicAdd(counter, count);
        *out_base = base;
    }
}

__global__ void stamp_kernel(uint64_t* stamps, const uint32_t* base, uint32_t offset) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (!stamps || !base) {
            return;
        }
        stamps[base[0] + offset] = clock64();
    }
}

__global__ void stamp_controlled_kernel(const RecorderGraphControl* control,
    uint64_t* stamps,
    const uint32_t* base,
    uint32_t offset) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (!graph_control_stamps_enabled(control)) {
            return;
        }
        if (!stamps || !base) {
            return;
        }
        stamps[base[0] + offset] = clock64();
    }
}

__device__ inline uint64_t hash_mix(uint64_t hash, uint64_t value) {
    hash ^= value;
    hash *= kHashPrime;
    return hash;
}

__global__ void hash_region_kernel(const uint8_t* data, uint32_t size_bytes, uint64_t* out_hash) {
    __shared__ uint64_t partial[kHashThreads];
    if (threadIdx.x >= kHashThreads) {
        return;
    }
    if (!out_hash) {
        return;
    }
    if (!data || size_bytes == 0u) {
        if (threadIdx.x == 0) {
            *out_hash = 0u;
        }
        return;
    }

    uint64_t local = 0u;
    const uint32_t word_count = size_bytes / 4u;
    const uint32_t* words = reinterpret_cast<const uint32_t*>(data);
    for (uint32_t i = threadIdx.x; i < word_count; i += blockDim.x) {
        local = hash_mix(local, static_cast<uint64_t>(words[i]));
    }
    partial[threadIdx.x] = local;
    __syncthreads();

    if (threadIdx.x == 0) {
        uint64_t hash = kHashOffset;
        for (uint32_t i = 0; i < kHashThreads; ++i) {
            hash = hash_mix(hash, partial[i]);
        }
        *out_hash = hash;
    }
}

__device__ inline void reserve_and_get_ring_offset(ControlBlock* control,
    uint8_t* ring,
    uint32_t ring_bytes,
    uint32_t total_bytes_aligned,
    uint32_t* out_ring_offset,
    uint32_t* out_used_wrap_marker) {
    uint64_t current_pos = atomicAdd(&control->write_pos, 0ull);
    uint32_t ring_offset = static_cast<uint32_t>(current_pos % ring_bytes);
    uint32_t used_wrap = 0;

    if (ring_offset + total_bytes_aligned > ring_bytes) {
        uint32_t marker_bytes = align_up(static_cast<uint32_t>(sizeof(ChunkHeader)));
        uint32_t remainder = ring_bytes - ring_offset;
        if (remainder >= marker_bytes) {
            uint64_t marker_pos = atomicAdd(&control->write_pos, static_cast<uint64_t>(marker_bytes));
            uint32_t marker_offset = static_cast<uint32_t>(marker_pos % ring_bytes);
            ChunkHeader* marker = reinterpret_cast<ChunkHeader*>(ring + marker_offset);
            marker->magic = kChunkMagic;
            marker->version = kChunkVersion;
            marker->header_bytes = kChunkHeaderBytes;
            marker->epoch_id = 0;
            marker->region_id = 0;
            marker->chunk_type = kChunkTypeWrapMarker;
            marker->payload_bytes = 0;
            marker->uncompressed_bytes = 0;
            marker->flags = kChunkFlagWrapMarker;

            if (remainder > marker_bytes) {
                uint32_t pad_bytes = remainder - marker_bytes;
                atomicAdd(&control->write_pos, static_cast<uint64_t>(pad_bytes));
            }
        } else if (remainder > 0u) {
            atomicAdd(&control->write_pos, static_cast<uint64_t>(remainder));
        }

        current_pos = atomicAdd(&control->write_pos, static_cast<uint64_t>(total_bytes_aligned));
        ring_offset = static_cast<uint32_t>(current_pos % ring_bytes);
        used_wrap = 1;
    } else {
        current_pos = atomicAdd(&control->write_pos, static_cast<uint64_t>(total_bytes_aligned));
        ring_offset = static_cast<uint32_t>(current_pos % ring_bytes);
    }

    if (out_ring_offset) {
        *out_ring_offset = ring_offset;
    }
    if (out_used_wrap_marker) {
        *out_used_wrap_marker = used_wrap;
    }
}

__global__ void begin_epoch_kernel(ControlBlock* control, DeviceEpochBegin* out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        out->write_pos_before = control->write_pos;
        uint32_t epoch_id = atomicAdd(&control->epoch_id, 1u);
        uint32_t epoch_index_pos = atomicAdd(&control->epoch_index_pos, 1u);
        out->epoch_id = epoch_id;
        out->epoch_index_pos = (control->epoch_capacity > 0) ? (epoch_index_pos % control->epoch_capacity) : 0u;
    }
}

__global__ void snapshot_region_kernel(ControlBlock* control,
    const TrackedRegion* region,
    const uint64_t* baseline_ptrs,
    uint8_t* ring,
    uint32_t ring_bytes,
    uint32_t epoch_id,
    uint32_t* first_ring_offset,
    uint32_t* first_was_written) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if ((region->options & 1u) == 0u || region->size_bytes == 0) {
            return;
        }

        uint32_t payload_bytes = region->size_bytes;
        uint32_t total_bytes = align_up(static_cast<uint32_t>(sizeof(ChunkHeader)) + payload_bytes);
        uint32_t ring_offset = 0;
        reserve_and_get_ring_offset(control, ring, ring_bytes, total_bytes, &ring_offset, nullptr);

        if (first_ring_offset) {
            if (!first_was_written || atomicCAS(first_was_written, 0u, 1u) == 0u) {
                *first_ring_offset = ring_offset;
            }
        }

        ChunkHeader* header = reinterpret_cast<ChunkHeader*>(ring + ring_offset);
        header->magic = kChunkMagic;
        header->version = kChunkVersion;
        header->header_bytes = kChunkHeaderBytes;
        header->epoch_id = epoch_id;
        header->region_id = region->region_id;
        header->chunk_type = kChunkTypeSnapshot;
        header->payload_bytes = payload_bytes;
        header->uncompressed_bytes = payload_bytes;
        header->flags = 0;

        uint8_t* payload_dst = ring + ring_offset + sizeof(ChunkHeader);
        const uint8_t* payload_src = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(region->base_ptr));

        const uint32_t word_count = payload_bytes / 4u;
        uint32_t* dst_words = reinterpret_cast<uint32_t*>(payload_dst);
        const uint32_t* src_words = reinterpret_cast<const uint32_t*>(payload_src);
        for (uint32_t i = 0; i < word_count; ++i) {
            dst_words[i] = src_words[i];
        }

        if (baseline_ptrs) {
            uint8_t* baseline_ptr = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(baseline_ptrs[region->region_id]));
            if (baseline_ptr) {
                uint32_t* baseline_words = reinterpret_cast<uint32_t*>(baseline_ptr);
                for (uint32_t i = 0; i < word_count; ++i) {
                    baseline_words[i] = src_words[i];
                }
            }
        }
    }
}

__global__ void snapshot_region_graph_kernel(ControlBlock* control,
    const TrackedRegion* region,
    const uint64_t* baseline_ptrs,
    uint8_t* ring,
    uint32_t ring_bytes,
    const DeviceEpochBegin* begin,
    const RecorderGraphControl* graph_control,
    uint32_t* first_ring_offset,
    uint32_t* first_was_written,
    uint32_t* enabled_count) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (!begin) {
            return;
        }
        if ((region->options & 1u) == 0u || region->size_bytes == 0) {
            return;
        }
        if (!graph_control_snapshot_allowed(graph_control, begin->epoch_id)) {
            return;
        }
        if (!graph_control_region_enabled(graph_control, region->region_id)) {
            return;
        }

        uint32_t payload_bytes = region->size_bytes;
        uint32_t total_bytes = align_up(static_cast<uint32_t>(sizeof(ChunkHeader)) + payload_bytes);
        uint32_t ring_offset = 0;
        reserve_and_get_ring_offset(control, ring, ring_bytes, total_bytes, &ring_offset, nullptr);

        if (first_ring_offset) {
            if (!first_was_written || atomicCAS(first_was_written, 0u, 1u) == 0u) {
                *first_ring_offset = ring_offset;
            }
        }

        ChunkHeader* header = reinterpret_cast<ChunkHeader*>(ring + ring_offset);
        header->magic = kChunkMagic;
        header->version = kChunkVersion;
        header->header_bytes = kChunkHeaderBytes;
        header->epoch_id = begin->epoch_id;
        header->region_id = region->region_id;
        header->chunk_type = kChunkTypeSnapshot;
        header->payload_bytes = payload_bytes;
        header->uncompressed_bytes = payload_bytes;
        header->flags = 0;

        uint8_t* payload_dst = ring + ring_offset + sizeof(ChunkHeader);
        const uint8_t* payload_src = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(region->base_ptr));

        const uint32_t word_count = payload_bytes / 4u;
        uint32_t* dst_words = reinterpret_cast<uint32_t*>(payload_dst);
        const uint32_t* src_words = reinterpret_cast<const uint32_t*>(payload_src);
        for (uint32_t i = 0; i < word_count; ++i) {
            dst_words[i] = src_words[i];
        }

        if (baseline_ptrs) {
            uint8_t* baseline_ptr = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(baseline_ptrs[region->region_id]));
            if (baseline_ptr) {
                uint32_t* baseline_words = reinterpret_cast<uint32_t*>(baseline_ptr);
                for (uint32_t i = 0; i < word_count; ++i) {
                    baseline_words[i] = src_words[i];
                }
            }
        }
        if (enabled_count) {
            atomicAdd(enabled_count, 1u);
        }
    }
}

__global__ void delta_prepare_kernel(const TrackedRegion* region,
    const uint64_t* baseline_ptrs,
    const uint64_t* scratch_ptrs,
    uint32_t* out_payload_bytes) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if ((region->options & 1u) == 0u || region->size_bytes == 0) {
            if (out_payload_bytes) {
                out_payload_bytes[region->region_id] = 0u;
            }
            return;
        }
        if ((region->size_bytes % 4u) != 0u) {
            if (out_payload_bytes) {
                out_payload_bytes[region->region_id] = 0u;
            }
            return;
        }

        const uint8_t* baseline_ptr = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(baseline_ptrs[region->region_id]));
        uint8_t* scratch_ptr = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(scratch_ptrs[region->region_id]));
        if (baseline_ptr == nullptr || scratch_ptr == nullptr) {
            if (out_payload_bytes) {
                out_payload_bytes[region->region_id] = 0u;
            }
            return;
        }

        const uint8_t* current_ptr = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(region->base_ptr));
        const uint32_t word_count = region->size_bytes / 4u;
        const uint32_t* baseline_words = reinterpret_cast<const uint32_t*>(baseline_ptr);
        const uint32_t* current_words = reinterpret_cast<const uint32_t*>(current_ptr);
        uint32_t* scratch_words = reinterpret_cast<uint32_t*>(scratch_ptr);

        uint32_t scratch_index = 2u;
        uint32_t block_count = 0u;
        uint32_t i = 0u;
        while (i < word_count) {
            uint32_t x = baseline_words[i] ^ current_words[i];
            if (x == 0u) {
                uint32_t run_len = 1u;
                ++i;
                while (i < word_count) {
                    x = baseline_words[i] ^ current_words[i];
                    if (x != 0u) {
                        break;
                    }
                    ++run_len;
                    ++i;
                }
                scratch_words[scratch_index++] = 0x80000000u | run_len;
                ++block_count;
            } else {
                uint32_t run_len = 1u;
                scratch_words[scratch_index++] = 0x00000000u | run_len;
                scratch_words[scratch_index++] = x;
                ++i;
                while (i < word_count) {
                    x = baseline_words[i] ^ current_words[i];
                    if (x == 0u) {
                        break;
                    }
                    ++run_len;
                    scratch_words[scratch_index++] = x;
                    ++i;
                }
                scratch_words[scratch_index - run_len - 1u] = 0x00000000u | run_len;
                ++block_count;
            }
        }

        scratch_words[0] = word_count;
        scratch_words[1] = block_count;
        if (out_payload_bytes) {
            out_payload_bytes[region->region_id] = scratch_index * sizeof(uint32_t);
        }
    }
}

__global__ void delta_write_kernel(ControlBlock* control,
    const TrackedRegion* region,
    const uint64_t* baseline_ptrs,
    const uint64_t* scratch_ptrs,
    uint8_t* ring,
    uint32_t ring_bytes,
    uint32_t epoch_id,
    uint32_t payload_bytes,
    uint32_t* first_ring_offset,
    uint32_t* first_was_written) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if ((region->options & 1u) == 0u || region->size_bytes == 0) {
            return;
        }
        if ((region->size_bytes % 4u) != 0u) {
            return;
        }
        if (payload_bytes == 0u) {
            return;
        }

        const uint8_t* scratch_ptr = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(scratch_ptrs[region->region_id]));
        if (scratch_ptr == nullptr) {
            return;
        }

        uint32_t total_bytes = align_up(static_cast<uint32_t>(sizeof(ChunkHeader)) + payload_bytes);
        uint32_t ring_offset = 0;
        reserve_and_get_ring_offset(control, ring, ring_bytes, total_bytes, &ring_offset, nullptr);

        if (first_ring_offset) {
            if (!first_was_written || atomicCAS(first_was_written, 0u, 1u) == 0u) {
                *first_ring_offset = ring_offset;
            }
        }

        ChunkHeader* header = reinterpret_cast<ChunkHeader*>(ring + ring_offset);
        header->magic = kChunkMagic;
        header->version = kChunkVersion;
        header->header_bytes = kChunkHeaderBytes;
        header->epoch_id = epoch_id;
        header->region_id = region->region_id;
        header->chunk_type = kChunkTypeDeltaXorRle0;
        header->payload_bytes = payload_bytes;
        header->uncompressed_bytes = region->size_bytes;
        header->flags = 0;

        uint8_t* payload_dst = ring + ring_offset + sizeof(ChunkHeader);
        for (uint32_t b = 0; b < payload_bytes; ++b) {
            payload_dst[b] = scratch_ptr[b];
        }

        if (baseline_ptrs) {
            const uint8_t* current_ptr = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(region->base_ptr));
            uint8_t* baseline_ptr = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(baseline_ptrs[region->region_id]));
            if (baseline_ptr && current_ptr) {
                const uint32_t word_count = region->size_bytes / 4u;
                const uint32_t* current_words = reinterpret_cast<const uint32_t*>(current_ptr);
                uint32_t* baseline_out = reinterpret_cast<uint32_t*>(baseline_ptr);
                for (uint32_t w = 0; w < word_count; ++w) {
                    baseline_out[w] = current_words[w];
                }
            }
        }
    }
}

__global__ void finalize_epoch_kernel(ControlBlock* control,
    const DeviceEpochBegin* begin,
    EpochRecord* epochs,
    uint32_t epoch_capacity,
    const uint32_t* enabled_count,
    const uint32_t* first_ring_offset) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (!begin || !epochs || epoch_capacity == 0u) {
            return;
        }

        uint32_t enabled = 0u;
        if (enabled_count) {
            enabled = *enabled_count;
        }
        uint64_t epoch_bytes = control->write_pos - begin->write_pos_before;
        EpochRecord record{};
        record.epoch_id = begin->epoch_id;
        record.chunk_count = enabled;
        record.region_count = enabled;
        record.reserved0 = (epoch_bytes > UINT32_MAX) ? 0u : static_cast<uint32_t>(epoch_bytes);
        record.ring_offset = (enabled > 0u && first_ring_offset) ? *first_ring_offset : 0u;
        record.reserved1 = begin->epoch_id / epoch_capacity;
        record.timestamp = 0;
        const uint32_t index = begin->epoch_index_pos % epoch_capacity;
        epochs[index] = record;
    }
}

__global__ void apply_chunk_kernel(const TrackedRegion* regions,
    uint32_t region_capacity,
    const uint64_t* baseline_ptrs,
    const uint8_t* ring,
    uint32_t ring_offset) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        const ChunkHeader* header = reinterpret_cast<const ChunkHeader*>(ring + ring_offset);
        if (header->magic != kChunkMagic || header->chunk_type != kChunkTypeSnapshot) {
            return;
        }

        const uint32_t region_id = header->region_id;
        if (region_id >= region_capacity) {
            return;
        }

        const TrackedRegion region = regions[region_id];
        if ((region.options & 1u) == 0u) {
            return;
        }

        uint32_t payload_bytes = header->payload_bytes;
        uint8_t* payload_dst = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(region.base_ptr));
        const uint8_t* payload_src = ring + ring_offset + sizeof(ChunkHeader);

        const uint32_t word_count = payload_bytes / 4u;
        uint32_t* dst_words = reinterpret_cast<uint32_t*>(payload_dst);
        const uint32_t* src_words = reinterpret_cast<const uint32_t*>(payload_src);
        for (uint32_t i = 0; i < word_count; ++i) {
            dst_words[i] = src_words[i];
        }

        if (baseline_ptrs) {
            uint8_t* baseline_ptr = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(baseline_ptrs[region_id]));
            if (baseline_ptr) {
                uint32_t* baseline_words = reinterpret_cast<uint32_t*>(baseline_ptr);
                for (uint32_t i = 0; i < word_count; ++i) {
                    baseline_words[i] = src_words[i];
                }
            }
        }
    }
}

__global__ void apply_delta_chunk_kernel(const TrackedRegion* regions,
    uint32_t region_capacity,
    const uint64_t* baseline_ptrs,
    const uint8_t* ring,
    uint32_t ring_offset) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        const ChunkHeader* header = reinterpret_cast<const ChunkHeader*>(ring + ring_offset);
        if (header->magic != kChunkMagic || header->chunk_type != kChunkTypeDeltaXorRle0) {
            return;
        }

        const uint32_t region_id = header->region_id;
        if (region_id >= region_capacity) {
            return;
        }

        const TrackedRegion region = regions[region_id];
        if ((region.options & 1u) == 0u) {
            return;
        }
        if ((region.size_bytes % 4u) != 0u) {
            return;
        }

        uint8_t* baseline_ptr = nullptr;
        if (baseline_ptrs) {
            baseline_ptr = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(baseline_ptrs[region_id]));
        }
        if (baseline_ptr == nullptr || region.base_ptr == 0) {
            return;
        }

        uint32_t* baseline_words = reinterpret_cast<uint32_t*>(baseline_ptr);
        uint32_t* region_words = reinterpret_cast<uint32_t*>(static_cast<uintptr_t>(region.base_ptr));
        const uint32_t* payload_words = reinterpret_cast<const uint32_t*>(ring + ring_offset + sizeof(ChunkHeader));

        const uint32_t word_count = payload_words[0];
        const uint32_t block_count = payload_words[1];
        if (word_count > (region.size_bytes / 4u)) {
            return;
        }

        uint32_t payload_index = 2u;
        uint32_t out_index = 0u;
        for (uint32_t block = 0; block < block_count && out_index < word_count; ++block) {
            const uint32_t tag = payload_words[payload_index++];
            const uint32_t run_len = tag & 0x7FFFFFFFu;
            const bool is_zero = ((tag & 0x80000000u) != 0u);
            if (run_len == 0u) {
                continue;
            }
            if (is_zero) {
                for (uint32_t j = 0; j < run_len && out_index < word_count; ++j) {
                    const uint32_t value = baseline_words[out_index];
                    region_words[out_index] = value;
                    baseline_words[out_index] = value;
                    ++out_index;
                }
            } else {
                for (uint32_t j = 0; j < run_len && out_index < word_count; ++j) {
                    const uint32_t xor_value = payload_words[payload_index++];
                    const uint32_t value = baseline_words[out_index] ^ xor_value;
                    region_words[out_index] = value;
                    baseline_words[out_index] = value;
                    ++out_index;
                }
            }
        }
    }
}

RecorderStatus read_chunk_header(const uint8_t* ring,
    uint32_t ring_bytes,
    uint32_t ring_offset,
    ChunkHeader* out_header,
    cudaStream_t stream) {
    if (ring_offset % kRingAlignment != 0u) {
        return RecorderStatus::kAlignmentError;
    }
    if (ring_offset + sizeof(ChunkHeader) > ring_bytes) {
        return RecorderStatus::kInvalidHeader;
    }

    cudaError_t err = cudaMemcpyAsync(out_header,
        ring + ring_offset,
        sizeof(ChunkHeader),
        cudaMemcpyDeviceToHost,
        stream);
    if (err != cudaSuccess) {
        return RecorderStatus::kCudaError;
    }
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        return RecorderStatus::kCudaError;
    }

    if (out_header->magic != kChunkMagic) {
        return RecorderStatus::kInvalidHeader;
    }
    if (out_header->version != kChunkVersion) {
        return RecorderStatus::kInvalidHeader;
    }
    if (out_header->header_bytes != kChunkHeaderBytes || out_header->header_bytes != sizeof(ChunkHeader)) {
        return RecorderStatus::kInvalidHeader;
    }
    if ((out_header->header_bytes % kRingAlignment) != 0u) {
        return RecorderStatus::kAlignmentError;
    }

    if (out_header->chunk_type != kChunkTypeSnapshot &&
        out_header->chunk_type != kChunkTypeDeltaXorRle0 &&
        out_header->chunk_type != kChunkTypeWrapMarker) {
        return RecorderStatus::kInvalidChunkType;
    }

    const uint32_t payload_bytes = out_header->payload_bytes;
    if (payload_bytes > ring_bytes - out_header->header_bytes) {
        return RecorderStatus::kInvalidPayload;
    }
    if (ring_offset + out_header->header_bytes + payload_bytes > ring_bytes) {
        return RecorderStatus::kInvalidPayload;
    }

    if (out_header->chunk_type == kChunkTypeWrapMarker) {
        if (payload_bytes != 0u) {
            return RecorderStatus::kInvalidPayload;
        }
        return RecorderStatus::kOk;
    }

    if (payload_bytes == 0u || (payload_bytes % 4u) != 0u) {
        return RecorderStatus::kInvalidPayload;
    }
    if ((out_header->chunk_type == kChunkTypeDeltaXorRle0) &&
        (out_header->uncompressed_bytes == 0u || (out_header->uncompressed_bytes % 4u) != 0u)) {
        return RecorderStatus::kInvalidPayload;
    }

    return RecorderStatus::kOk;
}

RecorderStatus rewind_apply_epoch(const EpochRecord& record,
    TrackedRegion* d_regions,
    uint32_t region_capacity,
    const uint64_t* baseline_ptrs,
    uint8_t* ring,
    uint32_t ring_bytes,
    cudaStream_t stream) {
    if (record.chunk_count == 0) {
        return RecorderStatus::kOk;
    }

    uint32_t ring_offset = record.ring_offset;
    uint32_t applied = 0;
    uint32_t guard = 0;
    while (applied < record.chunk_count) {
        if (guard++ > record.chunk_count + 4u) {
            return RecorderStatus::kRingCorrupt;
        }
        ChunkHeader host_header{};
        RecorderStatus status = read_chunk_header(ring, ring_bytes, ring_offset, &host_header, stream);
        if (status != RecorderStatus::kOk) {
            return status;
        }

        if (host_header.chunk_type == kChunkTypeWrapMarker || (host_header.flags & kChunkFlagWrapMarker) != 0u) {
            if (ring_offset == 0u) {
                return RecorderStatus::kRingCorrupt;
            }
            ring_offset = 0u;
            continue;
        }

        if (host_header.chunk_type == kChunkTypeSnapshot) {
            apply_chunk_kernel<<<1, 1, 0, stream>>>(d_regions, region_capacity, baseline_ptrs, ring, ring_offset);
        } else if (host_header.chunk_type == kChunkTypeDeltaXorRle0) {
            apply_delta_chunk_kernel<<<1, 1, 0, stream>>>(d_regions, region_capacity, baseline_ptrs, ring, ring_offset);
        } else {
            return RecorderStatus::kInvalidChunkType;
        }
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            return RecorderStatus::kCudaError;
        }

        uint32_t chunk_bytes = align_up(static_cast<uint32_t>(sizeof(ChunkHeader)) + host_header.payload_bytes);
        ring_offset += chunk_bytes;
        if (ring_offset >= ring_bytes) {
            ring_offset -= ring_bytes;
        }
        ++applied;
    }

    return RecorderStatus::kOk;
}

struct GraphKernelTagRegistrar {
    GraphKernelTagRegistrar() {
        RegisterGraphKernelTag(reinterpret_cast<const void*>(begin_epoch_kernel), "tt.begin_epoch");
        RegisterGraphKernelTag(reinterpret_cast<const void*>(snapshot_region_graph_kernel), "tt.snapshot_region_graph");
        RegisterGraphKernelTag(reinterpret_cast<const void*>(finalize_epoch_kernel), "tt.finalize_epoch");
        RegisterGraphKernelTag(reinterpret_cast<const void*>(stamp_controlled_kernel), "tt.graph_stamp");
        RegisterGraphKernelTag(reinterpret_cast<const void*>(reserve_stamp_range_controlled_kernel), "tt.graph_stamp_reserve");
    }
};

static GraphKernelTagRegistrar kGraphKernelTagRegistrar;

} // namespace

bool Recorder::init(const RecorderConfig& cfg) {
    if (initialized_) {
        shutdown();
    }

    last_status_ = RecorderStatus::kOk;
    if (cfg.ring_bytes == 0 || cfg.epoch_capacity == 0 || cfg.region_capacity == 0) {
        last_status_ = RecorderStatus::kInvalidConfig;
        return false;
    }
    if ((cfg.ring_bytes % kRingAlignment) != 0u) {
        last_status_ = RecorderStatus::kInvalidConfig;
        return false;
    }
    if (cfg.retention_epochs > 0 && cfg.retention_epochs > cfg.epoch_capacity) {
        last_status_ = RecorderStatus::kInvalidConfig;
        return false;
    }
    if (cfg.enable_graph_stamps && (!cfg.graph_stamps || !cfg.graph_stamp_counter)) {
        last_status_ = RecorderStatus::kInvalidConfig;
        return false;
    }

    cfg_ = cfg;
    enable_graph_stamps_ = cfg.enable_graph_stamps;
    d_graph_stamps_ = cfg.graph_stamps;
    d_graph_stamp_counter_ = cfg.graph_stamp_counter;
    deterministic_ = cfg.deterministic;
    enable_manifest_ = cfg.enable_manifest;
    deterministic_stream_ = nullptr;
    deterministic_stream_set_ = false;
    manifest_epochs_.clear();

    cudaError_t err = cudaSuccess;
    err = cudaMalloc(reinterpret_cast<void**>(&d_control_), sizeof(ControlBlock));
    if (err != cudaSuccess) {
        shutdown();
        return false;
    }
    err = cudaMalloc(reinterpret_cast<void**>(&d_ring_), cfg_.ring_bytes);
    if (err != cudaSuccess) {
        shutdown();
        return false;
    }
    err = cudaMalloc(reinterpret_cast<void**>(&d_epochs_), sizeof(EpochRecord) * cfg_.epoch_capacity);
    if (err != cudaSuccess) {
        shutdown();
        return false;
    }
    err = cudaMalloc(reinterpret_cast<void**>(&d_regions_), sizeof(TrackedRegion) * cfg_.region_capacity);
    if (err != cudaSuccess) {
        shutdown();
        return false;
    }
    err = cudaMalloc(reinterpret_cast<void**>(&d_baseline_ptrs_), sizeof(uint64_t) * cfg_.region_capacity);
    if (err != cudaSuccess) {
        shutdown();
        return false;
    }
    err = cudaMalloc(reinterpret_cast<void**>(&d_scratch_ptrs_), sizeof(uint64_t) * cfg_.region_capacity);
    if (err != cudaSuccess) {
        shutdown();
        return false;
    }
    err = cudaMalloc(reinterpret_cast<void**>(&d_first_ring_offset_), sizeof(uint32_t));
    if (err != cudaSuccess) {
        shutdown();
        return false;
    }
    err = cudaMalloc(reinterpret_cast<void**>(&d_first_was_written_), sizeof(uint32_t));
    if (err != cudaSuccess) {
        shutdown();
        return false;
    }
    err = cudaMalloc(reinterpret_cast<void**>(&d_delta_sizes_), sizeof(uint32_t) * cfg_.region_capacity);
    if (err != cudaSuccess) {
        shutdown();
        return false;
    }
    err = cudaMalloc(reinterpret_cast<void**>(&d_epoch_begin_), sizeof(DeviceEpochBegin));
    if (err != cudaSuccess) {
        shutdown();
        return false;
    }
    if (enable_graph_stamps_) {
        err = cudaMalloc(reinterpret_cast<void**>(&d_stamp_base_), sizeof(uint32_t));
        if (err != cudaSuccess) {
            shutdown();
            return false;
        }
    }
    if (enable_manifest_) {
        err = cudaMalloc(reinterpret_cast<void**>(&d_region_hashes_), sizeof(uint64_t) * cfg_.region_capacity);
        if (err != cudaSuccess) {
            shutdown();
            return false;
        }
    }
    err = cudaMalloc(reinterpret_cast<void**>(&d_graph_control_), sizeof(RecorderGraphControl));
    if (err != cudaSuccess) {
        shutdown();
        return false;
    }
    err = cudaMalloc(reinterpret_cast<void**>(&d_enabled_count_), sizeof(uint32_t));
    if (err != cudaSuccess) {
        shutdown();
        return false;
    }
    if (cfg_.region_capacity > 64u) {
        const uint32_t words = (cfg_.region_capacity + 31u) / 32u;
        err = cudaMalloc(reinterpret_cast<void**>(&d_region_enable_bitmap_), sizeof(uint32_t) * words);
        if (err != cudaSuccess) {
            shutdown();
            return false;
        }
    }

    ControlBlock host_control{};
    std::memset(&host_control, 0, sizeof(host_control));
    host_control.ring_bytes = cfg_.ring_bytes;
    host_control.epoch_capacity = cfg_.epoch_capacity;
    host_control.region_capacity = cfg_.region_capacity;
    host_control.min_valid_epoch = 0u;
    const OverwriteMode effective_overwrite = deterministic_ ? OverwriteMode::kBackpressure : cfg_.overwrite_mode;
    host_control.overwrite_mode = static_cast<uint32_t>(effective_overwrite);
    host_control.retention_epochs = cfg_.retention_epochs;

    err = cudaMemcpy(d_control_, &host_control, sizeof(ControlBlock), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        shutdown();
        return false;
    }

    err = cudaMemset(d_ring_, 0, cfg_.ring_bytes);
    if (err != cudaSuccess) {
        shutdown();
        return false;
    }
    err = cudaMemset(d_epochs_, 0, sizeof(EpochRecord) * cfg_.epoch_capacity);
    if (err != cudaSuccess) {
        shutdown();
        return false;
    }
    err = cudaMemset(d_regions_, 0, sizeof(TrackedRegion) * cfg_.region_capacity);
    if (err != cudaSuccess) {
        shutdown();
        return false;
    }
    err = cudaMemset(d_baseline_ptrs_, 0, sizeof(uint64_t) * cfg_.region_capacity);
    if (err != cudaSuccess) {
        shutdown();
        return false;
    }
    err = cudaMemset(d_scratch_ptrs_, 0, sizeof(uint64_t) * cfg_.region_capacity);
    if (err != cudaSuccess) {
        shutdown();
        return false;
    }
    err = cudaMemset(d_first_ring_offset_, 0, sizeof(uint32_t));
    if (err != cudaSuccess) {
        shutdown();
        return false;
    }
    err = cudaMemset(d_first_was_written_, 0, sizeof(uint32_t));
    if (err != cudaSuccess) {
        shutdown();
        return false;
    }
    err = cudaMemset(d_delta_sizes_, 0, sizeof(uint32_t) * cfg_.region_capacity);
    if (err != cudaSuccess) {
        shutdown();
        return false;
    }
    err = cudaMemset(d_epoch_begin_, 0, sizeof(DeviceEpochBegin));
    if (err != cudaSuccess) {
        shutdown();
        return false;
    }
    if (enable_graph_stamps_ && d_stamp_base_) {
        err = cudaMemset(d_stamp_base_, 0, sizeof(uint32_t));
        if (err != cudaSuccess) {
            shutdown();
            return false;
        }
    }
    if (enable_manifest_ && d_region_hashes_) {
        err = cudaMemset(d_region_hashes_, 0, sizeof(uint64_t) * cfg_.region_capacity);
        if (err != cudaSuccess) {
            shutdown();
            return false;
        }
    }
    if (d_enabled_count_) {
        err = cudaMemset(d_enabled_count_, 0, sizeof(uint32_t));
        if (err != cudaSuccess) {
            shutdown();
            return false;
        }
    }
    if (d_region_enable_bitmap_) {
        const uint32_t words = (cfg_.region_capacity + 31u) / 32u;
        err = cudaMemset(d_region_enable_bitmap_, 0xFF, sizeof(uint32_t) * words);
        if (err != cudaSuccess) {
            shutdown();
            return false;
        }
    }

    graph_control_host_ = RecorderGraphControl{};
    if (cfg_.region_capacity >= 64u) {
        graph_control_host_.region_mask = ~0ull;
    } else {
        graph_control_host_.region_mask = (1ull << cfg_.region_capacity) - 1ull;
    }
    graph_control_host_.snapshot_period = 0;
    graph_control_host_.flags = enable_graph_stamps_ ? kGraphControlStampsEnabled : 0u;
    graph_control_host_.region_bitmap = d_region_enable_bitmap_;
    graph_control_host_.bitmap_words = d_region_enable_bitmap_ ? (cfg_.region_capacity + 31u) / 32u : 0u;
    err = cudaMemcpy(d_graph_control_, &graph_control_host_, sizeof(RecorderGraphControl), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        shutdown();
        return false;
    }

    host_regions_.assign(cfg_.region_capacity, TrackedRegion{});
    initialized_ = true;
    min_valid_epoch_ = 0u;
    return true;
}

void Recorder::shutdown() {
    if (d_baseline_ptrs_) {
        std::vector<uint64_t> host_ptrs(cfg_.region_capacity, 0u);
        cudaMemcpy(host_ptrs.data(), d_baseline_ptrs_, sizeof(uint64_t) * cfg_.region_capacity, cudaMemcpyDeviceToHost);
        for (uint64_t ptr : host_ptrs) {
            if (ptr != 0u) {
                cudaFree(reinterpret_cast<void*>(static_cast<uintptr_t>(ptr)));
            }
        }
        cudaFree(d_baseline_ptrs_);
        d_baseline_ptrs_ = nullptr;
    }
    if (d_scratch_ptrs_) {
        std::vector<uint64_t> host_ptrs(cfg_.region_capacity, 0u);
        cudaMemcpy(host_ptrs.data(), d_scratch_ptrs_, sizeof(uint64_t) * cfg_.region_capacity, cudaMemcpyDeviceToHost);
        for (uint64_t ptr : host_ptrs) {
            if (ptr != 0u) {
                cudaFree(reinterpret_cast<void*>(static_cast<uintptr_t>(ptr)));
            }
        }
        cudaFree(d_scratch_ptrs_);
        d_scratch_ptrs_ = nullptr;
    }
    if (d_first_was_written_) {
        cudaFree(d_first_was_written_);
        d_first_was_written_ = nullptr;
    }
    if (d_first_ring_offset_) {
        cudaFree(d_first_ring_offset_);
        d_first_ring_offset_ = nullptr;
    }
    if (d_delta_sizes_) {
        cudaFree(d_delta_sizes_);
        d_delta_sizes_ = nullptr;
    }
    if (d_epoch_begin_) {
        cudaFree(d_epoch_begin_);
        d_epoch_begin_ = nullptr;
    }
    if (d_stamp_base_) {
        cudaFree(d_stamp_base_);
        d_stamp_base_ = nullptr;
    }
    if (d_region_hashes_) {
        cudaFree(d_region_hashes_);
        d_region_hashes_ = nullptr;
    }
    if (d_enabled_count_) {
        cudaFree(d_enabled_count_);
        d_enabled_count_ = nullptr;
    }
    if (d_graph_control_) {
        cudaFree(d_graph_control_);
        d_graph_control_ = nullptr;
    }
    if (d_region_enable_bitmap_) {
        cudaFree(d_region_enable_bitmap_);
        d_region_enable_bitmap_ = nullptr;
    }
    if (d_regions_) {
        cudaFree(d_regions_);
        d_regions_ = nullptr;
    }
    if (d_epochs_) {
        cudaFree(d_epochs_);
        d_epochs_ = nullptr;
    }
    if (d_ring_) {
        cudaFree(d_ring_);
        d_ring_ = nullptr;
    }
    if (d_control_) {
        cudaFree(d_control_);
        d_control_ = nullptr;
    }
    initialized_ = false;
    cfg_ = RecorderConfig{};
    host_regions_.clear();
    manifest_epochs_.clear();
    min_valid_epoch_ = 0u;
    last_status_ = RecorderStatus::kOk;
    enable_graph_stamps_ = false;
    deterministic_ = false;
    enable_manifest_ = false;
    deterministic_stream_ = nullptr;
    deterministic_stream_set_ = false;
    d_graph_stamps_ = nullptr;
    d_graph_stamp_counter_ = nullptr;
    graph_control_host_ = RecorderGraphControl{};
}

bool Recorder::register_region(uint32_t region_id, void* device_ptr, uint32_t size_bytes, uint32_t options) {
    if (!initialized_) {
        return false;
    }
    if (region_id >= cfg_.region_capacity) {
        return false;
    }
    if (device_ptr == nullptr) {
        return false;
    }
    if ((size_bytes % 4) != 0 || size_bytes == 0) {
        return false;
    }
    if (!d_baseline_ptrs_ || !d_scratch_ptrs_) {
        return false;
    }

    uint64_t prior_ptr = 0;
    cudaError_t err = cudaMemcpy(&prior_ptr,
        d_baseline_ptrs_ + region_id,
        sizeof(uint64_t),
        cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        return false;
    }
    if (prior_ptr != 0u) {
        cudaFree(reinterpret_cast<void*>(static_cast<uintptr_t>(prior_ptr)));
    }

    prior_ptr = 0;
    err = cudaMemcpy(&prior_ptr,
        d_scratch_ptrs_ + region_id,
        sizeof(uint64_t),
        cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        return false;
    }
    if (prior_ptr != 0u) {
        cudaFree(reinterpret_cast<void*>(static_cast<uintptr_t>(prior_ptr)));
    }

    void* baseline_ptr = nullptr;
    err = cudaMalloc(&baseline_ptr, size_bytes);
    if (err != cudaSuccess) {
        return false;
    }
    const size_t scratch_bytes = static_cast<size_t>(size_bytes) + 64u;
    void* scratch_ptr = nullptr;
    err = cudaMalloc(&scratch_ptr, scratch_bytes);
    if (err != cudaSuccess) {
        cudaFree(baseline_ptr);
        return false;
    }

    const uint64_t baseline_u64 = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(baseline_ptr));
    const uint64_t scratch_u64 = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(scratch_ptr));
    err = cudaMemcpy(d_baseline_ptrs_ + region_id, &baseline_u64, sizeof(uint64_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(baseline_ptr);
        cudaFree(scratch_ptr);
        return false;
    }
    err = cudaMemcpy(d_scratch_ptrs_ + region_id, &scratch_u64, sizeof(uint64_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(baseline_ptr);
        cudaFree(scratch_ptr);
        return false;
    }

    TrackedRegion host_region{};
    host_region.base_ptr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(device_ptr));
    host_region.size_bytes = size_bytes;
    host_region.region_id = region_id;
    host_region.options = options;
    host_region.full_snapshot_period = 0;
    host_region.user_tag = 0;
    if (region_id < host_regions_.size()) {
        host_regions_[region_id] = host_region;
    }

    const size_t offset_bytes = sizeof(TrackedRegion) * static_cast<size_t>(region_id);
    err = cudaMemcpy(reinterpret_cast<uint8_t*>(d_regions_) + offset_bytes,
        &host_region,
        sizeof(TrackedRegion),
        cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        const uint64_t zero = 0u;
        cudaMemcpy(d_baseline_ptrs_ + region_id, &zero, sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_scratch_ptrs_ + region_id, &zero, sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaFree(baseline_ptr);
        cudaFree(scratch_ptr);
        return false;
    }
    return true;
}

bool Recorder::set_region_full_snapshot_period(uint32_t region_id, uint32_t period) {
    if (!initialized_) {
        return false;
    }
    if (region_id >= cfg_.region_capacity) {
        return false;
    }

    TrackedRegion host_region = host_regions_[region_id];
    host_region.full_snapshot_period = period;
    host_regions_[region_id] = host_region;
    const size_t offset_bytes = sizeof(TrackedRegion) * static_cast<size_t>(region_id);
    cudaError_t err = cudaMemcpy(reinterpret_cast<uint8_t*>(d_regions_) + offset_bytes,
        &host_region,
        sizeof(TrackedRegion),
        cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        return false;
    }

    return true;
}

bool Recorder::update_graph_control(const RecorderGraphControl& control, cudaStream_t stream) {
    if (!initialized_ || !d_graph_control_) {
        return false;
    }
    graph_control_host_ = control;
    if (!enable_graph_stamps_) {
        graph_control_host_.flags &= ~kGraphControlStampsEnabled;
    }
    if (d_region_enable_bitmap_) {
        graph_control_host_.region_bitmap = d_region_enable_bitmap_;
        graph_control_host_.bitmap_words = (cfg_.region_capacity + 31u) / 32u;
    } else {
        graph_control_host_.region_bitmap = nullptr;
        graph_control_host_.bitmap_words = 0u;
    }
    cudaError_t err = cudaMemcpyAsync(d_graph_control_,
        &graph_control_host_,
        sizeof(RecorderGraphControl),
        cudaMemcpyHostToDevice,
        stream);
    return err == cudaSuccess;
}

bool Recorder::update_region_enable_bitmap(const uint32_t* bitmap, uint32_t words, cudaStream_t stream) {
    if (!initialized_ || !d_region_enable_bitmap_ || !bitmap || words == 0u) {
        return false;
    }
    const uint32_t expected_words = (cfg_.region_capacity + 31u) / 32u;
    if (words > expected_words) {
        return false;
    }
    cudaError_t err = cudaMemcpyAsync(d_region_enable_bitmap_,
        bitmap,
        sizeof(uint32_t) * words,
        cudaMemcpyHostToDevice,
        stream);
    if (err != cudaSuccess) {
        return false;
    }
    graph_control_host_.region_bitmap = d_region_enable_bitmap_;
    graph_control_host_.bitmap_words = words;
    err = cudaMemcpyAsync(d_graph_control_,
        &graph_control_host_,
        sizeof(RecorderGraphControl),
        cudaMemcpyHostToDevice,
        stream);
    return err == cudaSuccess;
}

bool Recorder::update_region_pointer(uint32_t region_id, void* device_ptr) {
    if (!initialized_) {
        return false;
    }
    if (region_id >= cfg_.region_capacity) {
        return false;
    }
    if (!device_ptr) {
        return false;
    }
    TrackedRegion host_region = host_regions_[region_id];
    host_region.base_ptr = reinterpret_cast<uint64_t>(device_ptr);
    host_regions_[region_id] = host_region;
    const size_t offset_bytes = sizeof(TrackedRegion) * static_cast<size_t>(region_id);
    cudaError_t err = cudaMemcpy(reinterpret_cast<uint8_t*>(d_regions_) + offset_bytes,
        &host_region,
        sizeof(TrackedRegion),
        cudaMemcpyHostToDevice);
    return err == cudaSuccess;
}

bool Recorder::capture_epoch(cudaStream_t stream) {
    last_status_ = RecorderStatus::kOk;
    if (!initialized_) {
        last_status_ = RecorderStatus::kNotInitialized;
        return false;
    }
    if (deterministic_) {
        if (!deterministic_stream_set_) {
            deterministic_stream_ = stream;
            deterministic_stream_set_ = true;
        } else if (stream != deterministic_stream_) {
            last_status_ = RecorderStatus::kDeterminismViolation;
            return false;
        }
    }

    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    cudaError_t err = cudaStreamIsCapturing(stream, &capture_status);
    if (err != cudaSuccess) {
        last_status_ = RecorderStatus::kCudaError;
        return false;
    }

    if (capture_status != cudaStreamCaptureStatusNone) {
        if (enable_manifest_) {
            last_status_ = RecorderStatus::kInvalidConfig;
            return false;
        }
        err = cudaMemsetAsync(d_first_ring_offset_, 0, sizeof(uint32_t), stream);
        if (err != cudaSuccess) {
            last_status_ = RecorderStatus::kCudaError;
            return false;
        }
        err = cudaMemsetAsync(d_first_was_written_, 0, sizeof(uint32_t), stream);
        if (err != cudaSuccess) {
            last_status_ = RecorderStatus::kCudaError;
            return false;
        }
        err = cudaMemsetAsync(d_enabled_count_, 0, sizeof(uint32_t), stream);
        if (err != cudaSuccess) {
            last_status_ = RecorderStatus::kCudaError;
            return false;
        }

        begin_epoch_kernel<<<1, 1, 0, stream>>>(d_control_, d_epoch_begin_);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            last_status_ = RecorderStatus::kCudaError;
            return false;
        }
        if (enable_graph_stamps_) {
            reserve_stamp_range_controlled_kernel<<<1, 1, 0, stream>>>(
                d_graph_control_,
                d_graph_stamp_counter_,
                d_stamp_base_,
                kStampsPerEpoch);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                last_status_ = RecorderStatus::kCudaError;
                return false;
            }
            stamp_controlled_kernel<<<1, 1, 0, stream>>>(
                d_graph_control_,
                d_graph_stamps_,
                d_stamp_base_,
                kStampEpochStart);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                last_status_ = RecorderStatus::kCudaError;
                return false;
            }
        }

        for (uint32_t i = 0; i < cfg_.region_capacity; ++i) {
            const TrackedRegion& region = host_regions_[i];
            if ((region.options & 1u) == 0u) {
                continue;
            }
            if (region.size_bytes == 0u || region.base_ptr == 0u) {
                continue;
            }

            snapshot_region_graph_kernel<<<1, 1, 0, stream>>>(
                d_control_,
                d_regions_ + i,
                d_baseline_ptrs_,
                d_ring_,
                cfg_.ring_bytes,
                d_epoch_begin_,
                d_graph_control_,
                d_first_ring_offset_,
                d_first_was_written_,
                d_enabled_count_);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                last_status_ = RecorderStatus::kCudaError;
                return false;
            }
        }
        if (enable_graph_stamps_) {
            stamp_controlled_kernel<<<1, 1, 0, stream>>>(
                d_graph_control_,
                d_graph_stamps_,
                d_stamp_base_,
                kStampAfterRegions);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                last_status_ = RecorderStatus::kCudaError;
                return false;
            }
        }

        finalize_epoch_kernel<<<1, 1, 0, stream>>>(
            d_control_,
            d_epoch_begin_,
            d_epochs_,
            cfg_.epoch_capacity,
            d_enabled_count_,
            d_first_ring_offset_);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            last_status_ = RecorderStatus::kCudaError;
            return false;
        }
        if (enable_graph_stamps_) {
            stamp_controlled_kernel<<<1, 1, 0, stream>>>(
                d_graph_control_,
                d_graph_stamps_,
                d_stamp_base_,
                kStampEpochEnd);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                last_status_ = RecorderStatus::kCudaError;
                return false;
            }
        }

        return true;
    }

    ControlBlock host_control{};
    err = cudaMemcpy(&host_control, d_control_, sizeof(ControlBlock), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        last_status_ = RecorderStatus::kCudaError;
        return false;
    }

    begin_epoch_kernel<<<1, 1, 0, stream>>>(d_control_, d_epoch_begin_);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        last_status_ = RecorderStatus::kCudaError;
        return false;
    }
    if (enable_graph_stamps_) {
        reserve_stamp_range_kernel<<<1, 1, 0, stream>>>(d_graph_stamp_counter_, d_stamp_base_, kStampsPerEpoch);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            last_status_ = RecorderStatus::kCudaError;
            return false;
        }
        stamp_kernel<<<1, 1, 0, stream>>>(d_graph_stamps_, d_stamp_base_, kStampEpochStart);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            last_status_ = RecorderStatus::kCudaError;
            return false;
        }
    }

    DeviceEpochBegin host_begin{};
    err = cudaMemcpyAsync(&host_begin, d_epoch_begin_, sizeof(DeviceEpochBegin), cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        last_status_ = RecorderStatus::kCudaError;
        return false;
    }
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        last_status_ = RecorderStatus::kCudaError;
        return false;
    }

    std::vector<uint32_t> delta_sizes(cfg_.region_capacity, 0u);
    if (enable_deltas_) {
        err = cudaMemset(d_delta_sizes_, 0, sizeof(uint32_t) * cfg_.region_capacity);
        if (err != cudaSuccess) {
            last_status_ = RecorderStatus::kCudaError;
            return false;
        }

        for (uint32_t i = 0; i < cfg_.region_capacity; ++i) {
            const TrackedRegion& region = host_regions_[i];
            if ((region.options & 1u) == 0u || region.size_bytes == 0u || region.base_ptr == 0u) {
                continue;
            }

            bool should_snapshot = (region.full_snapshot_period == 0u);
            if (!should_snapshot && !enable_deltas_) {
                should_snapshot = true;
            }
            if (!should_snapshot && region.full_snapshot_period > 0u) {
                should_snapshot = ((host_begin.epoch_id % region.full_snapshot_period) == 0u);
            }
            if (should_snapshot) {
                continue;
            }

            delta_prepare_kernel<<<1, 1, 0, stream>>>(
                d_regions_ + i,
                d_baseline_ptrs_,
                d_scratch_ptrs_,
                d_delta_sizes_);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                last_status_ = RecorderStatus::kCudaError;
                return false;
            }
        }

        err = cudaMemcpyAsync(delta_sizes.data(),
            d_delta_sizes_,
            sizeof(uint32_t) * cfg_.region_capacity,
            cudaMemcpyDeviceToHost,
            stream);
        if (err != cudaSuccess) {
            last_status_ = RecorderStatus::kCudaError;
            return false;
        }
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            last_status_ = RecorderStatus::kCudaError;
            return false;
        }
    }

    struct PendingChunk {
        uint32_t region_index;
        uint32_t chunk_type;
        uint32_t payload_bytes;
        uint32_t total_bytes;
    };
    std::vector<PendingChunk> pending;
    pending.reserve(cfg_.region_capacity);

    for (uint32_t i = 0; i < cfg_.region_capacity; ++i) {
        const TrackedRegion& region = host_regions_[i];
        if ((region.options & 1u) == 0u) {
            continue;
        }
        if (region.size_bytes == 0 || region.base_ptr == 0) {
            continue;
        }

        bool should_snapshot = (region.full_snapshot_period == 0u);
        if (!should_snapshot && !enable_deltas_) {
            should_snapshot = true;
        }
        if (!should_snapshot && region.full_snapshot_period > 0u) {
            should_snapshot = ((host_begin.epoch_id % region.full_snapshot_period) == 0u);
        }

        uint32_t payload_bytes = 0;
        uint32_t chunk_type = kChunkTypeSnapshot;
        if (should_snapshot) {
            payload_bytes = region.size_bytes;
            chunk_type = kChunkTypeSnapshot;
        } else {
            payload_bytes = delta_sizes[i];
            chunk_type = kChunkTypeDeltaXorRle0;
            if (payload_bytes == 0u) {
                last_status_ = RecorderStatus::kInvalidPayload;
                return false;
            }
        }

        uint32_t total_bytes = align_up(static_cast<uint32_t>(sizeof(ChunkHeader)) + payload_bytes);
        if (total_bytes > cfg_.ring_bytes) {
            last_status_ = RecorderStatus::kRingTooSmall;
            return false;
        }
        pending.push_back({i, chunk_type, payload_bytes, total_bytes});
    }

    uint32_t min_valid_epoch = min_valid_epoch_;
    if (cfg_.retention_epochs > 0 && host_begin.epoch_id + 1u > cfg_.retention_epochs) {
        uint32_t retention_min = host_begin.epoch_id - cfg_.retention_epochs + 1u;
        if (retention_min > min_valid_epoch) {
            min_valid_epoch = retention_min;
        }
    }

    std::vector<EpochRecord> host_epochs;
    host_epochs.resize(cfg_.epoch_capacity);
    err = cudaMemcpy(host_epochs.data(),
        d_epochs_,
        sizeof(EpochRecord) * cfg_.epoch_capacity,
        cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        last_status_ = RecorderStatus::kCudaError;
        return false;
    }

    struct EpochEntry {
        uint32_t index;
        uint32_t epoch_id;
        uint32_t bytes;
    };
    std::vector<EpochEntry> valid_epochs;
    valid_epochs.reserve(cfg_.epoch_capacity);
    std::vector<uint32_t> clear_indices;

    uint64_t used_bytes = 0;
    for (uint32_t i = 0; i < cfg_.epoch_capacity; ++i) {
        const EpochRecord& record = host_epochs[i];
        if (record.chunk_count == 0u) {
            continue;
        }
        if (record.epoch_id < min_valid_epoch) {
            clear_indices.push_back(i);
            continue;
        }
        if (record.chunk_count > cfg_.region_capacity || record.reserved0 == 0u) {
            clear_indices.push_back(i);
            continue;
        }
        uint32_t expected_generation = record.epoch_id / cfg_.epoch_capacity;
        if (record.reserved1 != expected_generation) {
            clear_indices.push_back(i);
            continue;
        }

        used_bytes += record.reserved0;
        valid_epochs.push_back({i, record.epoch_id, record.reserved0});
    }

    auto drop_oldest = [&](uint32_t* io_min_valid_epoch) {
        if (valid_epochs.empty()) {
            return false;
        }
        size_t oldest_index = 0;
        for (size_t idx = 1; idx < valid_epochs.size(); ++idx) {
            if (valid_epochs[idx].epoch_id < valid_epochs[oldest_index].epoch_id) {
                oldest_index = idx;
            }
        }
        const EpochEntry dropped = valid_epochs[oldest_index];
        clear_indices.push_back(dropped.index);
        used_bytes -= dropped.bytes;
        if (io_min_valid_epoch && dropped.epoch_id + 1u > *io_min_valid_epoch) {
            *io_min_valid_epoch = dropped.epoch_id + 1u;
        }
        valid_epochs.erase(valid_epochs.begin() + static_cast<std::ptrdiff_t>(oldest_index));
        return true;
    };

    const OverwriteMode effective_overwrite = deterministic_ ? OverwriteMode::kBackpressure : cfg_.overwrite_mode;
    if (effective_overwrite == OverwriteMode::kBackpressure) {
        if (cfg_.retention_epochs > 0 && valid_epochs.size() >= cfg_.retention_epochs) {
            last_status_ = RecorderStatus::kBackpressure;
            return false;
        }
    } else {
        if (cfg_.retention_epochs > 0) {
            while (valid_epochs.size() >= cfg_.retention_epochs) {
                drop_oldest(&min_valid_epoch);
            }
        }
    }

    uint64_t estimated_bytes = 0;
    uint32_t ring_offset = static_cast<uint32_t>(host_control.write_pos % cfg_.ring_bytes);
    for (const PendingChunk& chunk : pending) {
        if (ring_offset + chunk.total_bytes > cfg_.ring_bytes) {
            uint32_t remainder = cfg_.ring_bytes - ring_offset;
            estimated_bytes += remainder;
            ring_offset = 0u;
        }
        estimated_bytes += chunk.total_bytes;
        ring_offset += chunk.total_bytes;
        if (ring_offset >= cfg_.ring_bytes) {
            ring_offset -= cfg_.ring_bytes;
        }
    }

    if (estimated_bytes > cfg_.ring_bytes) {
        last_status_ = RecorderStatus::kRingTooSmall;
        return false;
    }

    if (effective_overwrite == OverwriteMode::kBackpressure) {
        if (estimated_bytes > (cfg_.ring_bytes - used_bytes)) {
            last_status_ = RecorderStatus::kBackpressure;
            return false;
        }
    } else {
        while (estimated_bytes > (cfg_.ring_bytes - used_bytes)) {
            if (!drop_oldest(&min_valid_epoch)) {
                last_status_ = RecorderStatus::kRingTooSmall;
                return false;
            }
        }
    }

    if (!clear_indices.empty() || min_valid_epoch != min_valid_epoch_) {
        EpochRecord zero_record{};
        for (uint32_t index : clear_indices) {
            const size_t epoch_offset_bytes = sizeof(EpochRecord) * static_cast<size_t>(index);
            err = cudaMemcpy(reinterpret_cast<uint8_t*>(d_epochs_) + epoch_offset_bytes,
                &zero_record,
                sizeof(EpochRecord),
                cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                last_status_ = RecorderStatus::kCudaError;
                return false;
            }
        }
        min_valid_epoch_ = min_valid_epoch;
        const size_t offset = offsetof(ControlBlock, min_valid_epoch);
        err = cudaMemcpy(reinterpret_cast<uint8_t*>(d_control_) + offset,
            &min_valid_epoch_,
            sizeof(uint32_t),
            cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            last_status_ = RecorderStatus::kCudaError;
            return false;
        }
    }

    uint32_t enabled_count = static_cast<uint32_t>(pending.size());
    uint32_t first_ring_offset = 0;
    bool first_written = false;
    if (enabled_count > 0) {
        err = cudaMemset(d_first_ring_offset_, 0, sizeof(uint32_t));
        if (err != cudaSuccess) {
            last_status_ = RecorderStatus::kCudaError;
            return false;
        }
    }

    for (const PendingChunk& chunk : pending) {
        uint32_t* first_offset_ptr = nullptr;
        if (!first_written) {
            first_offset_ptr = d_first_ring_offset_;
        }

        if (chunk.chunk_type == kChunkTypeSnapshot) {
            snapshot_region_kernel<<<1, 1, 0, stream>>>(
                d_control_,
                d_regions_ + chunk.region_index,
                d_baseline_ptrs_,
                d_ring_,
                cfg_.ring_bytes,
                host_begin.epoch_id,
                first_offset_ptr,
                nullptr);
        } else {
            delta_write_kernel<<<1, 1, 0, stream>>>(
                d_control_,
                d_regions_ + chunk.region_index,
                d_baseline_ptrs_,
                d_scratch_ptrs_,
                d_ring_,
                cfg_.ring_bytes,
                host_begin.epoch_id,
                chunk.payload_bytes,
                first_offset_ptr,
                nullptr);
        }
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            last_status_ = RecorderStatus::kCudaError;
            return false;
        }

        if (!first_written) {
            err = cudaMemcpyAsync(&first_ring_offset, d_first_ring_offset_, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
            if (err != cudaSuccess) {
                last_status_ = RecorderStatus::kCudaError;
                return false;
            }
            err = cudaStreamSynchronize(stream);
            if (err != cudaSuccess) {
                last_status_ = RecorderStatus::kCudaError;
                return false;
            }
            first_written = true;
        }
    }

    if (enable_graph_stamps_) {
        stamp_kernel<<<1, 1, 0, stream>>>(d_graph_stamps_, d_stamp_base_, kStampAfterRegions);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            last_status_ = RecorderStatus::kCudaError;
            return false;
        }
        stamp_kernel<<<1, 1, 0, stream>>>(d_graph_stamps_, d_stamp_base_, kStampEpochEnd);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            last_status_ = RecorderStatus::kCudaError;
            return false;
        }
    }

    std::vector<uint64_t> host_hashes;
    if (enable_manifest_) {
        host_hashes.assign(cfg_.region_capacity, 0u);
        err = cudaMemsetAsync(d_region_hashes_, 0, sizeof(uint64_t) * cfg_.region_capacity, stream);
        if (err != cudaSuccess) {
            last_status_ = RecorderStatus::kCudaError;
            return false;
        }
        for (const PendingChunk& chunk : pending) {
            const TrackedRegion& region = host_regions_[chunk.region_index];
            const uint8_t* region_ptr = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(region.base_ptr));
            hash_region_kernel<<<1, kHashThreads, 0, stream>>>(region_ptr, region.size_bytes, d_region_hashes_ + region.region_id);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                last_status_ = RecorderStatus::kCudaError;
                return false;
            }
        }
        err = cudaMemcpyAsync(host_hashes.data(),
            d_region_hashes_,
            sizeof(uint64_t) * cfg_.region_capacity,
            cudaMemcpyDeviceToHost,
            stream);
        if (err != cudaSuccess) {
            last_status_ = RecorderStatus::kCudaError;
            return false;
        }
    }

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        last_status_ = RecorderStatus::kCudaError;
        return false;
    }

    ControlBlock host_control_after{};
    err = cudaMemcpy(&host_control_after, d_control_, sizeof(ControlBlock), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        last_status_ = RecorderStatus::kCudaError;
        return false;
    }

    uint64_t epoch_bytes = host_control_after.write_pos - host_control.write_pos;
    if (epoch_bytes > UINT32_MAX) {
        last_status_ = RecorderStatus::kRingCorrupt;
        return false;
    }

    EpochRecord record{};
    record.epoch_id = host_begin.epoch_id;
    record.chunk_count = enabled_count;
    record.region_count = enabled_count;
    record.reserved0 = static_cast<uint32_t>(epoch_bytes);
    record.ring_offset = first_written ? first_ring_offset : 0u;
    record.reserved1 = host_begin.epoch_id / cfg_.epoch_capacity;
    record.timestamp = 0;

    const size_t epoch_offset_bytes = sizeof(EpochRecord) * static_cast<size_t>(host_begin.epoch_index_pos);
    err = cudaMemcpy(reinterpret_cast<uint8_t*>(d_epochs_) + epoch_offset_bytes,
        &record,
        sizeof(EpochRecord),
        cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        last_status_ = RecorderStatus::kCudaError;
        return false;
    }

    if (enable_manifest_) {
        ManifestEpoch manifest_epoch{};
        manifest_epoch.epoch_id = host_begin.epoch_id;
        manifest_epoch.ring_bytes_written = static_cast<uint32_t>(epoch_bytes);
        manifest_epoch.regions.reserve(pending.size());
        for (const PendingChunk& chunk : pending) {
            const TrackedRegion& region = host_regions_[chunk.region_index];
            ManifestRegion manifest_region{};
            manifest_region.region_id = region.region_id;
            manifest_region.size_bytes = region.size_bytes;
            manifest_region.hash64 = host_hashes[region.region_id];
            manifest_region.payload_bytes = chunk.payload_bytes;
            manifest_region.uncompressed_bytes = region.size_bytes;
            manifest_region.snapshot = (chunk.chunk_type == kChunkTypeSnapshot);
            manifest_epoch.regions.push_back(manifest_region);
        }
        manifest_epochs_.push_back(std::move(manifest_epoch));
    }

    return true;
}

bool Recorder::rewind_to_epoch(uint32_t target_epoch, cudaStream_t stream) {
    last_status_ = RecorderStatus::kOk;
    if (!initialized_) {
        last_status_ = RecorderStatus::kNotInitialized;
        return false;
    }
    if (target_epoch < min_valid_epoch_) {
        last_status_ = RecorderStatus::kEpochDropped;
        return false;
    }

    std::vector<EpochRecord> host_epochs;
    if (!read_epochs_to_host(host_epochs)) {
        return false;
    }

    bool found_target = false;
    bool any_chunks = false;
    uint32_t min_epoch = UINT32_MAX;
    for (const EpochRecord& record : host_epochs) {
        if (record.chunk_count == 0) {
            continue;
        }
        if (record.epoch_id < min_valid_epoch_) {
            continue;
        }
        if (record.reserved0 == 0u) {
            continue;
        }
        uint32_t expected_generation = record.epoch_id / cfg_.epoch_capacity;
        if (record.reserved1 != expected_generation) {
            continue;
        }
        any_chunks = true;
        if (record.epoch_id < min_epoch) {
            min_epoch = record.epoch_id;
        }
        if (record.epoch_id == target_epoch) {
            found_target = true;
        }
    }

    if (!found_target) {
        last_status_ = RecorderStatus::kEpochNotFound;
        return false;
    }
    if (!any_chunks) {
        return true;
    }
    if (target_epoch < min_epoch) {
        last_status_ = RecorderStatus::kEpochDropped;
        return false;
    }

    for (uint32_t epoch_id = min_epoch; epoch_id <= target_epoch; ++epoch_id) {
        const EpochRecord* record = nullptr;
        for (const EpochRecord& candidate : host_epochs) {
            if (candidate.epoch_id == epoch_id) {
                record = &candidate;
                break;
            }
        }
        if (!record) {
            continue;
        }
        if (record->epoch_id < min_valid_epoch_) {
            continue;
        }
        if (record->chunk_count == 0 || record->reserved0 == 0u) {
            continue;
        }
        uint32_t expected_generation = record->epoch_id / cfg_.epoch_capacity;
        if (record->reserved1 != expected_generation) {
            continue;
        }
        RecorderStatus status = rewind_apply_epoch(*record,
            d_regions_,
            cfg_.region_capacity,
            d_baseline_ptrs_,
            d_ring_,
            cfg_.ring_bytes,
            stream);
        if (status != RecorderStatus::kOk) {
            last_status_ = status;
            return false;
        }
    }

    return true;
}

bool Recorder::read_epochs_to_host(std::vector<EpochRecord>& out) {
    out.clear();
    if (!initialized_) {
        last_status_ = RecorderStatus::kNotInitialized;
        return false;
    }
    out.resize(cfg_.epoch_capacity);
    cudaError_t err = cudaMemcpy(out.data(),
        d_epochs_,
        sizeof(EpochRecord) * cfg_.epoch_capacity,
        cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        out.clear();
        last_status_ = RecorderStatus::kCudaError;
        return false;
    }
    last_status_ = RecorderStatus::kOk;
    return true;
}

bool Recorder::write_manifest_json(const char* path) const {
    if (!path || !enable_manifest_) {
        return false;
    }

    std::filesystem::path out_path(path);
    if (out_path.has_parent_path()) {
        std::filesystem::create_directories(out_path.parent_path());
    }

    std::ofstream out(path, std::ios::trunc);
    if (!out.is_open()) {
        return false;
    }

    out.imbue(std::locale::classic());
    out << "{ \"epochs\": [";
    for (size_t i = 0; i < manifest_epochs_.size(); ++i) {
        const ManifestEpoch& epoch = manifest_epochs_[i];
        if (i > 0) {
            out << ",";
        }
        out << "{";
        out << "\"epoch_id\":" << epoch.epoch_id;
        out << ",\"ring_bytes_written\":" << epoch.ring_bytes_written;
        out << ",\"regions\":[";
        for (size_t r = 0; r < epoch.regions.size(); ++r) {
            const ManifestRegion& region = epoch.regions[r];
            if (r > 0) {
                out << ",";
            }
            std::ostringstream hash_stream;
            hash_stream.imbue(std::locale::classic());
            hash_stream << "0x" << std::hex << std::setw(16) << std::setfill('0') << region.hash64;

            double ratio = 0.0;
            if (region.uncompressed_bytes > 0 && region.payload_bytes > 0) {
                ratio = static_cast<double>(region.payload_bytes) / static_cast<double>(region.uncompressed_bytes);
            }
            std::ostringstream ratio_stream;
            ratio_stream.imbue(std::locale::classic());
            ratio_stream << std::fixed << std::setprecision(6) << ratio;

            out << "{";
            out << "\"region_id\":" << region.region_id;
            out << ",\"size_bytes\":" << region.size_bytes;
            out << ",\"hash64\":\"" << hash_stream.str() << "\"";
            out << ",\"snapshot_or_delta\":\"" << (region.snapshot ? "snapshot" : "delta") << "\"";
            out << ",\"payload_bytes\":" << region.payload_bytes;
            out << ",\"uncompressed_bytes\":" << region.uncompressed_bytes;
            out << ",\"compression_ratio\":" << ratio_stream.str();
            out << "}";
        }
        out << "]}";
    }
    out << "] }";
    return true;
}

void Recorder::clear_manifest() {
    manifest_epochs_.clear();
}

} // namespace tt
