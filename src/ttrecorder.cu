#include "tt/ttrecorder.h"

#include <cstring>
#include <vector>

#include "tt/tt_layout.h"

namespace tt {

namespace {

struct EpochBegin {
    uint32_t epoch_id;
    uint32_t epoch_index_pos;
};

__host__ __device__ inline uint32_t align16(uint32_t value) {
    return (value + 15u) & ~15u;
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
        uint32_t marker_bytes = align16(static_cast<uint32_t>(sizeof(ChunkHeader)));
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

        uint32_t remainder = ring_bytes - ring_offset;
        if (remainder > marker_bytes) {
            uint32_t pad_bytes = remainder - marker_bytes;
            atomicAdd(&control->write_pos, static_cast<uint64_t>(pad_bytes));
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

__global__ void begin_epoch_kernel(ControlBlock* control, EpochBegin* out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
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
    uint32_t* first_ring_offset) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if ((region->options & 1u) == 0u || region->size_bytes == 0) {
            return;
        }

        uint32_t payload_bytes = region->size_bytes;
        uint32_t total_bytes = align16(static_cast<uint32_t>(sizeof(ChunkHeader)) + payload_bytes);
        uint32_t ring_offset = 0;
        reserve_and_get_ring_offset(control, ring, ring_bytes, total_bytes, &ring_offset, nullptr);

        if (first_ring_offset) {
            *first_ring_offset = ring_offset;
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

__global__ void delta_region_kernel(ControlBlock* control,
    const TrackedRegion* region,
    const uint64_t* baseline_ptrs,
    const uint64_t* scratch_ptrs,
    uint8_t* ring,
    uint32_t ring_bytes,
    uint32_t epoch_id,
    uint32_t* first_ring_offset) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if ((region->options & 1u) == 0u || region->size_bytes == 0) {
            return;
        }
        if ((region->size_bytes % 4u) != 0u) {
            return;
        }

        const uint8_t* baseline_ptr = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(baseline_ptrs[region->region_id]));
        uint8_t* scratch_ptr = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(scratch_ptrs[region->region_id]));
        if (baseline_ptr == nullptr || scratch_ptr == nullptr) {
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

        uint32_t payload_bytes = scratch_index * sizeof(uint32_t);
        uint32_t total_bytes = align16(static_cast<uint32_t>(sizeof(ChunkHeader)) + payload_bytes);
        uint32_t ring_offset = 0;
        reserve_and_get_ring_offset(control, ring, ring_bytes, total_bytes, &ring_offset, nullptr);

        if (first_ring_offset) {
            *first_ring_offset = ring_offset;
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
        const uint8_t* payload_src = scratch_ptr;
        for (uint32_t b = 0; b < payload_bytes; ++b) {
            payload_dst[b] = payload_src[b];
        }

        uint32_t* baseline_out = reinterpret_cast<uint32_t*>(const_cast<uint8_t*>(baseline_ptr));
        for (uint32_t w = 0; w < word_count; ++w) {
            baseline_out[w] = current_words[w];
        }
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

bool rewind_apply_epoch(const EpochRecord& record,
    TrackedRegion* d_regions,
    uint32_t region_capacity,
    const uint64_t* baseline_ptrs,
    uint8_t* ring,
    uint32_t ring_bytes,
    cudaStream_t stream) {
    if (record.chunk_count == 0) {
        return true;
    }

    uint32_t ring_offset = record.ring_offset;
    uint32_t applied = 0;
    while (applied < record.chunk_count) {
        ChunkHeader host_header{};
        cudaError_t err = cudaMemcpyAsync(&host_header,
            ring + ring_offset,
            sizeof(ChunkHeader),
            cudaMemcpyDeviceToHost,
            stream);
        if (err != cudaSuccess) {
            return false;
        }
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            return false;
        }

        if (host_header.magic != kChunkMagic) {
            return false;
        }

        if (host_header.chunk_type == kChunkTypeWrapMarker || (host_header.flags & kChunkFlagWrapMarker) != 0u) {
            ring_offset = 0;
            continue;
        }

        if (host_header.chunk_type == kChunkTypeSnapshot) {
            apply_chunk_kernel<<<1, 1, 0, stream>>>(d_regions, region_capacity, baseline_ptrs, ring, ring_offset);
        } else if (host_header.chunk_type == kChunkTypeDeltaXorRle0) {
            apply_delta_chunk_kernel<<<1, 1, 0, stream>>>(d_regions, region_capacity, baseline_ptrs, ring, ring_offset);
        } else {
            return false;
        }
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            return false;
        }

        uint32_t chunk_bytes = align16(static_cast<uint32_t>(sizeof(ChunkHeader)) + host_header.payload_bytes);
        ring_offset += chunk_bytes;
        if (ring_offset >= ring_bytes) {
            ring_offset -= ring_bytes;
        }
        ++applied;
    }

    return true;
}

} // namespace

bool Recorder::init(const RecorderConfig& cfg) {
    if (initialized_) {
        shutdown();
    }

    if (cfg.ring_bytes == 0 || cfg.epoch_capacity == 0 || cfg.region_capacity == 0) {
        return false;
    }

    cfg_ = cfg;

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

    ControlBlock host_control{};
    std::memset(&host_control, 0, sizeof(host_control));
    host_control.ring_bytes = cfg_.ring_bytes;
    host_control.epoch_capacity = cfg_.epoch_capacity;
    host_control.region_capacity = cfg_.region_capacity;

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

    initialized_ = true;
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

    TrackedRegion host_region{};
    const size_t offset_bytes = sizeof(TrackedRegion) * static_cast<size_t>(region_id);
    cudaError_t err = cudaMemcpy(&host_region,
        reinterpret_cast<uint8_t*>(d_regions_) + offset_bytes,
        sizeof(TrackedRegion),
        cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        return false;
    }

    host_region.full_snapshot_period = period;
    err = cudaMemcpy(reinterpret_cast<uint8_t*>(d_regions_) + offset_bytes,
        &host_region,
        sizeof(TrackedRegion),
        cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        return false;
    }

    return true;
}

bool Recorder::capture_epoch(cudaStream_t stream) {
    if (!initialized_) {
        return false;
    }

    EpochBegin* d_begin = nullptr;
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&d_begin), sizeof(EpochBegin));
    if (err != cudaSuccess) {
        return false;
    }

    begin_epoch_kernel<<<1, 1, 0, stream>>>(d_control_, d_begin);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_begin);
        return false;
    }

    EpochBegin host_begin{};
    err = cudaMemcpyAsync(&host_begin, d_begin, sizeof(EpochBegin), cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        cudaFree(d_begin);
        return false;
    }
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        cudaFree(d_begin);
        return false;
    }
    cudaFree(d_begin);
    d_begin = nullptr;

    std::vector<TrackedRegion> host_regions(cfg_.region_capacity);
    err = cudaMemcpy(host_regions.data(), d_regions_, sizeof(TrackedRegion) * cfg_.region_capacity, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        return false;
    }

    uint32_t enabled_count = 0;
    uint32_t first_ring_offset = 0;
    bool first_written = false;

    for (uint32_t i = 0; i < cfg_.region_capacity; ++i) {
        const TrackedRegion& region = host_regions[i];
        if ((region.options & 1u) == 0u) {
            continue;
        }
        if (region.size_bytes == 0 || region.base_ptr == 0) {
            continue;
        }

        uint32_t* first_offset_ptr = nullptr;
        if (!first_written) {
            first_offset_ptr = d_first_ring_offset_;
        }

        bool should_snapshot = (region.full_snapshot_period == 0u);
        if (!should_snapshot && !enable_deltas_) {
            should_snapshot = true;
        }
        if (!should_snapshot && region.full_snapshot_period > 0u) {
            should_snapshot = ((host_begin.epoch_id % region.full_snapshot_period) == 0u);
        }

        if (should_snapshot) {
            snapshot_region_kernel<<<1, 1, 0, stream>>>(
                d_control_,
                d_regions_ + i,
                d_baseline_ptrs_,
                d_ring_,
                cfg_.ring_bytes,
                host_begin.epoch_id,
                first_offset_ptr);
        } else {
            delta_region_kernel<<<1, 1, 0, stream>>>(
                d_control_,
                d_regions_ + i,
                d_baseline_ptrs_,
                d_scratch_ptrs_,
                d_ring_,
                cfg_.ring_bytes,
                host_begin.epoch_id,
                first_offset_ptr);
        }
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            return false;
        }

        if (!first_written) {
            err = cudaMemcpyAsync(&first_ring_offset, d_first_ring_offset_, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
            if (err != cudaSuccess) {
                return false;
            }
            err = cudaStreamSynchronize(stream);
            if (err != cudaSuccess) {
                return false;
            }
            first_written = true;
        }

        ++enabled_count;
    }

    EpochRecord record{};
    record.epoch_id = host_begin.epoch_id;
    record.chunk_count = enabled_count;
    record.region_count = enabled_count;
    record.reserved0 = 0;
    record.ring_offset = first_written ? first_ring_offset : 0u;
    record.reserved1 = 0;
    record.timestamp = 0;

    const size_t epoch_offset_bytes = sizeof(EpochRecord) * static_cast<size_t>(host_begin.epoch_index_pos);
    err = cudaMemcpy(reinterpret_cast<uint8_t*>(d_epochs_) + epoch_offset_bytes,
        &record,
        sizeof(EpochRecord),
        cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        return false;
    }

    return true;
}

bool Recorder::rewind_to_epoch(uint32_t target_epoch, cudaStream_t stream) {
    if (!initialized_) {
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
        if (record.epoch_id == target_epoch) {
            found_target = true;
        }
        if (record.chunk_count > 0) {
            any_chunks = true;
            if (record.epoch_id < min_epoch) {
                min_epoch = record.epoch_id;
            }
        }
    }
    if (!found_target) {
        return false;
    }
    if (!any_chunks) {
        return true;
    }
    if (target_epoch < min_epoch) {
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
        if (!rewind_apply_epoch(*record,
                d_regions_,
                cfg_.region_capacity,
                d_baseline_ptrs_,
                d_ring_,
                cfg_.ring_bytes,
                stream)) {
            return false;
        }
    }

    return true;
}

bool Recorder::read_epochs_to_host(std::vector<EpochRecord>& out) {
    out.clear();
    if (!initialized_) {
        return false;
    }
    out.resize(cfg_.epoch_capacity);
    cudaError_t err = cudaMemcpy(out.data(),
        d_epochs_,
        sizeof(EpochRecord) * cfg_.epoch_capacity,
        cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        out.clear();
        return false;
    }
    return true;
}

} // namespace tt
