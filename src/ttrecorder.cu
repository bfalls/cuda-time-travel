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
        uint64_t old_pos = atomicAdd(&control->write_pos, static_cast<uint64_t>(total_bytes));
        uint32_t ring_offset = static_cast<uint32_t>(old_pos % ring_bytes);

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
    }
}

__global__ void apply_chunk_kernel(const TrackedRegion* regions,
    uint32_t region_capacity,
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
    }
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

    initialized_ = true;
    return true;
}

void Recorder::shutdown() {
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

    TrackedRegion host_region{};
    host_region.base_ptr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(device_ptr));
    host_region.size_bytes = size_bytes;
    host_region.region_id = region_id;
    host_region.options = options;
    host_region.full_snapshot_period = 0;
    host_region.user_tag = 0;

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

bool Recorder::capture_epoch(cudaStream_t stream) {
    if (!initialized_) {
        return false;
    }

    EpochBegin* d_begin = nullptr;
    uint32_t* d_first_offset = nullptr;
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
            err = cudaMalloc(reinterpret_cast<void**>(&d_first_offset), sizeof(uint32_t));
            if (err != cudaSuccess) {
                return false;
            }
            first_offset_ptr = d_first_offset;
        }

        snapshot_region_kernel<<<1, 1, 0, stream>>>(d_control_, d_regions_ + i, d_ring_, cfg_.ring_bytes, host_begin.epoch_id, first_offset_ptr);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            if (d_first_offset) {
                cudaFree(d_first_offset);
            }
            return false;
        }

        if (!first_written) {
            err = cudaMemcpyAsync(&first_ring_offset, d_first_offset, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
            if (err != cudaSuccess) {
                cudaFree(d_first_offset);
                return false;
            }
            err = cudaStreamSynchronize(stream);
            if (err != cudaSuccess) {
                cudaFree(d_first_offset);
                return false;
            }
            cudaFree(d_first_offset);
            d_first_offset = nullptr;
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

    const EpochRecord* target = nullptr;
    for (const EpochRecord& record : host_epochs) {
        if (record.epoch_id == target_epoch) {
            target = &record;
            break;
        }
    }
    if (target == nullptr) {
        return false;
    }

    if (target->chunk_count == 0) {
        return true;
    }

    uint32_t ring_offset = target->ring_offset;
    for (uint32_t i = 0; i < target->chunk_count; ++i) {
        uint32_t header_fields[8] = {};
        cudaError_t err = cudaMemcpyAsync(header_fields,
            d_ring_ + ring_offset,
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

        ChunkHeader host_header{};
        std::memcpy(&host_header, header_fields, sizeof(ChunkHeader));

        apply_chunk_kernel<<<1, 1, 0, stream>>>(d_regions_, cfg_.region_capacity, d_ring_, ring_offset);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            return false;
        }

        uint32_t chunk_bytes = align16(static_cast<uint32_t>(sizeof(ChunkHeader)) + host_header.payload_bytes);
        ring_offset += chunk_bytes;
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
