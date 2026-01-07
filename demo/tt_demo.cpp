#include "tt/ttrecorder.h"

#include <cstdio>
#include <vector>
#include <cstring>
#include <string>

namespace {

bool check_cuda(cudaError_t err, const char* label) {
    if (err == cudaSuccess) {
        return true;
    }
    std::printf("CUDA error at %s: %s\n", label, cudaGetErrorString(err));
    return false;
}

void fill_pattern(std::vector<uint32_t>& data, uint32_t seed) {
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = seed ^ static_cast<uint32_t>(i * 1664525u + 1013904223u);
    }
}

bool verify_pattern(const std::vector<uint32_t>& data, uint32_t seed) {
    for (size_t i = 0; i < data.size(); ++i) {
        uint32_t expected = seed ^ static_cast<uint32_t>(i * 1664525u + 1013904223u);
        if (data[i] != expected) {
            return false;
        }
    }
    return true;
}

} // namespace

int main() {
    const uint32_t element_count = 256 * 1024;
    const uint32_t size_bytes = element_count * sizeof(uint32_t);
    uint32_t epoch_count = 10;
    const uint32_t snapshot_period = 4;

    uint32_t ring_bytes = size_bytes * 2 * epoch_count + 4096;
    bool enable_deltas = true;
    uint32_t retention_epochs = 0;
    tt::OverwriteMode overwrite_mode = tt::OverwriteMode::kDropOldest;
    for (int i = 1; i < __argc; ++i) {
        if (std::strcmp(__argv[i], "--no-delta") == 0) {
            enable_deltas = false;
        } else if (std::strncmp(__argv[i], "--ring-bytes=", 13) == 0) {
            ring_bytes = static_cast<uint32_t>(std::strtoul(__argv[i] + 13, nullptr, 10));
        } else if (std::strncmp(__argv[i], "--retention-epochs=", 19) == 0) {
            retention_epochs = static_cast<uint32_t>(std::strtoul(__argv[i] + 19, nullptr, 10));
        } else if (std::strncmp(__argv[i], "--overwrite-mode=", 17) == 0) {
            const char* mode = __argv[i] + 17;
            if (std::strcmp(mode, "backpressure") == 0) {
                overwrite_mode = tt::OverwriteMode::kBackpressure;
            } else if (std::strcmp(mode, "drop") == 0) {
                overwrite_mode = tt::OverwriteMode::kDropOldest;
            }
        }
    }

    void* d_buffer_a = nullptr;
    void* d_buffer_b = nullptr;
    if (!check_cuda(cudaMalloc(&d_buffer_a, size_bytes), "cudaMalloc A")) {
        return 1;
    }
    if (!check_cuda(cudaMalloc(&d_buffer_b, size_bytes), "cudaMalloc B")) {
        cudaFree(d_buffer_a);
        return 1;
    }

    const uint32_t per_chunk_bytes = (static_cast<uint32_t>(sizeof(tt::ChunkHeader)) + size_bytes + 31u) & ~31u;
    const uint32_t per_epoch_bytes = per_chunk_bytes * 2u;
    const uint64_t required_bytes = static_cast<uint64_t>(per_epoch_bytes) * static_cast<uint64_t>(epoch_count);
    bool can_rewind = true;
    if (ring_bytes < per_chunk_bytes) {
        std::printf("ring_bytes too small for a single chunk (%u required)\n", per_chunk_bytes);
        cudaFree(d_buffer_b);
        cudaFree(d_buffer_a);
        return 1;
    }
    if (static_cast<uint64_t>(ring_bytes) < required_bytes) {
        std::printf("warning: ring_bytes=%u smaller than required=%llu; wrap will overwrite old epochs\n",
            ring_bytes,
            static_cast<unsigned long long>(required_bytes));
        can_rewind = false;
    }

    tt::Recorder recorder;
    tt::RecorderConfig cfg{};
    cfg.ring_bytes = ring_bytes;
    cfg.epoch_capacity = 32;
    cfg.region_capacity = 8;
    cfg.retention_epochs = retention_epochs;
    cfg.overwrite_mode = overwrite_mode;
    if (!recorder.init(cfg)) {
        std::printf("recorder init failed\n");
        cudaFree(d_buffer_b);
        cudaFree(d_buffer_a);
        return 1;
    }

    if (!recorder.register_region(0, d_buffer_a, size_bytes, 1) ||
        !recorder.register_region(1, d_buffer_b, size_bytes, 1)) {
        std::printf("register_region failed\n");
        recorder.shutdown();
        cudaFree(d_buffer_b);
        cudaFree(d_buffer_a);
        return 1;
    }

    if (enable_deltas) {
        recorder.set_region_full_snapshot_period(0, snapshot_period);
        recorder.set_region_full_snapshot_period(1, snapshot_period);
    }

    std::vector<uint32_t> host_buffer(element_count);
    for (uint32_t epoch = 0; epoch < epoch_count; ++epoch) {
        fill_pattern(host_buffer, 0x1000u + epoch);
        if (!check_cuda(cudaMemcpy(d_buffer_a, host_buffer.data(), size_bytes, cudaMemcpyHostToDevice), "memcpy A")) {
            recorder.shutdown();
            cudaFree(d_buffer_b);
            cudaFree(d_buffer_a);
            return 1;
        }

        fill_pattern(host_buffer, 0x2000u + epoch);
        if (!check_cuda(cudaMemcpy(d_buffer_b, host_buffer.data(), size_bytes, cudaMemcpyHostToDevice), "memcpy B")) {
            recorder.shutdown();
            cudaFree(d_buffer_b);
            cudaFree(d_buffer_a);
            return 1;
        }

        if (!recorder.capture_epoch(0)) {
            std::printf("capture_epoch failed at %u\n", epoch);
            recorder.shutdown();
            cudaFree(d_buffer_b);
            cudaFree(d_buffer_a);
            return 1;
        }

        const bool is_snapshot = (!enable_deltas) || (snapshot_period == 0u) || ((epoch % snapshot_period) == 0u);
        const uint32_t payload_bytes = size_bytes;
        const uint32_t chunk_bytes = static_cast<uint32_t>(sizeof(tt::ChunkHeader)) + payload_bytes;
        const uint32_t aligned_bytes = (chunk_bytes + 31u) & ~31u;
        std::printf("epoch %u: %s payload=%u total=%u ring_bytes=%u\n",
            epoch,
            is_snapshot ? "snapshot" : "delta",
            payload_bytes,
            aligned_bytes,
            ring_bytes);
    }

    bool ok_a = true;
    bool ok_b = true;
    if (can_rewind) {
        const uint32_t target_epoch = 4;
        if (!recorder.rewind_to_epoch(target_epoch, 0)) {
            std::printf("rewind_to_epoch failed\n");
            recorder.shutdown();
            cudaFree(d_buffer_b);
            cudaFree(d_buffer_a);
            return 1;
        }

        std::vector<uint32_t> host_out(element_count);
        if (!check_cuda(cudaMemcpy(host_out.data(), d_buffer_a, size_bytes, cudaMemcpyDeviceToHost), "memcpy out A")) {
            recorder.shutdown();
            cudaFree(d_buffer_b);
            cudaFree(d_buffer_a);
            return 1;
        }
        ok_a = verify_pattern(host_out, 0x1000u + target_epoch);

        if (!check_cuda(cudaMemcpy(host_out.data(), d_buffer_b, size_bytes, cudaMemcpyDeviceToHost), "memcpy out B")) {
            recorder.shutdown();
            cudaFree(d_buffer_b);
            cudaFree(d_buffer_a);
            return 1;
        }
        ok_b = verify_pattern(host_out, 0x2000u + target_epoch);
    }

    recorder.shutdown();
    cudaFree(d_buffer_b);
    cudaFree(d_buffer_a);

    if (!ok_a || !ok_b) {
        std::printf("tt_demo failed verification\n");
        return 1;
    }
    if (!can_rewind) {
        std::printf("tt_demo complete (rewind skipped; ring too small)\n");
        return 0;
    }

    std::printf("tt_demo success\n");
    return 0;
}
