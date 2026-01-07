#include "tt/ttrecorder.h"

#include <cstdio>
#include <vector>

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
    const uint32_t epoch_count = 10;

    void* d_buffer_a = nullptr;
    void* d_buffer_b = nullptr;
    if (!check_cuda(cudaMalloc(&d_buffer_a, size_bytes), "cudaMalloc A")) {
        return 1;
    }
    if (!check_cuda(cudaMalloc(&d_buffer_b, size_bytes), "cudaMalloc B")) {
        cudaFree(d_buffer_a);
        return 1;
    }

    tt::Recorder recorder;
    tt::RecorderConfig cfg{};
    cfg.ring_bytes = size_bytes * 2 + 4096;
    cfg.epoch_capacity = 32;
    cfg.region_capacity = 8;
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
    }

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
    bool ok_a = verify_pattern(host_out, 0x1000u + target_epoch);

    if (!check_cuda(cudaMemcpy(host_out.data(), d_buffer_b, size_bytes, cudaMemcpyDeviceToHost), "memcpy out B")) {
        recorder.shutdown();
        cudaFree(d_buffer_b);
        cudaFree(d_buffer_a);
        return 1;
    }
    bool ok_b = verify_pattern(host_out, 0x2000u + target_epoch);

    recorder.shutdown();
    cudaFree(d_buffer_b);
    cudaFree(d_buffer_a);

    if (!ok_a || !ok_b) {
        std::printf("tt_demo failed verification\n");
        return 1;
    }

    std::printf("tt_demo success\n");
    return 0;
}
