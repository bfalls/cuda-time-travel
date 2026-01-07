#include "tt/ttrecorder.h"

#include <cstdio>
#include <cstdint>
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
        data[i] = seed ^ static_cast<uint32_t>(i * 2654435761u);
    }
}

bool verify_pattern(const std::vector<uint32_t>& data, uint32_t seed) {
    for (size_t i = 0; i < data.size(); ++i) {
        uint32_t expected = seed ^ static_cast<uint32_t>(i * 2654435761u);
        if (data[i] != expected) {
            return false;
        }
    }
    return true;
}

bool test_single_region() {
    const uint32_t element_count = 256;
    const uint32_t size_bytes = element_count * sizeof(uint32_t);

    void* d_buffer = nullptr;
    if (!check_cuda(cudaMalloc(&d_buffer, size_bytes), "cudaMalloc single")) {
        return false;
    }

    std::vector<uint32_t> host_a(element_count);
    std::vector<uint32_t> host_b(element_count);
    std::vector<uint32_t> host_out(element_count);
    fill_pattern(host_a, 0x11111111u);
    fill_pattern(host_b, 0x22222222u);

    tt::Recorder recorder;
    tt::RecorderConfig cfg{};
    cfg.ring_bytes = 4096;
    cfg.epoch_capacity = 8;
    cfg.region_capacity = 4;
    if (!recorder.init(cfg)) {
        cudaFree(d_buffer);
        return false;
    }

    if (!check_cuda(cudaMemcpy(d_buffer, host_a.data(), size_bytes, cudaMemcpyHostToDevice), "memcpy A")) {
        recorder.shutdown();
        cudaFree(d_buffer);
        return false;
    }

    if (!recorder.register_region(0, d_buffer, size_bytes, 1)) {
        recorder.shutdown();
        cudaFree(d_buffer);
        return false;
    }

    if (!recorder.capture_epoch(0)) {
        recorder.shutdown();
        cudaFree(d_buffer);
        return false;
    }

    if (!check_cuda(cudaMemcpy(d_buffer, host_b.data(), size_bytes, cudaMemcpyHostToDevice), "memcpy B")) {
        recorder.shutdown();
        cudaFree(d_buffer);
        return false;
    }

    if (!recorder.capture_epoch(0)) {
        recorder.shutdown();
        cudaFree(d_buffer);
        return false;
    }

    if (!recorder.rewind_to_epoch(0, 0)) {
        recorder.shutdown();
        cudaFree(d_buffer);
        return false;
    }

    if (!check_cuda(cudaMemcpy(host_out.data(), d_buffer, size_bytes, cudaMemcpyDeviceToHost), "memcpy out")) {
        recorder.shutdown();
        cudaFree(d_buffer);
        return false;
    }

    bool ok = verify_pattern(host_out, 0x11111111u);
    recorder.shutdown();
    cudaFree(d_buffer);
    return ok;
}

bool test_two_regions() {
    const uint32_t count_a = 128;
    const uint32_t count_b = 64;
    const uint32_t size_a = count_a * sizeof(uint32_t);
    const uint32_t size_b = count_b * sizeof(uint32_t);

    void* d_buffer_a = nullptr;
    void* d_buffer_b = nullptr;
    if (!check_cuda(cudaMalloc(&d_buffer_a, size_a), "cudaMalloc A")) {
        return false;
    }
    if (!check_cuda(cudaMalloc(&d_buffer_b, size_b), "cudaMalloc B")) {
        cudaFree(d_buffer_a);
        return false;
    }

    std::vector<uint32_t> host_a0(count_a);
    std::vector<uint32_t> host_b0(count_b);
    std::vector<uint32_t> host_a1(count_a);
    std::vector<uint32_t> host_b1(count_b);
    std::vector<uint32_t> host_out_a(count_a);
    std::vector<uint32_t> host_out_b(count_b);
    fill_pattern(host_a0, 0xAAAA0001u);
    fill_pattern(host_b0, 0xBBBB0002u);
    fill_pattern(host_a1, 0xCCCC0003u);
    fill_pattern(host_b1, 0xDDDD0004u);

    tt::Recorder recorder;
    tt::RecorderConfig cfg{};
    cfg.ring_bytes = 8192;
    cfg.epoch_capacity = 8;
    cfg.region_capacity = 4;
    if (!recorder.init(cfg)) {
        cudaFree(d_buffer_b);
        cudaFree(d_buffer_a);
        return false;
    }

    if (!check_cuda(cudaMemcpy(d_buffer_a, host_a0.data(), size_a, cudaMemcpyHostToDevice), "memcpy A0")) {
        recorder.shutdown();
        cudaFree(d_buffer_b);
        cudaFree(d_buffer_a);
        return false;
    }
    if (!check_cuda(cudaMemcpy(d_buffer_b, host_b0.data(), size_b, cudaMemcpyHostToDevice), "memcpy B0")) {
        recorder.shutdown();
        cudaFree(d_buffer_b);
        cudaFree(d_buffer_a);
        return false;
    }

    if (!recorder.register_region(0, d_buffer_a, size_a, 1)) {
        recorder.shutdown();
        cudaFree(d_buffer_b);
        cudaFree(d_buffer_a);
        return false;
    }
    if (!recorder.register_region(1, d_buffer_b, size_b, 1)) {
        recorder.shutdown();
        cudaFree(d_buffer_b);
        cudaFree(d_buffer_a);
        return false;
    }

    if (!recorder.capture_epoch(0)) {
        recorder.shutdown();
        cudaFree(d_buffer_b);
        cudaFree(d_buffer_a);
        return false;
    }

    if (!check_cuda(cudaMemcpy(d_buffer_a, host_a1.data(), size_a, cudaMemcpyHostToDevice), "memcpy A1")) {
        recorder.shutdown();
        cudaFree(d_buffer_b);
        cudaFree(d_buffer_a);
        return false;
    }
    if (!check_cuda(cudaMemcpy(d_buffer_b, host_b1.data(), size_b, cudaMemcpyHostToDevice), "memcpy B1")) {
        recorder.shutdown();
        cudaFree(d_buffer_b);
        cudaFree(d_buffer_a);
        return false;
    }

    if (!recorder.capture_epoch(0)) {
        recorder.shutdown();
        cudaFree(d_buffer_b);
        cudaFree(d_buffer_a);
        return false;
    }

    if (!recorder.rewind_to_epoch(0, 0)) {
        recorder.shutdown();
        cudaFree(d_buffer_b);
        cudaFree(d_buffer_a);
        return false;
    }

    if (!check_cuda(cudaMemcpy(host_out_a.data(), d_buffer_a, size_a, cudaMemcpyDeviceToHost), "memcpy out A")) {
        recorder.shutdown();
        cudaFree(d_buffer_b);
        cudaFree(d_buffer_a);
        return false;
    }
    if (!check_cuda(cudaMemcpy(host_out_b.data(), d_buffer_b, size_b, cudaMemcpyDeviceToHost), "memcpy out B")) {
        recorder.shutdown();
        cudaFree(d_buffer_b);
        cudaFree(d_buffer_a);
        return false;
    }

    bool ok = verify_pattern(host_out_a, 0xAAAA0001u) && verify_pattern(host_out_b, 0xBBBB0002u);
    recorder.shutdown();
    cudaFree(d_buffer_b);
    cudaFree(d_buffer_a);
    return ok;
}

} // namespace

int main() {
    bool ok_single = test_single_region();
    if (!ok_single) {
        std::printf("test_single_region failed\n");
        return 1;
    }

    bool ok_two = test_two_regions();
    if (!ok_two) {
        std::printf("test_two_regions failed\n");
        return 1;
    }

    std::printf("tt_tests passed\n");
    return 0;
}
