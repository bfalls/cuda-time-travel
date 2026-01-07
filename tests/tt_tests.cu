#include "tt/tt_graph.h"
#include "tt/tt_graph_patch.h"
#include "tt/tt_cupti.h"
#include "tt/tt_trace.h"
#include "tt/ttrecorder.h"

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace {

bool check_cuda(cudaError_t err, const char* label) {
    if (err == cudaSuccess) {
        return true;
    }
    std::printf("CUDA error at %s: %s\n", label, cudaGetErrorString(err));
    return false;
}

__global__ void set_epoch_kernel(uint32_t* counter, uint32_t value) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *counter = value;
    }
}

__global__ void write_pattern_kernel(uint32_t* data, uint32_t count, const uint32_t* counter) {
    const uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < count) {
        const uint32_t epoch = *counter;
        data[idx] = epoch ^ (idx * 2654435761u);
    }
}

__global__ void write_pattern_iter_kernel(uint32_t* data, uint32_t count, const tt::IterParams* params) {
    const uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < count && params) {
        const uint32_t mix = (idx * 2654435761u) ^ params->seed;
        const uint32_t value = params->epoch ^ mix;
        data[idx] = (params->flags & 1u) ? ~value : value;
    }
}

__global__ void write_pattern_patch_kernel(uint32_t* data, uint32_t count, uint32_t epoch, uint32_t seed) {
    const uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < count) {
        const uint32_t mix = (idx * 2654435761u) ^ seed;
        data[idx] = epoch ^ mix;
    }
}

bool trace_json_valid(const char* path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        return false;
    }
    std::string contents((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (contents.empty()) {
        return false;
    }
    size_t first = contents.find_first_not_of(" \t\r\n");
    size_t last = contents.find_last_not_of(" \t\r\n");
    if (first == std::string::npos || last == std::string::npos) {
        return false;
    }
    if (contents[first] != '{' || contents[last] != '}') {
        return false;
    }
    if (contents.find("\"traceEvents\"") == std::string::npos) {
        return false;
    }
    return true;
}

std::string read_file_bytes(const char* path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        return {};
    }
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
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

bool verify_pattern_params(const std::vector<uint32_t>& data, uint32_t epoch, uint32_t seed, uint32_t flags) {
    for (size_t i = 0; i < data.size(); ++i) {
        const uint32_t mix = (static_cast<uint32_t>(i) * 2654435761u) ^ seed;
        const uint32_t expected = epoch ^ mix;
        const uint32_t value = (flags & 1u) ? ~expected : expected;
        if (data[i] != value) {
            return false;
        }
    }
    return true;
}

void mutate_words(std::vector<uint32_t>& data, uint32_t epoch) {
    if (data.empty()) {
        return;
    }
    for (uint32_t j = 0; j < 4; ++j) {
        uint32_t index = (epoch * 7u + j * 13u) % static_cast<uint32_t>(data.size());
        data[index] ^= 0xA5A50000u + epoch * 17u + j;
    }
}

uint32_t get_env_u32(const char* name, uint32_t fallback) {
    const char* value = std::getenv(name);
    if (!value || !*value) {
        return fallback;
    }
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    if (!end || *end != '\0') {
        return fallback;
    }
    return static_cast<uint32_t>(parsed);
}

__host__ __device__ uint32_t multistream_pattern(uint32_t epoch, uint32_t seed, uint32_t idx) {
    return epoch ^ seed ^ (idx * 2654435761u);
}

__host__ __device__ uint32_t multistream_commit(uint32_t epoch, uint32_t seed) {
    return epoch ^ seed ^ 0xA5A5A5A5u;
}

__global__ void write_region_data_kernel(uint32_t* data, uint32_t count, uint32_t epoch, uint32_t seed) {
    const uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (count == 0u) {
        return;
    }
    const uint32_t commit_index = count - 1u;
    if (idx < commit_index) {
        data[idx] = multistream_pattern(epoch, seed, idx);
    }
}

__global__ void write_commit_kernel(uint32_t* data, uint32_t count, uint32_t epoch, uint32_t seed) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (count == 0u) {
            return;
        }
        data[count - 1u] = multistream_commit(epoch, seed);
    }
}

bool verify_multistream_region(const std::vector<uint32_t>& data, uint32_t epoch, uint32_t seed) {
    if (data.size() < 2) {
        return false;
    }
    const uint32_t commit_index = static_cast<uint32_t>(data.size() - 1u);
    for (uint32_t i = 0; i < commit_index; ++i) {
        if (data[i] != multistream_pattern(epoch, seed, i)) {
            return false;
        }
    }
    return data[commit_index] == multistream_commit(epoch, seed);
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
    cfg.retention_epochs = 0;
    cfg.overwrite_mode = tt::OverwriteMode::kDropOldest;
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
    cfg.retention_epochs = 0;
    cfg.overwrite_mode = tt::OverwriteMode::kDropOldest;
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

bool test_delta_single_region() {
    const uint32_t element_count = 512;
    const uint32_t size_bytes = element_count * sizeof(uint32_t);
    const uint32_t epoch_count = 16;

    void* d_buffer = nullptr;
    if (!check_cuda(cudaMalloc(&d_buffer, size_bytes), "cudaMalloc delta")) {
        return false;
    }

    tt::Recorder recorder;
    tt::RecorderConfig cfg{};
    cfg.ring_bytes = 65536;
    cfg.epoch_capacity = 32;
    cfg.region_capacity = 4;
    cfg.retention_epochs = 0;
    cfg.overwrite_mode = tt::OverwriteMode::kDropOldest;
    if (!recorder.init(cfg)) {
        cudaFree(d_buffer);
        return false;
    }

    if (!recorder.register_region(0, d_buffer, size_bytes, 1)) {
        recorder.shutdown();
        cudaFree(d_buffer);
        return false;
    }
    if (!recorder.set_region_full_snapshot_period(0, 4)) {
        recorder.shutdown();
        cudaFree(d_buffer);
        return false;
    }

    std::vector<uint32_t> host_model(element_count);
    std::vector<uint32_t> host_out(element_count);
    std::vector<std::vector<uint32_t>> expected(epoch_count, std::vector<uint32_t>(element_count));

    fill_pattern(host_model, 0x12345678u);
    for (uint32_t epoch = 0; epoch < epoch_count; ++epoch) {
        if (epoch != 0) {
            mutate_words(host_model, epoch);
        }
        if (!check_cuda(cudaMemcpy(d_buffer, host_model.data(), size_bytes, cudaMemcpyHostToDevice), "memcpy delta in")) {
            recorder.shutdown();
            cudaFree(d_buffer);
            return false;
        }
        if (!recorder.capture_epoch(0)) {
            recorder.shutdown();
            cudaFree(d_buffer);
            return false;
        }
        expected[epoch] = host_model;
    }

    for (uint32_t epoch = 0; epoch < epoch_count; ++epoch) {
        if (!recorder.rewind_to_epoch(epoch, 0)) {
            recorder.shutdown();
            cudaFree(d_buffer);
            return false;
        }
        if (!check_cuda(cudaMemcpy(host_out.data(), d_buffer, size_bytes, cudaMemcpyDeviceToHost), "memcpy delta out")) {
            recorder.shutdown();
            cudaFree(d_buffer);
            return false;
        }
        if (host_out != expected[epoch]) {
            recorder.shutdown();
            cudaFree(d_buffer);
            return false;
        }
    }

    recorder.shutdown();
    cudaFree(d_buffer);
    return true;
}

bool test_wrap_marker() {
    const uint32_t element_count = 256;
    const uint32_t size_bytes = element_count * sizeof(uint32_t);
    const uint32_t epoch_count = 6;
    const uint32_t target_epoch = 4;

    void* d_buffer = nullptr;
    if (!check_cuda(cudaMalloc(&d_buffer, size_bytes), "cudaMalloc wrap")) {
        return false;
    }

    tt::Recorder recorder;
    tt::RecorderConfig cfg{};
    cfg.ring_bytes = 4096;
    cfg.epoch_capacity = 16;
    cfg.region_capacity = 4;
    cfg.retention_epochs = 0;
    cfg.overwrite_mode = tt::OverwriteMode::kDropOldest;
    if (!recorder.init(cfg)) {
        cudaFree(d_buffer);
        return false;
    }

    if (!recorder.register_region(0, d_buffer, size_bytes, 1)) {
        recorder.shutdown();
        cudaFree(d_buffer);
        return false;
    }

    std::vector<uint32_t> host_model(element_count);
    std::vector<uint32_t> host_out(element_count);
    std::vector<std::vector<uint32_t>> expected(epoch_count, std::vector<uint32_t>(element_count));

    for (uint32_t epoch = 0; epoch < epoch_count; ++epoch) {
        fill_pattern(host_model, 0xCAFE0000u + epoch);
        if (!check_cuda(cudaMemcpy(d_buffer, host_model.data(), size_bytes, cudaMemcpyHostToDevice), "memcpy wrap in")) {
            recorder.shutdown();
            cudaFree(d_buffer);
            return false;
        }
        if (!recorder.capture_epoch(0)) {
            recorder.shutdown();
            cudaFree(d_buffer);
            return false;
        }
        expected[epoch] = host_model;
    }

    if (!recorder.rewind_to_epoch(target_epoch, 0)) {
        recorder.shutdown();
        cudaFree(d_buffer);
        return false;
    }
    if (!check_cuda(cudaMemcpy(host_out.data(), d_buffer, size_bytes, cudaMemcpyDeviceToHost), "memcpy wrap out")) {
        recorder.shutdown();
        cudaFree(d_buffer);
        return false;
    }
    if (host_out != expected[target_epoch]) {
        recorder.shutdown();
        cudaFree(d_buffer);
        return false;
    }

    recorder.shutdown();
    cudaFree(d_buffer);
    return true;
}

bool test_tiny_ring_wrap() {
    const uint32_t element_count = 64;
    const uint32_t size_bytes = element_count * sizeof(uint32_t);
    const uint32_t epoch_count = 5;
    const uint32_t target_epoch = epoch_count - 1;

    void* d_buffer = nullptr;
    if (!check_cuda(cudaMalloc(&d_buffer, size_bytes), "cudaMalloc tiny wrap")) {
        return false;
    }

    tt::Recorder recorder;
    tt::RecorderConfig cfg{};
    cfg.ring_bytes = 640;
    cfg.epoch_capacity = 8;
    cfg.region_capacity = 2;
    cfg.retention_epochs = 0;
    cfg.overwrite_mode = tt::OverwriteMode::kDropOldest;
    if (!recorder.init(cfg)) {
        cudaFree(d_buffer);
        return false;
    }

    if (!recorder.register_region(0, d_buffer, size_bytes, 1)) {
        recorder.shutdown();
        cudaFree(d_buffer);
        return false;
    }

    std::vector<uint32_t> host_model(element_count);
    std::vector<uint32_t> host_out(element_count);
    std::vector<uint32_t> expected_last(element_count);

    for (uint32_t epoch = 0; epoch < epoch_count; ++epoch) {
        fill_pattern(host_model, 0xBEEF0000u + epoch);
        if (!check_cuda(cudaMemcpy(d_buffer, host_model.data(), size_bytes, cudaMemcpyHostToDevice), "memcpy tiny wrap in")) {
            recorder.shutdown();
            cudaFree(d_buffer);
            return false;
        }
        if (!recorder.capture_epoch(0)) {
            recorder.shutdown();
            cudaFree(d_buffer);
            return false;
        }
        if (epoch == target_epoch) {
            expected_last = host_model;
        }
    }

    if (!recorder.rewind_to_epoch(target_epoch, 0)) {
        recorder.shutdown();
        cudaFree(d_buffer);
        return false;
    }
    if (!check_cuda(cudaMemcpy(host_out.data(), d_buffer, size_bytes, cudaMemcpyDeviceToHost), "memcpy tiny wrap out")) {
        recorder.shutdown();
        cudaFree(d_buffer);
        return false;
    }
    if (host_out != expected_last) {
        recorder.shutdown();
        cudaFree(d_buffer);
        return false;
    }

    recorder.shutdown();
    cudaFree(d_buffer);
    return true;
}

bool test_overwrite_retention() {
    const uint32_t element_count = 64;
    const uint32_t size_bytes = element_count * sizeof(uint32_t);
    const uint32_t epoch_count = 6;

    void* d_buffer = nullptr;
    if (!check_cuda(cudaMalloc(&d_buffer, size_bytes), "cudaMalloc retention")) {
        return false;
    }

    tt::Recorder recorder;
    tt::RecorderConfig cfg{};
    cfg.ring_bytes = 4096;
    cfg.epoch_capacity = 8;
    cfg.region_capacity = 2;
    cfg.retention_epochs = 3;
    cfg.overwrite_mode = tt::OverwriteMode::kDropOldest;
    if (!recorder.init(cfg)) {
        cudaFree(d_buffer);
        return false;
    }

    if (!recorder.register_region(0, d_buffer, size_bytes, 1)) {
        recorder.shutdown();
        cudaFree(d_buffer);
        return false;
    }

    std::vector<uint32_t> host_model(element_count);
    std::vector<uint32_t> host_out(element_count);
    std::vector<std::vector<uint32_t>> expected(epoch_count, std::vector<uint32_t>(element_count));

    for (uint32_t epoch = 0; epoch < epoch_count; ++epoch) {
        fill_pattern(host_model, 0xABCD0000u + epoch);
        if (!check_cuda(cudaMemcpy(d_buffer, host_model.data(), size_bytes, cudaMemcpyHostToDevice), "memcpy retention in")) {
            recorder.shutdown();
            cudaFree(d_buffer);
            return false;
        }
        if (!recorder.capture_epoch(0)) {
            recorder.shutdown();
            cudaFree(d_buffer);
            return false;
        }
        expected[epoch] = host_model;
    }

    if (recorder.rewind_to_epoch(2, 0)) {
        recorder.shutdown();
        cudaFree(d_buffer);
        return false;
    }

    if (!recorder.rewind_to_epoch(4, 0)) {
        recorder.shutdown();
        cudaFree(d_buffer);
        return false;
    }
    if (!check_cuda(cudaMemcpy(host_out.data(), d_buffer, size_bytes, cudaMemcpyDeviceToHost), "memcpy retention out")) {
        recorder.shutdown();
        cudaFree(d_buffer);
        return false;
    }
    if (host_out != expected[4]) {
        recorder.shutdown();
        cudaFree(d_buffer);
        return false;
    }

    recorder.shutdown();
    cudaFree(d_buffer);
    return true;
}

bool test_backpressure() {
    const uint32_t element_count = 64;
    const uint32_t size_bytes = element_count * sizeof(uint32_t);

    void* d_buffer = nullptr;
    if (!check_cuda(cudaMalloc(&d_buffer, size_bytes), "cudaMalloc backpressure")) {
        return false;
    }

    tt::Recorder recorder;
    tt::RecorderConfig cfg{};
    cfg.ring_bytes = 320;
    cfg.epoch_capacity = 4;
    cfg.region_capacity = 2;
    cfg.retention_epochs = 0;
    cfg.overwrite_mode = tt::OverwriteMode::kBackpressure;
    if (!recorder.init(cfg)) {
        cudaFree(d_buffer);
        return false;
    }

    if (!recorder.register_region(0, d_buffer, size_bytes, 1)) {
        recorder.shutdown();
        cudaFree(d_buffer);
        return false;
    }

    std::vector<uint32_t> host_a(element_count);
    std::vector<uint32_t> host_b(element_count);
    std::vector<uint32_t> host_out(element_count);
    fill_pattern(host_a, 0x12340000u);
    fill_pattern(host_b, 0x56780000u);

    if (!check_cuda(cudaMemcpy(d_buffer, host_a.data(), size_bytes, cudaMemcpyHostToDevice), "memcpy backpressure A")) {
        recorder.shutdown();
        cudaFree(d_buffer);
        return false;
    }
    if (!recorder.capture_epoch(0)) {
        recorder.shutdown();
        cudaFree(d_buffer);
        return false;
    }

    if (!check_cuda(cudaMemcpy(d_buffer, host_b.data(), size_bytes, cudaMemcpyHostToDevice), "memcpy backpressure B")) {
        recorder.shutdown();
        cudaFree(d_buffer);
        return false;
    }
    if (recorder.capture_epoch(0)) {
        recorder.shutdown();
        cudaFree(d_buffer);
        return false;
    }

    if (!recorder.rewind_to_epoch(0, 0)) {
        recorder.shutdown();
        cudaFree(d_buffer);
        return false;
    }
    if (!check_cuda(cudaMemcpy(host_out.data(), d_buffer, size_bytes, cudaMemcpyDeviceToHost), "memcpy backpressure out")) {
        recorder.shutdown();
        cudaFree(d_buffer);
        return false;
    }
    if (host_out != host_a) {
        recorder.shutdown();
        cudaFree(d_buffer);
        return false;
    }

    recorder.shutdown();
    cudaFree(d_buffer);
    return true;
}

bool test_graph_capture_and_trace() {
    const uint32_t element_count = 512;
    const uint32_t size_bytes = element_count * sizeof(uint32_t);
    const uint32_t epoch_count = 4;
    const uint32_t stamps_per_epoch = 3;

    uint32_t* d_buffer = nullptr;
    uint32_t* d_epoch_counter = nullptr;
    uint64_t* d_stamps = nullptr;
    uint32_t* d_stamp_counter = nullptr;
    if (!check_cuda(cudaMalloc(&d_buffer, size_bytes), "cudaMalloc graph buffer")) {
        return false;
    }
    if (!check_cuda(cudaMalloc(&d_epoch_counter, sizeof(uint32_t)), "cudaMalloc graph epoch counter")) {
        cudaFree(d_buffer);
        return false;
    }
    if (!check_cuda(cudaMalloc(&d_stamps, sizeof(uint64_t) * epoch_count * stamps_per_epoch), "cudaMalloc graph stamps")) {
        cudaFree(d_epoch_counter);
        cudaFree(d_buffer);
        return false;
    }
    if (!check_cuda(cudaMalloc(&d_stamp_counter, sizeof(uint32_t)), "cudaMalloc graph stamp counter")) {
        cudaFree(d_stamps);
        cudaFree(d_epoch_counter);
        cudaFree(d_buffer);
        return false;
    }
    if (!check_cuda(cudaMemset(d_stamp_counter, 0, sizeof(uint32_t)), "memset graph stamp counter")) {
        cudaFree(d_stamp_counter);
        cudaFree(d_stamps);
        cudaFree(d_epoch_counter);
        cudaFree(d_buffer);
        return false;
    }

    cudaStream_t stream = nullptr;
    if (!check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate graph")) {
        cudaFree(d_stamp_counter);
        cudaFree(d_stamps);
        cudaFree(d_epoch_counter);
        cudaFree(d_buffer);
        return false;
    }

    const uint32_t per_chunk_bytes = (static_cast<uint32_t>(sizeof(tt::ChunkHeader)) + size_bytes + 31u) & ~31u;
    const uint32_t ring_bytes = per_chunk_bytes * epoch_count + 4096u;

    tt::Recorder recorder;
    tt::RecorderConfig cfg{};
    cfg.ring_bytes = ring_bytes;
    cfg.epoch_capacity = 16;
    cfg.region_capacity = 4;
    cfg.retention_epochs = 0;
    cfg.overwrite_mode = tt::OverwriteMode::kDropOldest;
    cfg.enable_graph_stamps = true;
    cfg.graph_stamps = d_stamps;
    cfg.graph_stamp_counter = d_stamp_counter;
    if (!recorder.init(cfg)) {
        cudaStreamDestroy(stream);
        cudaFree(d_stamp_counter);
        cudaFree(d_stamps);
        cudaFree(d_epoch_counter);
        cudaFree(d_buffer);
        return false;
    }

    if (!recorder.register_region(0, d_buffer, size_bytes, 1)) {
        recorder.shutdown();
        cudaStreamDestroy(stream);
        cudaFree(d_stamp_counter);
        cudaFree(d_stamps);
        cudaFree(d_epoch_counter);
        cudaFree(d_buffer);
        return false;
    }

    tt::GraphSession graph;
    if (!graph.begin_capture(stream)) {
        recorder.shutdown();
        cudaStreamDestroy(stream);
        cudaFree(d_stamp_counter);
        cudaFree(d_stamps);
        cudaFree(d_epoch_counter);
        cudaFree(d_buffer);
        return false;
    }

    const uint32_t threads = 256;
    const uint32_t blocks = (element_count + threads - 1u) / threads;
    write_pattern_kernel<<<blocks, threads, 0, stream>>>(d_buffer, element_count, d_epoch_counter);
    if (!recorder.capture_epoch(stream)) {
        graph.destroy();
        recorder.shutdown();
        cudaStreamDestroy(stream);
        cudaFree(d_stamp_counter);
        cudaFree(d_stamps);
        cudaFree(d_epoch_counter);
        cudaFree(d_buffer);
        return false;
    }
    if (!graph.end_capture()) {
        graph.destroy();
        recorder.shutdown();
        cudaStreamDestroy(stream);
        cudaFree(d_stamp_counter);
        cudaFree(d_stamps);
        cudaFree(d_epoch_counter);
        cudaFree(d_buffer);
        return false;
    }

    for (uint32_t epoch = 0; epoch < epoch_count; ++epoch) {
        set_epoch_kernel<<<1, 1, 0, stream>>>(d_epoch_counter, epoch);
        if (!graph.launch(stream)) {
            graph.destroy();
            recorder.shutdown();
            cudaStreamDestroy(stream);
            cudaFree(d_stamp_counter);
            cudaFree(d_stamps);
            cudaFree(d_epoch_counter);
            cudaFree(d_buffer);
            return false;
        }
    }
    if (!check_cuda(cudaStreamSynchronize(stream), "graph sync test")) {
        graph.destroy();
        recorder.shutdown();
        cudaStreamDestroy(stream);
        cudaFree(d_stamp_counter);
        cudaFree(d_stamps);
        cudaFree(d_epoch_counter);
        cudaFree(d_buffer);
        return false;
    }

    for (uint32_t epoch = 0; epoch < epoch_count; ++epoch) {
        if (!recorder.rewind_to_epoch(epoch, stream)) {
            graph.destroy();
            recorder.shutdown();
            cudaStreamDestroy(stream);
            cudaFree(d_stamp_counter);
            cudaFree(d_stamps);
            cudaFree(d_epoch_counter);
            cudaFree(d_buffer);
            return false;
        }
        std::vector<uint32_t> host_out(element_count);
        if (!check_cuda(cudaMemcpy(host_out.data(), d_buffer, size_bytes, cudaMemcpyDeviceToHost), "memcpy graph out")) {
            graph.destroy();
            recorder.shutdown();
            cudaStreamDestroy(stream);
            cudaFree(d_stamp_counter);
            cudaFree(d_stamps);
            cudaFree(d_epoch_counter);
            cudaFree(d_buffer);
            return false;
        }
        if (!verify_pattern(host_out, epoch)) {
            graph.destroy();
            recorder.shutdown();
            cudaStreamDestroy(stream);
            cudaFree(d_stamp_counter);
            cudaFree(d_stamps);
            cudaFree(d_epoch_counter);
            cudaFree(d_buffer);
            return false;
        }
    }

    uint32_t stamp_count = 0;
    if (!check_cuda(cudaMemcpy(&stamp_count, d_stamp_counter, sizeof(uint32_t), cudaMemcpyDeviceToHost), "memcpy graph stamp count")) {
        graph.destroy();
        recorder.shutdown();
        cudaStreamDestroy(stream);
        cudaFree(d_stamp_counter);
        cudaFree(d_stamps);
        cudaFree(d_epoch_counter);
        cudaFree(d_buffer);
        return false;
    }
    std::vector<uint64_t> host_stamps(stamp_count);
    if (stamp_count > 0) {
        if (!check_cuda(cudaMemcpy(host_stamps.data(), d_stamps, sizeof(uint64_t) * stamp_count, cudaMemcpyDeviceToHost), "memcpy graph stamps")) {
            graph.destroy();
            recorder.shutdown();
            cudaStreamDestroy(stream);
            cudaFree(d_stamp_counter);
            cudaFree(d_stamps);
            cudaFree(d_epoch_counter);
            cudaFree(d_buffer);
            return false;
        }
    }

    tt::TraceCollector trace;
    if (!host_stamps.empty()) {
        const uint64_t base_stamp = host_stamps[0];
        int clock_rate_khz = 0;
        if (!check_cuda(cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate, 0), "device clock rate test")) {
            graph.destroy();
            recorder.shutdown();
            cudaStreamDestroy(stream);
            cudaFree(d_stamp_counter);
            cudaFree(d_stamps);
            cudaFree(d_epoch_counter);
            cudaFree(d_buffer);
            return false;
        }
        const double cycles_to_us = 1000.0 / static_cast<double>(clock_rate_khz);
        const uint32_t epoch_samples = stamp_count / stamps_per_epoch;
        for (uint32_t i = 0; i < epoch_samples; ++i) {
            const uint64_t start_stamp = host_stamps[i * stamps_per_epoch + 0];
            const uint64_t end_stamp = host_stamps[i * stamps_per_epoch + 2];
            tt::TraceEvent event{};
            event.name = "epoch_total";
            event.cat = "epoch";
            event.ts_us = static_cast<double>(start_stamp - base_stamp) * cycles_to_us;
            event.dur_us = static_cast<double>(end_stamp - start_stamp) * cycles_to_us;
            event.pid = 1;
            event.tid = 1;
            event.args.push_back({"epoch_id", std::to_string(i), false});
            trace.add_event(event);
        }
    }

    trace.add_event({"graph_launch", "graph", 0.0, 1.0, 1, 0, {{"epoch_id", "0", false}}});
    tt::GetCuptiKernelTracer().append_kernel_events(trace);
    trace.write("trace/tt_trace.json");

    const bool valid_trace = trace_json_valid("trace/tt_trace.json");
    graph.destroy();
    recorder.shutdown();
    cudaStreamDestroy(stream);
    cudaFree(d_stamp_counter);
    cudaFree(d_stamps);
    cudaFree(d_epoch_counter);
    cudaFree(d_buffer);
    return valid_trace;
}

bool test_graph_iter_params_update() {
    const uint32_t element_count = 256;
    const uint32_t size_bytes = element_count * sizeof(uint32_t);

    uint32_t* d_buffer = nullptr;
    tt::IterParams* d_params = nullptr;
    if (!check_cuda(cudaMalloc(&d_buffer, size_bytes), "cudaMalloc iter buffer")) {
        return false;
    }
    if (!check_cuda(cudaMalloc(&d_params, sizeof(tt::IterParams)), "cudaMalloc iter params")) {
        cudaFree(d_buffer);
        return false;
    }

    cudaStream_t stream = nullptr;
    if (!check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate iter params")) {
        cudaFree(d_params);
        cudaFree(d_buffer);
        return false;
    }

    tt::GraphSession graph;
    if (!graph.begin_capture(stream)) {
        cudaStreamDestroy(stream);
        cudaFree(d_params);
        cudaFree(d_buffer);
        return false;
    }
    const uint32_t threads = 256;
    const uint32_t blocks = (element_count + threads - 1u) / threads;
    write_pattern_iter_kernel<<<blocks, threads, 0, stream>>>(d_buffer, element_count, d_params);
    if (!graph.end_capture()) {
        graph.destroy();
        cudaStreamDestroy(stream);
        cudaFree(d_params);
        cudaFree(d_buffer);
        return false;
    }

    for (uint32_t epoch = 0; epoch < 3; ++epoch) {
        tt::IterParams host_params{};
        host_params.epoch = epoch;
        host_params.seed = 0x55u + epoch;
        host_params.flags = epoch & 1u;
        if (!check_cuda(cudaMemcpyAsync(d_params,
                &host_params,
                sizeof(tt::IterParams),
                cudaMemcpyHostToDevice,
                stream),
                "memcpy iter params test")) {
            graph.destroy();
            cudaStreamDestroy(stream);
            cudaFree(d_params);
            cudaFree(d_buffer);
            return false;
        }
        if (!graph.launch(stream)) {
            graph.destroy();
            cudaStreamDestroy(stream);
            cudaFree(d_params);
            cudaFree(d_buffer);
            return false;
        }
        if (!check_cuda(cudaStreamSynchronize(stream), "iter params sync")) {
            graph.destroy();
            cudaStreamDestroy(stream);
            cudaFree(d_params);
            cudaFree(d_buffer);
            return false;
        }
        std::vector<uint32_t> host_out(element_count);
        if (!check_cuda(cudaMemcpy(host_out.data(), d_buffer, size_bytes, cudaMemcpyDeviceToHost), "memcpy iter out")) {
            graph.destroy();
            cudaStreamDestroy(stream);
            cudaFree(d_params);
            cudaFree(d_buffer);
            return false;
        }
        if (!verify_pattern_params(host_out, host_params.epoch, host_params.seed, host_params.flags)) {
            graph.destroy();
            cudaStreamDestroy(stream);
            cudaFree(d_params);
            cudaFree(d_buffer);
            return false;
        }
    }

    graph.destroy();
    cudaStreamDestroy(stream);
    cudaFree(d_params);
    cudaFree(d_buffer);
    return true;
}

bool test_graph_kernel_param_patch() {
    tt::RegisterGraphKernelTag(reinterpret_cast<const void*>(write_pattern_patch_kernel), "test.write_pattern_patch");

    const uint32_t element_count = 256;
    const uint32_t size_bytes = element_count * sizeof(uint32_t);

    uint32_t* d_buffer = nullptr;
    if (!check_cuda(cudaMalloc(&d_buffer, size_bytes), "cudaMalloc patch buffer")) {
        return false;
    }

    cudaStream_t stream = nullptr;
    if (!check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate patch")) {
        cudaFree(d_buffer);
        return false;
    }

    tt::GraphSession graph;
    if (!graph.begin_capture(stream)) {
        cudaStreamDestroy(stream);
        cudaFree(d_buffer);
        return false;
    }
    const uint32_t threads = 256;
    const uint32_t blocks = (element_count + threads - 1u) / threads;
    write_pattern_patch_kernel<<<blocks, threads, 0, stream>>>(d_buffer, element_count, 0u, 0u);
    if (!graph.end_capture()) {
        graph.destroy();
        cudaStreamDestroy(stream);
        cudaFree(d_buffer);
        return false;
    }

    uint32_t node_id = UINT32_MAX;
    for (const auto& node : graph.get_nodes()) {
        if (node.name == "test.write_pattern_patch") {
            node_id = node.id;
            break;
        }
    }
    if (node_id == UINT32_MAX) {
        graph.destroy();
        cudaStreamDestroy(stream);
        cudaFree(d_buffer);
        return false;
    }

    cudaKernelNodeParams params{};
    if (!graph.get_kernel_node_params(node_id, &params)) {
        graph.destroy();
        cudaStreamDestroy(stream);
        cudaFree(d_buffer);
        return false;
    }

    uint32_t epoch = 2;
    uint32_t seed = 0x44u;
    uint32_t element_count_value = element_count;
    void* kernel_args[] = {&d_buffer, &element_count_value, &epoch, &seed};
    params.kernelParams = kernel_args;
    tt::GraphUpdateStatus status{};
    if (!graph.update_kernel_node_params(node_id, params, &status)) {
        std::printf("test_graph_kernel_param_patch skipped: %s (%s)\n",
            tt::GraphUpdateReasonString(status.reason),
            cudaGetErrorString(status.cuda_error));
        graph.destroy();
        cudaStreamDestroy(stream);
        cudaFree(d_buffer);
        return status.reason == tt::GraphUpdateReason::kGraphUpdateNotSupported;
    }

    if (!graph.launch(stream) || !check_cuda(cudaStreamSynchronize(stream), "patch sync 1")) {
        graph.destroy();
        cudaStreamDestroy(stream);
        cudaFree(d_buffer);
        return false;
    }
    std::vector<uint32_t> host_out(element_count);
    if (!check_cuda(cudaMemcpy(host_out.data(), d_buffer, size_bytes, cudaMemcpyDeviceToHost), "memcpy patch out 1")) {
        graph.destroy();
        cudaStreamDestroy(stream);
        cudaFree(d_buffer);
        return false;
    }
    if (!verify_pattern_params(host_out, epoch, seed, 0u)) {
        graph.destroy();
        cudaStreamDestroy(stream);
        cudaFree(d_buffer);
        return false;
    }

    epoch = 5;
    seed = 0x99u;
    void* kernel_args2[] = {&d_buffer, &element_count_value, &epoch, &seed};
    params.kernelParams = kernel_args2;
    if (!graph.update_kernel_node_params(node_id, params, &status)) {
        std::printf("test_graph_kernel_param_patch skipped: %s (%s)\n",
            tt::GraphUpdateReasonString(status.reason),
            cudaGetErrorString(status.cuda_error));
        graph.destroy();
        cudaStreamDestroy(stream);
        cudaFree(d_buffer);
        return status.reason == tt::GraphUpdateReason::kGraphUpdateNotSupported;
    }
    if (!graph.launch(stream) || !check_cuda(cudaStreamSynchronize(stream), "patch sync 2")) {
        graph.destroy();
        cudaStreamDestroy(stream);
        cudaFree(d_buffer);
        return false;
    }
    if (!check_cuda(cudaMemcpy(host_out.data(), d_buffer, size_bytes, cudaMemcpyDeviceToHost), "memcpy patch out 2")) {
        graph.destroy();
        cudaStreamDestroy(stream);
        cudaFree(d_buffer);
        return false;
    }
    if (!verify_pattern_params(host_out, epoch, seed, 0u)) {
        graph.destroy();
        cudaStreamDestroy(stream);
        cudaFree(d_buffer);
        return false;
    }

    graph.destroy();
    cudaStreamDestroy(stream);
    cudaFree(d_buffer);
    return true;
}

bool test_graph_control_updates() {
    const uint32_t element_count = 256;
    const uint32_t size_bytes = element_count * sizeof(uint32_t);
    const uint32_t epoch_count = 6;
    const uint32_t stamps_per_epoch = 3;

    uint32_t* d_buffer_a = nullptr;
    uint32_t* d_buffer_b = nullptr;
    uint32_t* d_epoch_counter = nullptr;
    uint64_t* d_stamps = nullptr;
    uint32_t* d_stamp_counter = nullptr;
    if (!check_cuda(cudaMalloc(&d_buffer_a, size_bytes), "cudaMalloc ctrl buffer A") ||
        !check_cuda(cudaMalloc(&d_buffer_b, size_bytes), "cudaMalloc ctrl buffer B") ||
        !check_cuda(cudaMalloc(&d_epoch_counter, sizeof(uint32_t)), "cudaMalloc ctrl epoch") ||
        !check_cuda(cudaMalloc(&d_stamps, sizeof(uint64_t) * epoch_count * stamps_per_epoch), "cudaMalloc ctrl stamps") ||
        !check_cuda(cudaMalloc(&d_stamp_counter, sizeof(uint32_t)), "cudaMalloc ctrl stamp counter")) {
        return false;
    }
    if (!check_cuda(cudaMemset(d_stamp_counter, 0, sizeof(uint32_t)), "memset ctrl stamp counter")) {
        return false;
    }

    cudaStream_t stream = nullptr;
    if (!check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate ctrl")) {
        return false;
    }

    const uint32_t per_chunk_bytes = (static_cast<uint32_t>(sizeof(tt::ChunkHeader)) + size_bytes + 31u) & ~31u;
    const uint32_t ring_bytes = per_chunk_bytes * epoch_count + 4096u;

    tt::Recorder recorder;
    tt::RecorderConfig cfg{};
    cfg.ring_bytes = ring_bytes;
    cfg.epoch_capacity = 16;
    cfg.region_capacity = 4;
    cfg.retention_epochs = 0;
    cfg.overwrite_mode = tt::OverwriteMode::kDropOldest;
    cfg.enable_graph_stamps = true;
    cfg.graph_stamps = d_stamps;
    cfg.graph_stamp_counter = d_stamp_counter;
    if (!recorder.init(cfg)) {
        return false;
    }
    if (!recorder.register_region(0, d_buffer_a, size_bytes, 1) ||
        !recorder.register_region(1, d_buffer_b, size_bytes, 1)) {
        recorder.shutdown();
        return false;
    }

    tt::GraphSession graph;
    if (!graph.begin_capture(stream)) {
        recorder.shutdown();
        return false;
    }
    const uint32_t threads = 256;
    const uint32_t blocks = (element_count + threads - 1u) / threads;
    write_pattern_kernel<<<blocks, threads, 0, stream>>>(d_buffer_a, element_count, d_epoch_counter);
    write_pattern_kernel<<<blocks, threads, 0, stream>>>(d_buffer_b, element_count, d_epoch_counter);
    if (!recorder.capture_epoch(stream)) {
        graph.destroy();
        recorder.shutdown();
        return false;
    }
    if (!graph.end_capture()) {
        graph.destroy();
        recorder.shutdown();
        return false;
    }

    tt::RecorderGraphControl control{};
    control.region_mask = (1ull << 2) - 1ull;
    control.snapshot_period = 0;
    control.flags = tt::kGraphControlStampsEnabled;
    if (!recorder.update_graph_control(control, stream)) {
        graph.destroy();
        recorder.shutdown();
        return false;
    }

    std::vector<uint32_t> expected_region1(epoch_count, 0u);
    uint32_t last_region1_epoch = 0u;
    for (uint32_t epoch = 0; epoch < epoch_count; ++epoch) {
        set_epoch_kernel<<<1, 1, 0, stream>>>(d_epoch_counter, epoch);
        if (epoch % 2 == 1) {
            control.region_mask = 0x1ull;
        } else {
            control.region_mask = 0x3ull;
            last_region1_epoch = epoch;
        }
        control.snapshot_period = (epoch % 3 == 0) ? 1u : 0u;
        control.flags = (epoch % 2 == 0) ? tt::kGraphControlStampsEnabled : 0u;
        if (!recorder.update_graph_control(control, stream)) {
            graph.destroy();
            recorder.shutdown();
            return false;
        }
        expected_region1[epoch] = last_region1_epoch;
        if (!graph.launch(stream)) {
            graph.destroy();
            recorder.shutdown();
            return false;
        }
    }
    if (!check_cuda(cudaStreamSynchronize(stream), "ctrl sync")) {
        graph.destroy();
        recorder.shutdown();
        return false;
    }

    for (uint32_t epoch = 0; epoch < epoch_count; ++epoch) {
        if (!recorder.rewind_to_epoch(epoch, stream)) {
            graph.destroy();
            recorder.shutdown();
            return false;
        }
        std::vector<uint32_t> host_a(element_count);
        std::vector<uint32_t> host_b(element_count);
        if (!check_cuda(cudaMemcpy(host_a.data(), d_buffer_a, size_bytes, cudaMemcpyDeviceToHost), "memcpy ctrl out A") ||
            !check_cuda(cudaMemcpy(host_b.data(), d_buffer_b, size_bytes, cudaMemcpyDeviceToHost), "memcpy ctrl out B")) {
            graph.destroy();
            recorder.shutdown();
            return false;
        }
        if (!verify_pattern(host_a, epoch)) {
            graph.destroy();
            recorder.shutdown();
            return false;
        }
        if (!verify_pattern(host_b, expected_region1[epoch])) {
            graph.destroy();
            recorder.shutdown();
            return false;
        }
    }

    graph.destroy();
    recorder.shutdown();
    cudaStreamDestroy(stream);
    cudaFree(d_stamp_counter);
    cudaFree(d_stamps);
    cudaFree(d_epoch_counter);
    cudaFree(d_buffer_b);
    cudaFree(d_buffer_a);
    return true;
}

bool run_deterministic_manifest_capture(const char* manifest_path, uint32_t epoch_count) {
    const uint32_t element_count = 1024;
    const uint32_t size_bytes = element_count * sizeof(uint32_t);

    uint32_t* d_buffer = nullptr;
    uint32_t* d_epoch_counter = nullptr;
    if (!check_cuda(cudaMalloc(&d_buffer, size_bytes), "cudaMalloc deterministic buffer")) {
        return false;
    }
    if (!check_cuda(cudaMalloc(&d_epoch_counter, sizeof(uint32_t)), "cudaMalloc deterministic counter")) {
        cudaFree(d_buffer);
        return false;
    }

    const uint32_t per_chunk_bytes = (static_cast<uint32_t>(sizeof(tt::ChunkHeader)) + size_bytes + 31u) & ~31u;
    const uint32_t ring_bytes = per_chunk_bytes * epoch_count + 4096u;

    tt::Recorder recorder;
    tt::RecorderConfig cfg{};
    cfg.ring_bytes = ring_bytes;
    cfg.epoch_capacity = 32;
    cfg.region_capacity = 4;
    cfg.retention_epochs = 0;
    cfg.overwrite_mode = tt::OverwriteMode::kDropOldest;
    cfg.deterministic = true;
    cfg.enable_manifest = true;
    if (!recorder.init(cfg)) {
        cudaFree(d_epoch_counter);
        cudaFree(d_buffer);
        return false;
    }

    if (!recorder.register_region(0, d_buffer, size_bytes, 1)) {
        recorder.shutdown();
        cudaFree(d_epoch_counter);
        cudaFree(d_buffer);
        return false;
    }

    const uint32_t threads = 256;
    const uint32_t blocks = (element_count + threads - 1u) / threads;
    for (uint32_t epoch = 0; epoch < epoch_count; ++epoch) {
        set_epoch_kernel<<<1, 1>>>(d_epoch_counter, epoch);
        write_pattern_kernel<<<blocks, threads>>>(d_buffer, element_count, d_epoch_counter);
        if (!recorder.capture_epoch(0)) {
            recorder.shutdown();
            cudaFree(d_epoch_counter);
            cudaFree(d_buffer);
            return false;
        }
    }

    bool wrote_manifest = recorder.write_manifest_json(manifest_path);
    recorder.shutdown();
    cudaFree(d_epoch_counter);
    cudaFree(d_buffer);
    return wrote_manifest;
}

bool test_deterministic_manifest_match() {
    const uint32_t epoch_count = 6;
    const char* path_a = "trace/tt_manifest_a.json";
    const char* path_b = "trace/tt_manifest_b.json";
    if (!run_deterministic_manifest_capture(path_a, epoch_count)) {
        return false;
    }
    if (!run_deterministic_manifest_capture(path_b, epoch_count)) {
        return false;
    }
    const std::string manifest_a = read_file_bytes(path_a);
    const std::string manifest_b = read_file_bytes(path_b);
    if (manifest_a.empty() || manifest_b.empty()) {
        return false;
    }
    return manifest_a == manifest_b;
}

bool test_deterministic_rewind() {
    const uint32_t element_count = 512;
    const uint32_t size_bytes = element_count * sizeof(uint32_t);
    const uint32_t epoch_count = 5;
    const uint32_t target_epoch = 3;

    void* d_buffer = nullptr;
    if (!check_cuda(cudaMalloc(&d_buffer, size_bytes), "cudaMalloc deterministic rewind")) {
        return false;
    }

    tt::Recorder recorder;
    tt::RecorderConfig cfg{};
    cfg.ring_bytes = 65536;
    cfg.epoch_capacity = 16;
    cfg.region_capacity = 4;
    cfg.retention_epochs = 0;
    cfg.overwrite_mode = tt::OverwriteMode::kDropOldest;
    cfg.deterministic = true;
    if (!recorder.init(cfg)) {
        cudaFree(d_buffer);
        return false;
    }
    if (!recorder.register_region(0, d_buffer, size_bytes, 1)) {
        recorder.shutdown();
        cudaFree(d_buffer);
        return false;
    }

    std::vector<uint32_t> host_model(element_count);
    std::vector<uint32_t> host_out(element_count);
    std::vector<uint32_t> expected(element_count);
    for (uint32_t epoch = 0; epoch < epoch_count; ++epoch) {
        fill_pattern(host_model, 0xBEEF0000u + epoch);
        if (epoch == target_epoch) {
            expected = host_model;
        }
        if (!check_cuda(cudaMemcpy(d_buffer, host_model.data(), size_bytes, cudaMemcpyHostToDevice), "memcpy deterministic in")) {
            recorder.shutdown();
            cudaFree(d_buffer);
            return false;
        }
        if (!recorder.capture_epoch(0)) {
            recorder.shutdown();
            cudaFree(d_buffer);
            return false;
        }
    }

    if (!recorder.rewind_to_epoch(target_epoch, 0)) {
        recorder.shutdown();
        cudaFree(d_buffer);
        return false;
    }
    if (!check_cuda(cudaMemcpy(host_out.data(), d_buffer, size_bytes, cudaMemcpyDeviceToHost), "memcpy deterministic out")) {
        recorder.shutdown();
        cudaFree(d_buffer);
        return false;
    }
    bool ok = (host_out == expected);
    if (!ok) {
        recorder.shutdown();
        cudaFree(d_buffer);
        return false;
    }

    recorder.shutdown();
    cudaFree(d_buffer);
    return true;
}

bool test_multistream_no_deps_failure() {
    const uint32_t element_count = 256;
    const uint32_t size_bytes = element_count * sizeof(uint32_t);
    const uint32_t epoch_count = get_env_u32("TT_TEST_MULTISTREAM_ITERS", 6);

    void* d_buffer = nullptr;
    if (!check_cuda(cudaMalloc(&d_buffer, size_bytes), "cudaMalloc multistream buffer")) {
        return false;
    }

    cudaStream_t producer_stream = nullptr;
    cudaStream_t capture_stream = nullptr;
    cudaEvent_t data_event = nullptr;
    if (!check_cuda(cudaStreamCreate(&producer_stream), "producer stream create") ||
        !check_cuda(cudaStreamCreate(&capture_stream), "capture stream create") ||
        !check_cuda(cudaEventCreateWithFlags(&data_event, cudaEventDisableTiming), "data event create")) {
        cudaFree(d_buffer);
        return false;
    }

    tt::Recorder recorder;
    tt::RecorderConfig cfg{};
    cfg.ring_bytes = size_bytes * (epoch_count + 2u) + 4096u;
    cfg.epoch_capacity = epoch_count + 2u;
    cfg.region_capacity = 1;
    cfg.retention_epochs = 0;
    cfg.overwrite_mode = tt::OverwriteMode::kDropOldest;
    if (!recorder.init(cfg)) {
        cudaEventDestroy(data_event);
        cudaStreamDestroy(capture_stream);
        cudaStreamDestroy(producer_stream);
        cudaFree(d_buffer);
        return false;
    }
    if (!recorder.register_region(0, d_buffer, size_bytes, 1)) {
        recorder.shutdown();
        cudaEventDestroy(data_event);
        cudaStreamDestroy(capture_stream);
        cudaStreamDestroy(producer_stream);
        cudaFree(d_buffer);
        return false;
    }

    const uint32_t seed = 0xABCDEF01u;
    const uint32_t threads = 128;
    const uint32_t blocks = (element_count + threads - 1u) / threads;
    for (uint32_t epoch = 0; epoch < epoch_count; ++epoch) {
        write_region_data_kernel<<<blocks, threads, 0, producer_stream>>>(
            static_cast<uint32_t*>(d_buffer),
            element_count,
            epoch,
            seed);
        if (!check_cuda(cudaGetLastError(), "launch data kernel")) {
            recorder.shutdown();
            cudaEventDestroy(data_event);
            cudaStreamDestroy(capture_stream);
            cudaStreamDestroy(producer_stream);
            cudaFree(d_buffer);
            return false;
        }
        if (!check_cuda(cudaEventRecord(data_event, producer_stream), "record data event")) {
            recorder.shutdown();
            cudaEventDestroy(data_event);
            cudaStreamDestroy(capture_stream);
            cudaStreamDestroy(producer_stream);
            cudaFree(d_buffer);
            return false;
        }
        if (!check_cuda(cudaEventSynchronize(data_event), "sync data event")) {
            recorder.shutdown();
            cudaEventDestroy(data_event);
            cudaStreamDestroy(capture_stream);
            cudaStreamDestroy(producer_stream);
            cudaFree(d_buffer);
            return false;
        }

        if (!recorder.capture_epoch(capture_stream)) {
            recorder.shutdown();
            cudaEventDestroy(data_event);
            cudaStreamDestroy(capture_stream);
            cudaStreamDestroy(producer_stream);
            cudaFree(d_buffer);
            return false;
        }

        write_commit_kernel<<<1, 1, 0, producer_stream>>>(
            static_cast<uint32_t*>(d_buffer),
            element_count,
            epoch,
            seed);
        if (!check_cuda(cudaGetLastError(), "launch commit kernel")) {
            recorder.shutdown();
            cudaEventDestroy(data_event);
            cudaStreamDestroy(capture_stream);
            cudaStreamDestroy(producer_stream);
            cudaFree(d_buffer);
            return false;
        }
        if (!check_cuda(cudaStreamSynchronize(producer_stream), "producer sync")) {
            recorder.shutdown();
            cudaEventDestroy(data_event);
            cudaStreamDestroy(capture_stream);
            cudaStreamDestroy(producer_stream);
            cudaFree(d_buffer);
            return false;
        }
    }

    std::vector<uint32_t> host_data(element_count, 0u);
    uint32_t mismatches = 0;
    for (uint32_t epoch = 0; epoch < epoch_count; ++epoch) {
        if (!recorder.rewind_to_epoch(epoch, capture_stream)) {
            recorder.shutdown();
            cudaEventDestroy(data_event);
            cudaStreamDestroy(capture_stream);
            cudaStreamDestroy(producer_stream);
            cudaFree(d_buffer);
            return false;
        }
        if (!check_cuda(cudaMemcpy(host_data.data(), d_buffer, size_bytes, cudaMemcpyDeviceToHost), "memcpy multistream")) {
            recorder.shutdown();
            cudaEventDestroy(data_event);
            cudaStreamDestroy(capture_stream);
            cudaStreamDestroy(producer_stream);
            cudaFree(d_buffer);
            return false;
        }
        if (!verify_multistream_region(host_data, epoch, seed)) {
            ++mismatches;
        }
    }

    recorder.shutdown();
    cudaEventDestroy(data_event);
    cudaStreamDestroy(capture_stream);
    cudaStreamDestroy(producer_stream);
    cudaFree(d_buffer);

    return mismatches > 0u;
}

bool test_multistream_with_deps_success() {
    const uint32_t element_count = 256;
    const uint32_t size_bytes = element_count * sizeof(uint32_t);
    const uint32_t epoch_count = get_env_u32("TT_TEST_MULTISTREAM_ITERS", 6);

    void* d_buffer = nullptr;
    if (!check_cuda(cudaMalloc(&d_buffer, size_bytes), "cudaMalloc multistream buffer deps")) {
        return false;
    }

    cudaStream_t producer_stream = nullptr;
    cudaStream_t capture_stream = nullptr;
    cudaEvent_t commit_event = nullptr;
    if (!check_cuda(cudaStreamCreate(&producer_stream), "producer stream create deps") ||
        !check_cuda(cudaStreamCreate(&capture_stream), "capture stream create deps") ||
        !check_cuda(cudaEventCreateWithFlags(&commit_event, cudaEventDisableTiming), "commit event create deps")) {
        cudaFree(d_buffer);
        return false;
    }

    tt::Recorder recorder;
    tt::RecorderConfig cfg{};
    cfg.ring_bytes = size_bytes * (epoch_count + 2u) + 4096u;
    cfg.epoch_capacity = epoch_count + 2u;
    cfg.region_capacity = 1;
    cfg.retention_epochs = 0;
    cfg.overwrite_mode = tt::OverwriteMode::kDropOldest;
    if (!recorder.init(cfg)) {
        cudaEventDestroy(commit_event);
        cudaStreamDestroy(capture_stream);
        cudaStreamDestroy(producer_stream);
        cudaFree(d_buffer);
        return false;
    }
    if (!recorder.register_region(0, d_buffer, size_bytes, 1)) {
        recorder.shutdown();
        cudaEventDestroy(commit_event);
        cudaStreamDestroy(capture_stream);
        cudaStreamDestroy(producer_stream);
        cudaFree(d_buffer);
        return false;
    }

    const uint32_t seed = 0x12345678u;
    const uint32_t threads = 128;
    const uint32_t blocks = (element_count + threads - 1u) / threads;
    tt::CaptureDependency dep{};
    dep.region_id = 0;
    dep.producer_stream = producer_stream;
    for (uint32_t epoch = 0; epoch < epoch_count; ++epoch) {
        write_region_data_kernel<<<blocks, threads, 0, producer_stream>>>(
            static_cast<uint32_t*>(d_buffer),
            element_count,
            epoch,
            seed);
        if (!check_cuda(cudaGetLastError(), "launch data kernel deps")) {
            recorder.shutdown();
            cudaEventDestroy(commit_event);
            cudaStreamDestroy(capture_stream);
            cudaStreamDestroy(producer_stream);
            cudaFree(d_buffer);
            return false;
        }
        write_commit_kernel<<<1, 1, 0, producer_stream>>>(
            static_cast<uint32_t*>(d_buffer),
            element_count,
            epoch,
            seed);
        if (!check_cuda(cudaGetLastError(), "launch commit kernel deps")) {
            recorder.shutdown();
            cudaEventDestroy(commit_event);
            cudaStreamDestroy(capture_stream);
            cudaStreamDestroy(producer_stream);
            cudaFree(d_buffer);
            return false;
        }
        if (!check_cuda(cudaEventRecord(commit_event, producer_stream), "record commit event")) {
            recorder.shutdown();
            cudaEventDestroy(commit_event);
            cudaStreamDestroy(capture_stream);
            cudaStreamDestroy(producer_stream);
            cudaFree(d_buffer);
            return false;
        }

        dep.event = commit_event;
        tt::CaptureDeps deps{&dep, 1};
        if (!recorder.capture_epoch(capture_stream, deps)) {
            recorder.shutdown();
            cudaEventDestroy(commit_event);
            cudaStreamDestroy(capture_stream);
            cudaStreamDestroy(producer_stream);
            cudaFree(d_buffer);
            return false;
        }
        if (!check_cuda(cudaStreamSynchronize(producer_stream), "producer sync deps")) {
            recorder.shutdown();
            cudaEventDestroy(commit_event);
            cudaStreamDestroy(capture_stream);
            cudaStreamDestroy(producer_stream);
            cudaFree(d_buffer);
            return false;
        }
    }

    std::vector<uint32_t> host_data(element_count, 0u);
    bool ok = true;
    for (uint32_t epoch = 0; epoch < epoch_count; ++epoch) {
        if (!recorder.rewind_to_epoch(epoch, capture_stream)) {
            ok = false;
            break;
        }
        if (!check_cuda(cudaMemcpy(host_data.data(), d_buffer, size_bytes, cudaMemcpyDeviceToHost), "memcpy multistream deps")) {
            ok = false;
            break;
        }
        if (!verify_multistream_region(host_data, epoch, seed)) {
            ok = false;
            break;
        }
    }

    recorder.shutdown();
    cudaEventDestroy(commit_event);
    cudaStreamDestroy(capture_stream);
    cudaStreamDestroy(producer_stream);
    cudaFree(d_buffer);

    return ok;
}

bool test_multistream_multi_region() {
    const uint32_t element_count = 256;
    const uint32_t size_bytes = element_count * sizeof(uint32_t);
    const uint32_t epoch_count = get_env_u32("TT_TEST_MULTISTREAM_ITERS", 6);

    void* d_buffers[2]{};
    for (int i = 0; i < 2; ++i) {
        if (!check_cuda(cudaMalloc(&d_buffers[i], size_bytes), "cudaMalloc multistream region")) {
            return false;
        }
    }

    cudaStream_t producer_streams[2]{};
    cudaEvent_t commit_events[2]{};
    cudaStream_t capture_stream = nullptr;
    for (int i = 0; i < 2; ++i) {
        if (!check_cuda(cudaStreamCreate(&producer_streams[i]), "producer stream create multi") ||
            !check_cuda(cudaEventCreateWithFlags(&commit_events[i], cudaEventDisableTiming), "commit event create multi")) {
            return false;
        }
    }
    if (!check_cuda(cudaStreamCreate(&capture_stream), "capture stream create multi")) {
        return false;
    }

    tt::Recorder recorder;
    tt::RecorderConfig cfg{};
    cfg.ring_bytes = size_bytes * 2u * (epoch_count + 2u) + 4096u;
    cfg.epoch_capacity = epoch_count + 2u;
    cfg.region_capacity = 2;
    cfg.retention_epochs = 0;
    cfg.overwrite_mode = tt::OverwriteMode::kDropOldest;
    if (!recorder.init(cfg)) {
        return false;
    }
    for (uint32_t i = 0; i < 2; ++i) {
        if (!recorder.register_region(i, d_buffers[i], size_bytes, 1)) {
            recorder.shutdown();
            return false;
        }
    }

    const uint32_t seeds[2] = {0x0BADF00Du, 0x0D15EA5Eu};
    const uint32_t threads = 128;
    const uint32_t blocks = (element_count + threads - 1u) / threads;
    tt::CaptureDependency deps[2]{};
    for (uint32_t epoch = 0; epoch < epoch_count; ++epoch) {
        for (uint32_t r = 0; r < 2; ++r) {
            write_region_data_kernel<<<blocks, threads, 0, producer_streams[r]>>>(
                static_cast<uint32_t*>(d_buffers[r]),
                element_count,
                epoch,
                seeds[r]);
            if (!check_cuda(cudaGetLastError(), "launch data kernel multi")) {
                recorder.shutdown();
                return false;
            }
            write_commit_kernel<<<1, 1, 0, producer_streams[r]>>>(
                static_cast<uint32_t*>(d_buffers[r]),
                element_count,
                epoch,
                seeds[r]);
            if (!check_cuda(cudaGetLastError(), "launch commit kernel multi")) {
                recorder.shutdown();
                return false;
            }
            if (!check_cuda(cudaEventRecord(commit_events[r], producer_streams[r]), "record commit event multi")) {
                recorder.shutdown();
                return false;
            }
            deps[r].region_id = r;
            deps[r].event = commit_events[r];
            deps[r].producer_stream = producer_streams[r];
        }

        tt::CaptureDeps dep_list{deps, 2};
        if (!recorder.capture_epoch(capture_stream, dep_list)) {
            recorder.shutdown();
            return false;
        }
        for (uint32_t r = 0; r < 2; ++r) {
            if (!check_cuda(cudaStreamSynchronize(producer_streams[r]), "producer sync multi")) {
                recorder.shutdown();
                return false;
            }
        }
    }

    bool ok = true;
    std::vector<uint32_t> host_data(element_count, 0u);
    const uint32_t check_epoch = epoch_count > 0 ? (epoch_count - 1u) : 0u;
    if (!recorder.rewind_to_epoch(check_epoch, capture_stream)) {
        ok = false;
    } else {
        for (uint32_t r = 0; r < 2; ++r) {
            if (!check_cuda(cudaMemcpy(host_data.data(), d_buffers[r], size_bytes, cudaMemcpyDeviceToHost), "memcpy multistream multi")) {
                ok = false;
                break;
            }
            if (!verify_multistream_region(host_data, check_epoch, seeds[r])) {
                ok = false;
                break;
            }
        }
    }

    recorder.shutdown();
    for (uint32_t r = 0; r < 2; ++r) {
        cudaEventDestroy(commit_events[r]);
        cudaStreamDestroy(producer_streams[r]);
        cudaFree(d_buffers[r]);
    }
    cudaStreamDestroy(capture_stream);

    return ok;
}

} // namespace

int main() {
    tt::GetCuptiKernelTracer();
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

    bool ok_delta = test_delta_single_region();
    if (!ok_delta) {
        std::printf("test_delta_single_region failed\n");
        return 1;
    }

    bool ok_wrap = test_wrap_marker();
    if (!ok_wrap) {
        std::printf("test_wrap_marker failed\n");
        return 1;
    }

    bool ok_tiny_wrap = test_tiny_ring_wrap();
    if (!ok_tiny_wrap) {
        std::printf("test_tiny_ring_wrap failed\n");
        return 1;
    }

    bool ok_retention = test_overwrite_retention();
    if (!ok_retention) {
        std::printf("test_overwrite_retention failed\n");
        return 1;
    }

    bool ok_backpressure = test_backpressure();
    if (!ok_backpressure) {
        std::printf("test_backpressure failed\n");
        return 1;
    }

    bool ok_graph = test_graph_capture_and_trace();
    if (!ok_graph) {
        std::printf("test_graph_capture_and_trace failed\n");
        return 1;
    }

    bool ok_graph_params = test_graph_iter_params_update();
    if (!ok_graph_params) {
        std::printf("test_graph_iter_params_update failed\n");
        return 1;
    }

    bool ok_graph_patch = test_graph_kernel_param_patch();
    if (!ok_graph_patch) {
        std::printf("test_graph_kernel_param_patch failed\n");
        return 1;
    }

    bool ok_graph_controls = test_graph_control_updates();
    if (!ok_graph_controls) {
        std::printf("test_graph_control_updates failed\n");
        return 1;
    }

    bool ok_manifest = test_deterministic_manifest_match();
    if (!ok_manifest) {
        std::printf("test_deterministic_manifest_match failed\n");
        return 1;
    }

    bool ok_det_rewind = test_deterministic_rewind();
    if (!ok_det_rewind) {
        std::printf("test_deterministic_rewind failed\n");
        return 1;
    }

    bool ok_multistream_fail = test_multistream_no_deps_failure();
    if (!ok_multistream_fail) {
        std::printf("test_multistream_no_deps_failure failed\n");
        return 1;
    }

    bool ok_multistream_deps = test_multistream_with_deps_success();
    if (!ok_multistream_deps) {
        std::printf("test_multistream_with_deps_success failed\n");
        return 1;
    }

    bool ok_multistream_multi = test_multistream_multi_region();
    if (!ok_multistream_multi) {
        std::printf("test_multistream_multi_region failed\n");
        return 1;
    }

    std::printf("tt_tests passed\n");
    return 0;
}
