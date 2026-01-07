#include "tt/tt_graph.h"
#include "tt/tt_cupti.h"
#include "tt/tt_trace.h"
#include "tt/ttrecorder.h"

#include <cstdio>
#include <cstdint>
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

void mutate_words(std::vector<uint32_t>& data, uint32_t epoch) {
    if (data.empty()) {
        return;
    }
    for (uint32_t j = 0; j < 4; ++j) {
        uint32_t index = (epoch * 7u + j * 13u) % static_cast<uint32_t>(data.size());
        data[index] ^= 0xA5A50000u + epoch * 17u + j;
    }
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

    std::printf("tt_tests passed\n");
    return 0;
}
