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

std::string read_file_bytes(const char* path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        return {};
    }
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
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

} // namespace

int main() {
    const uint32_t element_count = 1024;
    const uint32_t size_bytes = element_count * sizeof(uint32_t);
    const uint32_t epoch_count = 6;

    uint32_t* d_buffer = nullptr;
    uint32_t* d_epoch_counter = nullptr;
    if (!check_cuda(cudaMalloc(&d_buffer, size_bytes), "cudaMalloc buffer")) {
        return 1;
    }
    if (!check_cuda(cudaMalloc(&d_epoch_counter, sizeof(uint32_t)), "cudaMalloc counter")) {
        cudaFree(d_buffer);
        return 1;
    }

    const uint32_t per_chunk_bytes = (static_cast<uint32_t>(sizeof(tt::ChunkHeader)) + size_bytes + 31u) & ~31u;
    const uint32_t ring_bytes = per_chunk_bytes * epoch_count + 4096u;
    const uint32_t threads = 256;
    const uint32_t blocks = (element_count + threads - 1u) / threads;

    auto run_capture = [&](const char* manifest_path, bool write_trace) -> bool {
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
            return false;
        }
        if (!recorder.register_region(0, d_buffer, size_bytes, 1)) {
            recorder.shutdown();
            return false;
        }

        tt::TraceCollector trace;
        for (uint32_t epoch = 0; epoch < epoch_count; ++epoch) {
            set_epoch_kernel<<<1, 1>>>(d_epoch_counter, epoch);
            write_pattern_kernel<<<blocks, threads>>>(d_buffer, element_count, d_epoch_counter);
            if (!recorder.capture_epoch(0)) {
                recorder.shutdown();
                return false;
            }
            if (write_trace) {
                tt::TraceEvent event{};
                event.name = "epoch";
                event.cat = "deterministic";
                event.ts_us = static_cast<double>(epoch) * 1000.0;
                event.dur_us = 500.0;
                event.pid = 1;
                event.tid = 0;
                event.args.push_back({"epoch_id", std::to_string(epoch), false});
                trace.add_event(event);
            }
        }

        if (write_trace) {
            trace.write("trace/tt_trace.json");
        }

        bool wrote_manifest = recorder.write_manifest_json(manifest_path);
        recorder.shutdown();
        return wrote_manifest;
    };

    const char* manifest_a = "trace/tt_manifest_a.json";
    const char* manifest_b = "trace/tt_manifest_b.json";
    if (!run_capture(manifest_a, true) || !run_capture(manifest_b, false)) {
        cudaFree(d_epoch_counter);
        cudaFree(d_buffer);
        std::printf("tt_demo_determinism failed\n");
        return 1;
    }

    const std::string a = read_file_bytes(manifest_a);
    const std::string b = read_file_bytes(manifest_b);
    const bool match = (!a.empty() && a == b);
    std::printf("tt_demo_determinism manifests match: %s\n", match ? "yes" : "no");

    cudaFree(d_epoch_counter);
    cudaFree(d_buffer);
    return match ? 0 : 1;
}
