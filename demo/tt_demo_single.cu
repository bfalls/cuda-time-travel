#include "tt/tt_trace.h"
#include "tt/ttrecorder.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <chrono>
#include <cstring>

namespace {

bool check_cuda(cudaError_t err, const char* label) {
    if (err == cudaSuccess) {
        return true;
    }
    std::printf("CUDA error at %s: %s\n", label, cudaGetErrorString(err));
    return false;
}

__global__ void write_pattern_kernel(uint32_t* data, uint32_t count, uint32_t epoch) {
    const uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < count) {
        data[idx] = epoch ^ (idx * 2654435761u);
    }
}

bool verify_pattern(const std::vector<uint32_t>& data, uint32_t epoch) {
    for (uint32_t i = 0; i < data.size(); ++i) {
        const uint32_t expected = epoch ^ (i * 2654435761u);
        if (data[i] != expected) {
            return false;
        }
    }
    return true;
}

void add_event(tt::TraceCollector& trace,
    const char* name,
    const char* cat,
    double ts_us,
    double dur_us,
    uint32_t epoch) {
    tt::TraceEvent event{};
    event.name = name;
    event.cat = cat;
    event.ts_us = ts_us;
    event.dur_us = dur_us;
    event.pid = 1;
    event.tid = 0;
    event.args.push_back({"epoch_id", std::to_string(epoch), false});
    trace.add_event(event);
}

bool run_epoch(uint32_t epoch,
    tt::TraceCollector& trace,
    tt::Recorder& recorder,
    cudaStream_t stream,
    uint32_t* d_buffer,
    uint32_t element_count,
    std::chrono::steady_clock::time_point trace_start) {
    const uint32_t threads = 256;
    const uint32_t blocks = (element_count + threads - 1u) / threads;
    const auto now_us = [&]() {
        return std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - trace_start).count();
    };

    write_pattern_kernel<<<blocks, threads, 0, stream>>>(d_buffer, element_count, epoch);
    if (!check_cuda(cudaGetLastError(), "launch write_pattern_kernel")) {
        return false;
    }
    add_event(trace, "write_kernel", "epoch", now_us(), 0.0, epoch);

    if (!recorder.capture_epoch(stream)) {
        std::printf("capture_epoch failed at %u\n", epoch);
        return false;
    }
    add_event(trace, "capture_epoch", "epoch", now_us(), 0.0, epoch);
    return true;
}

} // namespace

int main() {
    const bool bug_mode = (__argc > 1 && std::strcmp(__argv[1], "--bug") == 0);
    const uint32_t element_count = 512;
    const uint32_t size_bytes = element_count * sizeof(uint32_t);
    const uint32_t epoch_count = 6;

    uint32_t* d_buffer = nullptr;
    if (!check_cuda(cudaMalloc(&d_buffer, size_bytes), "cudaMalloc buffer")) {
        return 1;
    }

    cudaStream_t stream = nullptr;
    if (!check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate")) {
        cudaFree(d_buffer);
        return 1;
    }

    const uint32_t per_chunk_bytes = (static_cast<uint32_t>(sizeof(tt::ChunkHeader)) + size_bytes + 31u) & ~31u;
    const uint32_t ring_bytes = per_chunk_bytes * epoch_count + 4096u;

    tt::Recorder recorder;
    tt::RecorderConfig cfg{};
    cfg.ring_bytes = ring_bytes;
    cfg.epoch_capacity = epoch_count + 2u;
    cfg.region_capacity = 1u;
    cfg.retention_epochs = 0u;
    cfg.overwrite_mode = tt::OverwriteMode::kDropOldest;
    if (!recorder.init(cfg)) {
        std::printf("recorder init failed\n");
        cudaStreamDestroy(stream);
        cudaFree(d_buffer);
        return 1;
    }
    if (!recorder.register_region(0, d_buffer, size_bytes, 1)) {
        std::printf("register_region failed\n");
        recorder.shutdown();
        cudaStreamDestroy(stream);
        cudaFree(d_buffer);
        return 1;
    }

    tt::TraceCollector trace;
    const auto trace_start = std::chrono::steady_clock::now();

    if (bug_mode) {
        // BUG: off-by-one loop boundary.
        for (uint32_t epoch = 0; epoch <= epoch_count; ++epoch) {
            if (!run_epoch(epoch, trace, recorder, stream, d_buffer, element_count, trace_start)) {
                recorder.shutdown();
                cudaStreamDestroy(stream);
                cudaFree(d_buffer);
                return 1;
            }
        }
    } else {
        for (uint32_t epoch = 0; epoch < epoch_count; ++epoch) {
            if (!run_epoch(epoch, trace, recorder, stream, d_buffer, element_count, trace_start)) {
                recorder.shutdown();
                cudaStreamDestroy(stream);
                cudaFree(d_buffer);
                return 1;
            }
        }
    }

    if (!check_cuda(cudaStreamSynchronize(stream), "stream sync")) {
        recorder.shutdown();
        cudaStreamDestroy(stream);
        cudaFree(d_buffer);
        return 1;
    }

    trace.write(bug_mode ? "trace/tt_demo_single-bug.json" : "trace/tt_demo_single.json");

    // 1) Notice the extra epoch_id==epoch_count in the trace timeline.
    // 2) Rewind to the first epoch; it fails because the ring dropped it.
    // 3) The extra loop iteration above is the off-by-one that caused the drop.
    std::vector<uint32_t> host_out(element_count);
    bool ok = true;
    for (uint32_t epoch = 0; epoch < epoch_count; ++epoch) {
        if (!recorder.rewind_to_epoch(epoch, stream)) {
            std::printf("rewind_to_epoch failed at %u\n", epoch);
            ok = false;
            break;
        }
        if (!check_cuda(cudaMemcpy(host_out.data(), d_buffer, size_bytes, cudaMemcpyDeviceToHost), "memcpy out")) {
            ok = false;
            break;
        }
        if (!verify_pattern(host_out, epoch)) {
            std::printf("verify failed at epoch %u\n", epoch);
            ok = false;
            break;
        }
    }

    recorder.shutdown();
    cudaStreamDestroy(stream);
    cudaFree(d_buffer);

    if (!ok) {
        std::printf("tt_demo_single failed verification%s\n", bug_mode ? " (bug mode)" : "");
        return 1;
    }

    std::printf("tt_demo_single success\n");
    return 0;
}
