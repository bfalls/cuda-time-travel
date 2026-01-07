#include "tt/tt_graph.h"
#include "tt/tt_cupti.h"
#include "tt/tt_trace.h"
#include "tt/ttrecorder.h"

#include <cstdio>
#include <chrono>
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

bool verify_pattern(const std::vector<uint32_t>& data, uint32_t epoch) {
    for (size_t i = 0; i < data.size(); ++i) {
        uint32_t expected = epoch ^ (static_cast<uint32_t>(i) * 2654435761u);
        if (data[i] != expected) {
            return false;
        }
    }
    return true;
}

} // namespace

int main() {
    tt::GetCuptiKernelTracer();
    const uint32_t element_count = 1024;
    const uint32_t size_bytes = element_count * sizeof(uint32_t);
    const uint32_t epoch_count = 8;
    const uint32_t target_epoch = 3;
    const uint32_t stamps_per_epoch = 3;

    uint32_t* d_buffer = nullptr;
    uint32_t* d_epoch_counter = nullptr;
    uint64_t* d_stamps = nullptr;
    uint32_t* d_stamp_counter = nullptr;
    if (!check_cuda(cudaMalloc(&d_buffer, size_bytes), "cudaMalloc buffer")) {
        return 1;
    }
    if (!check_cuda(cudaMalloc(&d_epoch_counter, sizeof(uint32_t)), "cudaMalloc epoch counter")) {
        cudaFree(d_buffer);
        return 1;
    }
    if (!check_cuda(cudaMalloc(&d_stamps, sizeof(uint64_t) * epoch_count * stamps_per_epoch), "cudaMalloc stamps")) {
        cudaFree(d_epoch_counter);
        cudaFree(d_buffer);
        return 1;
    }
    if (!check_cuda(cudaMalloc(&d_stamp_counter, sizeof(uint32_t)), "cudaMalloc stamp counter")) {
        cudaFree(d_stamps);
        cudaFree(d_epoch_counter);
        cudaFree(d_buffer);
        return 1;
    }
    if (!check_cuda(cudaMemset(d_stamp_counter, 0, sizeof(uint32_t)), "memset stamp counter")) {
        cudaFree(d_stamp_counter);
        cudaFree(d_stamps);
        cudaFree(d_epoch_counter);
        cudaFree(d_buffer);
        return 1;
    }

    cudaStream_t stream = nullptr;
    if (!check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate")) {
        cudaFree(d_stamp_counter);
        cudaFree(d_stamps);
        cudaFree(d_epoch_counter);
        cudaFree(d_buffer);
        return 1;
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
    cfg.enable_graph_stamps = true;
    cfg.graph_stamps = d_stamps;
    cfg.graph_stamp_counter = d_stamp_counter;
    if (!recorder.init(cfg)) {
        std::printf("recorder init failed\n");
        cudaStreamDestroy(stream);
        cudaFree(d_stamp_counter);
        cudaFree(d_stamps);
        cudaFree(d_epoch_counter);
        cudaFree(d_buffer);
        return 1;
    }

    if (!recorder.register_region(0, d_buffer, size_bytes, 1)) {
        std::printf("register_region failed\n");
        recorder.shutdown();
        cudaStreamDestroy(stream);
        cudaFree(d_stamp_counter);
        cudaFree(d_stamps);
        cudaFree(d_epoch_counter);
        cudaFree(d_buffer);
        return 1;
    }

    tt::GraphSession graph;
    if (!graph.begin_capture(stream)) {
        std::printf("begin_capture failed\n");
        recorder.shutdown();
        cudaStreamDestroy(stream);
        cudaFree(d_epoch_counter);
        cudaFree(d_buffer);
        return 1;
    }

    const uint32_t threads = 256;
    const uint32_t blocks = (element_count + threads - 1u) / threads;
    write_pattern_kernel<<<blocks, threads, 0, stream>>>(d_buffer, element_count, d_epoch_counter);
    if (!recorder.capture_epoch(stream)) {
        std::printf("capture_epoch failed during graph capture\n");
        graph.destroy();
        recorder.shutdown();
        cudaStreamDestroy(stream);
        cudaFree(d_stamp_counter);
        cudaFree(d_stamps);
        cudaFree(d_epoch_counter);
        cudaFree(d_buffer);
        return 1;
    }

    if (!graph.end_capture()) {
        std::printf("end_capture failed\n");
        graph.destroy();
        recorder.shutdown();
        cudaStreamDestroy(stream);
        cudaFree(d_stamp_counter);
        cudaFree(d_stamps);
        cudaFree(d_epoch_counter);
        cudaFree(d_buffer);
        return 1;
    }

    cudaEvent_t event_start = nullptr;
    cudaEvent_t event_end = nullptr;
    if (!check_cuda(cudaEventCreate(&event_start), "event create start") ||
        !check_cuda(cudaEventCreate(&event_end), "event create end")) {
        graph.destroy();
        recorder.shutdown();
        cudaStreamDestroy(stream);
        cudaFree(d_stamp_counter);
        cudaFree(d_stamps);
        cudaFree(d_epoch_counter);
        cudaFree(d_buffer);
        return 1;
    }

    tt::TraceCollector trace;
    const auto trace_start = std::chrono::steady_clock::now();

    for (uint32_t epoch = 0; epoch < epoch_count; ++epoch) {
        set_epoch_kernel<<<1, 1, 0, stream>>>(d_epoch_counter, epoch);
        auto cpu_start = std::chrono::steady_clock::now();
        if (!check_cuda(cudaEventRecord(event_start, stream), "event record start")) {
            cudaEventDestroy(event_end);
            cudaEventDestroy(event_start);
            graph.destroy();
            recorder.shutdown();
            cudaStreamDestroy(stream);
            cudaFree(d_stamp_counter);
            cudaFree(d_stamps);
            cudaFree(d_epoch_counter);
            cudaFree(d_buffer);
            return 1;
        }
        if (!graph.launch(stream)) {
            std::printf("graph launch failed at epoch %u\n", epoch);
            cudaEventDestroy(event_end);
            cudaEventDestroy(event_start);
            graph.destroy();
            recorder.shutdown();
            cudaStreamDestroy(stream);
            cudaFree(d_stamp_counter);
            cudaFree(d_stamps);
            cudaFree(d_epoch_counter);
            cudaFree(d_buffer);
            return 1;
        }
        if (!check_cuda(cudaEventRecord(event_end, stream), "event record end")) {
            cudaEventDestroy(event_end);
            cudaEventDestroy(event_start);
            graph.destroy();
            recorder.shutdown();
            cudaStreamDestroy(stream);
            cudaFree(d_stamp_counter);
            cudaFree(d_stamps);
            cudaFree(d_epoch_counter);
            cudaFree(d_buffer);
            return 1;
        }
        if (!check_cuda(cudaEventSynchronize(event_end), "event sync end")) {
            cudaEventDestroy(event_end);
            cudaEventDestroy(event_start);
            graph.destroy();
            recorder.shutdown();
            cudaStreamDestroy(stream);
            cudaFree(d_stamp_counter);
            cudaFree(d_stamps);
            cudaFree(d_epoch_counter);
            cudaFree(d_buffer);
            return 1;
        }
        float elapsed_ms = 0.0f;
        if (!check_cuda(cudaEventElapsedTime(&elapsed_ms, event_start, event_end), "event elapsed")) {
            cudaEventDestroy(event_end);
            cudaEventDestroy(event_start);
            graph.destroy();
            recorder.shutdown();
            cudaStreamDestroy(stream);
            cudaFree(d_stamp_counter);
            cudaFree(d_stamps);
            cudaFree(d_epoch_counter);
            cudaFree(d_buffer);
            return 1;
        }
        const auto cpu_delta = std::chrono::duration<double, std::micro>(cpu_start - trace_start).count();
        tt::TraceEvent graph_event{};
        graph_event.name = "graph_launch";
        graph_event.cat = "graph";
        graph_event.ts_us = cpu_delta;
        graph_event.dur_us = static_cast<double>(elapsed_ms) * 1000.0;
        graph_event.pid = 1;
        graph_event.tid = 0;
        graph_event.args.push_back({"epoch_id", std::to_string(epoch), false});
        trace.add_event(graph_event);
    }

    if (!check_cuda(cudaStreamSynchronize(stream), "graph sync")) {
        cudaEventDestroy(event_end);
        cudaEventDestroy(event_start);
        graph.destroy();
        recorder.shutdown();
        cudaStreamDestroy(stream);
        cudaFree(d_stamp_counter);
        cudaFree(d_stamps);
        cudaFree(d_epoch_counter);
        cudaFree(d_buffer);
        return 1;
    }

    cudaEventDestroy(event_end);
    cudaEventDestroy(event_start);

    uint32_t stamp_count = 0;
    if (!check_cuda(cudaMemcpy(&stamp_count, d_stamp_counter, sizeof(uint32_t), cudaMemcpyDeviceToHost), "memcpy stamp count")) {
        graph.destroy();
        recorder.shutdown();
        cudaStreamDestroy(stream);
        cudaFree(d_stamp_counter);
        cudaFree(d_stamps);
        cudaFree(d_epoch_counter);
        cudaFree(d_buffer);
        return 1;
    }

    std::vector<uint64_t> host_stamps(stamp_count);
    if (stamp_count > 0) {
        if (!check_cuda(cudaMemcpy(host_stamps.data(), d_stamps, sizeof(uint64_t) * stamp_count, cudaMemcpyDeviceToHost), "memcpy stamps")) {
            graph.destroy();
            recorder.shutdown();
            cudaStreamDestroy(stream);
            cudaFree(d_stamp_counter);
            cudaFree(d_stamps);
            cudaFree(d_epoch_counter);
            cudaFree(d_buffer);
            return 1;
        }
    }

    int clock_rate_khz = 0;
    if (!check_cuda(cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate, 0), "device clock rate")) {
        graph.destroy();
        recorder.shutdown();
        cudaStreamDestroy(stream);
        cudaFree(d_stamp_counter);
        cudaFree(d_stamps);
        cudaFree(d_epoch_counter);
        cudaFree(d_buffer);
        return 1;
    }
    const double cycles_to_us = 1000.0 / static_cast<double>(clock_rate_khz);

    std::vector<tt::EpochRecord> epochs;
    recorder.read_epochs_to_host(epochs);
    const uint64_t base_stamp = host_stamps.empty() ? 0u : host_stamps[0];
    const uint32_t epoch_samples = stamp_count / stamps_per_epoch;
    for (uint32_t i = 0; i < epoch_samples; ++i) {
        const uint64_t start_stamp = host_stamps[i * stamps_per_epoch + 0];
        const uint64_t mid_stamp = host_stamps[i * stamps_per_epoch + 1];
        const uint64_t end_stamp = host_stamps[i * stamps_per_epoch + 2];
        const uint32_t epoch_id = i;
        uint32_t ring_bytes_written = 0;
        for (const auto& record : epochs) {
            if (record.epoch_id == epoch_id) {
                ring_bytes_written = record.reserved0;
                break;
            }
        }
        const double start_us = static_cast<double>(start_stamp - base_stamp) * cycles_to_us;
        const double mid_us = static_cast<double>(mid_stamp - base_stamp) * cycles_to_us;
        const double end_us = static_cast<double>(end_stamp - base_stamp) * cycles_to_us;

        tt::TraceEvent epoch_total{};
        epoch_total.name = "epoch_total";
        epoch_total.cat = "epoch";
        epoch_total.ts_us = start_us;
        epoch_total.dur_us = end_us - start_us;
        epoch_total.pid = 1;
        epoch_total.tid = 1;
        epoch_total.args.push_back({"epoch_id", std::to_string(epoch_id), false});
        epoch_total.args.push_back({"ring_bytes_written", std::to_string(ring_bytes_written), false});
        trace.add_event(epoch_total);

        tt::TraceEvent epoch_regions{};
        epoch_regions.name = "epoch_regions";
        epoch_regions.cat = "epoch";
        epoch_regions.ts_us = start_us;
        epoch_regions.dur_us = mid_us - start_us;
        epoch_regions.pid = 1;
        epoch_regions.tid = 1;
        epoch_regions.args.push_back({"epoch_id", std::to_string(epoch_id), false});
        trace.add_event(epoch_regions);
    }

    tt::GetCuptiKernelTracer().append_kernel_events(trace);
    trace.write("trace/tt_trace.json");

    if (!recorder.rewind_to_epoch(target_epoch, stream)) {
        std::printf("rewind_to_epoch failed\n");
        graph.destroy();
        recorder.shutdown();
        cudaStreamDestroy(stream);
        cudaFree(d_stamp_counter);
        cudaFree(d_stamps);
        cudaFree(d_epoch_counter);
        cudaFree(d_buffer);
        return 1;
    }

    std::vector<uint32_t> host_out(element_count);
    if (!check_cuda(cudaMemcpy(host_out.data(), d_buffer, size_bytes, cudaMemcpyDeviceToHost), "memcpy out")) {
        graph.destroy();
        recorder.shutdown();
        cudaStreamDestroy(stream);
        cudaFree(d_stamp_counter);
        cudaFree(d_stamps);
        cudaFree(d_epoch_counter);
        cudaFree(d_buffer);
        return 1;
    }

    bool ok = verify_pattern(host_out, target_epoch);
    graph.destroy();
    recorder.shutdown();
    cudaStreamDestroy(stream);
    cudaFree(d_stamp_counter);
    cudaFree(d_stamps);
    cudaFree(d_epoch_counter);
    cudaFree(d_buffer);

    if (!ok) {
        std::printf("tt_demo_graph failed verification\n");
        return 1;
    }

    std::printf("tt_demo_graph success\n");
    return 0;
}
