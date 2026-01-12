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

    bool has_flag(const char* flag) {
        for (int i = 1; i < __argc; ++i) {
            if (std::strcmp(__argv[i], flag) == 0) return true;
        }
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
        uint32_t epoch,
        uint32_t tid) {
        tt::TraceEvent event{};
        event.name = name;
        event.cat = cat;
        event.ts_us = ts_us;
        event.dur_us = dur_us;
        event.pid = 1;
        event.tid = tid;
        event.args.push_back({ "epoch_id", std::to_string(epoch), false });
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

        const double t0 = now_us();

        write_pattern_kernel << <blocks, threads, 0, stream >> > (d_buffer, element_count, epoch);
        if (!check_cuda(cudaGetLastError(), "launch write_pattern_kernel")) {
            return false;
        }
        add_event(trace, "write_kernel", "epoch", now_us(), 0.0, epoch, 0);

        if (!recorder.capture_epoch(stream)) {
            std::printf("capture_epoch failed at %u\n", epoch);
            return false;
        }
        add_event(trace, "capture_epoch", "epoch", now_us(), 0.0, epoch, 1);

        add_event(trace, "epoch", "epoch", t0, (now_us() - t0), epoch, 2);
        return true;
    }

    // Demo 2: small, realistic "why is this so slow?"
    // A debug-only sync accidentally runs every iteration due to '&' vs '&&'.
    bool run_sync_demo(bool sync_bug,
        tt::TraceCollector& trace,
        cudaStream_t stream,
        std::chrono::steady_clock::time_point trace_start) {

        const auto now_us = [&]() {
            return std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - trace_start).count();
            };

        // Make kernels take a noticeable amount of time so an accidental sync is obvious.
        // (Realistic: "work is heavy enough that a sync fence hurts.")
        const uint32_t big_elements = 4u * 1024u * 1024u; // 4M uint32_t = 16MB
        const uint32_t big_bytes = big_elements * sizeof(uint32_t);

        uint32_t* d_big = nullptr;
        if (!check_cuda(cudaMalloc(&d_big, big_bytes), "cudaMalloc d_big")) {
            return false;
        }

        const uint32_t threads = 256;
        const uint32_t blocks = (big_elements + threads - 1u) / threads;

        // Keep debug_mode OFF in both runs. The only difference is '&' vs '&&'.
        const bool debug_mode = false;

        // Keep the demo short but visible.
        const uint32_t iters = 12;

        for (uint32_t i = 0; i < iters; ++i) {
            const double t_epoch0 = now_us();

            write_pattern_kernel << <blocks, threads, 0, stream >> > (d_big, big_elements, i);
            if (!check_cuda(cudaGetLastError(), "launch write_pattern_kernel (sync demo)")) {
                cudaFree(d_big);
                return false;
            }
            add_event(trace, "launch_kernel", "sync_demo", now_us(), 0.0, i, 0);

            // Measure how much time we spend in this "debug fence" decision.
            // Good: fence never runs (debug_mode=false) => tiny duration.
            // Bug: '&' forces evaluation of RHS => cudaStreamSynchronize runs every iter => big duration.
            const double t_fence0 = now_us();

            if (sync_bug) {
                // BUG: '&' does NOT short-circuit; RHS always executes.
                if (debug_mode & (cudaSuccess == cudaStreamSynchronize(stream))) {
                    // no-op
                }
            }
            else {
                // Correct: '&&' short-circuits; RHS never executes when debug_mode=false.
                if (debug_mode && (cudaSuccess == cudaStreamSynchronize(stream))) {
                    // no-op
                }
            }

            const double t_fence1 = now_us();
            add_event(trace, "debug_sync_fence", "sync_demo", t_fence0, (t_fence1 - t_fence0), i, 1);

            const double t_epoch1 = now_us();
            add_event(trace, "iter", "sync_demo", t_epoch0, (t_epoch1 - t_epoch0), i, 2);
        }

        if (!check_cuda(cudaStreamSynchronize(stream), "final sync (sync demo)")) {
            cudaFree(d_big);
            return false;
        }

        cudaFree(d_big);
        return true;
    }

} // namespace

int main() {
    const bool bug_mode = has_flag("--bug");           // off-by-one demo bug
    const bool sync_bug = has_flag("--sync-bug");      // sync fence demo bug

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

    const uint32_t per_chunk_bytes =
        (static_cast<uint32_t>(sizeof(tt::ChunkHeader)) + size_bytes + 31u) & ~31u;
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

    // Demo 1: off-by-one epoch bug (correctness + trace count)
    tt::TraceCollector trace;
    const auto trace_start = std::chrono::steady_clock::now();

    int32_t remaining_work = static_cast<int32_t>(epoch_count);

    auto run_and_account = [&](uint32_t epoch) -> bool {
        if (!run_epoch(epoch, trace, recorder, stream, d_buffer, element_count, trace_start)) {
            recorder.shutdown();
            cudaStreamDestroy(stream);
            cudaFree(d_buffer);
            return false;
        }
        --remaining_work;
        return true;
        };

    if (bug_mode) {
        // BUG: off-by-one loop boundary.
        for (uint32_t epoch = 0; epoch <= epoch_count; ++epoch) {
            if (!run_and_account(epoch))
                return 1;
        }
    }
    else {
        for (uint32_t epoch = 0; epoch < epoch_count; ++epoch) {
            if (!run_and_account(epoch))
                return 1;
        }
    }

    // clean mode => 0, bug mode => -1
    std::printf("remaining_work=%d (expected 0)\n", remaining_work);

    if (!check_cuda(cudaStreamSynchronize(stream), "stream sync")) {
        recorder.shutdown();
        cudaStreamDestroy(stream);
        cudaFree(d_buffer);
        return 1;
    }

    trace.write(bug_mode ? "trace/tt_demo_single-bug.json" : "trace/tt_demo_single.json");

    std::vector<uint32_t> host_out(element_count);
    bool ok = true;
    for (uint32_t epoch = 0; epoch < epoch_count; ++epoch) {
        if (!recorder.rewind_to_epoch(epoch, stream)) {
            std::printf("rewind_to_epoch failed at %u\n", epoch);
            ok = false;
            break;
        }
        if (!check_cuda(cudaMemcpy(host_out.data(), d_buffer, size_bytes,
            cudaMemcpyDeviceToHost), "memcpy out")) {
            ok = false;
            break;
        }
        if (!verify_pattern(host_out, epoch)) {
            std::printf("verify failed at epoch %u\n", epoch);
            ok = false;
            break;
        }
    }

    // Demo 2: mysterious slowdown from '&' vs '&&' causing unconditional sync
    {
        tt::TraceCollector trace2;
        const auto trace2_start = std::chrono::steady_clock::now();

        if (!run_sync_demo(sync_bug, trace2, stream, trace2_start)) {
            std::printf("tt_demo_sync failed%s\n", sync_bug ? " (sync bug)" : "");
            recorder.shutdown();
            cudaStreamDestroy(stream);
            cudaFree(d_buffer);
            return 1;
        }

        trace2.write(sync_bug ? "trace/tt_demo_sync-bug.json"
            : "trace/tt_demo_sync.json");

        std::printf("tt_demo_sync wrote trace (%s)\n",
            sync_bug ? "sync bug" : "clean");
    }

    recorder.shutdown();
    cudaStreamDestroy(stream);
    cudaFree(d_buffer);

    if (!ok) {
        std::printf("tt_demo_single failed verification%s\n",
            bug_mode ? " (bug mode)" : "");
        return 1;
    }

    std::printf("tt_demo_single success\n");
    return 0;
}
