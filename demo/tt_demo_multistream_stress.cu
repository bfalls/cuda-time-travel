#include "tt/tt_trace.h"
#include "tt/ttrecorder.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sstream>
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

__host__ __device__ uint32_t pattern_value(uint32_t checkpoint, uint32_t seed, uint32_t idx) {
    return checkpoint ^ seed ^ (idx * 2654435761u);
}

__host__ __device__ uint32_t commit_value(uint32_t checkpoint, uint32_t seed) {
    return checkpoint ^ seed ^ 0xA5A5A5A5u;
}

__global__ void write_region_phased_kernel(uint32_t* data,
    uint32_t count,
    uint32_t checkpoint,
    uint32_t seed,
    uint32_t spin_cycles) {
    const uint32_t idx = threadIdx.x;
    if (idx >= count) {
        return;
    }
    const uint32_t commit_index = count - 1u;
    const uint32_t half = commit_index / 2u;

    if (idx < commit_index && idx < half) {
        data[idx] = pattern_value(checkpoint, seed, idx);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        const uint64_t start = clock64();
        while ((clock64() - start) < static_cast<uint64_t>(spin_cycles)) {
        }
    }
    __syncthreads();

    if (idx < commit_index && idx >= half) {
        data[idx] = pattern_value(checkpoint, seed, idx);
    }
    __syncthreads();

    if (idx == commit_index) {
        data[idx] = commit_value(checkpoint, seed);
    }
}

bool verify_region(const std::vector<uint32_t>& data, uint32_t checkpoint, uint32_t seed) {
    if (data.size() < 2) {
        return false;
    }
    const uint32_t commit_index = static_cast<uint32_t>(data.size() - 1);
    for (uint32_t i = 0; i < commit_index; ++i) {
        const uint32_t expected = pattern_value(checkpoint, seed, i);
        if (data[i] != expected) {
            return false;
        }
    }
    return data[commit_index] == commit_value(checkpoint, seed);
}

std::string hex_ptr(uint64_t value) {
    std::ostringstream out;
    out << "0x" << std::hex << value;
    return out.str();
}

uint32_t parse_u32(const char* arg, const char* prefix, uint32_t fallback) {
    const size_t prefix_len = std::strlen(prefix);
    if (std::strncmp(arg, prefix, prefix_len) != 0) {
        return fallback;
    }
    return static_cast<uint32_t>(std::strtoul(arg + prefix_len, nullptr, 10));
}

} // namespace

int main() {
    bool use_deps = true;
    uint32_t checkpoint_count = 64;
    uint32_t verify_count = 8;
    uint32_t spin_cycles = 200000;
    std::string trace_path = "trace/tt_multistream_deps.json";

    for (int i = 1; i < __argc; ++i) {
        if (std::strcmp(__argv[i], "--no-deps") == 0) {
            use_deps = false;
            trace_path = "trace/tt_multistream_no_deps.json";
        } else if (std::strcmp(__argv[i], "--deps") == 0) {
            use_deps = true;
            trace_path = "trace/tt_multistream_deps.json";
        } else if (std::strncmp(__argv[i], "--checkpoints=", 9) == 0) {
            checkpoint_count = parse_u32(__argv[i], "--checkpoints=", checkpoint_count);
        } else if (std::strncmp(__argv[i], "--verify=", 9) == 0) {
            verify_count = parse_u32(__argv[i], "--verify=", verify_count);
        } else if (std::strncmp(__argv[i], "--spin=", 7) == 0) {
            spin_cycles = parse_u32(__argv[i], "--spin=", spin_cycles);
        } else if (std::strncmp(__argv[i], "--trace-out=", 12) == 0) {
            trace_path = __argv[i] + 12;
        }
    }

    const uint32_t region_count = 3;
    const uint32_t element_count = 256;
    const uint32_t size_bytes = element_count * sizeof(uint32_t);
    const uint32_t block_threads = element_count;

    cudaStream_t producer_streams[region_count]{};
    cudaEvent_t producer_events[region_count]{};
    for (uint32_t i = 0; i < region_count; ++i) {
        if (!check_cuda(cudaStreamCreate(&producer_streams[i]), "stream create")) {
            return 1;
        }
        if (!check_cuda(cudaEventCreateWithFlags(&producer_events[i], cudaEventDisableTiming), "event create")) {
            return 1;
        }
    }
    cudaStream_t capture_stream = nullptr;
    if (!check_cuda(cudaStreamCreate(&capture_stream), "capture stream create")) {
        return 1;
    }

    void* d_regions[region_count]{};
    for (uint32_t i = 0; i < region_count; ++i) {
        if (!check_cuda(cudaMalloc(&d_regions[i], size_bytes), "cudaMalloc region")) {
            return 1;
        }
    }

    uint64_t* d_dep_stamps = nullptr;
    uint32_t* d_dep_stamp_counter = nullptr;
    const uint32_t dep_count = region_count;
    const uint32_t dep_stamps_needed = checkpoint_count * dep_count * 2u;
    if (use_deps) {
        if (!check_cuda(cudaMalloc(&d_dep_stamps, sizeof(uint64_t) * dep_stamps_needed), "cudaMalloc dep stamps") ||
            !check_cuda(cudaMalloc(&d_dep_stamp_counter, sizeof(uint32_t)), "cudaMalloc dep stamp counter")) {
            return 1;
        }
        if (!check_cuda(cudaMemset(d_dep_stamp_counter, 0, sizeof(uint32_t)), "memset dep stamp counter")) {
            return 1;
        }
    }

    tt::Recorder recorder;
    tt::RecorderConfig cfg{};
    cfg.ring_bytes = size_bytes * region_count * (checkpoint_count + 2u) + 4096u;
    cfg.checkpoint_capacity = checkpoint_count + 4u;
    cfg.region_capacity = region_count;
    cfg.retention_checkpoints = 0;
    cfg.overwrite_mode = tt::OverwriteMode::kDropOldest;
    cfg.enable_dep_stamps = use_deps;
    cfg.dep_stamps = d_dep_stamps;
    cfg.dep_stamp_counter = d_dep_stamp_counter;
    cfg.dep_stamp_capacity = use_deps ? dep_stamps_needed : 0u;
    if (!recorder.init(cfg)) {
        std::printf("recorder init failed\n");
        return 1;
    }

    for (uint32_t i = 0; i < region_count; ++i) {
        if (!recorder.register_region(i, d_regions[i], size_bytes, 1)) {
            std::printf("register_region failed\n");
            return 1;
        }
    }

    std::vector<double> capture_submit_us;
    capture_submit_us.reserve(checkpoint_count);
    const auto trace_start = std::chrono::steady_clock::now();

    const uint32_t seeds[region_count] = {0x11110000u, 0x22220000u, 0x33330000u};
    tt::CaptureDependency deps[region_count]{};

    for (uint32_t checkpoint = 0; checkpoint < checkpoint_count; ++checkpoint) {
        for (uint32_t r = 0; r < region_count; ++r) {
            write_region_phased_kernel<<<1, block_threads, 0, producer_streams[r]>>>(
                static_cast<uint32_t*>(d_regions[r]),
                element_count,
                checkpoint,
                seeds[r],
                spin_cycles);
            if (!check_cuda(cudaGetLastError(), "launch write_region_phased_kernel")) {
                return 1;
            }
            if (!check_cuda(cudaEventRecord(producer_events[r], producer_streams[r]), "event record")) {
                return 1;
            }
        }

        const auto submit_time = std::chrono::steady_clock::now();
        if (use_deps) {
            for (uint32_t r = 0; r < region_count; ++r) {
                deps[r].region_id = r;
                deps[r].event = producer_events[r];
                deps[r].producer_stream = producer_streams[r];
            }
            tt::CaptureDeps dep_list{deps, region_count};
            if (!recorder.capture_checkpoint(capture_stream, dep_list)) {
                std::printf("capture_checkpoint failed at %u\n", checkpoint);
                return 1;
            }
        } else {
            if (!recorder.capture_checkpoint(capture_stream)) {
                std::printf("capture_checkpoint failed at %u\n", checkpoint);
                return 1;
            }
        }
        const double submit_us = std::chrono::duration<double, std::micro>(submit_time - trace_start).count();
        capture_submit_us.push_back(submit_us);

        for (uint32_t r = 0; r < region_count; ++r) {
            if (!check_cuda(cudaStreamSynchronize(producer_streams[r]), "producer sync")) {
                return 1;
            }
        }
    }

    std::vector<uint32_t> host_region(element_count);
    std::mt19937 rng(0xC0FFEEu);
    std::uniform_int_distribution<uint32_t> dist(0, checkpoint_count - 1u);
    uint32_t mismatches = 0;
    const uint32_t checks = (verify_count > checkpoint_count) ? checkpoint_count : verify_count;
    for (uint32_t i = 0; i < checks; ++i) {
        const uint32_t checkpoint = dist(rng);
        if (!recorder.rewind_to_checkpoint(checkpoint, capture_stream)) {
            std::printf("rewind_to_checkpoint failed at %u\n", checkpoint);
            return 1;
        }
        for (uint32_t r = 0; r < region_count; ++r) {
            if (!check_cuda(cudaMemcpy(host_region.data(), d_regions[r], size_bytes, cudaMemcpyDeviceToHost), "memcpy region")) {
                return 1;
            }
            if (!verify_region(host_region, checkpoint, seeds[r])) {
                ++mismatches;
                break;
            }
        }
    }

    tt::TraceCollector trace;
    if (use_deps) {
        uint32_t stamp_count = 0;
        if (!check_cuda(cudaMemcpy(&stamp_count, d_dep_stamp_counter, sizeof(uint32_t), cudaMemcpyDeviceToHost), "memcpy dep stamp count")) {
            return 1;
        }
        std::vector<uint64_t> host_stamps(stamp_count, 0u);
        if (stamp_count > 0) {
            if (!check_cuda(cudaMemcpy(host_stamps.data(), d_dep_stamps, sizeof(uint64_t) * stamp_count, cudaMemcpyDeviceToHost), "memcpy dep stamps")) {
                return 1;
            }
        }

        int clock_rate_khz = 0;
        if (!check_cuda(cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate, 0), "device clock rate")) {
            return 1;
        }
        const double cycles_to_us = 1000.0 / static_cast<double>(clock_rate_khz);

        uint64_t base_stamp = 0;
        bool base_set = false;
        for (const auto& wait : recorder.dep_wait_records()) {
            if (wait.stamp_index_begin >= host_stamps.size() || wait.stamp_index_end >= host_stamps.size()) {
                continue;
            }
            const uint64_t begin_stamp = host_stamps[wait.stamp_index_begin];
            if (!base_set || begin_stamp < base_stamp) {
                base_stamp = begin_stamp;
                base_set = true;
            }
        }

        std::vector<uint64_t> checkpoint_min_stamp(checkpoint_count, 0);
        std::vector<bool> checkpoint_seen(checkpoint_count, false);
        for (const auto& wait : recorder.dep_wait_records()) {
            if (wait.checkpoint_id >= checkpoint_count) {
                continue;
            }
            if (wait.stamp_index_begin >= host_stamps.size() || wait.stamp_index_end >= host_stamps.size()) {
                continue;
            }
            const uint64_t begin_stamp = host_stamps[wait.stamp_index_begin];
            const uint64_t end_stamp = host_stamps[wait.stamp_index_end];
            if (!base_set) {
                base_stamp = begin_stamp;
                base_set = true;
            }

            if (!checkpoint_seen[wait.checkpoint_id] || begin_stamp < checkpoint_min_stamp[wait.checkpoint_id]) {
                checkpoint_min_stamp[wait.checkpoint_id] = begin_stamp;
                checkpoint_seen[wait.checkpoint_id] = true;
            }

            tt::TraceEvent event{};
            event.name = "wait_event";
            event.cat = "deps";
            event.ts_us = static_cast<double>(begin_stamp - base_stamp) * cycles_to_us;
            event.dur_us = static_cast<double>(end_stamp - begin_stamp) * cycles_to_us;
            event.pid = 1;
            event.tid = 0;
            event.args.push_back({"checkpoint_id", std::to_string(wait.checkpoint_id), false});
            event.args.push_back({"region_id", std::to_string(wait.region_id), false});
            event.args.push_back({"stream_id_capture", std::to_string(wait.capture_stream), false});
            if (wait.producer_stream != 0u) {
                event.args.push_back({"stream_id_producer", std::to_string(wait.producer_stream), false});
            }
            event.args.push_back({"event_ptr", hex_ptr(wait.event_ptr), true});
            trace.add_event(event);
        }

        for (uint32_t checkpoint = 0; checkpoint < checkpoint_count; ++checkpoint) {
            if (!checkpoint_seen[checkpoint]) {
                continue;
            }
            tt::TraceEvent submit{};
            submit.name = "capture_checkpoint_submit";
            submit.cat = "deps";
            submit.ts_us = static_cast<double>(checkpoint_min_stamp[checkpoint] - base_stamp) * cycles_to_us;
            submit.dur_us = 0.0;
            submit.pid = 1;
            submit.tid = 0;
            submit.args.push_back({"checkpoint_id", std::to_string(checkpoint), false});
            trace.add_event(submit);
        }
    } else {
        for (uint32_t checkpoint = 0; checkpoint < capture_submit_us.size(); ++checkpoint) {
            tt::TraceEvent submit{};
            submit.name = "capture_checkpoint_submit";
            submit.cat = "deps";
            submit.ts_us = capture_submit_us[checkpoint];
            submit.dur_us = 0.0;
            submit.pid = 1;
            submit.tid = 0;
            submit.args.push_back({"checkpoint_id", std::to_string(checkpoint), false});
            trace.add_event(submit);
        }
    }

    trace.write(trace_path.c_str());

    recorder.shutdown();
    for (uint32_t r = 0; r < region_count; ++r) {
        cudaFree(d_regions[r]);
    }
    if (use_deps) {
        cudaFree(d_dep_stamp_counter);
        cudaFree(d_dep_stamps);
    }
    for (uint32_t r = 0; r < region_count; ++r) {
        cudaEventDestroy(producer_events[r]);
        cudaStreamDestroy(producer_streams[r]);
    }
    cudaStreamDestroy(capture_stream);

    if (!use_deps) {
        std::printf("no-deps mismatches: %u\n", mismatches);
        std::printf("trace written: %s\n", trace_path.c_str());
        return 0;
    }
    if (mismatches > 0u) {
        std::printf("deps mode mismatches: %u\n", mismatches);
        return 1;
    }

    std::printf("deps mode success (trace written: %s)\n", trace_path.c_str());
    return 0;
}
