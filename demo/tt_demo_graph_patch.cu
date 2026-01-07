#include "tt/tt_graph.h"
#include "tt/tt_graph_patch.h"
#include "tt/tt_cupti.h"
#include "tt/tt_trace.h"
#include "tt/ttrecorder.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
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

__global__ void write_pattern_iter_kernel(uint32_t* data, uint32_t count, const tt::IterParams* params) {
    const uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < count && params) {
        const uint32_t epoch = params->epoch;
        const uint32_t seed = params->seed;
        const uint32_t mix = (idx * 2654435761u) ^ seed;
        const uint32_t value = epoch ^ mix;
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

bool verify_pattern(const std::vector<uint32_t>& data, uint32_t epoch, uint32_t seed, uint32_t flags) {
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

uint32_t parse_u32(const char* arg, const char* prefix, uint32_t fallback) {
    const size_t len = std::strlen(prefix);
    if (std::strncmp(arg, prefix, len) != 0) {
        return fallback;
    }
    return static_cast<uint32_t>(std::strtoul(arg + len, nullptr, 10));
}

struct PatchNode {
    uint32_t node_id = 0;
    cudaKernelNodeParams params{};
    uint32_t* buffer_ptr = nullptr;
    uint32_t epoch_value = 0;
    uint32_t seed_value = 0;
};

bool build_graph(tt::GraphSession& graph,
    cudaStream_t stream,
    tt::Recorder& recorder,
    bool use_kernel_patch,
    uint32_t* d_buffer_a,
    uint32_t* d_buffer_b,
    uint32_t element_count,
    tt::IterParams* d_iter_params,
    std::vector<PatchNode>& patch_nodes) {
    if (!graph.begin_capture(stream)) {
        return false;
    }

    const uint32_t threads = 256;
    const uint32_t blocks = (element_count + threads - 1u) / threads;
    if (use_kernel_patch) {
        write_pattern_patch_kernel<<<blocks, threads, 0, stream>>>(d_buffer_a, element_count, 0u, 0u);
        write_pattern_patch_kernel<<<blocks, threads, 0, stream>>>(d_buffer_b, element_count, 0u, 0u);
    } else {
        write_pattern_iter_kernel<<<blocks, threads, 0, stream>>>(d_buffer_a, element_count, d_iter_params);
        write_pattern_iter_kernel<<<blocks, threads, 0, stream>>>(d_buffer_b, element_count, d_iter_params);
    }
    if (!recorder.capture_epoch(stream)) {
        graph.destroy();
        return false;
    }
    if (!graph.end_capture()) {
        graph.destroy();
        return false;
    }

    patch_nodes.clear();
    if (use_kernel_patch) {
        for (const auto& node : graph.get_nodes()) {
            if (node.name != "demo.write_pattern_patch") {
                continue;
            }
            cudaKernelNodeParams params{};
            if (!graph.get_kernel_node_params(node.id, &params)) {
                continue;
            }
            uint32_t* buffer_ptr = nullptr;
            if (params.kernelParams && params.kernelParams[0]) {
                buffer_ptr = *reinterpret_cast<uint32_t**>(params.kernelParams[0]);
            }
            PatchNode patch{};
            patch.node_id = node.id;
            patch.params = params;
            patch.buffer_ptr = buffer_ptr;
            patch_nodes.push_back(patch);
        }
    }
    return true;
}

bool apply_patch_updates(tt::GraphSession& graph,
    std::vector<PatchNode>& patch_nodes,
    uint32_t epoch,
    uint32_t seed,
    bool allow_fallback,
    bool& did_fallback,
    tt::Recorder& recorder,
    cudaStream_t stream,
    bool use_kernel_patch,
    uint32_t* d_buffer_a,
    uint32_t* d_buffer_b,
    uint32_t element_count,
    tt::IterParams* d_iter_params) {
    if (!use_kernel_patch) {
        return true;
    }
    tt::GraphUpdateStatus status{};
    bool printed = false;
    for (auto& patch : patch_nodes) {
        patch.epoch_value = epoch;
        patch.seed_value = seed;
        uint32_t* buffer_ptr = patch.buffer_ptr ? patch.buffer_ptr : d_buffer_a;
        void* kernel_args[] = {&buffer_ptr, &element_count, &patch.epoch_value, &patch.seed_value};
        patch.params.kernelParams = kernel_args;
        if (!graph.update_kernel_node_params(patch.node_id, patch.params, &status)) {
            std::printf("kernel patch failed: %s (%s)\n",
                tt::GraphUpdateReasonString(status.reason),
                cudaGetErrorString(status.cuda_error));
            if (allow_fallback && !did_fallback) {
                did_fallback = true;
                graph.destroy();
                std::vector<PatchNode> rebuilt;
                if (!build_graph(graph,
                        stream,
                        recorder,
                        use_kernel_patch,
                        d_buffer_a,
                        d_buffer_b,
                        element_count,
                        d_iter_params,
                        rebuilt)) {
                    return false;
                }
                patch_nodes = rebuilt;
                return apply_patch_updates(graph,
                    patch_nodes,
                    epoch,
                    seed,
                    false,
                    did_fallback,
                    recorder,
                    stream,
                    use_kernel_patch,
                    d_buffer_a,
                    d_buffer_b,
                    element_count,
                    d_iter_params);
            }
            return false;
        }
        if (!printed) {
            std::printf("kernel patch applied via fast path\n");
            printed = true;
        }
    }
    return true;
}

} // namespace

int main(int argc, char** argv) {
    tt::GetCuptiKernelTracer();
    tt::RegisterGraphKernelTag(reinterpret_cast<const void*>(write_pattern_iter_kernel), "demo.write_pattern_iter");
    tt::RegisterGraphKernelTag(reinterpret_cast<const void*>(write_pattern_patch_kernel), "demo.write_pattern_patch");

    uint32_t iterations = 12;
    uint32_t toggle_every = 3;
    bool use_kernel_patch = false;
    bool deterministic = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--kernel-patch") {
            use_kernel_patch = true;
        } else if (std::string(argv[i]) == "--deterministic") {
            deterministic = true;
        } else if (std::string(argv[i]).rfind("--iterations=", 0) == 0) {
            iterations = parse_u32(argv[i], "--iterations=", iterations);
        } else if (std::string(argv[i]).rfind("--toggle-every=", 0) == 0) {
            toggle_every = parse_u32(argv[i], "--toggle-every=", toggle_every);
        }
    }

    const uint32_t element_count = 1024;
    const uint32_t size_bytes = element_count * sizeof(uint32_t);
    const uint32_t stamps_per_epoch = 3;

    uint32_t* d_buffer_a = nullptr;
    uint32_t* d_buffer_b = nullptr;
    tt::IterParams* d_iter_params = nullptr;
    uint64_t* d_stamps = nullptr;
    uint32_t* d_stamp_counter = nullptr;
    if (!check_cuda(cudaMalloc(&d_buffer_a, size_bytes), "cudaMalloc buffer A") ||
        !check_cuda(cudaMalloc(&d_buffer_b, size_bytes), "cudaMalloc buffer B") ||
        !check_cuda(cudaMalloc(&d_iter_params, sizeof(tt::IterParams)), "cudaMalloc iter params") ||
        !check_cuda(cudaMalloc(&d_stamps, sizeof(uint64_t) * iterations * stamps_per_epoch), "cudaMalloc stamps") ||
        !check_cuda(cudaMalloc(&d_stamp_counter, sizeof(uint32_t)), "cudaMalloc stamp counter")) {
        return 1;
    }
    if (!check_cuda(cudaMemset(d_stamp_counter, 0, sizeof(uint32_t)), "memset stamp counter")) {
        return 1;
    }

    cudaStream_t stream = nullptr;
    if (!check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate")) {
        return 1;
    }

    const uint32_t per_chunk_bytes = (static_cast<uint32_t>(sizeof(tt::ChunkHeader)) + size_bytes + 31u) & ~31u;
    const uint32_t ring_bytes = per_chunk_bytes * iterations + 4096u;

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
    cfg.deterministic = deterministic;
    if (!recorder.init(cfg)) {
        std::printf("recorder init failed\n");
        return 1;
    }

    if (!recorder.register_region(0, d_buffer_a, size_bytes, 1) ||
        !recorder.register_region(1, d_buffer_b, size_bytes, 1)) {
        std::printf("register_region failed\n");
        recorder.shutdown();
        return 1;
    }

    tt::GraphSession graph;
    std::vector<PatchNode> patch_nodes;
    if (!build_graph(graph,
            stream,
            recorder,
            use_kernel_patch,
            d_buffer_a,
            d_buffer_b,
            element_count,
            d_iter_params,
            patch_nodes)) {
        std::printf("graph build failed\n");
        recorder.shutdown();
        return 1;
    }
    if (use_kernel_patch && patch_nodes.empty()) {
        std::printf("graph patch nodes not found\n");
        graph.destroy();
        recorder.shutdown();
        return 1;
    }

    tt::RecorderGraphControl control{};
    control.region_mask = (1ull << 2) - 1ull;
    control.snapshot_period = 0;
    control.flags = tt::kGraphControlStampsEnabled;
    if (!recorder.update_graph_control(control, stream)) {
        std::printf("graph control update failed\n");
        graph.destroy();
        recorder.shutdown();
        return 1;
    }

    tt::TraceCollector trace;
    const auto trace_start = std::chrono::steady_clock::now();

    bool did_fallback = false;
    for (uint32_t iter = 0; iter < iterations; ++iter) {
        if (toggle_every > 0 && (iter % toggle_every) == 0) {
            control.flags ^= tt::kGraphControlStampsEnabled;
            control.snapshot_period = (control.snapshot_period == 0u) ? 2u : 0u;
            control.region_mask ^= (1ull << 1);
            if (!recorder.update_graph_control(control, stream)) {
                std::printf("graph control update failed\n");
                graph.destroy();
                recorder.shutdown();
                return 1;
            }
            std::printf("update controls: stamps=%u snapshot_period=%u region_mask=0x%llx\n",
                (control.flags & tt::kGraphControlStampsEnabled) ? 1u : 0u,
                control.snapshot_period,
                static_cast<unsigned long long>(control.region_mask));
        }

        tt::IterParams host_params{};
        host_params.epoch = iter;
        host_params.seed = 0x1234u + iter;
        host_params.flags = (iter & 1u);
        if (!use_kernel_patch) {
            if (!check_cuda(cudaMemcpyAsync(d_iter_params,
                    &host_params,
                    sizeof(tt::IterParams),
                    cudaMemcpyHostToDevice,
                    stream),
                    "memcpy iter params")) {
                graph.destroy();
                recorder.shutdown();
                return 1;
            }
        }

        if (!apply_patch_updates(graph,
                patch_nodes,
                host_params.epoch,
                host_params.seed,
                true,
                did_fallback,
                recorder,
                stream,
                use_kernel_patch,
                d_buffer_a,
                d_buffer_b,
                element_count,
                d_iter_params)) {
            std::printf("graph patching failed at iter %u\n", iter);
            graph.destroy();
            recorder.shutdown();
            return 1;
        }

        if (!graph.launch(stream)) {
            std::printf("graph launch failed at iter %u\n", iter);
            graph.destroy();
            recorder.shutdown();
            return 1;
        }

        const auto cpu_delta = std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - trace_start).count();
        tt::TraceEvent event{};
        event.name = "graph_launch";
        event.cat = "graph";
        event.ts_us = cpu_delta;
        event.dur_us = 0.0;
        event.pid = 1;
        event.tid = 0;
        event.args.push_back({"epoch_id", std::to_string(iter), false});
        trace.add_event(event);
    }

    if (!check_cuda(cudaStreamSynchronize(stream), "graph sync")) {
        graph.destroy();
        recorder.shutdown();
        return 1;
    }

    const uint32_t target_epoch = iterations > 0 ? (iterations - 1u) : 0u;
    if (!recorder.rewind_to_epoch(target_epoch, stream)) {
        std::printf("rewind_to_epoch failed\n");
        graph.destroy();
        recorder.shutdown();
        return 1;
    }
    std::vector<uint32_t> host_out(element_count);
    if (!check_cuda(cudaMemcpy(host_out.data(), d_buffer_a, size_bytes, cudaMemcpyDeviceToHost), "memcpy out")) {
        graph.destroy();
        recorder.shutdown();
        return 1;
    }

    tt::IterParams final_params{};
    final_params.epoch = target_epoch;
    final_params.seed = 0x1234u + target_epoch;
    final_params.flags = (target_epoch & 1u);
    if (!verify_pattern(host_out, final_params.epoch, final_params.seed, final_params.flags)) {
        std::printf("tt_demo_graph_patch verification failed\n");
        graph.destroy();
        recorder.shutdown();
        return 1;
    }

    tt::GetCuptiKernelTracer().append_kernel_events(trace);
    trace.write("trace/tt_graph_patch_trace.json");

    graph.destroy();
    recorder.shutdown();
    cudaStreamDestroy(stream);
    cudaFree(d_stamp_counter);
    cudaFree(d_stamps);
    cudaFree(d_iter_params);
    cudaFree(d_buffer_b);
    cudaFree(d_buffer_a);

    std::printf("tt_demo_graph_patch success\n");
    return 0;
}
