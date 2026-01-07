#include "tt/ttrecorder.h"
#include "tt/tt_trace.h"

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

} // namespace

int main() {
    const uint32_t element_count = 1024;
    const uint32_t size_bytes = element_count * sizeof(uint32_t);
    const uint32_t epoch_count = 6;
    const uint32_t snapshot_period = 2;
    const char* manifest_path = "trace/tt_manifest_verify_demo.json";
    const char* report_pass = "trace/tt_verify_report_pass.json";
    const char* report_fail = "trace/tt_verify_report_fail.json";
    const char* trace_path = "trace/tt_verify_trace.json";

    void* d_buffer_a = nullptr;
    void* d_buffer_b = nullptr;
    if (!check_cuda(cudaMalloc(&d_buffer_a, size_bytes), "cudaMalloc A")) {
        return 1;
    }
    if (!check_cuda(cudaMalloc(&d_buffer_b, size_bytes), "cudaMalloc B")) {
        cudaFree(d_buffer_a);
        return 1;
    }

    const uint32_t per_chunk_bytes = (static_cast<uint32_t>(sizeof(tt::ChunkHeader)) + size_bytes + 31u) & ~31u;
    const uint32_t per_epoch_bytes = per_chunk_bytes * 2u;
    const uint32_t ring_bytes = per_epoch_bytes * epoch_count + 4096u;

    tt::Recorder recorder;
    tt::RecorderConfig cfg{};
    cfg.ring_bytes = ring_bytes;
    cfg.epoch_capacity = 32;
    cfg.region_capacity = 8;
    cfg.retention_epochs = 0;
    cfg.overwrite_mode = tt::OverwriteMode::kDropOldest;
    cfg.deterministic = true;
    cfg.enable_manifest = true;
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
    recorder.set_region_full_snapshot_period(0, snapshot_period);
    recorder.set_region_full_snapshot_period(1, snapshot_period);

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
            std::printf("capture_epoch failed\n");
            recorder.shutdown();
            cudaFree(d_buffer_b);
            cudaFree(d_buffer_a);
            return 1;
        }
    }

    if (!recorder.write_manifest_json(manifest_path)) {
        std::printf("manifest write failed\n");
        recorder.shutdown();
        cudaFree(d_buffer_b);
        cudaFree(d_buffer_a);
        return 1;
    }

    tt::TraceCollector trace;
    tt::VerifyOptions pass_options{};
    pass_options.report_path = report_pass;
    pass_options.trace_annotate = true;
    pass_options.trace = &trace;

    tt::VerifyReport pass_report{};
    bool pass_ok = recorder.verify_manifest_json(manifest_path, pass_options, &pass_report);
    std::printf("verify pass: %s\n", pass_ok ? "ok" : "fail");

    tt::VerifyOptions fail_options{};
    fail_options.report_path = report_fail;
    fail_options.trace_annotate = true;
    fail_options.trace = &trace;
    fail_options.tamper.enabled = true;
    fail_options.tamper.epoch_id = 3;
    fail_options.tamper.region_id = 1;
    fail_options.tamper.byte_offset = 8;
    fail_options.tamper.xor_mask = 0x1;

    tt::VerifyReport fail_report{};
    bool fail_ok = recorder.verify_manifest_json(manifest_path, fail_options, &fail_report);
    std::printf("verify fail (expected): %s\n", fail_ok ? "unexpected pass" : "mismatch detected");

    trace.write(trace_path);
    std::printf("verify trace written: %s\n", trace_path);

    recorder.shutdown();
    cudaFree(d_buffer_b);
    cudaFree(d_buffer_a);

    return (pass_ok && !fail_ok) ? 0 : 1;
}
