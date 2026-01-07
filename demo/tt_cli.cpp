#include "tt/ttrecorder.h"
#include "tt/tt_trace.h"

#include <cstdio>
#include <cstring>
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

void fill_pattern(std::vector<uint32_t>& data, uint32_t seed) {
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = seed ^ static_cast<uint32_t>(i * 1664525u + 1013904223u);
    }
}

bool parse_u32(const char* text, uint32_t& out) {
    if (!text || !*text) {
        return false;
    }
    char* end = nullptr;
    unsigned long value = std::strtoul(text, &end, 10);
    if (!end || *end != '\0') {
        return false;
    }
    out = static_cast<uint32_t>(value);
    return true;
}

bool parse_epoch_range(const char* text, uint32_t& begin, uint32_t& end) {
    if (!text || !*text) {
        return false;
    }
    const char* dash = std::strchr(text, '-');
    if (!dash) {
        uint32_t value = 0;
        if (!parse_u32(text, value)) {
            return false;
        }
        begin = value;
        end = value;
        return true;
    }
    std::string start(text, dash - text);
    std::string stop(dash + 1);
    uint32_t b = 0;
    uint32_t e = 0;
    if (!parse_u32(start.c_str(), b) || !parse_u32(stop.c_str(), e)) {
        return false;
    }
    begin = b;
    end = e;
    return true;
}

bool parse_regions(const char* text, std::vector<uint32_t>& out) {
    out.clear();
    if (!text || !*text) {
        return false;
    }
    const char* cursor = text;
    while (*cursor) {
        const char* comma = std::strchr(cursor, ',');
        std::string token = comma ? std::string(cursor, comma - cursor) : std::string(cursor);
        uint32_t value = 0;
        if (!parse_u32(token.c_str(), value)) {
            return false;
        }
        out.push_back(value);
        if (!comma) {
            break;
        }
        cursor = comma + 1;
    }
    return !out.empty();
}

void print_usage() {
    std::printf("Usage:\\n");
    std::printf("  tt verify --manifest <path> [--epochs <start-end>] [--regions <list>] [--out <report.json>] [--trace-annotate] [--trace-out <path>] [--continue] [--localize] [--tamper <epoch,region,offset>]\\n");
}

} // namespace

int main() {
    if (__argc < 2) {
        print_usage();
        return 1;
    }

    if (std::strcmp(__argv[1], "verify") != 0) {
        print_usage();
        return 1;
    }

    const char* manifest_path = nullptr;
    const char* report_path = nullptr;
    const char* trace_path = "trace/tt_verify_trace.json";
    bool trace_annotate = false;
    bool continue_on_mismatch = false;
    bool localize = false;
    bool epoch_range_set = false;
    uint32_t epoch_begin = 0;
    uint32_t epoch_end = 0;
    std::vector<uint32_t> regions;
    bool tamper_enabled = false;
    uint32_t tamper_epoch = 0;
    uint32_t tamper_region = 0;
    uint32_t tamper_offset = 0;

    for (int i = 2; i < __argc; ++i) {
        if (std::strcmp(__argv[i], "--trace-annotate") == 0) {
            trace_annotate = true;
        } else if (std::strcmp(__argv[i], "--continue") == 0) {
            continue_on_mismatch = true;
        } else if (std::strcmp(__argv[i], "--localize") == 0) {
            localize = true;
        } else if (std::strcmp(__argv[i], "--manifest") == 0 && i + 1 < __argc) {
            manifest_path = __argv[++i];
        } else if (std::strncmp(__argv[i], "--manifest=", 11) == 0) {
            manifest_path = __argv[i] + 11;
        } else if (std::strcmp(__argv[i], "--out") == 0 && i + 1 < __argc) {
            report_path = __argv[++i];
        } else if (std::strncmp(__argv[i], "--out=", 6) == 0) {
            report_path = __argv[i] + 6;
        } else if (std::strcmp(__argv[i], "--trace-out") == 0 && i + 1 < __argc) {
            trace_path = __argv[++i];
        } else if (std::strncmp(__argv[i], "--trace-out=", 12) == 0) {
            trace_path = __argv[i] + 12;
        } else if (std::strcmp(__argv[i], "--epochs") == 0 && i + 1 < __argc) {
            epoch_range_set = parse_epoch_range(__argv[++i], epoch_begin, epoch_end);
        } else if (std::strncmp(__argv[i], "--epochs=", 9) == 0) {
            epoch_range_set = parse_epoch_range(__argv[i] + 9, epoch_begin, epoch_end);
        } else if (std::strcmp(__argv[i], "--regions") == 0 && i + 1 < __argc) {
            if (!parse_regions(__argv[++i], regions)) {
                std::printf("invalid --regions value\\n");
                return 1;
            }
        } else if (std::strncmp(__argv[i], "--regions=", 10) == 0) {
            if (!parse_regions(__argv[i] + 10, regions)) {
                std::printf("invalid --regions value\\n");
                return 1;
            }
        } else if (std::strcmp(__argv[i], "--tamper") == 0 && i + 1 < __argc) {
            const char* value = __argv[++i];
            const char* first = std::strchr(value, ',');
            const char* second = first ? std::strchr(first + 1, ',') : nullptr;
            if (!first || !second) {
                std::printf("invalid --tamper value\\n");
                return 1;
            }
            std::string epoch_text(value, first - value);
            std::string region_text(first + 1, second - first - 1);
            std::string offset_text(second + 1);
            if (!parse_u32(epoch_text.c_str(), tamper_epoch) ||
                !parse_u32(region_text.c_str(), tamper_region) ||
                !parse_u32(offset_text.c_str(), tamper_offset)) {
                std::printf("invalid --tamper value\\n");
                return 1;
            }
            tamper_enabled = true;
        } else if (std::strncmp(__argv[i], "--tamper=", 9) == 0) {
            const char* value = __argv[i] + 9;
            const char* first = std::strchr(value, ',');
            const char* second = first ? std::strchr(first + 1, ',') : nullptr;
            if (!first || !second) {
                std::printf("invalid --tamper value\\n");
                return 1;
            }
            std::string epoch_text(value, first - value);
            std::string region_text(first + 1, second - first - 1);
            std::string offset_text(second + 1);
            if (!parse_u32(epoch_text.c_str(), tamper_epoch) ||
                !parse_u32(region_text.c_str(), tamper_region) ||
                !parse_u32(offset_text.c_str(), tamper_offset)) {
                std::printf("invalid --tamper value\\n");
                return 1;
            }
            tamper_enabled = true;
        }
    }

    if (!manifest_path) {
        std::printf("--manifest is required\\n");
        return 1;
    }

    const uint32_t element_count = 1024;
    const uint32_t size_bytes = element_count * sizeof(uint32_t);
    const uint32_t epoch_count = 6;
    const uint32_t snapshot_period = 2;

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
        std::printf("recorder init failed\\n");
        cudaFree(d_buffer_b);
        cudaFree(d_buffer_a);
        return 1;
    }
    if (!recorder.register_region(0, d_buffer_a, size_bytes, 1) ||
        !recorder.register_region(1, d_buffer_b, size_bytes, 1)) {
        std::printf("register_region failed\\n");
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
            std::printf("capture_epoch failed\\n");
            recorder.shutdown();
            cudaFree(d_buffer_b);
            cudaFree(d_buffer_a);
            return 1;
        }
    }

    tt::TraceCollector trace;
    tt::VerifyOptions options{};
    options.report_path = report_path;
    options.trace_annotate = trace_annotate;
    options.trace = trace_annotate ? &trace : nullptr;
    options.continue_on_mismatch = continue_on_mismatch;
    options.localize = localize;
    options.epoch_range_set = epoch_range_set;
    options.epoch_begin = epoch_begin;
    options.epoch_end = epoch_end;
    options.region_ids = regions;
    if (tamper_enabled) {
        options.tamper.enabled = true;
        options.tamper.epoch_id = tamper_epoch;
        options.tamper.region_id = tamper_region;
        options.tamper.byte_offset = tamper_offset;
        options.tamper.xor_mask = 0x1;
    }

    tt::VerifyReport report{};
    const bool ok = recorder.verify_manifest_json(manifest_path, options, &report);

    if (trace_annotate) {
        trace.write(trace_path);
    }

    recorder.shutdown();
    cudaFree(d_buffer_b);
    cudaFree(d_buffer_a);

    return ok ? 0 : 1;
}
