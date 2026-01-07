#ifdef TT_ENABLE_CUPTI

#include "tt/tt_cupti.h"

#include <cupti.h>
#include <cupti_activity.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <windows.h>

namespace tt {

namespace {

constexpr size_t kBufferSize = 16 * 1024;

CuptiKernelTracer* g_tracer_ptr = nullptr;

struct CuptiApi {
    using RegisterCallbacksFn = CUptiResult(CUPTIAPI*)(CUpti_BuffersCallbackRequestFunc,
        CUpti_BuffersCallbackCompleteFunc);
    using EnableFn = CUptiResult(CUPTIAPI*)(CUpti_ActivityKind);
    using DisableFn = CUptiResult(CUPTIAPI*)(CUpti_ActivityKind);
    using FlushAllFn = CUptiResult(CUPTIAPI*)(uint32_t);
    using GetNextRecordFn = CUptiResult(CUPTIAPI*)(uint8_t*, size_t, CUpti_Activity**);
    using GetDroppedFn = CUptiResult(CUPTIAPI*)(CUcontext, uint32_t, size_t*);

    HMODULE module = nullptr;
    RegisterCallbacksFn register_callbacks = nullptr;
    EnableFn enable = nullptr;
    DisableFn disable = nullptr;
    FlushAllFn flush_all = nullptr;
    GetNextRecordFn get_next_record = nullptr;
    GetDroppedFn get_num_dropped = nullptr;

    bool load() {
        if (module) {
            return true;
        }
        char cuda_path[MAX_PATH] = {};
        DWORD len = GetEnvironmentVariableA("CUDA_PATH", cuda_path, MAX_PATH);
        if (len > 0 && len < MAX_PATH) {
            std::string pattern = std::string(cuda_path) + "\\extras\\CUPTI\\lib64\\cupti64_*.dll";
            WIN32_FIND_DATAA data{};
            HANDLE handle = FindFirstFileA(pattern.c_str(), &data);
            if (handle != INVALID_HANDLE_VALUE) {
                std::string full_path = std::string(cuda_path) + "\\extras\\CUPTI\\lib64\\" + data.cFileName;
                FindClose(handle);
                module = LoadLibraryA(full_path.c_str());
            }
        }
        if (!module) {
            module = LoadLibraryA("cupti64.dll");
        }
        if (!module) {
            return false;
        }

        register_callbacks = reinterpret_cast<RegisterCallbacksFn>(
            GetProcAddress(module, "cuptiActivityRegisterCallbacks"));
        enable = reinterpret_cast<EnableFn>(GetProcAddress(module, "cuptiActivityEnable"));
        disable = reinterpret_cast<DisableFn>(GetProcAddress(module, "cuptiActivityDisable"));
        flush_all = reinterpret_cast<FlushAllFn>(GetProcAddress(module, "cuptiActivityFlushAll"));
        get_next_record = reinterpret_cast<GetNextRecordFn>(
            GetProcAddress(module, "cuptiActivityGetNextRecord"));
        get_num_dropped = reinterpret_cast<GetDroppedFn>(
            GetProcAddress(module, "cuptiActivityGetNumDroppedRecords"));

        if (!register_callbacks || !enable || !disable || !flush_all || !get_next_record || !get_num_dropped) {
            FreeLibrary(module);
            module = nullptr;
            return false;
        }
        return true;
    }
};

CuptiApi& GetCuptiApi() {
    static CuptiApi api;
    return api;
}

void CUPTIAPI buffer_requested(uint8_t** buffer, size_t* size, size_t* max_num_records) {
    if (!buffer || !size || !max_num_records) {
        return;
    }
    *size = kBufferSize;
    *buffer = static_cast<uint8_t*>(std::malloc(*size));
    *max_num_records = 0;
}

void CUPTIAPI buffer_completed(CUcontext ctx, uint32_t stream_id, uint8_t* buffer, size_t size, size_t valid_size) {
    if (g_tracer_ptr && buffer && valid_size > 0) {
        g_tracer_ptr->consume_activity_buffer(buffer, valid_size);
    }
    if (buffer) {
        std::free(buffer);
    }
    size_t dropped = 0;
    CuptiApi& api = GetCuptiApi();
    if (api.get_num_dropped) {
        (void)api.get_num_dropped(ctx, stream_id, &dropped);
    }
}

} // namespace

CuptiKernelTracer::CuptiKernelTracer() {
    g_tracer_ptr = this;
    CuptiApi& api = GetCuptiApi();
    if (!api.load()) {
        enabled_ = false;
        return;
    }
    if (api.register_callbacks(buffer_requested, buffer_completed) != CUPTI_SUCCESS) {
        enabled_ = false;
        return;
    }
    if (api.enable(CUPTI_ACTIVITY_KIND_KERNEL) != CUPTI_SUCCESS) {
        enabled_ = false;
        return;
    }
    if (api.enable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) != CUPTI_SUCCESS) {
        enabled_ = false;
        return;
    }
    enabled_ = true;
}

CuptiKernelTracer::~CuptiKernelTracer() {
    if (enabled_) {
        CuptiApi& api = GetCuptiApi();
        api.flush_all(0);
        api.disable(CUPTI_ACTIVITY_KIND_KERNEL);
        api.disable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    }
    g_tracer_ptr = nullptr;
}

void CuptiKernelTracer::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    events_.clear();
}

void CuptiKernelTracer::consume_activity_buffer(uint8_t* buffer, size_t valid_size) {
    CUpti_Activity* record = nullptr;
    CuptiApi& api = GetCuptiApi();
    if (!api.get_next_record) {
        return;
    }
    while (api.get_next_record(buffer, valid_size, &record) == CUPTI_SUCCESS) {
        if (!record) {
            continue;
        }
        if (record->kind != CUPTI_ACTIVITY_KIND_KERNEL &&
            record->kind != CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
            continue;
        }
        const auto* kernel = reinterpret_cast<const CUpti_ActivityKernel4*>(record);
        CuptiKernelEvent event{};
        if (kernel->name) {
            event.name = kernel->name;
        }
        event.start = kernel->start;
        event.end = kernel->end;
        event.stream_id = kernel->streamId;
        event.correlation_id = kernel->correlationId;
        std::lock_guard<std::mutex> lock(mutex_);
        events_.push_back(std::move(event));
    }
}

void CuptiKernelTracer::append_kernel_events(TraceCollector& trace) {
    if (!enabled_) {
        return;
    }

    CuptiApi& api = GetCuptiApi();
    if (api.flush_all) {
        api.flush_all(0);
    }

    std::vector<CuptiKernelEvent> snapshot;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (events_.empty()) {
            return;
        }
        snapshot.swap(events_);
    }

    uint64_t base_start = snapshot[0].start;
    for (const auto& event : snapshot) {
        base_start = std::min(base_start, event.start);
    }

    for (const auto& event : snapshot) {
        if (event.end <= event.start) {
            continue;
        }
        TraceEvent trace_event{};
        trace_event.name = event.name.empty() ? "kernel" : event.name;
        trace_event.cat = "kernel";
        trace_event.ts_us = static_cast<double>(event.start - base_start) / 1000.0;
        trace_event.dur_us = static_cast<double>(event.end - event.start) / 1000.0;
        trace_event.pid = 1;
        trace_event.tid = static_cast<int>(event.stream_id);
        if (event.correlation_id != 0) {
            trace_event.args.push_back({"correlation_id", std::to_string(event.correlation_id), false});
        }
        trace.add_event(trace_event);
    }
}

CuptiKernelTracer& GetCuptiKernelTracer() {
    static CuptiKernelTracer tracer;
    return tracer;
}

} // namespace tt

#endif
