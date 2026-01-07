#ifndef TT_CUPTI_H
#define TT_CUPTI_H

#include "tt/tt_trace.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#ifdef TT_ENABLE_CUPTI
#include <mutex>
#endif

namespace tt {

#ifdef TT_ENABLE_CUPTI

struct CuptiKernelEvent {
    std::string name;
    uint64_t start = 0;
    uint64_t end = 0;
    uint32_t stream_id = 0;
    uint32_t correlation_id = 0;
};

class CuptiKernelTracer {
public:
    CuptiKernelTracer();
    ~CuptiKernelTracer();

    void append_kernel_events(TraceCollector& trace);
    void clear();
    void consume_activity_buffer(uint8_t* buffer, size_t valid_size);

private:
    std::vector<CuptiKernelEvent> events_;
    std::mutex mutex_;
    bool enabled_ = false;
};

CuptiKernelTracer& GetCuptiKernelTracer();

#else

class CuptiKernelTracer {
public:
    void append_kernel_events(TraceCollector&) {}
    void clear() {}
};

inline CuptiKernelTracer& GetCuptiKernelTracer() {
    static CuptiKernelTracer tracer;
    return tracer;
}

#endif

} // namespace tt

#endif // TT_CUPTI_H
