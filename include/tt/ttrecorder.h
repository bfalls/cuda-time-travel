#ifndef TTRECORDER_H
#define TTRECORDER_H

#include <cstdint>
#include <cuda_runtime.h>

namespace tt {

struct RecorderConfig {
    uint32_t ring_bytes = 0;
    uint32_t epoch_capacity = 0;
    uint32_t region_capacity = 0;
};

class Recorder {
public:
    bool init(const RecorderConfig& cfg);
    void shutdown();

    bool register_region(uint32_t region_id, void* device_ptr, uint32_t size_bytes, uint32_t options = 1);

    bool capture_epoch(cudaStream_t stream);
    bool rewind_to_epoch(uint32_t target_epoch, cudaStream_t stream);
};

} // namespace tt

#endif // TTRECORDER_H
