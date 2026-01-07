#ifndef TTRECORDER_H
#define TTRECORDER_H

#include <cstdint>
#include <vector>
#include <cuda_runtime.h>

#include "tt/tt_layout.h"

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
    bool read_epochs_to_host(std::vector<EpochRecord>& out);

private:
    RecorderConfig cfg_{};
    ControlBlock* d_control_ = nullptr;
    uint8_t* d_ring_ = nullptr;
    EpochRecord* d_epochs_ = nullptr;
    TrackedRegion* d_regions_ = nullptr;
    bool initialized_ = false;
};

} // namespace tt

#endif // TTRECORDER_H
