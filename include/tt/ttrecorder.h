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
    bool set_region_full_snapshot_period(uint32_t region_id, uint32_t period);

    bool capture_epoch(cudaStream_t stream);
    bool rewind_to_epoch(uint32_t target_epoch, cudaStream_t stream);
    bool read_epochs_to_host(std::vector<EpochRecord>& out);

private:
    RecorderConfig cfg_{};
    ControlBlock* d_control_ = nullptr;
    uint8_t* d_ring_ = nullptr;
    EpochRecord* d_epochs_ = nullptr;
    TrackedRegion* d_regions_ = nullptr;
    uint8_t* d_baseline_arena_ = nullptr;
    uint64_t* d_baseline_ptrs_ = nullptr;
    uint8_t* d_scratch_arena_ = nullptr;
    uint64_t* d_scratch_ptrs_ = nullptr;
    uint32_t* d_first_ring_offset_ = nullptr;
    uint32_t* d_first_was_written_ = nullptr;
    bool enable_deltas_ = true;
    bool initialized_ = false;
};

} // namespace tt

#endif // TTRECORDER_H
