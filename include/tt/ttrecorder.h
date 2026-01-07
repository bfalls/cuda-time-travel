#ifndef TTRECORDER_H
#define TTRECORDER_H

#include <cstdint>
#include <vector>
#include <cuda_runtime.h>

#include "tt/tt_layout.h"

namespace tt {

struct DeviceEpochBegin;

enum class OverwriteMode : uint32_t {
    kDropOldest = 0,
    kBackpressure = 1
};

enum class RecorderStatus : uint32_t {
    kOk = 0,
    kNotInitialized,
    kInvalidConfig,
    kCudaError,
    kBackpressure,
    kRingTooSmall,
    kRingCorrupt,
    kInvalidHeader,
    kInvalidChunkType,
    kInvalidPayload,
    kAlignmentError,
    kEpochNotFound,
    kEpochDropped
};

struct RecorderConfig {
    uint32_t ring_bytes = 0;
    uint32_t epoch_capacity = 0;
    uint32_t region_capacity = 0;
    uint32_t retention_epochs = 0;
    OverwriteMode overwrite_mode = OverwriteMode::kDropOldest;
    bool enable_graph_stamps = false;
    uint64_t* graph_stamps = nullptr;
    uint32_t* graph_stamp_counter = nullptr;
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
    RecorderStatus last_status() const { return last_status_; }

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
    uint32_t* d_delta_sizes_ = nullptr;
    DeviceEpochBegin* d_epoch_begin_ = nullptr;
    uint32_t* d_stamp_base_ = nullptr;
    std::vector<TrackedRegion> host_regions_{};
    bool enable_deltas_ = true;
    bool enable_graph_stamps_ = false;
    uint64_t* d_graph_stamps_ = nullptr;
    uint32_t* d_graph_stamp_counter_ = nullptr;
    bool initialized_ = false;
    uint32_t min_valid_epoch_ = 0;
    RecorderStatus last_status_ = RecorderStatus::kOk;
};

} // namespace tt

#endif // TTRECORDER_H
