#ifndef TTRECORDER_H
#define TTRECORDER_H

#include <cstdint>
#include <vector>
#include <cuda_runtime.h>

#include "tt/tt_graph_patch.h"
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
    kEpochDropped,
    kDeterminismViolation
};

struct RecorderConfig {
    uint32_t ring_bytes = 0;
    uint32_t epoch_capacity = 0;
    uint32_t region_capacity = 0;
    uint32_t retention_epochs = 0;
    OverwriteMode overwrite_mode = OverwriteMode::kDropOldest;
    bool deterministic = false;
    bool enable_manifest = false;
    bool enable_graph_stamps = false;
    uint64_t* graph_stamps = nullptr;
    uint32_t* graph_stamp_counter = nullptr;
};

struct ManifestRegion {
    uint32_t region_id = 0;
    uint32_t size_bytes = 0;
    uint64_t hash64 = 0;
    uint32_t payload_bytes = 0;
    uint32_t uncompressed_bytes = 0;
    bool snapshot = true;
};

struct ManifestEpoch {
    uint32_t epoch_id = 0;
    uint32_t ring_bytes_written = 0;
    std::vector<ManifestRegion> regions{};
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
    bool write_manifest_json(const char* path) const;
    void clear_manifest();
    const std::vector<ManifestEpoch>& manifest() const { return manifest_epochs_; }
    RecorderStatus last_status() const { return last_status_; }
    const RecorderGraphControl* graph_control_device() const { return d_graph_control_; }
    bool update_graph_control(const RecorderGraphControl& control, cudaStream_t stream);
    bool update_region_enable_bitmap(const uint32_t* bitmap, uint32_t words, cudaStream_t stream);
    bool update_region_pointer(uint32_t region_id, void* device_ptr);

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
    uint64_t* d_region_hashes_ = nullptr;
    RecorderGraphControl* d_graph_control_ = nullptr;
    uint32_t* d_region_enable_bitmap_ = nullptr;
    uint32_t* d_enabled_count_ = nullptr;
    RecorderGraphControl graph_control_host_{};
    std::vector<TrackedRegion> host_regions_{};
    std::vector<ManifestEpoch> manifest_epochs_{};
    bool enable_deltas_ = true;
    bool enable_graph_stamps_ = false;
    bool deterministic_ = false;
    bool enable_manifest_ = false;
    cudaStream_t deterministic_stream_ = nullptr;
    bool deterministic_stream_set_ = false;
    uint64_t* d_graph_stamps_ = nullptr;
    uint32_t* d_graph_stamp_counter_ = nullptr;
    bool initialized_ = false;
    uint32_t min_valid_epoch_ = 0;
    RecorderStatus last_status_ = RecorderStatus::kOk;
};

} // namespace tt

#endif // TTRECORDER_H
