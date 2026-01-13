#ifndef TTRECORDER_H
#define TTRECORDER_H

#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include <cuda_runtime.h>

#include "tt/tt_graph_patch.h"
#include "tt/tt_layout.h"

namespace tt {

struct DeviceCheckpointBegin;
class TraceCollector;

enum class OverwriteMode : uint32_t {
    kDropOldest = 0,
    kBackpressure = 1
};

enum class RecorderStatus : uint32_t {
    kOk = 0,
    kNotInitialized,
    kInvalidConfig,
    kInvalidDependency,
    kCudaError,
    kBackpressure,
    kRingTooSmall,
    kRingCorrupt,
    kInvalidHeader,
    kInvalidChunkType,
    kInvalidPayload,
    kAlignmentError,
    kCheckpointNotFound,
    kCheckpointDropped,
    kDeterminismViolation
};

struct RecorderConfig {
    uint32_t ring_bytes = 0;
    uint32_t checkpoint_capacity = 0;
    uint32_t region_capacity = 0;
    uint32_t retention_checkpoints = 0;
    OverwriteMode overwrite_mode = OverwriteMode::kDropOldest;
    bool deterministic = false;
    bool enable_manifest = false;
    bool enable_graph_stamps = false;
    uint64_t* graph_stamps = nullptr;
    uint32_t* graph_stamp_counter = nullptr;
    bool enable_dep_stamps = false;
    uint64_t* dep_stamps = nullptr;
    uint32_t* dep_stamp_counter = nullptr;
    uint32_t dep_stamp_capacity = 0;
};

struct ManifestRegion {
    uint32_t region_id = 0;
    uint32_t size_bytes = 0;
    uint64_t hash64 = 0;
    uint32_t payload_bytes = 0;
    uint32_t uncompressed_bytes = 0;
    bool snapshot = true;
};

struct ManifestCheckpoint {
    uint32_t checkpoint_id = 0;
    uint32_t ring_bytes_written = 0;
    std::vector<ManifestRegion> regions{};
};

struct CaptureDependency {
    uint32_t region_id = 0;
    cudaEvent_t event = nullptr;
    cudaStream_t producer_stream = nullptr;
};

struct CaptureDeps {
    const CaptureDependency* deps = nullptr;
    size_t count = 0;
};

struct DepWaitRecord {
    uint32_t checkpoint_id = 0;
    uint32_t region_id = 0;
    uint64_t event_ptr = 0;
    uint64_t producer_stream = 0;
    uint64_t capture_stream = 0;
    uint32_t stamp_index_begin = 0;
    uint32_t stamp_index_end = 0;
};

struct VerifyTamper {
    bool enabled = false;
    uint32_t checkpoint_id = 0;
    uint32_t region_id = 0;
    uint32_t byte_offset = 0;
    uint8_t xor_mask = 0;
};

struct VerifyMismatch {
    uint32_t checkpoint_id = 0;
    uint32_t region_id = 0;
    uint64_t expected_hash64 = 0;
    uint64_t actual_hash64 = 0;
    bool localized = false;
    uint32_t first_diff_offset_bytes = 0;
    std::string expected_window_hex{};
    std::string actual_window_hex{};
    std::string localization_error{};
};

struct VerifyReport {
    std::string status{};
    uint32_t tested_checkpoint_begin = 0;
    uint32_t tested_checkpoint_end = 0;
    std::vector<uint32_t> tested_regions{};
    bool has_mismatch = false;
    VerifyMismatch first_mismatch{};
    std::vector<VerifyMismatch> mismatches{};
    std::string error{};
    std::string device_name{};
    int device_cc_major = 0;
    int device_cc_minor = 0;
    int driver_version = 0;
    int runtime_version = 0;
};

struct VerifyOptions {
    bool continue_on_mismatch = false;
    bool localize = false;
    bool trace_annotate = false;
    bool checkpoint_range_set = false;
    uint32_t checkpoint_begin = 0;
    uint32_t checkpoint_end = 0;
    std::vector<uint32_t> region_ids{};
    const char* report_path = nullptr;
    cudaStream_t stream = nullptr;
    TraceCollector* trace = nullptr;
    VerifyTamper tamper{};
};

class Recorder {
public:
    bool init(const RecorderConfig& cfg);
    void shutdown();

    bool register_region(uint32_t region_id, void* device_ptr, uint32_t size_bytes, uint32_t options = 1);
    bool set_region_full_snapshot_period(uint32_t region_id, uint32_t period);

    bool capture_checkpoint(cudaStream_t stream);
    bool capture_checkpoint(cudaStream_t stream, const CaptureDeps& deps);
    bool rewind_to_checkpoint(uint32_t target_checkpoint, cudaStream_t stream);
    bool read_checkpoints_to_host(std::vector<CheckpointRecord>& out);
    bool write_manifest_json(const char* path) const;
    void clear_manifest();
    const std::vector<ManifestCheckpoint>& manifest() const { return manifest_checkpoints_; }
    RecorderStatus last_status() const { return last_status_; }
    const RecorderGraphControl* graph_control_device() const { return d_graph_control_; }
    bool update_graph_control(const RecorderGraphControl& control, cudaStream_t stream);
    bool update_region_enable_bitmap(const uint32_t* bitmap, uint32_t words, cudaStream_t stream);
    bool update_region_pointer(uint32_t region_id, void* device_ptr);
    bool compute_region_hashes(cudaStream_t stream, const std::vector<uint32_t>& region_ids, uint64_t* d_out_hashes);
    bool copy_region_hashes_to_host(cudaStream_t stream, const uint64_t* d_hashes, size_t count, std::vector<uint64_t>& out);
    bool verify_manifest_json(const char* manifest_path, const VerifyOptions& options, VerifyReport* out_report);

    const std::vector<DepWaitRecord>& dep_wait_records() const { return dep_wait_records_; }

private:
    RecorderConfig cfg_{};
    ControlBlock* d_control_ = nullptr;
    uint8_t* d_ring_ = nullptr;
    CheckpointRecord* d_checkpoints_ = nullptr;
    TrackedRegion* d_regions_ = nullptr;
    uint8_t* d_baseline_arena_ = nullptr;
    uint64_t* d_baseline_ptrs_ = nullptr;
    uint8_t* d_scratch_arena_ = nullptr;
    uint64_t* d_scratch_ptrs_ = nullptr;
    uint32_t* d_first_ring_offset_ = nullptr;
    uint32_t* d_first_was_written_ = nullptr;
    uint32_t* d_delta_sizes_ = nullptr;
    DeviceCheckpointBegin* d_checkpoint_begin_ = nullptr;
    uint32_t* d_stamp_base_ = nullptr;
    uint32_t* d_dep_stamp_base_ = nullptr;
    uint64_t* d_region_hashes_ = nullptr;
    RecorderGraphControl* d_graph_control_ = nullptr;
    uint32_t* d_region_enable_bitmap_ = nullptr;
    uint32_t* d_enabled_count_ = nullptr;
    RecorderGraphControl graph_control_host_{};
    std::vector<TrackedRegion> host_regions_{};
    std::vector<ManifestCheckpoint> manifest_checkpoints_{};
    bool enable_deltas_ = true;
    bool enable_graph_stamps_ = false;
    bool deterministic_ = false;
    bool enable_manifest_ = false;
    cudaStream_t deterministic_stream_ = nullptr;
    bool deterministic_stream_set_ = false;
    uint64_t* d_graph_stamps_ = nullptr;
    uint32_t* d_graph_stamp_counter_ = nullptr;
    uint64_t* d_dep_stamps_ = nullptr;
    uint32_t* d_dep_stamp_counter_ = nullptr;
    uint32_t dep_stamp_capacity_ = 0;
    bool initialized_ = false;
    uint32_t min_valid_checkpoint_ = 0;
    RecorderStatus last_status_ = RecorderStatus::kOk;
    std::vector<DepWaitRecord> dep_wait_records_{};
};

} // namespace tt

#endif // TTRECORDER_H
