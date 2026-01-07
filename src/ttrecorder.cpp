#include "tt/ttrecorder.h"

namespace tt {

bool Recorder::init(const RecorderConfig& cfg) {
    (void)cfg;
    return false;
}

void Recorder::shutdown() {}

bool Recorder::register_region(uint32_t region_id, void* device_ptr, uint32_t size_bytes, uint32_t options) {
    (void)region_id;
    (void)device_ptr;
    (void)size_bytes;
    (void)options;
    return false;
}

bool Recorder::capture_epoch(cudaStream_t stream) {
    (void)stream;
    return false;
}

bool Recorder::rewind_to_epoch(uint32_t target_epoch, cudaStream_t stream) {
    (void)target_epoch;
    (void)stream;
    return false;
}

} // namespace tt
