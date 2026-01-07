# Multi-stream capture correctness

This project supports per-region dependencies for capture epochs. The dependency model makes capture ordering explicit and avoids observing partially-written data when producers run on different streams.

## Dependency model

- Producers write into tracked regions on their own streams.
- Each producer records a `cudaEvent_t` after the region update completes.
- The capture stream waits on that event before reading the region.
- The wait is scheduled with `cudaStreamWaitEvent` on the capture stream (no host blocking).

Without dependencies, the capture stream may read a region while a producer is still writing it, which can yield partially-updated snapshots or deltas.

## Minimal usage

```cpp
cudaStream_t producer = nullptr;
cudaStream_t capture = nullptr;
cudaEvent_t done = nullptr;

cudaStreamCreate(&producer);
cudaStreamCreate(&capture);
cudaEventCreateWithFlags(&done, cudaEventDisableTiming);

// Launch producer work for region 0.
write_region_kernel<<<grid, block, 0, producer>>>(region_ptr, ...);
cudaEventRecord(done, producer);

tt::CaptureDependency dep{};
dep.region_id = 0;
dep.event = done;
dep.producer_stream = producer;

tt::CaptureDeps deps{&dep, 1};
recorder.capture_epoch(capture, deps);
```

## Trace and stamps

If you enable dependency stamps in `RecorderConfig` (`enable_dep_stamps`, `dep_stamps`, `dep_stamp_counter`, `dep_stamp_capacity`), the recorder emits GPU clock stamps before/after each wait. These stamps can be converted into Chrome trace events. See the demo below for a full example that produces `wait_event` traces with per-region metadata.

## Determinism interaction

Deterministic mode pins captures to a single stream. Multi-stream producers are still allowed, but capture itself is serialized and will wait on dependencies in that fixed stream. If you need maximum concurrency, keep deterministic mode off.

## Demo

`tt_demo_multistream_stress` shows dependency waits and correctness under concurrent producers:

- `tt_demo_multistream_stress --deps` records with per-region dependencies and validates correctness.
- `tt_demo_multistream_stress --no-deps` records without dependencies and reports any mismatches it finds.

## Test scaling

The multistream tests honor `TT_TEST_MULTISTREAM_ITERS` to reduce the number of epochs when needed.
