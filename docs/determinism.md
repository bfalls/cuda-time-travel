# Deterministic replay mode

Deterministic mode ensures repeated runs of the same workload on the same GPU/driver produce identical per-epoch buffer hashes and manifest output. This mode is intended for reproducibility checks, CI validation, and debugging.

## Contract

- Deterministic mode guarantees identical manifest output when:
  - The same GPU model and driver version are used.
  - The same deterministic workload is captured with the same launch order.
  - Capture uses a single CUDA stream for all `capture_epoch` calls.
- Epochs are never dropped: deterministic mode enforces backpressure and fails capture when the ring would overflow.
- Manifest output has stable key ordering and consistent numeric formatting for byte-for-byte comparisons.

## Limits and behavior

- Determinism is scoped to device buffer state hashing and manifest output. It does not cover host timing, CUPTI traces, or kernel execution timestamps.
- Stream ordering must be consistent. If `capture_epoch` is called with multiple streams, capture fails.
- Manifest output is only supported for non-graph capture paths; enable manifest only when the stream is not being captured into a CUDA graph.
- Reproducibility is not guaranteed across different GPUs, drivers, or changes in the captured workload.

## Usage

- Enable deterministic mode and manifest collection via `RecorderConfig`:
  - `deterministic = true`
  - `enable_manifest = true`
- Use `Recorder::write_manifest_json` to emit the manifest.

Example:

```
.\build\Release\tt_demo.exe --deterministic --manifest-out=trace\tt_manifest.json
```
