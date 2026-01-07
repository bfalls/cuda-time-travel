# CUDA Graphs

## GraphSession usage
1) Create a stream and a `tt::GraphSession`.
2) Call `begin_capture(stream)`.
3) Enqueue your work and `Recorder::capture_epoch(stream)`.
4) Call `end_capture()`.
5) Replay with `launch(stream)` and synchronize as needed.

Minimal example:

```cpp
tt::GraphSession graph;
graph.begin_capture(stream);
// enqueue work
recorder.capture_epoch(stream);
graph.end_capture();

graph.launch(stream);
```

## Limitations
- `capture_epoch(stream)` is graph-safe (no host reads or device syncs during capture).
- During capture, `capture_epoch` records snapshots for enabled regions; delta logic,
  full-snapshot periods, and retention/backpressure checks are host-side and are not
  enforced inside graph replay.
- Timestamp stamps use `clock64()` and are device-clock based; see `docs/trace.md`.
- No CUPTI integration.
