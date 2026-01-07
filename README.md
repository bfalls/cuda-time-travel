# cuda-time-travel
Record and rewind CUDA GPU memory state with low overhead, deterministic replay, and Chrome trace export (CUPTI kernel timeline + graph-aware stamps).

## v1 scope and limitations

Snapshot-only recorder for CUDA device buffers. Assumptions:
- Full snapshots only by default; delta chunks are available in Phase 2.
- Ring buffer wrap markers are supported.
- Retention policy supports DROP_OLDEST and BACKPRESSURE modes.
- Chunks for an epoch are contiguous in ring order.
- Rewind lookup scans epoch table on host.
- No persistence; device memory only.
- No instruction-level replay or CUPTI tracing.
- Tracked region sizes must be multiples of 4 bytes.

## Build (Windows, Visual Studio)

```
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

## Run

```
.\build\Release\tt_demo.exe
.\build\Release\tt_demo_graph.exe
.\build\Release\tt_demo_graph_patch.exe
.\build\Release\tt_demo_determinism.exe
.\build\Release\tt_demo_multistream_stress.exe --deps
.\build\Release\tt_demo_multistream_stress.exe --no-deps
.\build\Release\tt_tests.exe
```

Demo flags:
- `--no-delta` disables delta capture (snapshots only).
- `--ring-bytes=<n>` overrides ring size. If too small for all epochs, rewind verification is skipped.
- `--deterministic` enables deterministic capture mode (see `docs/determinism.md`).
- `--manifest-out=<path>` writes a deterministic manifest JSON.

Graph + trace:
- `tt_demo_graph` captures `(app work + capture_epoch)` as a CUDA Graph and replays it.
- Chrome trace output is written to `trace/tt_trace.json`.
- See `docs/graphs.md` and `docs/trace.md` for usage and limitations.

CUDA Graph patching:
- `tt_demo_graph_patch` replays a captured graph and updates per-iteration parameters and recorder controls without rebuilding.
- Example: `.\build\Release\tt_demo_graph_patch.exe --iterations=12 --toggle-every=3`
- Use `--kernel-patch` to exercise kernel node param updates (falls back to recapture if needed).

Open trace in Chrome:
```
start chrome "chrome://tracing"
```

## Recorder config

`RecorderConfig` supports:
- `retention_epochs`: keep the last N epochs (0 keeps all until space pressure).
- `overwrite_mode`: `DROP_OLDEST` (discard old epochs to make space) or `BACKPRESSURE` (fail capture when space is insufficient).
- `deterministic`: enforce deterministic capture ordering and prevent epoch drops.
- `enable_manifest`: collect per-epoch region hashes for manifest output.

Notes:
- `ring_bytes` must be a multiple of 32 bytes.

## Multi-stream correctness

When producers write tracked regions on different streams, the capture stream should wait on a producer-completed event per region. Without dependencies, captures can observe partially-written data. The recorder supports per-region dependencies via `CaptureDeps` and `cudaStreamWaitEvent`.

See `docs/multistream.md` for the dependency model, trace stamps, and a minimal code snippet.

Example commands:

```
.\build\Release\tt_demo_multistream_stress.exe --deps
.\build\Release\tt_demo_multistream_stress.exe --no-deps
```

## Deterministic replay mode

Use deterministic mode to ensure repeated runs generate identical per-epoch buffer hashes and manifest output:

```
.\build\Release\tt_demo.exe --deterministic --manifest-out=trace\tt_manifest.json
.\build\Release\tt_demo_determinism.exe
```

See `docs/determinism.md` for the full contract and limitations.
