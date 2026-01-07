# cuda-time-travel

## v1 scope and limitations

Snapshot-only recorder for CUDA device buffers. Assumptions:
- Full snapshots only by default; delta chunks are available in Phase 2.
- Ring buffer wrap markers are supported.
- Retention policy supports DROP_OLDEST and BACKPRESSURE modes.
- Chunks for an epoch are contiguous in ring order.
- Rewind lookup scans epoch table on host.
- No persistence; device memory only.
- No instruction-level replay, tracing, or CUDA Graph capture.
- Tracked region sizes must be multiples of 4 bytes.

## Build (Windows, Visual Studio)

```
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

## Run

```
.\build\Release\tt_demo.exe
.\build\Release\tt_tests.exe
```

Demo flags:
- `--no-delta` disables delta capture (snapshots only).
- `--ring-bytes=<n>` overrides ring size. If too small for all epochs, rewind verification is skipped.

## Recorder config

`RecorderConfig` supports:
- `retention_epochs`: keep the last N epochs (0 keeps all until space pressure).
- `overwrite_mode`: `DROP_OLDEST` (discard old epochs to make space) or `BACKPRESSURE` (fail capture when space is insufficient).

Notes:
- `ring_bytes` must be a multiple of 32 bytes.
