# cuda-time-travel

## v1 scope and limitations

Snapshot-only recorder for CUDA device buffers. Assumptions:
- Full snapshots only; no delta/compression.
- Ring buffer is large enough to avoid wrap or overwrite.
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
