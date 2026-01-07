# CUDA graph patching

Phase 9 introduces two supported ways to update per-iteration values while replaying a captured CUDA graph:

## Pattern 1: device-side parameter block (recommended)

Allocate a small device struct and have graph-captured kernels read it:

```
struct IterParams {
  uint32_t epoch;
  uint32_t seed;
  uint32_t flags;
};
```

Update the struct between graph launches (outside the graph) with a tiny H2D copy. This is the most robust approach because the graph node parameters do not change between launches. Deterministic mode uses this pattern.

## Pattern 2: kernel node param patching (demo)

For kernels with by-value parameters, you can update `cudaKernelNodeParams` on the existing exec:

```
cudaGraphExecKernelNodeSetParams(exec, node, &params);
```

This is less robust because kernel signatures and parameter layouts must match exactly. If the runtime rejects the update, fall back to re-capture + re-instantiate.

## Recorder control updates

The recorder exposes a device-side control block (`RecorderGraphControl`) that graph capture reads each launch:
- `region_mask` / `region_bitmap`: enable or disable regions without rebuilding the graph.
- `snapshot_period`: override the snapshot cadence. A period of 0 disables the override.
- `flags`: toggle graph stamps (`kGraphControlStampsEnabled`).

Update the control block between launches to change capture behavior without re-capturing.

## When a rebuild is required

Some updates cannot be applied in-place:
- The kernel parameter layout changes (different signature or parameter count).
- Graph topology changes (node types added/removed).
- The runtime reports update not supported.

When this happens, recapture and reinstantiate the graph once and continue.

## Deterministic mode interaction

Deterministic mode relies on the device-side parameter block (Pattern 1). Avoid changing graph topology or per-node parameter layouts when deterministic replay is enabled.
