# Verification (Phase 12)

The verifier replays a recorded run from the in-memory ring buffer and compares per-epoch region hashes against a manifest. When a mismatch is detected, it attempts to localize the first differing byte offset by reconstructing the expected bytes from the ring payloads and diffing against the replayed region state.

## How verification works

1. Load a manifest JSON file containing per-epoch, per-region hash64 values.
2. Replay epochs in order (via `Recorder::rewind_to_epoch`) and compute hashes for selected regions after each epoch.
3. Compare computed hashes to manifest hashes.
4. On mismatch, reconstruct expected bytes from ring contents and run a GPU diff to localize the first difference.

## Ground truth

The ring buffer contents are treated as ground truth. Localization uses snapshot and delta payloads from the ring to reconstruct the expected bytes for a specific epoch and region. This works even when a manifest only stores hashes.

## Limitations

- Verification requires the recorded run to be present in memory (the ring is not persisted to disk).
- Localization is best-effort: if required snapshot/delta chunks are missing or corrupted, localization will be reported as unavailable.
- Manifest hashing is deterministic on a given GPU/driver but does not validate host timing or kernel scheduling.

## CLI usage

The `tt` CLI includes a `verify` command that records a deterministic demo workload in-process and verifies it against a manifest file:

```
.\build\Release\tt.exe verify --manifest trace\tt_manifest_verify_demo.json
```

Optional flags:
- `--epochs <start-end>` verifies only a subset of epochs (inclusive).
- `--regions <list>` verifies a comma-separated list of region IDs.
- `--out <report.json>` writes a verifier report.
- `--trace-annotate` emits Chrome trace events in `trace/tt_verify_trace.json`.
- `--trace-out <path>` overrides the trace output path (default: `trace/tt_verify_trace.json`).
- `--continue` collects all mismatches instead of stopping at the first.
- `--localize` forces localization on mismatch (also auto-enabled on mismatch).
- `--tamper <epoch,region,offset>` flips a byte after replay to force a mismatch (debug/demo).

## Demo usage

The `tt_demo_verify` demo records, verifies a passing run, then introduces a controlled corruption and verifies a failing run with localization:

```
.\build\Release\tt_demo_verify.exe
```

Trace output is written to:
```
trace\tt_verify_trace.json
```
