# Chrome Trace Output

The recorder and demo can emit a Chrome Trace JSON file at:

```
trace/tt_trace.json
```

## Events
- `graph` category: one event per graph launch (CUDA event elapsed time).
- `epoch` category: internal timing from device stamps recorded inside graphs.

Each event uses the `"X"` (complete) phase and includes:
- `ts` and `dur` in microseconds
- `pid = 1`
- `tid` for grouping lanes
- `args` such as `epoch_id` and `ring_bytes_written` when available

## Timestamp conversion
Device stamps use `clock64()` and are converted with `cudaGetDeviceProperties().clockRate`
(`kHz`), where:

```
time_us = cycles * 1000 / clockRate_khz
```

The stamp-based timeline is relative to the first stamp (starts at 0).

## Viewing
Open `chrome://tracing` in Chrome and load `trace/tt_trace.json`.
