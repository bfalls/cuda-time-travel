# Ring Format and WRAP Behavior

## Chunk layout

Each chunk starts with a 32-byte `ChunkHeader` followed by a payload:

- `magic` must be `0x54545243`
- `version` must be `1`
- `header_bytes` must be `32`
- `chunk_type` is one of:
  - `kChunkTypeSnapshot`
  - `kChunkTypeDeltaXorRle0`
  - `kChunkTypeWrapMarker`

Chunks are aligned to 32 bytes in the ring. Payloads are contiguous and never span the ring boundary.

## WRAP marker chunks

If an append would cross the end of the ring, the recorder writes a WRAP marker chunk at the current ring offset, pads the remainder of the ring, and continues writing the next chunk at ring offset 0. WRAP markers have:

- `chunk_type = kChunkTypeWrapMarker`
- `payload_bytes = 0`
- `flags` includes `kChunkFlagWrapMarker`

Readers must recognize WRAP markers, reset the ring offset to 0, and continue parsing without consuming an epoch chunk count.

## Corruption-safe parsing checks

Readers validate each header before applying:

- magic/version/header size must match expected constants
- header alignment is 32 bytes
- payload size is non-zero for data chunks
- payload size does not exceed ring bounds
- chunk type is known

On failure, readers return a structured error code instead of dereferencing invalid data.
