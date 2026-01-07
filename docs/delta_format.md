# Delta XOR RLE0 Format

This format encodes the XOR of a region snapshot against a baseline.
All values are little-endian uint32 words.

Header (payload layout):
- uint32 word_count
- uint32 block_count
- blocks...

Each block starts with a uint32 tag:
- Zero run:    tag = 0x80000000 | run_len, followed by no data.
- Literal run: tag = 0x00000000 | run_len, followed by run_len uint32 XOR values.

Example:
Baseline words: [0x1, 0x2, 0x3, 0x4]
Current  words: [0x1, 0x7, 0x3, 0x5]
XOR      words: [0x0, 0x5, 0x0, 0x1]

Encoding:
word_count = 4
blocks:
  zero run of 1
  literal run of 1: 0x5
  zero run of 1
  literal run of 1: 0x1

Payload words:
  [4, 4, 0x80000001, 0x00000001, 0x00000005, 0x80000001, 0x00000001, 0x00000001]
