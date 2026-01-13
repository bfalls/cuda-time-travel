#include "tt/ttrecorder.h"
#include "tt/tt_trace.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string_view>

namespace tt {

namespace {

static constexpr uint32_t kRingAlignment = 32u;
static constexpr uint32_t kHashThreads = 256u;

struct JsonCursor {
    const std::string* text = nullptr;
    size_t pos = 0;
};

static void skip_ws(JsonCursor& cursor) {
    const std::string& text = *cursor.text;
    while (cursor.pos < text.size() && std::isspace(static_cast<unsigned char>(text[cursor.pos])) != 0) {
        ++cursor.pos;
    }
}

static bool consume(JsonCursor& cursor, char c) {
    skip_ws(cursor);
    const std::string& text = *cursor.text;
    if (cursor.pos >= text.size() || text[cursor.pos] != c) {
        return false;
    }
    ++cursor.pos;
    return true;
}

static bool parse_string(JsonCursor& cursor, std::string& out) {
    skip_ws(cursor);
    const std::string& text = *cursor.text;
    if (cursor.pos >= text.size() || text[cursor.pos] != '"') {
        return false;
    }
    ++cursor.pos;
    std::string result;
    while (cursor.pos < text.size()) {
        char ch = text[cursor.pos++];
        if (ch == '"') {
            out = result;
            return true;
        }
        if (ch == '\\') {
            if (cursor.pos >= text.size()) {
                return false;
            }
            char escaped = text[cursor.pos++];
            switch (escaped) {
                case '"': result.push_back('"'); break;
                case '\\': result.push_back('\\'); break;
                case '/': result.push_back('/'); break;
                case 'b': result.push_back('\b'); break;
                case 'f': result.push_back('\f'); break;
                case 'n': result.push_back('\n'); break;
                case 'r': result.push_back('\r'); break;
                case 't': result.push_back('\t'); break;
                default: return false;
            }
        } else {
            result.push_back(ch);
        }
    }
    return false;
}

static bool parse_u32(JsonCursor& cursor, uint32_t& out) {
    skip_ws(cursor);
    const std::string& text = *cursor.text;
    size_t start = cursor.pos;
    while (cursor.pos < text.size() && std::isdigit(static_cast<unsigned char>(text[cursor.pos])) != 0) {
        ++cursor.pos;
    }
    if (start == cursor.pos) {
        return false;
    }
    const std::string_view slice(text.data() + start, cursor.pos - start);
    unsigned long value = 0;
    for (char ch : slice) {
        value = value * 10u + static_cast<unsigned long>(ch - '0');
        if (value > std::numeric_limits<uint32_t>::max()) {
            return false;
        }
    }
    out = static_cast<uint32_t>(value);
    return true;
}

static bool skip_number(JsonCursor& cursor) {
    skip_ws(cursor);
    const std::string& text = *cursor.text;
    if (cursor.pos >= text.size()) {
        return false;
    }
    size_t start = cursor.pos;
    if (text[cursor.pos] == '-') {
        ++cursor.pos;
    }
    bool saw_digit = false;
    while (cursor.pos < text.size() && std::isdigit(static_cast<unsigned char>(text[cursor.pos])) != 0) {
        saw_digit = true;
        ++cursor.pos;
    }
    if (cursor.pos < text.size() && text[cursor.pos] == '.') {
        ++cursor.pos;
        while (cursor.pos < text.size() && std::isdigit(static_cast<unsigned char>(text[cursor.pos])) != 0) {
            saw_digit = true;
            ++cursor.pos;
        }
    }
    if (cursor.pos < text.size() && (text[cursor.pos] == 'e' || text[cursor.pos] == 'E')) {
        ++cursor.pos;
        if (cursor.pos < text.size() && (text[cursor.pos] == '+' || text[cursor.pos] == '-')) {
            ++cursor.pos;
        }
        while (cursor.pos < text.size() && std::isdigit(static_cast<unsigned char>(text[cursor.pos])) != 0) {
            saw_digit = true;
            ++cursor.pos;
        }
    }
    return saw_digit && cursor.pos > start;
}

static bool parse_uint64_hex(const std::string& text, uint64_t& out) {
    if (text.size() < 3 || text[0] != '0' || (text[1] != 'x' && text[1] != 'X')) {
        return false;
    }
    uint64_t value = 0;
    for (size_t i = 2; i < text.size(); ++i) {
        char ch = text[i];
        uint32_t digit = 0;
        if (ch >= '0' && ch <= '9') {
            digit = static_cast<uint32_t>(ch - '0');
        } else if (ch >= 'a' && ch <= 'f') {
            digit = static_cast<uint32_t>(ch - 'a' + 10);
        } else if (ch >= 'A' && ch <= 'F') {
            digit = static_cast<uint32_t>(ch - 'A' + 10);
        } else {
            return false;
        }
        value = (value << 4u) | digit;
    }
    out = value;
    return true;
}

static bool skip_value(JsonCursor& cursor);

static bool skip_array(JsonCursor& cursor) {
    if (!consume(cursor, '[')) {
        return false;
    }
    skip_ws(cursor);
    const std::string& text = *cursor.text;
    if (cursor.pos < text.size() && text[cursor.pos] == ']') {
        ++cursor.pos;
        return true;
    }
    while (cursor.pos < text.size()) {
        if (!skip_value(cursor)) {
            return false;
        }
        skip_ws(cursor);
        if (cursor.pos < text.size() && text[cursor.pos] == ',') {
            ++cursor.pos;
            continue;
        }
        break;
    }
    return consume(cursor, ']');
}

static bool skip_object(JsonCursor& cursor) {
    if (!consume(cursor, '{')) {
        return false;
    }
    skip_ws(cursor);
    const std::string& text = *cursor.text;
    if (cursor.pos < text.size() && text[cursor.pos] == '}') {
        ++cursor.pos;
        return true;
    }
    while (cursor.pos < text.size()) {
        std::string key;
        if (!parse_string(cursor, key)) {
            return false;
        }
        if (!consume(cursor, ':')) {
            return false;
        }
        if (!skip_value(cursor)) {
            return false;
        }
        skip_ws(cursor);
        if (cursor.pos < text.size() && text[cursor.pos] == ',') {
            ++cursor.pos;
            continue;
        }
        break;
    }
    return consume(cursor, '}');
}

static bool skip_value(JsonCursor& cursor) {
    skip_ws(cursor);
    const std::string& text = *cursor.text;
    if (cursor.pos >= text.size()) {
        return false;
    }
    char ch = text[cursor.pos];
    if (ch == '"') {
        std::string ignored;
        return parse_string(cursor, ignored);
    }
    if (ch == '{') {
        return skip_object(cursor);
    }
    if (ch == '[') {
        return skip_array(cursor);
    }
    if (std::isdigit(static_cast<unsigned char>(ch)) != 0 || ch == '-') {
        return skip_number(cursor);
    }
    if (text.compare(cursor.pos, 4, "true") == 0) {
        cursor.pos += 4;
        return true;
    }
    if (text.compare(cursor.pos, 5, "false") == 0) {
        cursor.pos += 5;
        return true;
    }
    if (text.compare(cursor.pos, 4, "null") == 0) {
        cursor.pos += 4;
        return true;
    }
    return false;
}

static bool parse_region(JsonCursor& cursor, ManifestRegion& out, std::string& error) {
    if (!consume(cursor, '{')) {
        error = "expected '{' for region object";
        return false;
    }
    bool has_region_id = false;
    bool has_size_bytes = false;
    bool has_hash = false;
    while (true) {
        skip_ws(cursor);
        const std::string& text = *cursor.text;
        if (cursor.pos < text.size() && text[cursor.pos] == '}') {
            ++cursor.pos;
            break;
        }
        std::string key;
        if (!parse_string(cursor, key)) {
            error = "failed to parse region key";
            return false;
        }
        if (!consume(cursor, ':')) {
            error = "expected ':' after region key";
            return false;
        }
        if (key == "region_id") {
            uint32_t value = 0;
            if (!parse_u32(cursor, value)) {
                error = "failed to parse region_id";
                return false;
            }
            out.region_id = value;
            has_region_id = true;
        } else if (key == "size_bytes") {
            uint32_t value = 0;
            if (!parse_u32(cursor, value)) {
                error = "failed to parse size_bytes";
                return false;
            }
            out.size_bytes = value;
            has_size_bytes = true;
        } else if (key == "hash64") {
            std::string value;
            if (!parse_string(cursor, value)) {
                error = "failed to parse hash64";
                return false;
            }
            uint64_t parsed = 0;
            if (!parse_uint64_hex(value, parsed)) {
                error = "failed to parse hash64 hex";
                return false;
            }
            out.hash64 = parsed;
            has_hash = true;
        } else if (key == "payload_bytes") {
            uint32_t value = 0;
            if (!parse_u32(cursor, value)) {
                error = "failed to parse payload_bytes";
                return false;
            }
            out.payload_bytes = value;
        } else if (key == "uncompressed_bytes") {
            uint32_t value = 0;
            if (!parse_u32(cursor, value)) {
                error = "failed to parse uncompressed_bytes";
                return false;
            }
            out.uncompressed_bytes = value;
        } else if (key == "snapshot_or_delta") {
            std::string value;
            if (!parse_string(cursor, value)) {
                error = "failed to parse snapshot_or_delta";
                return false;
            }
            out.snapshot = (value == "snapshot");
        } else {
            if (!skip_value(cursor)) {
                error = "failed to skip region value";
                return false;
            }
        }
        skip_ws(cursor);
        if (cursor.pos < text.size() && text[cursor.pos] == ',') {
            ++cursor.pos;
            continue;
        }
        if (cursor.pos < text.size() && text[cursor.pos] == '}') {
            ++cursor.pos;
            break;
        }
    }
    if (!has_region_id || !has_size_bytes || !has_hash) {
        error = "missing region fields";
        return false;
    }
    return true;
}

static bool parse_checkpoint(JsonCursor& cursor, ManifestCheckpoint& out, std::string& error) {
    if (!consume(cursor, '{')) {
        error = "expected '{' for checkpoint object";
        return false;
    }
    bool has_checkpoint_id = false;
    while (true) {
        skip_ws(cursor);
        const std::string& text = *cursor.text;
        if (cursor.pos < text.size() && text[cursor.pos] == '}') {
            ++cursor.pos;
            break;
        }
        std::string key;
        if (!parse_string(cursor, key)) {
            error = "failed to parse checkpoint key";
            return false;
        }
        if (!consume(cursor, ':')) {
            error = "expected ':' after checkpoint key";
            return false;
        }
        if (key == "checkpoint_id") {
            uint32_t value = 0;
            if (!parse_u32(cursor, value)) {
                error = "failed to parse checkpoint_id";
                return false;
            }
            out.checkpoint_id = value;
            has_checkpoint_id = true;
        } else if (key == "ring_bytes_written") {
            uint32_t value = 0;
            if (!parse_u32(cursor, value)) {
                error = "failed to parse ring_bytes_written";
                return false;
            }
            out.ring_bytes_written = value;
        } else if (key == "regions") {
            if (!consume(cursor, '[')) {
                error = "expected '[' for regions";
                return false;
            }
            skip_ws(cursor);
            out.regions.clear();
            if (cursor.pos < text.size() && text[cursor.pos] == ']') {
                ++cursor.pos;
            } else {
                while (cursor.pos < text.size()) {
                    ManifestRegion region{};
                    if (!parse_region(cursor, region, error)) {
                        return false;
                    }
                    out.regions.push_back(region);
                    skip_ws(cursor);
                    if (cursor.pos < text.size() && text[cursor.pos] == ',') {
                        ++cursor.pos;
                        continue;
                    }
                    break;
                }
                if (!consume(cursor, ']')) {
                    error = "expected closing ']' for regions";
                    return false;
                }
            }
        } else {
            if (!skip_value(cursor)) {
                error = "failed to skip checkpoint value";
                return false;
            }
        }
        skip_ws(cursor);
        if (cursor.pos < text.size() && text[cursor.pos] == ',') {
            ++cursor.pos;
            continue;
        }
        if (cursor.pos < text.size() && text[cursor.pos] == '}') {
            ++cursor.pos;
            break;
        }
    }
    if (!has_checkpoint_id) {
        error = "missing checkpoint_id";
        return false;
    }
    return true;
}

static bool parse_manifest_json(const std::string& text, std::vector<ManifestCheckpoint>& out_checkpoints, std::string& error) {
    JsonCursor cursor{&text, 0};
    if (!consume(cursor, '{')) {
        error = "expected '{' at top-level";
        return false;
    }
    bool found_checkpoints = false;
    while (true) {
        skip_ws(cursor);
        if (cursor.pos >= text.size()) {
            error = "unexpected end of manifest";
            return false;
        }
        if (text[cursor.pos] == '}') {
            ++cursor.pos;
            break;
        }
        std::string key;
        if (!parse_string(cursor, key)) {
            error = "failed to parse top-level key";
            return false;
        }
        if (!consume(cursor, ':')) {
            error = "expected ':' after top-level key";
            return false;
        }
        if (key == "checkpoints") {
            if (!consume(cursor, '[')) {
                error = "expected '[' for checkpoints";
                return false;
            }
            skip_ws(cursor);
            out_checkpoints.clear();
            if (cursor.pos < text.size() && text[cursor.pos] == ']') {
                ++cursor.pos;
            } else {
                while (cursor.pos < text.size()) {
                    ManifestCheckpoint checkpoint{};
                    if (!parse_checkpoint(cursor, checkpoint, error)) {
                        return false;
                    }
                    out_checkpoints.push_back(checkpoint);
                    skip_ws(cursor);
                    if (cursor.pos < text.size() && text[cursor.pos] == ',') {
                        ++cursor.pos;
                        continue;
                    }
                    break;
                }
                if (!consume(cursor, ']')) {
                    error = "expected closing ']' for checkpoints";
                    return false;
                }
            }
            found_checkpoints = true;
        } else {
            if (!skip_value(cursor)) {
                error = "failed to skip top-level value";
                return false;
            }
        }
        skip_ws(cursor);
        if (cursor.pos < text.size() && text[cursor.pos] == ',') {
            ++cursor.pos;
            continue;
        }
        if (cursor.pos < text.size() && text[cursor.pos] == '}') {
            ++cursor.pos;
            break;
        }
    }
    if (!found_checkpoints) {
        error = "manifest missing checkpoints";
        return false;
    }
    return true;
}

static std::string read_file_bytes(const char* path) {
    if (!path) {
        return {};
    }
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        return {};
    }
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

static std::string format_hash64(uint64_t value) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << "0x" << std::hex << std::setw(16) << std::setfill('0') << value;
    return out.str();
}

static std::string format_hex_window(const std::vector<uint8_t>& data, size_t start, size_t count) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    for (size_t i = 0; i < count; ++i) {
        if (i > 0) {
            out << " ";
        }
        const uint8_t value = data[start + i];
        out << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(value);
    }
    return out.str();
}

static uint32_t align_up(uint32_t value) {
    return (value + (kRingAlignment - 1u)) & ~(kRingAlignment - 1u);
}

static bool read_chunk_header_host(const uint8_t* ring,
    uint32_t ring_bytes,
    uint32_t ring_offset,
    ChunkHeader& out_header,
    cudaStream_t stream,
    std::string& error) {
    if (ring_offset % kRingAlignment != 0u) {
        error = "ring alignment error";
        return false;
    }
    if (ring_offset + sizeof(ChunkHeader) > ring_bytes) {
        error = "chunk header out of bounds";
        return false;
    }
    cudaError_t err = cudaMemcpyAsync(&out_header,
        ring + ring_offset,
        sizeof(ChunkHeader),
        cudaMemcpyDeviceToHost,
        stream);
    if (err != cudaSuccess) {
        error = "cudaMemcpy header failed";
        return false;
    }
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        error = "cudaStreamSynchronize header failed";
        return false;
    }
    if (out_header.magic != kChunkMagic || out_header.version != kChunkVersion ||
        out_header.header_bytes != kChunkHeaderBytes || out_header.header_bytes != sizeof(ChunkHeader)) {
        error = "invalid chunk header";
        return false;
    }
    if (out_header.chunk_type != kChunkTypeSnapshot &&
        out_header.chunk_type != kChunkTypeDeltaXorRle0 &&
        out_header.chunk_type != kChunkTypeWrapMarker) {
        error = "invalid chunk type";
        return false;
    }
    const uint32_t payload_bytes = out_header.payload_bytes;
    if (payload_bytes > ring_bytes - out_header.header_bytes) {
        error = "payload bytes too large";
        return false;
    }
    if (ring_offset + out_header.header_bytes + payload_bytes > ring_bytes) {
        error = "payload out of bounds";
        return false;
    }
    if (out_header.chunk_type == kChunkTypeWrapMarker) {
        if (payload_bytes != 0u) {
            error = "wrap marker payload non-zero";
            return false;
        }
        return true;
    }
    if (payload_bytes == 0u || (payload_bytes % 4u) != 0u) {
        error = "payload alignment error";
        return false;
    }
    if (out_header.chunk_type == kChunkTypeDeltaXorRle0 &&
        (out_header.uncompressed_bytes == 0u || (out_header.uncompressed_bytes % 4u) != 0u)) {
        error = "delta payload invalid";
        return false;
    }
    return true;
}

static bool read_chunk_payload_host(const uint8_t* ring,
    uint32_t ring_offset,
    uint32_t payload_bytes,
    cudaStream_t stream,
    std::vector<uint8_t>& out_payload,
    std::string& error) {
    out_payload.assign(payload_bytes, 0u);
    if (payload_bytes == 0u) {
        return true;
    }
    cudaError_t err = cudaMemcpyAsync(out_payload.data(),
        ring + ring_offset + sizeof(ChunkHeader),
        payload_bytes,
        cudaMemcpyDeviceToHost,
        stream);
    if (err != cudaSuccess) {
        error = "cudaMemcpy payload failed";
        out_payload.clear();
        return false;
    }
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        error = "cudaStreamSynchronize payload failed";
        out_payload.clear();
        return false;
    }
    return true;
}

static bool apply_delta_payload(const std::vector<uint8_t>& payload, std::vector<uint8_t>& expected, std::string& error) {
    if (payload.size() < sizeof(uint32_t) * 2u) {
        error = "delta payload too small";
        return false;
    }
    if (expected.size() % 4u != 0u) {
        error = "expected buffer alignment error";
        return false;
    }
    const uint32_t* payload_words = reinterpret_cast<const uint32_t*>(payload.data());
    const uint32_t word_count = payload_words[0];
    const uint32_t block_count = payload_words[1];
    const uint32_t expected_words = static_cast<uint32_t>(expected.size() / 4u);
    if (word_count > expected_words) {
        error = "delta word count exceeds region size";
        return false;
    }
    const uint32_t payload_words_count = static_cast<uint32_t>(payload.size() / 4u);
    uint32_t payload_index = 2u;
    uint32_t out_index = 0u;
    uint32_t* expected_words_ptr = reinterpret_cast<uint32_t*>(expected.data());
    for (uint32_t block = 0; block < block_count && out_index < word_count; ++block) {
        if (payload_index >= payload_words_count) {
            error = "delta payload truncated";
            return false;
        }
        const uint32_t tag = payload_words[payload_index++];
        const uint32_t run_len = tag & 0x7FFFFFFFu;
        const bool is_zero = ((tag & 0x80000000u) != 0u);
        if (run_len == 0u) {
            continue;
        }
        if (is_zero) {
            out_index = std::min(out_index + run_len, word_count);
        } else {
            for (uint32_t j = 0; j < run_len && out_index < word_count; ++j) {
                if (payload_index >= payload_words_count) {
                    error = "delta payload truncated";
                    return false;
                }
                const uint32_t xor_value = payload_words[payload_index++];
                expected_words_ptr[out_index] ^= xor_value;
                ++out_index;
            }
        }
    }
    return true;
}

static bool find_region_chunk(const CheckpointRecord& record,
    uint32_t region_id,
    const uint8_t* ring,
    uint32_t ring_bytes,
    cudaStream_t stream,
    ChunkHeader& out_header,
    std::vector<uint8_t>& out_payload,
    std::string& error) {
    uint32_t ring_offset = record.ring_offset;
    uint32_t applied = 0;
    uint32_t guard = 0;
    while (applied < record.chunk_count) {
        if (guard++ > record.chunk_count + 4u) {
            error = "ring corrupt during scan";
            return false;
        }
        ChunkHeader header{};
        if (!read_chunk_header_host(ring, ring_bytes, ring_offset, header, stream, error)) {
            return false;
        }
        if (header.chunk_type == kChunkTypeWrapMarker || (header.flags & kChunkFlagWrapMarker) != 0u) {
            ring_offset = 0u;
            continue;
        }
        if (header.region_id == region_id) {
            out_header = header;
            if (!read_chunk_payload_host(ring, ring_offset, header.payload_bytes, stream, out_payload, error)) {
                return false;
            }
            return true;
        }
        uint32_t chunk_bytes = align_up(static_cast<uint32_t>(sizeof(ChunkHeader)) + header.payload_bytes);
        ring_offset += chunk_bytes;
        if (ring_offset >= ring_bytes) {
            ring_offset -= ring_bytes;
        }
        ++applied;
    }
    return false;
}

static bool build_expected_region_bytes(uint32_t target_checkpoint,
    uint32_t region_id,
    uint32_t region_size_bytes,
    const std::vector<CheckpointRecord>& valid_records,
    const uint8_t* ring,
    uint32_t ring_bytes,
    cudaStream_t stream,
    std::vector<uint8_t>& out_expected,
    std::string& error) {
    bool have_snapshot = false;
    out_expected.assign(region_size_bytes, 0u);
    for (const CheckpointRecord& record : valid_records) {
        if (record.checkpoint_id > target_checkpoint) {
            break;
        }
        ChunkHeader header{};
        std::vector<uint8_t> payload;
        std::string local_error;
        if (!find_region_chunk(record, region_id, ring, ring_bytes, stream, header, payload, local_error)) {
            if (!local_error.empty()) {
                error = local_error;
                return false;
            }
            continue;
        }
        if (header.chunk_type == kChunkTypeSnapshot) {
            if (payload.size() != region_size_bytes) {
                error = "snapshot payload size mismatch";
                return false;
            }
            out_expected = payload;
            have_snapshot = true;
        } else if (header.chunk_type == kChunkTypeDeltaXorRle0) {
            if (!have_snapshot) {
                error = "delta without prior snapshot";
                return false;
            }
            if (!apply_delta_payload(payload, out_expected, error)) {
                return false;
            }
        }
    }
    if (!have_snapshot) {
        error = "no snapshot found before target checkpoint";
        return false;
    }
    return true;
}

__global__ void diff_first_word_kernel(const uint32_t* expected,
    const uint32_t* actual,
    uint32_t word_count,
    uint32_t* out_index) {
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t stride = blockDim.x * gridDim.x;
    for (uint32_t i = idx; i < word_count; i += stride) {
        if (expected[i] != actual[i]) {
            atomicMin(out_index, i);
        }
    }
}

__global__ void xor_byte_kernel(uint8_t* data, uint32_t size_bytes, uint32_t offset, uint8_t mask) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (!data || offset >= size_bytes) {
            return;
        }
        data[offset] ^= mask;
    }
}

static void append_trace_event(TraceCollector* trace, const char* name, double ts_us,
    uint32_t checkpoint_id, uint32_t region_id, uint32_t offset_bytes, bool include_args) {
    if (!trace) {
        return;
    }
    TraceEvent event{};
    event.name = name;
    event.cat = "verify";
    event.ts_us = ts_us;
    event.dur_us = 0.0;
    event.pid = 1;
    event.tid = 0;
    if (include_args) {
        event.args.push_back({"checkpoint_id", std::to_string(checkpoint_id), false});
        event.args.push_back({"region_id", std::to_string(region_id), false});
        event.args.push_back({"offset", std::to_string(offset_bytes), false});
    }
    trace->add_event(event);
}

static void write_report_json(const VerifyReport& report, const char* path, bool include_mismatches) {
    if (!path || !*path) {
        return;
    }
    std::ofstream out(path, std::ios::trunc);
    if (!out.is_open()) {
        return;
    }
    out.imbue(std::locale::classic());
    out << "{";
    out << "\"status\":\"" << report.status << "\"";
    out << ",\"tested_checkpoint_range\":{";
    out << "\"begin\":" << report.tested_checkpoint_begin;
    out << ",\"end\":" << report.tested_checkpoint_end;
    out << "}";
    out << ",\"tested_regions\":[";
    for (size_t i = 0; i < report.tested_regions.size(); ++i) {
        if (i > 0) {
            out << ",";
        }
        out << report.tested_regions[i];
    }
    out << "]";
    out << ",\"first_mismatch\":";
    if (!report.has_mismatch) {
        out << "null";
    } else {
        const VerifyMismatch& mm = report.first_mismatch;
        out << "{";
        out << "\"checkpoint_id\":" << mm.checkpoint_id;
        out << ",\"region_id\":" << mm.region_id;
        out << ",\"expected_hash64\":\"" << format_hash64(mm.expected_hash64) << "\"";
        out << ",\"actual_hash64\":\"" << format_hash64(mm.actual_hash64) << "\"";
        out << ",\"localized\":" << (mm.localized ? "true" : "false");
        if (mm.localized) {
            out << ",\"first_diff_offset_bytes\":" << mm.first_diff_offset_bytes;
            if (!mm.expected_window_hex.empty()) {
                out << ",\"expected_window_hex\":\"" << mm.expected_window_hex << "\"";
            }
            if (!mm.actual_window_hex.empty()) {
                out << ",\"actual_window_hex\":\"" << mm.actual_window_hex << "\"";
            }
        } else if (!mm.localization_error.empty()) {
            out << ",\"localization_error\":\"" << mm.localization_error << "\"";
        }
        out << "}";
    }
    if (include_mismatches) {
        out << ",\"mismatches\":[";
        for (size_t i = 0; i < report.mismatches.size(); ++i) {
            if (i > 0) {
                out << ",";
            }
            const VerifyMismatch& mm = report.mismatches[i];
            out << "{";
            out << "\"checkpoint_id\":" << mm.checkpoint_id;
            out << ",\"region_id\":" << mm.region_id;
            out << ",\"expected_hash64\":\"" << format_hash64(mm.expected_hash64) << "\"";
            out << ",\"actual_hash64\":\"" << format_hash64(mm.actual_hash64) << "\"";
            out << ",\"localized\":" << (mm.localized ? "true" : "false");
            if (mm.localized) {
                out << ",\"first_diff_offset_bytes\":" << mm.first_diff_offset_bytes;
                if (!mm.expected_window_hex.empty()) {
                    out << ",\"expected_window_hex\":\"" << mm.expected_window_hex << "\"";
                }
                if (!mm.actual_window_hex.empty()) {
                    out << ",\"actual_window_hex\":\"" << mm.actual_window_hex << "\"";
                }
            } else if (!mm.localization_error.empty()) {
                out << ",\"localization_error\":\"" << mm.localization_error << "\"";
            }
            out << "}";
        }
        out << "]";
    }
    out << ",\"environment\":{";
    out << "\"device_name\":\"" << report.device_name << "\"";
    out << ",\"compute_capability\":\"" << report.device_cc_major << "." << report.device_cc_minor << "\"";
    out << ",\"driver_version\":" << report.driver_version;
    out << ",\"runtime_version\":" << report.runtime_version;
    out << "}";
    if (!report.error.empty()) {
        out << ",\"error\":\"" << report.error << "\"";
    }
    out << "}";
}

static void fill_device_info(VerifyReport& report) {
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device) == cudaSuccess) {
        report.device_name = prop.name ? prop.name : "";
        report.device_cc_major = prop.major;
        report.device_cc_minor = prop.minor;
    }
    cudaDriverGetVersion(&report.driver_version);
    cudaRuntimeGetVersion(&report.runtime_version);
}

} // namespace

bool Recorder::verify_manifest_json(const char* manifest_path, const VerifyOptions& options, VerifyReport* out_report) {
    VerifyReport local_report{};
    VerifyReport& report = out_report ? *out_report : local_report;
    report = VerifyReport{};
    report.status = "error";
    fill_device_info(report);

    if (!initialized_) {
        report.error = "recorder not initialized";
        write_report_json(report, options.report_path, options.continue_on_mismatch);
        return false;
    }
    const std::string manifest_text = read_file_bytes(manifest_path);
    if (manifest_text.empty()) {
        report.error = "failed to read manifest";
        write_report_json(report, options.report_path, options.continue_on_mismatch);
        return false;
    }
    std::vector<ManifestCheckpoint> manifest_checkpoints;
    std::string parse_error;
    if (!parse_manifest_json(manifest_text, manifest_checkpoints, parse_error)) {
        report.error = parse_error.empty() ? "manifest parse failed" : parse_error;
        write_report_json(report, options.report_path, options.continue_on_mismatch);
        return false;
    }
    if (manifest_checkpoints.empty()) {
        report.error = "manifest contains no checkpoints";
        write_report_json(report, options.report_path, options.continue_on_mismatch);
        return false;
    }

    std::vector<CheckpointRecord> host_checkpoints;
    if (!read_checkpoints_to_host(host_checkpoints)) {
        report.error = "failed to read checkpoints";
        write_report_json(report, options.report_path, options.continue_on_mismatch);
        return false;
    }
    std::vector<CheckpointRecord> valid_records;
    valid_records.reserve(host_checkpoints.size());
    uint32_t min_checkpoint = UINT32_MAX;
    uint32_t max_checkpoint = 0u;
    for (const CheckpointRecord& record : host_checkpoints) {
        if (record.chunk_count == 0u || record.reserved0 == 0u) {
            continue;
        }
        if (record.checkpoint_id < min_valid_checkpoint_) {
            continue;
        }
        const uint32_t expected_generation = record.checkpoint_id / cfg_.checkpoint_capacity;
        if (record.reserved1 != expected_generation) {
            continue;
        }
        valid_records.push_back(record);
        min_checkpoint = std::min(min_checkpoint, record.checkpoint_id);
        max_checkpoint = std::max(max_checkpoint, record.checkpoint_id);
    }
    if (valid_records.empty()) {
        report.error = "no valid checkpoints available in ring";
        write_report_json(report, options.report_path, options.continue_on_mismatch);
        return false;
    }
    std::sort(valid_records.begin(), valid_records.end(),
        [](const CheckpointRecord& a, const CheckpointRecord& b) { return a.checkpoint_id < b.checkpoint_id; });

    std::vector<ManifestCheckpoint> filtered_checkpoints;
    filtered_checkpoints.reserve(manifest_checkpoints.size());
    for (const ManifestCheckpoint& checkpoint : manifest_checkpoints) {
        if (options.checkpoint_range_set) {
            if (checkpoint.checkpoint_id < options.checkpoint_begin || checkpoint.checkpoint_id > options.checkpoint_end) {
                continue;
            }
        }
        filtered_checkpoints.push_back(checkpoint);
    }
    if (filtered_checkpoints.empty()) {
        report.error = "no checkpoints matched the requested range";
        write_report_json(report, options.report_path, options.continue_on_mismatch);
        return false;
    }
    for (const ManifestCheckpoint& checkpoint : filtered_checkpoints) {
        if (checkpoint.checkpoint_id < min_checkpoint || checkpoint.checkpoint_id > max_checkpoint) {
            std::ostringstream msg;
            msg << "manifest checkpoint " << checkpoint.checkpoint_id << " not available; valid range "
                << min_checkpoint << "-" << max_checkpoint;
            report.error = msg.str();
            write_report_json(report, options.report_path, options.continue_on_mismatch);
            return false;
        }
    }

    std::vector<uint32_t> region_filter = options.region_ids;
    if (region_filter.empty()) {
        std::vector<uint32_t> collected;
        for (const ManifestCheckpoint& checkpoint : filtered_checkpoints) {
            for (const ManifestRegion& region : checkpoint.regions) {
                collected.push_back(region.region_id);
            }
        }
        std::sort(collected.begin(), collected.end());
        collected.erase(std::unique(collected.begin(), collected.end()), collected.end());
        region_filter = collected;
    }
    if (region_filter.empty()) {
        report.error = "no regions selected for verification";
        write_report_json(report, options.report_path, options.continue_on_mismatch);
        return false;
    }
    for (uint32_t region_id : region_filter) {
        if (region_id >= cfg_.region_capacity) {
            report.error = "region id out of range";
            write_report_json(report, options.report_path, options.continue_on_mismatch);
            return false;
        }
    }

    report.tested_checkpoint_begin = filtered_checkpoints.front().checkpoint_id;
    report.tested_checkpoint_end = filtered_checkpoints.back().checkpoint_id;
    report.tested_regions = region_filter;

    cudaStream_t stream = options.stream;
    if (!stream) {
        stream = 0;
    }

    uint64_t* d_hashes = nullptr;
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&d_hashes), sizeof(uint64_t) * region_filter.size());
    if (err != cudaSuccess) {
        report.error = "cudaMalloc for hashes failed";
        write_report_json(report, options.report_path, options.continue_on_mismatch);
        return false;
    }

    const auto trace_start = std::chrono::steady_clock::now();
    if (options.trace_annotate) {
        append_trace_event(options.trace, "verify_start", 0.0, 0, 0, 0, false);
    }

    bool any_mismatch = false;
    for (const ManifestCheckpoint& checkpoint : filtered_checkpoints) {
        if (!rewind_to_checkpoint(checkpoint.checkpoint_id, stream)) {
            report.error = "rewind_to_checkpoint failed";
            cudaFree(d_hashes);
            write_report_json(report, options.report_path, options.continue_on_mismatch);
            return false;
        }
        if (options.tamper.enabled && options.tamper.checkpoint_id == checkpoint.checkpoint_id) {
            const uint32_t region_id = options.tamper.region_id;
            if (region_id < cfg_.region_capacity) {
                const TrackedRegion& region = host_regions_[region_id];
                if (region.base_ptr != 0u && region.size_bytes > 0u) {
                    uint8_t* region_ptr = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(region.base_ptr));
                    xor_byte_kernel<<<1, 1, 0, stream>>>(region_ptr, region.size_bytes, options.tamper.byte_offset, options.tamper.xor_mask);
                    cudaStreamSynchronize(stream);
                }
            }
        }
        if (!compute_region_hashes(stream, region_filter, d_hashes)) {
            report.error = "compute_region_hashes failed";
            cudaFree(d_hashes);
            write_report_json(report, options.report_path, options.continue_on_mismatch);
            return false;
        }
        std::vector<uint64_t> host_hashes;
        if (!copy_region_hashes_to_host(stream, d_hashes, region_filter.size(), host_hashes)) {
            report.error = "copy_region_hashes failed";
            cudaFree(d_hashes);
            write_report_json(report, options.report_path, options.continue_on_mismatch);
            return false;
        }

        for (size_t i = 0; i < region_filter.size(); ++i) {
            const uint32_t region_id = region_filter[i];
            const uint64_t actual_hash = host_hashes[i];
            auto manifest_it = std::find_if(checkpoint.regions.begin(), checkpoint.regions.end(),
                [region_id](const ManifestRegion& region) { return region.region_id == region_id; });
            if (manifest_it == checkpoint.regions.end()) {
                report.error = "region missing in manifest checkpoint";
                cudaFree(d_hashes);
                write_report_json(report, options.report_path, options.continue_on_mismatch);
                return false;
            }
            const uint64_t expected_hash = manifest_it->hash64;
            if (actual_hash == expected_hash) {
                continue;
            }
            any_mismatch = true;
            VerifyMismatch mismatch{};
            mismatch.checkpoint_id = checkpoint.checkpoint_id;
            mismatch.region_id = region_id;
            mismatch.expected_hash64 = expected_hash;
            mismatch.actual_hash64 = actual_hash;

            std::vector<uint8_t> expected_bytes;
            std::string localize_error;
            const TrackedRegion& region = host_regions_[region_id];
            if (region.size_bytes == 0u || region.base_ptr == 0u) {
                localize_error = "region not available for localization";
            } else if (!build_expected_region_bytes(checkpoint.checkpoint_id,
                region_id,
                region.size_bytes,
                valid_records,
                d_ring_,
                cfg_.ring_bytes,
                stream,
                expected_bytes,
                localize_error)) {
                if (localize_error.empty()) {
                    localize_error = "localization unavailable";
                }
            } else {
                uint8_t* d_expected = nullptr;
                err = cudaMalloc(reinterpret_cast<void**>(&d_expected), expected_bytes.size());
                if (err != cudaSuccess) {
                    localize_error = "cudaMalloc expected buffer failed";
                } else {
                    err = cudaMemcpyAsync(d_expected, expected_bytes.data(), expected_bytes.size(), cudaMemcpyHostToDevice, stream);
                    if (err != cudaSuccess) {
                        localize_error = "cudaMemcpy expected buffer failed";
                    } else {
                        uint32_t* d_first_index = nullptr;
                        err = cudaMalloc(reinterpret_cast<void**>(&d_first_index), sizeof(uint32_t));
                        if (err != cudaSuccess) {
                            localize_error = "cudaMalloc diff index failed";
                        } else {
                            const uint32_t max_index = std::numeric_limits<uint32_t>::max();
                            err = cudaMemcpyAsync(d_first_index, &max_index, sizeof(uint32_t), cudaMemcpyHostToDevice, stream);
                            if (err != cudaSuccess) {
                                localize_error = "cudaMemcpy diff index failed";
                            } else {
                                const uint32_t word_count = static_cast<uint32_t>(expected_bytes.size() / 4u);
                                const uint32_t blocks = (word_count + kHashThreads - 1u) / kHashThreads;
                                const uint32_t* expected_words = reinterpret_cast<const uint32_t*>(d_expected);
                                const uint32_t* actual_words = reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(region.base_ptr));
                                diff_first_word_kernel<<<blocks, kHashThreads, 0, stream>>>(expected_words, actual_words, word_count, d_first_index);
                                if (cudaGetLastError() != cudaSuccess) {
                                    localize_error = "diff kernel launch failed";
                                } else {
                                    uint32_t first_index = max_index;
                                    err = cudaMemcpyAsync(&first_index, d_first_index, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
                                    if (err != cudaSuccess) {
                                        localize_error = "cudaMemcpy diff index result failed";
                                    } else if (cudaStreamSynchronize(stream) != cudaSuccess) {
                                        localize_error = "cudaStreamSynchronize diff failed";
                                    } else if (first_index == max_index || first_index >= word_count) {
                                        localize_error = "no differences found for localization";
                                    } else {
                                        uint32_t expected_word = reinterpret_cast<const uint32_t*>(expected_bytes.data())[first_index];
                                        uint32_t actual_word = 0u;
                                        err = cudaMemcpy(&actual_word,
                                            reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(region.base_ptr)) + first_index,
                                            sizeof(uint32_t),
                                            cudaMemcpyDeviceToHost);
                                        if (err != cudaSuccess) {
                                            localize_error = "cudaMemcpy actual word failed";
                                        } else {
                                            uint32_t diff_byte = 0u;
                                            const uint32_t xor_word = expected_word ^ actual_word;
                                            for (uint32_t b = 0; b < 4u; ++b) {
                                                if ((xor_word >> (b * 8u)) & 0xFFu) {
                                                    diff_byte = b;
                                                    break;
                                                }
                                            }
                                            const uint32_t offset_bytes = first_index * 4u + diff_byte;
                                            mismatch.localized = true;
                                            mismatch.first_diff_offset_bytes = offset_bytes;
                                            const uint32_t window_before = 8u;
                                            const uint32_t window_after = 8u;
                                            const uint32_t total = static_cast<uint32_t>(expected_bytes.size());
                                            const uint32_t start = (offset_bytes > window_before) ? (offset_bytes - window_before) : 0u;
                                            const uint32_t end = std::min(total, offset_bytes + window_after);
                                            const uint32_t window_size = (end > start) ? (end - start) : 0u;
                                            std::vector<uint8_t> actual_window(window_size, 0u);
                                            if (window_size > 0u) {
                                                err = cudaMemcpy(actual_window.data(),
                                                    reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(region.base_ptr)) + start,
                                                    window_size,
                                                    cudaMemcpyDeviceToHost);
                                                if (err == cudaSuccess) {
                                                    mismatch.expected_window_hex = format_hex_window(expected_bytes, start, window_size);
                                                    mismatch.actual_window_hex = format_hex_window(actual_window, 0u, window_size);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            cudaFree(d_first_index);
                        }
                    }
                    cudaFree(d_expected);
                }
            }
            if (!mismatch.localized) {
                mismatch.localization_error = localize_error.empty() ? "localization unavailable" : localize_error;
            }

            if (options.trace_annotate) {
                const auto now = std::chrono::steady_clock::now();
                const double ts_us = std::chrono::duration<double, std::micro>(now - trace_start).count();
                append_trace_event(options.trace, "verify_mismatch", ts_us, checkpoint.checkpoint_id, region_id,
                    mismatch.localized ? mismatch.first_diff_offset_bytes : 0u, true);
            }

            if (!report.has_mismatch) {
                report.has_mismatch = true;
                report.first_mismatch = mismatch;
            }
            if (options.continue_on_mismatch) {
                report.mismatches.push_back(mismatch);
            }
            if (!options.continue_on_mismatch) {
                break;
            }
        }
        if (any_mismatch && !options.continue_on_mismatch) {
            break;
        }
    }

    cudaFree(d_hashes);
    report.status = any_mismatch ? "fail" : "pass";
    write_report_json(report, options.report_path, options.continue_on_mismatch);
    return !any_mismatch;
}

} // namespace tt
